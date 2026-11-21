#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""read_blackdecline_v1.py -- registered reader for BLACK-VIDEO products
(12p4n ... --video-op black; also accepts 12p6n rows).

Why: parse_rate is not a refusal rate.  In these products an unparsed row can be (a) an explicit refusal,
(b) a degenerate/gibberish generation, (c) a truncated answer, or (d) an empty generation; and a PARSED row
can still carry a refusal sentence with a placeholder window stuffed into <answer> (verdict C4).  This reader
decomposes the black column into those buckets under two refusal word lists (wide / strict), reports the
accuracy that survives conditioning on an answer, measures how stereotyped the predictions are, and splits
the blind score into a duration/position PRIOR component and a QUERY-SPECIFIC component (verdict C13).

It also provides the no-model constant-window blind baseline for the same test set (--blind-baseline),
because every black R@0.5 must be read against it.

usage:
  read_blackdecline_v1.py --label L --files black.json [--clean clean.json] [--blind-baseline]
                          [--train-anno charades_sta_train.txt] [--B 2000] [--perm 200] --out out.json
  read_blackdecline_v1.py --selftest
"""
import argparse, json, hashlib, os, sys, re, random, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_FROZEN = os.path.join(HERE, "read_p4nat_v1.py")
_SEL = os.path.join(HERE, "read_p4nat_selective_v1.py")


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


V1 = _imp("read_p4nat_v1_frozen", _FROZEN)
SEL = _imp("read_p4nat_selective_v1_reader", _SEL)
iou_1d = V1.iou_1d
load_rows = V1.load_rows

# ------------------------------------------------------------------ refusal word lists (both reported)
# STRICT: an explicit inability statement bound to a grounding verb, or one of the verbatim refusal forms.
# WIDE  : STRICT plus the softer hedges ("please provide", "does not contain", "sorry", ...).
# The lists live here at the top of the file on purpose -- they are part of the reported result.
REFUSE_STRICT = SEL.REFUSE_STRICT
REFUSE_WIDE = re.compile(SEL.REFUSE_WIDE.pattern.replace(r"n/?a\b", r"\bn/?a\b"), re.I)
is_gibberish = SEL.is_gibberish
is_truncated = SEL.is_truncated
classify_abstain = SEL.classify_abstain


def clusters(vid):
    order = sorted(set(vid))
    ix = {v: i for i, v in enumerate(order)}
    g = [[] for _ in order]
    for i, v in enumerate(vid):
        g[ix[v]].append(i)
    return [np.array(x) for x in g]


def boot_mean(flag, groups, B, seed):
    rng = np.random.default_rng(seed)
    G = len(groups)
    d = []
    for _ in range(B):
        idx = np.concatenate([groups[k] for k in rng.integers(0, G, G)])
        d.append(flag[idx].mean())
    d = np.asarray(d, float)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def iou_vec(s, e, gs, ge):
    inter = np.maximum(0.0, np.minimum(e, ge) - np.maximum(s, gs))
    uni = np.maximum(e, ge) - np.minimum(s, gs)
    return np.where(uni > 0, inter / np.maximum(uni, 1e-9), 0.0)


# ------------------------------------------------------------------ constant-window blind baseline
def _grid(dur_max, step=0.5):
    """search grid for the constant window: start in [0, dur_max], length in (0, 0.9*dur_max].
    The step is fixed at 0.5 s so the grid does not depend on the dataset (C13 used the same)."""
    a = np.arange(0.0, dur_max + step, step)
    L = np.arange(step, 0.9 * dur_max + step, step)
    return a, L


def best_const(gs, ge, dur_max):
    a_g, L_g = _grid(dur_max)
    best = (-1.0, None, None)
    for a in a_g:
        for L in L_g:
            r = float((iou_vec(a, a + L, gs, ge) >= 0.5).mean())
            if r > best[0]:
                best = (r, float(a), float(a + L))
    return best


def blind_baseline(rows, seed=0, n_random=200, train_anno=None, fixed=((0, 5), (0, 6), (0, 10))):
    gs = np.array([float(r["gsA"]) for r in rows]); ge = np.array([float(r["geA"]) for r in rows])
    vid = [str(r["video"]) for r in rows]
    dur_max = float(max(float(r["dur"]) for r in rows))
    out = {"n": len(rows), "dur_max": dur_max, "grid_step": 0.5}
    # 2-fold cross-fit BY VIDEO on the test set itself (honest: the window is never fit on the rows it scores)
    vs = sorted(set(vid)); rnd = random.Random(seed); rnd.shuffle(vs)
    half = set(vs[:len(vs) // 2])
    m = np.array([v in half for v in vid])
    bA = best_const(gs[m], ge[m], dur_max); bB = best_const(gs[~m], ge[~m], dur_max)
    tB = float((iou_vec(bA[1], bA[2], gs[~m], ge[~m]) >= 0.5).mean())
    tA = float((iou_vec(bB[1], bB[2], gs[m], ge[m]) >= 0.5).mean())
    pooled = (tB * int((~m).sum()) + tA * int(m.sum())) / len(rows)
    out["crossfit_2fold_by_video"] = {"fitA_window": [bA[1], bA[2]], "testB_R@0.5": tB,
                                      "fitB_window": [bB[1], bB[2]], "testA_R@0.5": tA,
                                      "pooled_R@0.5": float(pooled), "seed": seed}
    ob = best_const(gs, ge, dur_max)
    out["oracle_in_test_upper_bound"] = {"window": [ob[1], ob[2]], "R@0.5": ob[0]}
    out["fixed_windows"] = {"(%g,%g)" % (a, b): {"R@0.5": float((iou_vec(a, b, gs, ge) >= 0.5).mean()),
                                                 "mIoU": float(iou_vec(a, b, gs, ge).mean())} for a, b in fixed}
    # negative control: windows drawn at random from the same grid, fit to nothing
    a_g, L_g = _grid(dur_max)
    rng = np.random.default_rng(seed); d = []
    for _ in range(n_random):
        a = float(rng.choice(a_g)); L = float(rng.choice(L_g))
        d.append(float((iou_vec(a, a + L, gs, ge) >= 0.5).mean()))
    d = np.asarray(d, float)
    out["random_window_control"] = {"draws": n_random, "R@0.5_mean": float(d.mean()),
                                    "R@0.5_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                    "R@0.5_max": float(d.max())}
    if train_anno and os.path.exists(train_anno):
        tg, tv = [], []
        for ln in open(train_anno, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            head = ln.split("##")[0].split()
            if len(head) < 3:
                continue
            tv.append(head[0]); tg.append((float(head[1]), float(head[2])))
        if tg:
            tgs = np.array([x[0] for x in tg]); tge = np.array([x[1] for x in tg])
            bt = best_const(tgs, tge, dur_max)
            out["train_fit"] = {"anno": os.path.basename(train_anno), "anno_sha16": sha16(train_anno),
                                "n_train": len(tg), "n_train_videos": len(set(tv)),
                                "window": [bt[1], bt[2]], "train_R@0.5": bt[0],
                                "test_R@0.5": float((iou_vec(bt[1], bt[2], gs, ge) >= 0.5).mean())}
    else:
        out["train_fit"] = {"reason": "train annotation file not given or not found (absent, not zero)",
                            "path_tried": train_anno}
    return out


# ------------------------------------------------------------------ blind score decomposition
def decompose(rows, perm=200, seed=0):
    """Within-duration-stratum permutation of the predicted windows.  The permuted score is what a predictor
    that knows only the video duration (and the marginal distribution of emitted windows) would get; the
    residual is the part of the blind score that actually tracks the query."""
    n = len(rows)
    s = np.array([float(r["pred"][0]) if r.get("pred") else np.nan for r in rows])
    e = np.array([float(r["pred"][1]) if r.get("pred") else np.nan for r in rows])
    gs = np.array([float(r["gsA"]) for r in rows]); ge = np.array([float(r["geA"]) for r in rows])
    dur = np.array([float(r["dur"]) for r in rows])
    m = ~np.isnan(s)
    if m.sum() < 5:
        return {"reason": "fewer than 5 parsed rows: decomposition not defined (absent, not zero)",
                "n_parsed": int(m.sum()), "R@0.5": float((iou_vec(np.nan_to_num(s), np.nan_to_num(e), gs, ge) >= 0.5).mean() if m.sum() else 0.0)}
    S, E, GS, GE, D = s[m], e[m], gs[m], ge[m], dur[m]
    total = float((iou_vec(S, E, GS, GE) >= 0.5).sum() / n)
    strata = np.round(D).astype(int)
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(perm):
        Sp, Ep = S.copy(), E.copy()
        for v in np.unique(strata):
            i = np.where(strata == v)[0]
            if len(i) > 1:
                j = rng.permutation(i)
                Sp[i] = S[j]; Ep[i] = E[j]
        res.append((iou_vec(Sp, Ep, GS, GE) >= 0.5).sum() / n)
    res = np.asarray(res, float)
    sd = float(res.std())
    return {"perm": perm, "seed": seed, "n_parsed": int(m.sum()),
            "R@0.5": total,
            "prior_component": float(res.mean()), "prior_sd": sd,
            "prior_ci95": [float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))],
            "query_component": float(total - res.mean()),
            "query_ci95": [float(total - np.percentile(res, 97.5)), float(total - np.percentile(res, 2.5))],
            "query_sigma": float((total - res.mean()) / sd) if sd > 0 else None,
            "n_duration_strata": int(len(np.unique(strata)))}


# ------------------------------------------------------------------ main read
def read_black(rows, B=2000, seed=0, perm=200):
    n = len(rows)
    iou = np.array([iou_1d(r.get("pred"), (r["gsA"], r["geA"])) for r in rows])
    parsed = np.array([1 if r.get("pred") else 0 for r in rows], dtype=bool)
    raws = [(r.get("raw") or "") for r in rows]
    vid = [str(r["video"]) for r in rows]
    g = clusters(vid)
    out = {"n": n, "n_videos": len(g), "parse_rate": float(parsed.mean()),
           "R@0.5": float((iou >= 0.5).mean()), "R@0.3": float((iou >= 0.3).mean()),
           "R@0.7": float((iou >= 0.7).mean()), "mIoU": float(iou.mean())}
    out["R@0.5_parsed"] = float((iou[parsed] >= 0.5).mean()) if parsed.sum() else None
    out["mIoU_parsed"] = float(iou[parsed].mean()) if parsed.sum() else None
    if not parsed.sum():
        out["R@0.5_parsed_reason"] = "parse_rate 0: no parsed row (absent, not zero)"
    for tag, rx in (("wide", REFUSE_WIDE), ("strict", REFUSE_STRICT)):
        any_r = np.array([bool(rx.search(t)) for t in raws])
        hard = any_r & (~parsed)
        plc = any_r & parsed
        out[tag] = {
            "any_refusal": float(any_r.mean()), "any_refusal_ci95": boot_mean(any_r.astype(float), g, B, seed),
            "hard_refusal": float(hard.mean()), "hard_refusal_ci95": boot_mean(hard.astype(float), g, B, seed),
            "placeholder_refusal": float(plc.mean()),
            "unparsed_without_refusal_language": float(((~parsed) & (~any_r)).mean()),
        }
    trunc = np.array([is_truncated(t) for t in raws])
    gib = np.array([is_gibberish(t) for t in raws])
    out["truncation_heuristic"] = {"rate": float(trunc.mean()), "rate_unparsed_only": float((trunc & ~parsed).mean()),
                                   "rule": "no </answer> anywhere AND last non-space char is not terminal punctuation"}
    out["gibberish"] = {"rate": float(gib.mean()), "rate_unparsed_only": float((gib & ~parsed).mean())}
    out["empty_generation_rate"] = float(np.mean([1.0 if t.strip() == "" else 0.0 for t in raws]))
    # exclusive taxonomy over the unparsed rows (same buckets as read_p4nat_selective_v1)
    wb, sb, ex = {}, {}, {}
    for t, p in zip(raws, parsed):
        if p:
            continue
        w, s = classify_abstain(t)
        wb[w] = wb.get(w, 0) + 1; sb[s] = sb.get(s, 0) + 1
        ex.setdefault(w, t[:160])
    nun = int((~parsed).sum())
    out["unparsed_taxonomy"] = {"n_unparsed": nun, "wide": wb, "strict": sb, "examples": ex}
    # stereotypy of the emitted windows
    from collections import Counter
    wins = Counter((round(float(r["pred"][0]), 2), round(float(r["pred"][1]), 2)) for r in rows if r.get("pred"))
    out["stereotypy"] = {"n_distinct_windows": len(wins),
                         "top1_window": list(wins.most_common(1)[0][0]) if wins else None,
                         "top1_share_of_parsed": (wins.most_common(1)[0][1] / max(sum(wins.values()), 1)) if wins else None,
                         "top5": [[list(k), v] for k, v in wins.most_common(5)]}
    out["decomposition"] = decompose(rows, perm=perm, seed=seed)
    out["bootstrap"] = {"B": B, "seed": seed, "cluster": "video"}
    out["styles"] = sorted({str(r.get("style")) for r in rows})
    out["text_first"] = sorted({str(r.get("text_first")) for r in rows})
    out["video_op"] = sorted({str(r.get("video_op")) for r in rows})
    return out


def retention(black, clean):
    qb = [int(r["qid"]) for r in black]; qc = [int(r["qid"]) for r in clean]
    if qb != qc:
        return {"reason": "clean product has a different qid set; no paired retention (absent, not zero)"}
    # a retention ratio across two different prompt templates or two different message orders is not a
    # retention ratio -- refuse it rather than print a number (this guard has bitten before).
    sb = sorted({str(r.get("style")) for r in black}); sc = sorted({str(r.get("style")) for r in clean})
    tb = sorted({str(r.get("text_first")) for r in black}); tc = sorted({str(r.get("text_first")) for r in clean})
    if sb != sc or tb != tc:
        return {"reason": "clean counterpart differs in prompt style or message order; not a retention pair "
                          "(absent, not zero)", "black_style": sb, "clean_style": sc,
                "black_text_first": tb, "clean_text_first": tc}
    ib = np.array([iou_1d(r.get("pred"), (r["gsA"], r["geA"])) for r in black])
    ic = np.array([iou_1d(r.get("pred"), (r["gsA"], r["geA"])) for r in clean])
    pb = np.array([1 if r.get("pred") else 0 for r in black], dtype=float)
    pc = np.array([1 if r.get("pred") else 0 for r in clean], dtype=float)
    hb = ib >= 0.5; hc = ic >= 0.5
    r5c = float(hc.mean())
    return {"clean_R@0.5": r5c, "black_R@0.5": float(hb.mean()),
            "retention_R@0.5": float(hb.mean() / r5c) if r5c > 0 else None,
            "clean_parse_rate": float(pc.mean()), "black_parse_rate": float(pb.mean()),
            "retention_parse": float(pb.mean() / pc.mean()) if pc.mean() > 0 else None,
            "of_clean_correct_also_black_correct": float((hb & hc).sum() / hc.sum()) if hc.sum() else None,
            "black_correct_but_clean_wrong_n": int((hb & ~hc).sum())}


# ------------------------------------------------------------------ selftest
def _mk(q, v, pred, gs, ge, raw="", dur=30.0):
    return {"set": "orig", "qid": q, "video": v, "caption": "c", "dur": dur, "pred": pred,
            "gsA": gs, "geA": ge, "gsB": None, "geB": None, "raw": raw, "style": "timer1_default",
            "fps": 2.0, "video_op": "black"}


def selftest():
    ok = True

    def chk(c, m):
        nonlocal ok
        ok &= bool(c)
        print(("  PASS " if c else "  FAIL ") + m)

    rnd = random.Random(0)
    base = []
    for i in range(400):
        d = 20.0 + (i % 5) * 5.0
        gsv = round(rnd.uniform(0, d - 6), 1); gev = round(gsv + rnd.uniform(2, 10), 1)
        base.append(_mk(i, "v%d" % (i % 60), None, gsv, min(gev, d), "", d))

    # ---- POSITIVE CONTROL 1: every prediction is one constant window -> query component ~ 0
    const = [dict(r) for r in base]
    for r in const:
        r["pred"] = [0.0, 5.0]; r["raw"] = "<answer>0.00 to 5.00</answer>"
    dc = decompose(const, perm=200)
    chk(abs(dc["query_component"]) < 1e-9,
        "POS: a single constant window carries no query-specific component (exactly 0)")
    chk(dc["prior_component"] > 0.0, "POS: ... while its prior component is the non-zero blind score")

    # ---- POSITIVE CONTROL 2: refusal sentences replaced by numbers -> hard refusal 0
    ref = [dict(r) for r in base]
    for i, r in enumerate(ref):
        r["raw"] = "The video is completely black, so it is not possible to pinpoint the moment."
    a = read_black(ref, B=100, perm=20)
    chk(abs(a["wide"]["hard_refusal"] - 1.0) < 1e-12 and abs(a["parse_rate"]) < 1e-12,
        "POS: unparsed rows carrying refusal text are 100% hard refusals")
    num = [dict(r) for r in ref]
    for r in num:
        r["pred"] = [0.0, 5.0]; r["raw"] = "<answer>0.00 to 5.00</answer>"
    b = read_black(num, B=100, perm=20)
    chk(b["wide"]["hard_refusal"] == 0.0 and b["wide"]["any_refusal"] == 0.0,
        "POS: replacing every refusal sentence with a number drives hard refusal and any-refusal to 0")

    # ---- NEGATIVE CONTROL 1: pred == GT -> query component is maximal
    gt = [dict(r) for r in base]
    for r in gt:
        r["pred"] = [r["gsA"], r["geA"]]; r["raw"] = "<answer>ok</answer>"
    dg = decompose(gt, perm=200)
    chk(abs(dg["R@0.5"] - 1.0) < 1e-12, "NEG: pred==GT scores R@0.5 = 1")
    chk(dg["query_component"] > dc["query_component"] + 0.5,
        "NEG: pred==GT gives a far larger query component than the constant window")
    chk(dg["query_sigma"] is not None and dg["query_sigma"] > 5, "NEG: ... and it is many sigma above the permuted null")

    # ---- NEGATIVE CONTROL 2: a text-first-style base that never refuses must score exactly 0.0000
    tf = [dict(r) for r in base]
    for r in tf:
        r["pred"] = [0.0, 5.0]; r["raw"] = "<answer>\n0.00 to 5.00\n</answer>"
    t = read_black(tf, B=100, perm=20)
    chk(t["wide"]["any_refusal"] == 0.0 and t["strict"]["any_refusal"] == 0.0,
        "NEG: a 0-refusal synthetic base scores exactly 0.0000 on BOTH word lists (no word-list false positive)")
    chk(t["truncation_heuristic"]["rate"] == 0.0 and t["gibberish"]["rate"] == 0.0,
        "NEG: ... and neither the truncation nor the gibberish heuristic fires on it")
    chk(t["stereotypy"]["n_distinct_windows"] == 1 and abs(t["stereotypy"]["top1_share_of_parsed"] - 1.0) < 1e-12,
        "NEG: ... and its stereotypy is 1 window at 100% share")

    # ---- wide vs strict really differ, and placeholder refusals are separated from hard ones
    mix = [dict(r) for r in base]
    for i, r in enumerate(mix):
        if i % 2:
            r["pred"] = [0.0, 6.0]
            r["raw"] = "Please provide a longer video clip to analyze.\n<answer>The given query happens in 0.00 to 6.00 seconds.</answer>"
        else:
            r["pred"] = None
            r["raw"] = "I cannot determine the moment from this video."
    mm = read_black(mix, B=100, perm=20)
    chk(abs(mm["wide"]["placeholder_refusal"] - 0.5) < 1e-9 and abs(mm["wide"]["hard_refusal"] - 0.5) < 1e-9,
        "TAXONOMY: placeholder refusals (parsed + refusal text) are not counted as hard refusals")
    chk(mm["strict"]["placeholder_refusal"] == 0.0 and abs(mm["strict"]["hard_refusal"] - 0.5) < 1e-9,
        "TAXONOMY: the strict list drops the 'please provide' placeholder, the wide list keeps it")

    # ---- constant-window baseline: the cross-fit number must not beat the in-test oracle
    bb = blind_baseline(base, n_random=50)
    chk(bb["crossfit_2fold_by_video"]["pooled_R@0.5"] <= bb["oracle_in_test_upper_bound"]["R@0.5"] + 1e-12,
        "BASELINE: the honest 2-fold cross-fit never exceeds the in-test oracle upper bound")
    chk(bb["random_window_control"]["R@0.5_mean"] < bb["oracle_in_test_upper_bound"]["R@0.5"],
        "BASELINE: random windows (fit to nothing) score below the fitted constant window")
    chk("reason" in bb["train_fit"], "BASELINE: a missing train annotation is reported as a reason, not as 0")

    # ---- guard: parse_rate 0 must not silently report conditional accuracy 0
    z = read_black(ref, B=50, perm=20)
    chk(z["R@0.5_parsed"] is None and "R@0.5_parsed_reason" in z,
        "GUARD: parse 0 -> R@0.5|parsed is null with a reason, never 0.0")
    chk("reason" in z["decomposition"], "GUARD: parse 0 -> decomposition reports a reason, never 0")

    print("[selftest]", "ALL PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label"); ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--clean", default=None)
    ap.add_argument("--blind-baseline", action="store_true")
    ap.add_argument("--train-anno", default="<data_root>/charades_sta/anno/charades_sta_train.txt")
    ap.add_argument("--expect-n", type=int, default=None)
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--random-draws", type=int, default=200)
    ap.add_argument("--note", default=None)
    ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    rows = load_rows(a.files)
    if a.expect_n is not None and len(rows) != a.expect_n:
        raise SystemExit("expected n=%d, got %d" % (a.expect_n, len(rows)))
    res = read_black(rows, B=a.B, seed=a.seed, perm=a.perm)
    files = list(a.files)
    if a.clean:
        cl = load_rows([a.clean])
        res["retention"] = retention(rows, cl)
        res["retention"]["clean_file"] = os.path.basename(a.clean)
        res["retention"]["clean_sha16"] = sha16(a.clean)
        files = files + [a.clean]
    else:
        res["retention"] = {"reason": "no clean counterpart given (absent, not zero)"}
    if a.blind_baseline:
        res["blind_baseline"] = blind_baseline(rows, seed=a.seed, n_random=a.random_draws, train_anno=a.train_anno)
    res["label"] = a.label; res["note"] = a.note; res["mode"] = "black"
    res["inputs"] = [{"file": os.path.basename(f), "sha16": sha16(f)} for f in files]
    res["reader"] = "read_blackdecline_v1"
    res["reader_sha16"] = sha16(os.path.abspath(__file__))
    res["frozen_reader"] = {"file": "read_p4nat_v1.py", "sha16": sha16(_FROZEN), "used_for": "iou_1d, load_rows"}
    res["sibling_reader"] = {"file": "read_p4nat_selective_v1.py", "sha16": sha16(_SEL),
                             "used_for": "refusal word lists, gibberish/truncation heuristics, abstention taxonomy"}
    res["refusal_lists"] = {"wide": REFUSE_WIDE.pattern, "strict": REFUSE_STRICT.pattern}
    brief = {"label": a.label, "n": res["n"], "parse": res["parse_rate"],
             "hard_refusal_wide": res["wide"]["hard_refusal"], "any_refusal_wide": res["wide"]["any_refusal"],
             "any_refusal_strict": res["strict"]["any_refusal"],
             "R@0.5": res["R@0.5"], "R@0.5|parsed": res["R@0.5_parsed"],
             "prior": res["decomposition"].get("prior_component"), "query": res["decomposition"].get("query_component"),
             "retention_R@0.5": res["retention"].get("retention_R@0.5")}
    print(json.dumps(brief, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
