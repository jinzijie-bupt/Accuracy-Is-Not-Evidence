#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""read_p4nat_selective_v1.py -- registered reader: selective prediction on native-protocol
grounding products (12p4n Charades-STA n=3720 / 12p6n QVH n=1541 / 12p6n chtl n=3363).

Why: the frozen reader analysis/read_p4nat_v1.py scores unparsed rows as IoU=0, so "abstained" and "answered
wrong" collapse into one column (verdicts C6/C1).  This reader keeps that marginal number untouched (it IMPORTS
iou_1d / metrics / load_rows from the frozen file and records its sha16; it never edits it) and adds the
selective-prediction axis: coverage, conditional accuracy on the answered subset, the abstention-string
taxonomy, predicted-span median, GT-length-tertile strata, and (in --pair mode) the paired reference contrasts.

usage:
  single  : read_p4nat_selective_v1.py --label L --files f1.json [f2 ...] [--expect-n N] --out out.json
  paired  : read_p4nat_selective_v1.py --label L --pair REF.json ARM.json [--B 4000] [--seed 0] --out out.json
  selftest: read_p4nat_selective_v1.py --selftest

Conventions kept identical to read_p4nat_v1: unparsed row -> IoU 0 for the *marginal* numbers; video-cluster
bootstrap; percentile CIs.  Conditional numbers are computed on the answered subset only (pred non-empty).
"""
import argparse, json, hashlib, os, sys, re, random, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_FROZEN = os.path.join(HERE, "read_p4nat_v1.py")


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def _load_frozen():
    spec = importlib.util.spec_from_file_location("read_p4nat_v1_frozen", _FROZEN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


V1 = _load_frozen()
iou_1d = V1.iou_1d
load_rows = V1.load_rows
metrics = V1.metrics

# ---------------------------------------------------------------- abstention string taxonomy
# Two word lists are reported side by side (wide / strict).  STRICT demands an explicit inability
# statement bound to a grounding verb (or one of three verbatim refusal forms); WIDE adds the softer
# hedges that also appear in these products.  Both are applied to the whole raw string, case-insensitive.
REFUSE_STRICT = re.compile(
    r"(?:\b(?:cannot|can\s?not|can't|unable|not\s+possible|impossible|no\s+way)\b[^.]{0,70}?"
    r"\b(?:determin|identif|pinpoint|provid|answer|tell|analyz|extract|locat|specif|find|detect|ground)\w*)"
    r"|(?:\bthere\s+are\s+none\b)"
    r"|(?:\bno\s+visual\s+(?:information|content|cues?|details?)\b)"
    r"|(?:\bnot\s+possible\s+to\s+pinpoint\b)"
    r"|(?:\bthere\s+(?:are|is)\s+no\s+(?:events?|such\s+\w+|matching\s+\w+)\b)", re.I)
REFUSE_WIDE = re.compile(
    r"(cannot|can\s?not|can't|unable|not\s+possible|impossible|no\s+visual|no\s+video|no\s+image|"
    r"please\s+provide|please\s+upload|not\s+applicable|does\s+not\s+contain|no\s+event|insufficient|"
    r"there\s+is\s+no|there\s+are\s+none|no\s+discernible|completely\s+black|entirely\s+black|"
    r"sorry|n/?a\b|without\s+(?:any\s+)?visual|no\s+content|does\s+not\s+match|please\s+check|"
    r"let\s+me\s+know|provide\s+more\s+details|there\s+are\s+no\s+events?|too\s+short\s+to|"
    r"more\s+frames|additional\s+context|please\s+share|based\s+(?:solely\s+)?on\s+this\s+image\s+alone)", re.I)

_ANS = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_TERM = set(".!?\"'>)]}\u3002\uff01\uff1f")


def _inner(raw):
    """content of the last <answer> block, else the whole string"""
    ms = _ANS.findall(raw or "")
    return ms[-1] if ms else (raw or "")


def is_gibberish(raw):
    """degenerate/junk generation.  Three disjunctive signals, all seen in these products:
       (a) a whitespace token repeated >=4x and making up >=25% of the tokens;
       (b) a long run of blank lines with almost no distinct tokens (the 'addCriterion\\n ...' collapse);
       (c) a long string whose token diversity is below 0.30."""
    s = raw or ""
    t = s.split()
    from collections import Counter
    if len(t) >= 6:
        c = Counter(t).most_common(1)[0][1]
        if c >= 4 and c / len(t) >= 0.25:
            return True
    if s.count("\n") >= 8 and 0 < len(set(t)) <= 4:
        return True
    if len(t) >= 8 and len(set(t)) / len(t) < 0.30:
        return True
    return False


def is_truncated(raw):
    """C4's heuristic: no closing </answer> anywhere AND the last non-space char is not terminal punctuation"""
    s = (raw or "").rstrip()
    if not s:
        return False
    return ("</answer>" not in s.lower()) and (s[-1] not in _TERM)


def classify_abstain(raw):
    """mutually exclusive bucket for a row with pred=None.  Returns (wide_bucket, strict_bucket)."""
    s = (raw or "").strip()
    inner = _inner(s).strip()
    low = re.sub(r"\s+", " ", inner.lower()).strip().rstrip(".")
    if low == "there are none":
        b = "literal_there_are_none"
    elif s == "":
        b = "empty_output"
    elif inner == "":
        b = "empty_answer_tag"
    elif REFUSE_WIDE.search(s):
        b = "nl_refusal"
    elif is_gibberish(raw):
        b = "gibberish"
    else:
        b = "other"
    if low == "there are none":
        bs = "literal_there_are_none"
    elif s == "":
        bs = "empty_output"
    elif inner == "":
        bs = "empty_answer_tag"
    elif REFUSE_STRICT.search(s):
        bs = "nl_refusal"
    elif is_gibberish(raw):
        bs = "gibberish"
    else:
        bs = "other"
    return b, bs


# ---------------------------------------------------------------- core vectors
def vecs(rows):
    n = len(rows)
    iou = np.array([iou_1d(r.get("pred"), (r["gsA"], r["geA"])) for r in rows])
    ans = np.array([1 if r.get("pred") else 0 for r in rows], dtype=bool)
    gl = np.array([float(r["geA"]) - float(r["gsA"]) for r in rows])
    span = np.array([(float(r["pred"][1]) - float(r["pred"][0])) if r.get("pred") else np.nan for r in rows])
    vid = [str(r["video"]) for r in rows]
    return dict(n=n, iou=iou, ans=ans, gl=gl, span=span, vid=vid)


def clusters(vid):
    order = sorted(set(vid))
    idx = {v: i for i, v in enumerate(order)}
    g = [[] for _ in order]
    for i, v in enumerate(vid):
        g[idx[v]].append(i)
    return [np.array(x) for x in g]


def boot_idx(groups, B, seed):
    rng = np.random.default_rng(seed)
    G = len(groups)
    for _ in range(B):
        yield np.concatenate([groups[k] for k in rng.integers(0, G, G)])


def two_sided_p(draws, null=0.0):
    d = np.asarray(draws, dtype=float)
    lo = float((d <= null).mean()); hi = float((d >= null).mean())
    return float(min(1.0, 2.0 * min(lo, hi)))


def tertile_table(iou, gl, ans, cuts):
    """R@0.5 by GT-length tertile (fixed cut points, default 8 s / 25 s; C1 uses the same on QVH)"""
    lab = ["short(<=%g)" % cuts[0], "mid(%g-%g]" % (cuts[0], cuts[1]), "long(>%g)" % cuts[1]]
    b = np.digitize(gl, cuts, right=True)
    out = {}
    for k, name in enumerate(lab):
        m = b == k
        if m.sum() == 0:
            out[name] = {"n": 0, "R@0.5": None, "R@0.5_answered": None, "coverage": None, "reason": "empty stratum"}
            continue
        ma = m & ans
        out[name] = {"n": int(m.sum()),
                     "R@0.5": float((iou[m] >= 0.5).mean()),
                     "coverage": float(ans[m].mean()),
                     "R@0.5_answered": float((iou[ma] >= 0.5).mean()) if ma.sum() else None,
                     "n_answered": int(ma.sum())}
    return out


def read_one(rows, B=4000, seed=0, cuts=(8.0, 25.0)):
    v = vecs(rows)
    n, iou, ans = v["n"], v["iou"], v["ans"]
    na = int(ans.sum())
    m = metrics(rows, B=min(B, 2000), seed=seed)          # frozen marginal numbers (unparsed = IoU 0)
    out = {"n": n, "coverage": float(ans.mean()), "n_answered": na,
           "marginal": {"R@0.3": m["R@0.3"], "R@0.5": m["R@0.5"], "R@0.7": m["R@0.7"], "mIoU": m["mIoU"],
                        "R@0.5_ci95": m["R@0.5_ci95"], "parse_rate": m["parse_rate"]},
           "conditional": {}}
    if na:
        out["conditional"] = {"R@0.3": float((iou[ans] >= 0.3).mean()), "R@0.5": float((iou[ans] >= 0.5).mean()),
                              "R@0.7": float((iou[ans] >= 0.7).mean()), "mIoU": float(iou[ans].mean())}
    else:
        out["conditional"] = {"R@0.3": None, "R@0.5": None, "R@0.7": None, "mIoU": None,
                              "reason": "coverage 0: no answered row (absent, not zero)"}
    # cluster bootstrap CI for the conditional number
    g = clusters(v["vid"])
    if na:
        d5, dm = [], []
        for idx in boot_idx(g, B, seed):
            a = idx[ans[idx]]
            if a.size:
                d5.append((iou[a] >= 0.5).mean()); dm.append(iou[a].mean())
        if d5:
            out["conditional"]["R@0.5_ci95"] = [float(np.percentile(d5, 2.5)), float(np.percentile(d5, 97.5))]
            out["conditional"]["mIoU_ci95"] = [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))]
    # abstention taxonomy
    wide, strict = {}, {}
    ex = {}
    for r in rows:
        if r.get("pred"):
            continue
        bw, bs = classify_abstain(r.get("raw") or "")
        wide[bw] = wide.get(bw, 0) + 1
        strict[bs] = strict.get(bs, 0) + 1
        ex.setdefault(bw, (r.get("raw") or "")[:160])
    nab = n - na
    out["abstain"] = {"n_abstained": nab,
                      "wide": {k: {"n": c, "frac_of_abstained": c / nab, "frac_of_all": c / n} for k, c in sorted(wide.items())},
                      "strict": {k: {"n": c, "frac_of_abstained": c / nab, "frac_of_all": c / n} for k, c in sorted(strict.items())},
                      "examples": ex} if nab else {"n_abstained": 0, "wide": {}, "strict": {}, "examples": {}}
    # span / GT length
    sp = v["span"][~np.isnan(v["span"])]
    out["pred_span_median"] = float(np.median(sp)) if sp.size else None
    out["pred_span_mean"] = float(sp.mean()) if sp.size else None
    out["gt_len_median"] = float(np.median(v["gl"]))
    out["gt_len_mean_abstained"] = float(v["gl"][~ans].mean()) if nab else None
    out["gt_len_mean_answered"] = float(v["gl"][ans].mean()) if na else None
    out["dur_mean_abstained"] = float(np.mean([float(r["dur"]) for r in rows if not r.get("pred")])) if nab else None
    out["dur_mean_answered"] = float(np.mean([float(r["dur"]) for r in rows if r.get("pred")])) if na else None
    out["tertile_cuts"] = list(cuts)
    out["tertiles"] = tertile_table(iou, v["gl"], ans, cuts)
    out["n_videos"] = len(g)
    out["bootstrap"] = {"B": B, "seed": seed, "cluster": "video"}
    out["styles"] = sorted({str(r.get("style")) for r in rows})
    return out


def read_pair(ref_rows, arm_rows, B=4000, seed=0, n_random=200):
    qr = [int(r["qid"]) for r in ref_rows]; qa = [int(r["qid"]) for r in arm_rows]
    if qr != qa:
        raise SystemExit("pair: qid sets differ (ref n=%d arm n=%d); only same-qid pairs are comparable" % (len(qr), len(qa)))
    R = vecs(ref_rows); A = vecs(arm_rows)
    if R["vid"] != A["vid"]:
        raise SystemExit("pair: video column differs between ref and arm")
    n = R["n"]; ans = A["ans"]; na = int(ans.sum())
    ri, ai = R["iou"], A["iou"]
    rh, ah = ri >= 0.5, ai >= 0.5
    out = {"n": n, "coverage_arm": float(ans.mean()), "coverage_ref": float(R["ans"].mean()), "n_answered_arm": na}
    out["point"] = {
        "ref_marginal_R@0.5": float(rh.mean()), "arm_marginal_R@0.5": float(ah.mean()),
        "delta_marginal": float(ah.mean() - rh.mean()),
        "arm_conditional_R@0.5": float(ah[ans].mean()) if na else None,
        "ref_on_arm_answered_R@0.5": float(rh[ans].mean()) if na else None,
        "ref_on_arm_abstained_R@0.5": float(rh[~ans].mean()) if na < n else None,
        "delta_conditional_paired": float(ah[ans].mean() - rh[ans].mean()) if na else None,
        "delta_arm_conditional_minus_ref_marginal": float(ah[ans].mean() - rh.mean()) if na else None,
        "ref_abstained_minus_answered": float(rh[~ans].mean() - rh[ans].mean()) if (na and na < n) else None,
        "arm_parse_on_ref": None,
    }
    out["point"]["ref_parse_on_arm_abstained"] = float(R["ans"][~ans].mean()) if na < n else None
    g = clusters(R["vid"])
    dm, dc, dsel, cond = [], [], [], []
    for idx in boot_idx(g, B, seed):
        a = idx[ans[idx]]; b = idx[~ans[idx]]
        dm.append(ah[idx].mean() - rh[idx].mean())
        if a.size:
            dc.append(ah[a].mean() - rh[a].mean()); cond.append(ah[a].mean())
        if a.size and b.size:
            dsel.append(rh[b].mean() - rh[a].mean())

    def ci(d, key):
        if not d:
            out[key] = {"reason": "no bootstrap draw with a non-empty subset (absent, not zero)"}
            return
        d = np.asarray(d, float)
        out[key] = {"mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    "p_two_sided": two_sided_p(d)}
    ci(dm, "delta_marginal_boot"); ci(dc, "delta_conditional_paired_boot"); ci(dsel, "ref_abstained_minus_answered_boot")
    if cond:
        c = np.asarray(cond, float)
        out["arm_conditional_R@0.5_ci95"] = [float(np.percentile(c, 2.5)), float(np.percentile(c, 97.5))]
    # coverage-matched RANDOM abstention control: REF keeps the same coverage but abstains at random
    k = int(round(float(ans.mean()) * n))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_random):
        sel = rng.choice(n, size=max(k, 1), replace=False)
        draws.append(rh[sel].mean())
    d = np.asarray(draws, float)
    out["random_abstention_control"] = {"draws": n_random, "k": k, "coverage": k / n,
                                        "ref_R@0.5_mean": float(d.mean()),
                                        "ref_R@0.5_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                        "ref_R@0.5_marginal": float(rh.mean())}
    out["n_videos"] = len(g); out["bootstrap"] = {"B": B, "seed": seed, "cluster": "video"}
    return out


# ---------------------------------------------------------------- selftest
def _mk(q, v, pred, gs, ge, raw="", dur=30.0):
    return {"set": "orig", "qid": q, "video": v, "caption": "c", "dur": dur, "pred": pred,
            "gsA": gs, "geA": ge, "gsB": None, "geB": None, "raw": raw, "style": "timer1_default", "fps": 2.0}


def selftest():
    ok = True

    def chk(c, m):
        nonlocal ok
        ok &= bool(c)
        print(("  PASS " if c else "  FAIL ") + m)

    # ---- taxonomy
    chk(classify_abstain("There are none.") == ("literal_there_are_none",) * 2, "taxonomy: bare 'There are none.'")
    chk(classify_abstain("<answer> There are none. </answer>") == ("literal_there_are_none",) * 2,
        "taxonomy: wrapped 'There are none.' -> same bucket")
    chk(classify_abstain("<answer></answer>") == ("empty_answer_tag",) * 2, "taxonomy: empty tag")
    chk(classify_abstain("<answer> </answer>") == ("empty_answer_tag",) * 2, "taxonomy: whitespace-only tag")
    w, s = classify_abstain("The image is completely black, so it is not possible to pinpoint the moment.")
    chk(w == "nl_refusal" and s == "nl_refusal", "taxonomy: prose refusal in both lists")
    w, s = classify_abstain("Sorry, here is my best guess at the scene.")
    chk(w == "nl_refusal" and s == "other", "taxonomy: soft hedge is WIDE-only (wide/strict really differ)")
    chk(classify_abstain("addCriterion addCriterion addCriterion addCriterion addCriterion x y")[0] == "gibberish",
        "taxonomy: repeated-token junk -> gibberish")
    chk(classify_abstain("\n addCriterion\n auto\n" + "\n" * 40)[0] == "gibberish",
        "taxonomy: blank-line collapse junk -> gibberish")
    chk(classify_abstain("")[0] == "empty_output" and classify_abstain("<answer></answer>")[0] == "empty_answer_tag",
        "taxonomy: empty generation and empty <answer> tag are different buckets")
    chk(is_truncated("The event appears to start around 12.0 and") and not is_truncated("<answer>1 to 2</answer>"),
        "truncation heuristic fires only without </answer> and without terminal punctuation")

    # ---- positive control: replacing every unparsed row with a number makes coverage 1 and conditional == marginal
    rows = [_mk(i, "v%d" % (i % 20), [0, 10] if i % 3 else None, 0, 10 if i % 2 else 4,
                "<answer>0.00 to 10.00</answer>" if i % 3 else "There are none.") for i in range(120)]
    a = read_one(rows, B=200)
    fixed = [dict(r) for r in rows]
    for r in fixed:
        if not r["pred"]:
            r["pred"] = [0.0, 10.0]; r["raw"] = "<answer>0.00 to 10.00</answer>"
    b = read_one(fixed, B=200)
    chk(abs(b["coverage"] - 1.0) < 1e-12, "POS: after filling every abstention, coverage == 1")
    chk(abs(b["conditional"]["R@0.5"] - b["marginal"]["R@0.5"]) < 1e-12,
        "POS: at coverage 1 the conditional and marginal R@0.5 coincide")
    chk(a["conditional"]["R@0.5"] >= a["marginal"]["R@0.5"] - 1e-12,
        "POS: with abstentions the conditional number is not below the marginal one")
    chk(a["abstain"]["n_abstained"] == sum(1 for r in rows if not r["pred"]) and
        a["abstain"]["wide"]["literal_there_are_none"]["n"] == a["abstain"]["n_abstained"],
        "POS: every abstained row is classified, all into the literal bucket here")

    # ---- negative control 1: arm == ref  =>  every paired delta is exactly 0
    ref = [_mk(i, "v%d" % (i % 20), [0, 10] if i % 4 else None, 0, 10 if i % 2 else 4) for i in range(120)]
    p0 = read_pair(ref, [dict(r) for r in ref], B=200)
    chk(p0["point"]["delta_marginal"] == 0.0 and p0["point"]["delta_conditional_paired"] == 0.0,
        "NEG: arm==ref gives paired delta exactly 0 (both marginal and conditional)")
    chk(abs(p0["delta_conditional_paired_boot"]["mean"]) < 1e-12,
        "NEG: arm==ref bootstrap mean of the paired delta is 0")

    # ---- negative control 2: abstaining ONLY on rows the arm got wrong raises the conditional number,
    #      while the coverage-matched random-abstention control on the reference does NOT move.
    rng = random.Random(0)
    refr, armr = [], []
    for i in range(300):
        v = "v%d" % (i % 40)
        hit_ref = rng.random() < 0.40
        hit_arm = rng.random() < 0.40
        refr.append(_mk(i, v, [0, 10] if hit_ref else [50, 60], 0, 10))
        armr.append(_mk(i, v, [0, 10] if hit_arm else [50, 60], 0, 10))
    arm_ab = []
    for r in armr:
        r2 = dict(r)
        if r2["pred"] == [50, 60] and rng.random() < 0.6:       # drop only wrong rows
            r2["pred"] = None; r2["raw"] = "There are none."
        arm_ab.append(r2)
    m_all = read_one(armr, B=200); m_ab = read_one(arm_ab, B=200)
    chk(m_ab["conditional"]["R@0.5"] > m_all["conditional"]["R@0.5"] + 0.05,
        "NEG: abstaining only on wrong rows raises the conditional accuracy")
    chk(abs(m_ab["marginal"]["R@0.5"] - m_all["marginal"]["R@0.5"]) < 1e-12,
        "NEG: ... and leaves the marginal accuracy untouched (unparsed already counted as 0)")
    pr = read_pair(refr, arm_ab, B=200)
    rc = pr["random_abstention_control"]
    chk(abs(rc["ref_R@0.5_mean"] - rc["ref_R@0.5_marginal"]) < 0.03,
        "NEG: coverage-matched RANDOM abstention leaves the reference at its marginal accuracy")
    chk(rc["ci95"[0:0] or "ref_R@0.5_ci95"][0] <= rc["ref_R@0.5_marginal"] <= rc["ref_R@0.5_ci95"][1],
        "NEG: ... its marginal accuracy sits inside the random-abstention CI")
    # the arm's abstentions here are informative BY CONSTRUCTION -> ref must do worse on them
    chk(pr["point"]["ref_on_arm_abstained_R@0.5"] is not None, "pair: ref scored on the arm's abstained rows")

    # ---- guard: a reader that silently reports 0 for an empty subset would pass everything above
    empty = [_mk(i, "v%d" % (i % 5), None, 0, 10, "There are none.") for i in range(20)]
    e = read_one(empty, B=50)
    chk(e["conditional"]["R@0.5"] is None and "reason" in e["conditional"],
        "GUARD: coverage 0 -> conditional reported as null with a reason, never as 0.0")

    # ---- guard: qid mismatch must be a hard error, not a silent truncation
    try:
        read_pair(ref, ref[:-1], B=20); chk(False, "GUARD: qid mismatch rejected")
    except SystemExit:
        chk(True, "GUARD: qid mismatch rejected")

    print("[selftest]", "ALL PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label"); ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--pair", nargs=2, metavar=("REF", "ARM"))
    ap.add_argument("--expect-n", type=int, default=None)
    ap.add_argument("--B", type=int, default=4000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-draws", type=int, default=200)
    ap.add_argument("--tertile-cuts", default="8,25")
    ap.add_argument("--note", default=None)
    ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    cuts = tuple(float(x) for x in a.tertile_cuts.split(","))
    if a.pair:
        ref = load_rows([a.pair[0]]); arm = load_rows([a.pair[1]])
        res = read_pair(ref, arm, B=a.B, seed=a.seed, n_random=a.random_draws)
        res["mode"] = "pair"
        res["ref_single"] = read_one(ref, B=a.B, seed=a.seed, cuts=cuts)
        res["arm_single"] = read_one(arm, B=a.B, seed=a.seed, cuts=cuts)
        files = list(a.pair)
        res["ref_file"] = os.path.basename(a.pair[0]); res["arm_file"] = os.path.basename(a.pair[1])
        brief = {"label": a.label, "cov_arm": res["coverage_arm"],
                 "d_marg": res["point"]["delta_marginal"], "d_cond": res["point"]["delta_conditional_paired"],
                 "cond": res["point"]["arm_conditional_R@0.5"],
                 "ref_on_abst": res["point"]["ref_on_arm_abstained_R@0.5"],
                 "ref_on_ans": res["point"]["ref_on_arm_answered_R@0.5"]}
    else:
        rows = load_rows(a.files)
        if a.expect_n is not None and len(rows) != a.expect_n:
            raise SystemExit("expected n=%d, got %d" % (a.expect_n, len(rows)))
        res = read_one(rows, B=a.B, seed=a.seed, cuts=cuts); res["mode"] = "single"
        files = list(a.files)
        brief = {"label": a.label, "n": res["n"], "coverage": res["coverage"],
                 "marg_R@0.5": res["marginal"]["R@0.5"], "cond_R@0.5": res["conditional"]["R@0.5"],
                 "cond_mIoU": res["conditional"]["mIoU"],
                 "abstain_wide": {k: v["n"] for k, v in res["abstain"]["wide"].items()}}
    res["label"] = a.label; res["note"] = a.note
    res["inputs"] = [{"file": os.path.basename(f), "sha16": sha16(f)} for f in files]
    res["reader"] = "read_p4nat_selective_v1"
    res["reader_sha16"] = sha16(os.path.abspath(__file__))
    res["frozen_reader"] = {"file": "read_p4nat_v1.py", "sha16": sha16(_FROZEN),
                            "used_for": "iou_1d, load_rows, metrics (marginal numbers); file not modified"}
    print(json.dumps(brief, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
