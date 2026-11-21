#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""read_forced_v1.py (rebuttal 2026-11-06, A3-2 / R2 Q1) -- paired reader for 12p5n_native_perturb_eval_forced.py
products run with --forced prefix (every condition yields an interval, so refusals cannot carry the contrast).

  --base B.json [B2.json ...] --arm A.json [A2.json ...]    shards of the same 1000-question run are concatenated
  per model, per condition       R@0.5 (mean hit), mIoU (mean iou), parse rate (must be 1.0 under forcing)
  per model, per axis mask/swap  E-vs-O drop = R(O) - R(E)  and  mIoU(O) - mIoU(E)   (O = matched placebo *_rand,
                                 E = evidence edit *_evid); plus clean-referenced drops none - E and none - O
  arm - base                     every per-condition value and every drop (DiD), same questions
  CIs: video-cluster bootstrap, ONE resampling shared by every quantity (pairing exact), B=2000, seed 0;
  p = two-sided bootstrap 2*min(P(d<=0), P(d>=0)).

Hard checks (error, not a silent n=0):
  * rows are forced=prefix (unless --allow-unforced) and one forced_sha16 / prompt_style / seed / instrument_sha16
    across base and arm (same grammar table, same instrument file);
  * pairing: n_estar, nframes, swap_donor, *_idx of every edit identical question by question;
  * every non-degenerate, decodable question present in both products has every requested condition;
  * forced products: parse rate exactly 1 in every condition.
Question set = qids in both, not degenerate (E/O undefined there), no decode error.  Same-machine pairing is the
caller's responsibility (parse/decoding behaviour is not portable across machines); recorded inputs make it auditable.

usage:
  read_forced_v1.py --base BASE.json [...] --arm ARM.json [...] --label NAME --out OUT_PREFIX   (-> .json + .tsv)
                    [--ops none,mask_evid,mask_rand,swap_evid,swap_rand,black] [--B 2000] [--seed 0]
  read_forced_v1.py --selftest
"""
import argparse, datetime, hashlib, json, os, sys
import numpy as np

READER = "read_forced_v1"
AXES = {"mask": ("mask_evid", "mask_rand"), "swap": ("swap_evid", "swap_rand")}
PAIR_KEYS = ("n_estar", "nframes", "swap_donor", "mask_evid_idx", "mask_rand_idx", "swap_evid_idx", "swap_rand_idx",
             "gsA", "geA")
SAME_KEYS = ("forced", "forced_sha16", "prompt_style", "seed", "instrument_sha16")
TSV_COLS = ("label", "section", "model", "metric", "cond", "n", "n_videos", "est", "ci_lo", "ci_hi", "p_two_sided")


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def ci95(d):
    d = np.asarray(d, float)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d.size else [None, None]


def two_sided_p(d):
    d = np.asarray(d, float)
    return float(min(1.0, 2.0 * min((d <= 0).mean(), (d >= 0).mean()))) if d.size else None


def est(point, draws):
    return {"est": float(point), "ci95": ci95(draws), "p_two_sided": two_sided_p(draws)}


def boot_means(arrays, vids, B, seed):
    """resample video clusters once; every array's mean per draw (all estimators are linear in condition means)."""
    groups = {}
    for i, v in enumerate(vids):
        groups.setdefault(v, []).append(i)
    groups = [np.asarray(g, int) for g in groups.values()]
    A = {k: np.asarray(v, float) for k, v in arrays.items()}
    point = {k: float(v.mean()) for k, v in A.items()}
    rng = np.random.default_rng(seed); G = len(groups)
    draws = {k: np.empty(B) for k in A}
    for b in range(B):
        ii = np.concatenate([groups[p] for p in rng.integers(0, G, G)])
        for k in A:
            draws[k][b] = A[k][ii].mean()
    return point, draws, G


def load_side(files, label):
    rows = {}; info = []
    for f in files:
        data = json.load(open(f, encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit("%s: %s is not a 12p5n row list" % (label, f))
        m = f + ".meta.json"
        info.append({"file": os.path.basename(f), "path": os.path.abspath(f), "sha16": sha16(f), "n_rows": len(data),
                     "meta_sha16": sha16(m) if os.path.exists(m) else None})
        for r in data:
            q = int(r["qid"])
            if q in rows and rows[q] != r:
                raise SystemExit("%s: qid %d appears in two shards with different content" % (label, q))
            rows[q] = r
    return rows, info


def check_same(B, A, allow_unforced):
    vals = {}
    for side, rows in (("base", B), ("arm", A)):
        for r in rows.values():
            for k in SAME_KEYS:
                vals.setdefault(k, {}).setdefault(side, set()).add(json.dumps(r.get(k, "<absent>")))
    for side, rows in (("base", B), ("arm", A)):
        if not allow_unforced and vals["forced"][side] != {json.dumps("prefix")}:
            raise SystemExit("%s rows are not all forced=prefix (%s); pass --allow-unforced to read r3 products"
                             % (side, sorted(vals["forced"][side])))
    out = {}
    for k in SAME_KEYS:
        u = vals[k]["base"] | vals[k]["arm"]
        if len(u) != 1:
            raise SystemExit("%s differs within/between products: base %s arm %s" % (k, sorted(vals[k]["base"]), sorted(vals[k]["arm"])))
        out[k] = json.loads(next(iter(u)))
    return out


def question_set(B, A, conds):
    excl = {"only_in_one": 0, "decode_error": 0, "degenerate": 0}
    qs = []
    for q in sorted(set(B) | set(A)):
        if q not in B or q not in A:
            excl["only_in_one"] += 1; continue
        b, a = B[q], A[q]
        if "decode_error" in b or "decode_error" in a:
            if ("decode_error" in b) != ("decode_error" in a):
                raise SystemExit("qid %d: decode_error in one product only" % q)
            excl["decode_error"] += 1; continue
        if b.get("degenerate") or a.get("degenerate"):
            if bool(b.get("degenerate")) != bool(a.get("degenerate")):
                raise SystemExit("qid %d: degenerate in one product only" % q)
            excl["degenerate"] += 1; continue
        for side, r in (("base", b), ("arm", a)):
            miss = [c for c in conds if "%s_hit" % c not in r or "%s_iou" % c not in r]
            if miss:
                raise SystemExit("qid %d: %s row lacks %s (absence is an error here, not n=0)" % (q, side, miss))
        bad = [k for k in PAIR_KEYS if b.get(k, "<absent>") != a.get(k, "<absent>")]
        if bad:
            raise SystemExit("PAIRING FAILED qid %d on %s -- the two products edited different frames" % (q, bad))
        qs.append(q)
    return qs, excl


def analyse(B, A, conds, nB, seed, allow_unforced=False):
    same = check_same(B, A, allow_unforced)
    qs, excl = question_set(B, A, conds)
    if not qs:
        raise SystemExit("no paired question (recorded as an error, not as 0)")
    vids = [B[q]["video"] for q in qs]
    arrays = {}
    for tag, rows in (("base", B), ("arm", A)):
        for c in conds:
            arrays["%s|%s|hit" % (tag, c)] = [float(rows[q]["%s_hit" % c]) for q in qs]
            arrays["%s|%s|iou" % (tag, c)] = [float(rows[q]["%s_iou" % c]) for q in qs]
            arrays["%s|%s|parsed" % (tag, c)] = [float(rows[q]["%s_parsed" % c]) for q in qs]
    pt, dr, G = boot_means(arrays, vids, nB, seed)
    if same["forced"] == "prefix":
        low = {k: v for k, v in pt.items() if k.endswith("|parsed") and v != 1.0}
        if low:
            raise SystemExit("forced product with parse rate < 1: %s -- the grammar did not hold" % low)
    P = lambda t, c, m: pt["%s|%s|%s" % (t, c, m)]
    D = lambda t, c, m: dr["%s|%s|%s" % (t, c, m)]
    MET = (("R@0.5", "hit"), ("mIoU", "iou"))
    res = {"n": len(qs), "n_videos": G, "per_model": {}, "arm_minus_base": {"conditions": {}, "drops": {}}}
    for t in ("base", "arm"):
        pm = {"conditions": {}, "drops": {}}
        for c in conds:
            pm["conditions"][c] = {name: est(P(t, c, m), D(t, c, m)) for name, m in MET}
            pm["conditions"][c]["parse_rate"] = P(t, c, "parsed")
        res["per_model"][t] = pm
    for ax, (ev, rd) in AXES.items():
        if ev not in conds or rd not in conds:
            for t in ("base", "arm"):
                res["per_model"][t]["drops"][ax] = {"note": "condition absent (n/a, not 0)"}
            res["arm_minus_base"]["drops"][ax] = {"note": "condition absent (n/a, not 0)"}
            continue
        pairs = [("O_minus_E", rd, ev)]
        if "none" in conds:
            pairs += [("clean_minus_E", "none", ev), ("clean_minus_O", "none", rd)]
        for t in ("base", "arm"):
            res["per_model"][t]["drops"][ax] = {
                kind: {name: est(P(t, hi, m) - P(t, lo, m), D(t, hi, m) - D(t, lo, m)) for name, m in MET}
                for kind, hi, lo in pairs}
        res["arm_minus_base"]["drops"][ax] = {
            kind: {name: est((P("arm", hi, m) - P("arm", lo, m)) - (P("base", hi, m) - P("base", lo, m)),
                             (D("arm", hi, m) - D("arm", lo, m)) - (D("base", hi, m) - D("base", lo, m))) for name, m in MET}
            for kind, hi, lo in pairs}
    for c in conds:
        res["arm_minus_base"]["conditions"][c] = {
            name: est(P("arm", c, m) - P("base", c, m), D("arm", c, m) - D("base", c, m)) for name, m in MET}
    return {"same": same, "question_set": {"n_paired": len(qs), "n_videos": G, "excluded": excl,
                                          "pairing_keys_asserted": list(PAIR_KEYS)}, "results": res}


def tsv_rows(out, label):
    R = out["results"]; n, G = R["n"], R["n_videos"]; rows = []
    def add(section, model, metric, cond, e):
        rows.append([label, section, model, metric, cond, n, G, e["est"], e["ci95"][0], e["ci95"][1], e["p_two_sided"]])
    for t in ("base", "arm"):
        for c, d in R["per_model"][t]["conditions"].items():
            for name in ("R@0.5", "mIoU"):
                add("condition", t, name, c, d[name])
        for ax, d in R["per_model"][t]["drops"].items():
            for kind, dd in d.items():
                if isinstance(dd, dict) and "R@0.5" in dd:
                    for name in ("R@0.5", "mIoU"):
                        add("drop_" + kind, t, name, ax, dd[name])
    for c, d in R["arm_minus_base"]["conditions"].items():
        for name in ("R@0.5", "mIoU"):
            add("condition", "arm-base", name, c, d[name])
    for ax, d in R["arm_minus_base"]["drops"].items():
        for kind, dd in d.items():
            if isinstance(dd, dict) and "R@0.5" in dd:
                for name in ("R@0.5", "mIoU"):
                    add("drop_" + kind, "arm-base", name, ax, dd[name])
    return rows


def write_tsv(path, rows, inputs):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for side in ("base", "arm"):
            for i in inputs[side]:
                f.write("# input %s %s sha16=%s n_rows=%d meta_sha16=%s\n" % (side, i["file"], i["sha16"], i["n_rows"], i["meta_sha16"]))
        f.write("\t".join(TSV_COLS) + "\n")
        for r in rows:
            f.write("\t".join("" if v is None else ("%.6f" % v if isinstance(v, float) else str(v)) for v in r) + "\n")


# ---------------------------------------------------------------- selftest
def _mk(nq=160, nv=40, seed=3):
    rng = np.random.default_rng(seed); rows = {}
    pr = {"none": .6, "mask_evid": .3, "mask_rand": .55, "swap_evid": .35, "swap_rand": .55, "black": .1}
    for q in range(nq):
        r = {"qid": q, "video": "v%d" % (q % nv), "degenerate": 0, "n_estar": 5, "nframes": 30, "swap_donor": "d%d" % q,
             "mask_evid_idx": [1, 2], "mask_rand_idx": [7, 9], "swap_evid_idx": [1, 2], "swap_rand_idx": [5, 8],
             "gsA": 1.0, "geA": 2.0, "forced": "prefix", "forced_sha16": "f" * 16, "prompt_style": "timer1_default",
             "seed": 1000, "instrument_sha16": "i" * 16}
        for c, p in pr.items():
            iou = float(rng.random()); h = int(rng.random() < p)
            r["%s_iou" % c] = max(iou, .5) if h else min(iou, .49); r["%s_hit" % c] = h; r["%s_parsed" % c] = 1
        rows[q] = r
    rows[nq] = dict(rows[0], qid=nq, degenerate=1)          # degenerate: excluded, counted
    for c in ("mask_evid", "mask_rand", "swap_evid", "swap_rand"):
        rows[nq].pop("%s_hit" % c); rows[nq].pop("%s_iou" % c)
    return rows


def selftest():
    import copy, tempfile
    ok = True
    def chk(c, m):
        nonlocal ok; ok &= bool(c); print(("  PASS " if c else "  FAIL ") + m)
    conds = ["none", "mask_evid", "mask_rand", "swap_evid", "swap_rand", "black"]
    base = _mk(); arm = copy.deepcopy(base)
    o = analyse(base, arm, conds, 300, 0)
    R = o["results"]
    z = [d[n] for d in R["arm_minus_base"]["conditions"].values() for n in ("R@0.5", "mIoU")]
    z += [dd[n] for ax in R["arm_minus_base"]["drops"].values() for dd in ax.values() for n in ("R@0.5", "mIoU")]
    chk(all(abs(e["est"]) < 1e-12 and e["ci95"] == [0.0, 0.0] for e in z), "base == arm -> every arm-base quantity 0, CI [0,0]")
    chk(o["question_set"]["n_paired"] == 160 and o["question_set"]["excluded"]["degenerate"] == 1 and R["n_videos"] == 40,
        "degenerate question excluded and counted; 40 video clusters")
    qs = sorted(q for q in base if not base[q]["degenerate"])
    chk(abs(R["per_model"]["base"]["conditions"]["mask_evid"]["R@0.5"]["est"] - np.mean([base[q]["mask_evid_hit"] for q in qs])) < 1e-12
        and abs(R["per_model"]["base"]["conditions"]["mask_evid"]["mIoU"]["est"] - np.mean([base[q]["mask_evid_iou"] for q in qs])) < 1e-12,
        "R@0.5 = mean hit, mIoU = mean iou")
    d = R["per_model"]["base"]["drops"]["mask"]["O_minus_E"]["R@0.5"]["est"]
    chk(abs(d - (np.mean([base[q]["mask_rand_hit"] for q in qs]) - np.mean([base[q]["mask_evid_hit"] for q in qs]))) < 1e-12,
        "E-vs-O drop = R(mask_rand) - R(mask_evid)")
    arm2 = copy.deepcopy(base); k = 0
    for q in qs:
        if arm2[q]["swap_evid_hit"] and k < 30:
            arm2[q]["swap_evid_hit"] = 0; arm2[q]["swap_evid_iou"] = 0.0; k += 1
    o2 = analyse(base, arm2, conds, 1000, 0)["results"]["arm_minus_base"]["drops"]
    e2 = o2["swap"]["O_minus_E"]["R@0.5"]; m2 = o2["mask"]["O_minus_E"]["R@0.5"]
    chk(e2["est"] > 0 and e2["ci95"][0] > 0 and e2["p_two_sided"] < 0.05 and abs(m2["est"]) < 1e-12,
        "evidence-specific arm errors on swap -> positive swap DiD with CI > 0, mask DiD untouched")
    chk(o2["swap"]["O_minus_E"]["mIoU"]["est"] > 0, "same injection raises the mIoU DiD")
    chk(analyse(base, arm2, conds, 200, 0)["results"]["arm_minus_base"]["drops"]["swap"]["O_minus_E"]["R@0.5"]["ci95"]
        == analyse(base, arm2, conds, 200, 0)["results"]["arm_minus_base"]["drops"]["swap"]["O_minus_E"]["R@0.5"]["ci95"],
        "bootstrap deterministic under a fixed seed")
    def raises(fn, m):
        try:
            fn(); chk(False, m)
        except SystemExit:
            chk(True, m)
    bad = copy.deepcopy(base); bad[5]["mask_rand_idx"] = [8, 9]
    raises(lambda: analyse(base, bad, conds, 50, 0), "different RAND draw on one question -> pairing error")
    bad = copy.deepcopy(base); bad[6]["swap_donor"] = "other"
    raises(lambda: analyse(base, bad, conds, 50, 0), "different donor -> pairing error")
    bad = copy.deepcopy(base); bad[7]["forced_sha16"] = "0" * 16
    raises(lambda: analyse(base, bad, conds, 50, 0), "forced_sha16 differs -> error")
    bad = copy.deepcopy(base)
    for r in bad.values():
        r.pop("forced"); r.pop("forced_sha16")
    raises(lambda: analyse(bad, copy.deepcopy(bad), conds, 50, 0), "unforced (r3) products rejected without --allow-unforced")
    chk(analyse(bad, copy.deepcopy(bad), conds, 50, 0, allow_unforced=True)["same"]["forced"] == "<absent>",
        "--allow-unforced reads r3 products")
    bad = copy.deepcopy(base); bad[8]["mask_evid_parsed"] = 0
    raises(lambda: analyse(base, bad, conds, 50, 0), "forced product with an unparsed cell -> error")
    bad = copy.deepcopy(base); bad[9].pop("swap_rand_hit")
    raises(lambda: analyse(base, bad, conds, 50, 0), "missing condition on a non-degenerate row -> error (absence is a value)")
    td = tempfile.mkdtemp(); fa, fb, fc = (os.path.join(td, n) for n in ("all.json", "s0.json", "s1.json"))
    L = [base[q] for q in sorted(base)]
    json.dump(L, open(fa, "w")); json.dump(L[:70], open(fb, "w")); json.dump(L[70:], open(fc, "w"))
    b1, _ = load_side([fa], "base"); b2, _ = load_side([fb, fc], "base")
    chk(analyse(b1, arm2, conds, 200, 0) == analyse(b2, arm2, conds, 200, 0), "two shards == one file")
    json.dump([dict(L[3], none_hit=1 - L[3]["none_hit"])], open(fc, "w"))
    raises(lambda: load_side([fa, fc], "base"), "same qid with different content in two shards -> error")
    print("[selftest]", "ALL PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+"); ap.add_argument("--arm", nargs="+")
    ap.add_argument("--ops", default="none,mask_evid,mask_rand,swap_evid,swap_rand,black")
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None); ap.add_argument("--out", default=None, help="prefix: writes OUT.json and OUT.tsv")
    ap.add_argument("--allow-unforced", action="store_true", help="read r3 (unforced) products with the same tables")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not (a.base and a.arm and a.out):
        ap.error("--base, --arm and --out required")
    conds = [c for c in a.ops.split(",") if c]
    B, ib = load_side(a.base, "base"); A, ia = load_side(a.arm, "arm")
    out = {"reader": READER, "reader_sha16": sha16(os.path.abspath(__file__)), "label": a.label,
           "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "params": {"ops": conds, "B": a.B, "seed": a.seed, "bootstrap_cluster": "video", "allow_unforced": a.allow_unforced,
                      "drop": "O_minus_E = cond(*_rand) - cond(*_evid); arm_minus_base = arm - base on the same questions",
                      "p": "two-sided bootstrap: 2*min(P(d<=0), P(d>=0))"},
           "inputs": {"base": ib, "arm": ia}}
    out.update(analyse(B, A, conds, a.B, a.seed, a.allow_unforced))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    write_tsv(a.out + ".tsv", tsv_rows(out, a.label or ""), out["inputs"])
    R = out["results"]
    print("[%s] %s n=%d (%d videos) forced=%s sha=%s excluded=%s" % (READER, a.label or "", R["n"], R["n_videos"],
          out["same"]["forced"], out["same"]["forced_sha16"], out["question_set"]["excluded"]))
    for c in conds:
        b, r, d = R["per_model"]["base"]["conditions"][c], R["per_model"]["arm"]["conditions"][c], R["arm_minus_base"]["conditions"][c]
        print("  %-10s R@.5 base %.3f arm %.3f (%+.3f [%+.3f,%+.3f]) | mIoU base %.3f arm %.3f (%+.3f [%+.3f,%+.3f])" %
              (c, b["R@0.5"]["est"], r["R@0.5"]["est"], d["R@0.5"]["est"], d["R@0.5"]["ci95"][0], d["R@0.5"]["ci95"][1],
               b["mIoU"]["est"], r["mIoU"]["est"], d["mIoU"]["est"], d["mIoU"]["ci95"][0], d["mIoU"]["ci95"][1]))
    for ax in AXES:
        dd = R["arm_minus_base"]["drops"][ax]
        if "O_minus_E" not in dd:
            print("  %s n/a" % ax); continue
        for name in ("R@0.5", "mIoU"):
            e = dd["O_minus_E"][name]
            print("  %-4s %-5s drop O-E base %+.3f arm %+.3f | arm-base %+.3f [%+.3f,%+.3f] p=%.4f" %
                  (ax, name, R["per_model"]["base"]["drops"][ax]["O_minus_E"][name]["est"],
                   R["per_model"]["arm"]["drops"][ax]["O_minus_E"][name]["est"], e["est"], e["ci95"][0], e["ci95"][1], e["p_two_sided"]))
    print("  -> %s.json / %s.tsv" % (a.out, a.out))


if __name__ == "__main__":
    main()
