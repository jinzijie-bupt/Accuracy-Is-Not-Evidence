#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""read_p5n_v1.py — reader for 12p5n_native_perturb_eval products (behavioral evidence test).
Per condition op: R@0.5 (hit rate) and parse rate on the NON-degenerate questions that carry every requested op
(same question set for all ops), paired deltas vs 'none' with video-cluster bootstrap (B draws, seed).
Evidence sensitivity = R@0.5(none) - R@0.5(mask_evid); placebo sensitivity = R@0.5(none) - R@0.5(mask_rand);
behavioral contrast = placebo-adjusted drop = [R(none)-R(evid)] - [R(none)-R(rand)] = R(rand) - R(evid).
usage: read_p5n_v1.py --rep FILE [FILE ...] [--ops none,mask_evid,mask_rand] [--B 2000] [--seed 0] [--out out.json]
--selftest runs the negative-control bench. Shards are concatenated by qid (duplicates / gaps rejected).
"""
import argparse, json, hashlib, os, sys
import numpy as np


def sha16(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def load_rows(files):
    rows = {}
    for f in files:
        for r in json.load(open(f, encoding="utf-8")):
            q = int(r["qid"])
            if q in rows: raise SystemExit(f"duplicate qid {q} in {f}")
            rows[q] = r
    qq = sorted(rows)
    if qq and qq != list(range(qq[0], qq[-1] + 1)): raise SystemExit("qid gap in concatenated shards")
    return rows


def usable(rows, ops):
    return {q: r for q, r in rows.items() if not r.get("degenerate") and "decode_error" not in r and all(f"{o}_hit" in r for o in ops)}


def boot(vals, vids, B, seed):
    vals = np.asarray(vals, float); uv = sorted(set(vids)); idx = {v: [i for i, x in enumerate(vids) if x == v] for v in uv}
    if not len(vals): return None, [None, None]
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        pick = rng.integers(0, len(uv), len(uv)); ii = np.concatenate([idx[uv[p]] for p in pick]); bs.append(vals[ii].mean())
    return float(vals.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def summarize(rows, ops, B, seed):
    U = usable(rows, ops); qq = sorted(U); vids = [U[q]["video"] for q in qq]
    out = {"n_rows": len(rows), "n_usable": len(qq), "n_videos": len(set(vids)), "n_degenerate": sum(1 for r in rows.values() if r.get("degenerate")),
           "n_decode_error": sum(1 for r in rows.values() if "decode_error" in r), "ops": {}}
    for o in ops:
        h = [U[q][f"{o}_hit"] for q in qq]; m, ci = boot(h, vids, B, seed)
        out["ops"][o] = {"R@0.5": m, "R@0.5_ci": ci, "parse_rate": float(np.mean([U[q][f"{o}_parsed"] for q in qq])) if qq else None,
                         "mIoU": float(np.mean([U[q][f"{o}_iou"] for q in qq])) if qq else None}
        if o != "none" and "none" in ops:
            d = [U[q]["none_hit"] - U[q][f"{o}_hit"] for q in qq]; dm, dci = boot(d, vids, B, seed)
            out["ops"][o]["drop_vs_none"] = dm; out["ops"][o]["drop_vs_none_ci"] = dci
    if "mask_evid" in ops and "mask_rand" in ops:
        c = [U[q]["mask_rand_hit"] - U[q]["mask_evid_hit"] for q in qq]; cm, cci = boot(c, vids, B, seed)
        out["mask_contrast"] = cm; out["mask_contrast_ci"] = cci
    if "swap_evid" in ops and "swap_rand" in ops:
        c = [U[q]["swap_rand_hit"] - U[q]["swap_evid_hit"] for q in qq]; cm, cci = boot(c, vids, B, seed)
        out["swap_contrast"] = cm; out["swap_contrast_ci"] = cci
    return out


def selftest():
    ok = True
    def chk(c, m):
        nonlocal ok; ok &= bool(c); print(("  PASS " if c else "  FAIL ") + m)
    def row(q, v, none, ev, rd, deg=0):
        return {"qid": q, "video": v, "degenerate": deg, "none_hit": none, "none_parsed": 1, "none_iou": none, "mask_evid_hit": ev, "mask_evid_parsed": 1, "mask_evid_iou": ev, "mask_rand_hit": rd, "mask_rand_parsed": 1, "mask_rand_iou": rd}
    rows = {r["qid"]: r for r in [row(0, "a", 1, 0, 1), row(1, "a", 1, 0, 0), row(2, "b", 0, 0, 0), row(3, "b", 1, 1, 1), row(4, "c", 1, 0, 1, deg=1)]}
    ops = ["none", "mask_evid", "mask_rand"]; s = summarize(rows, ops, 50, 0)
    chk(s["n_usable"] == 4 and s["n_degenerate"] == 1, "degenerate rows excluded from every condition")
    chk(abs(s["ops"]["none"]["R@0.5"] - 0.75) < 1e-12 and abs(s["ops"]["mask_evid"]["R@0.5"] - 0.25) < 1e-12 and abs(s["ops"]["mask_rand"]["R@0.5"] - 0.5) < 1e-12, "per-condition R@0.5 exact")
    chk(abs(s["ops"]["mask_evid"]["drop_vs_none"] - 0.5) < 1e-12 and abs(s["ops"]["mask_rand"]["drop_vs_none"] - 0.25) < 1e-12, "paired drops exact")
    chk(abs(s["mask_contrast"] - 0.25) < 1e-12, "contrast = R(rand) - R(evid) exact")
    rows2 = {q: dict(r) for q, r in rows.items()}; del rows2[1]["mask_rand_hit"]
    chk(summarize(rows2, ops, 10, 0)["n_usable"] == 3, "row missing a condition is dropped from all conditions")
    try:
        load_rows_dup = {0: rows[0]}; import tempfile
        d = tempfile.mkdtemp(); p1 = os.path.join(d, "a.json"); p2 = os.path.join(d, "b.json")
        json.dump([rows[0], rows[1]], open(p1, "w")); json.dump([rows[1], rows[2]], open(p2, "w"))
        load_rows([p1, p2]); chk(False, "duplicate qid across shards rejected")
    except SystemExit:
        chk(True, "duplicate qid across shards rejected")
    print("[selftest]", "ALL PASS" if ok else "FAIL"); return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", nargs="+"); ap.add_argument("--ops", default="none,mask_evid,mask_rand"); ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None); ap.add_argument("--label", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(0 if selftest() else 1)
    ops = a.ops.split(","); rows = load_rows(a.rep)
    out = {"reader": "read_p5n_v1", "reader_sha16": sha16(os.path.abspath(__file__)), "label": a.label, "files": [{"file": os.path.basename(f), "sha16": sha16(f)} for f in a.rep], "B": a.B, "seed": a.seed}
    out.update(summarize(rows, ops, a.B, a.seed))
    line = f"{a.label or ''} n_usable={out['n_usable']}/{out['n_rows']} (deg {out['n_degenerate']}) " + " | ".join(f"{o}: R@0.5 {out['ops'][o]['R@0.5']:.3f} parse {out['ops'][o]['parse_rate']:.3f}" + (f" drop {out['ops'][o]['drop_vs_none']:+.3f} {[round(x,3) for x in out['ops'][o]['drop_vs_none_ci']]}" if 'drop_vs_none' in out['ops'][o] else "") for o in ops)
    if "mask_contrast" in out: line += f" || mask contrast {out['mask_contrast']:+.3f} {[round(x,3) for x in out['mask_contrast_ci']]}"
    if "swap_contrast" in out: line += f" || swap contrast {out['swap_contrast']:+.3f} {[round(x,3) for x in out['swap_contrast_ci']]}"
    print(line)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
