#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""read_p4nat_v1.py — registered reader for native-protocol grounding products
(12p4n_native_acc_eval_tf5meta.py / _tf451.py rows: qid, video, pred=[s,e]|None, gsA, geA, dur, style, fps).
usage: read_p4nat_v1.py --label <name> --files f1.json [f2.json ...] [--expect-n 3720] [--B 2000] [--seed 0] --out out.json
Shards are concatenated by qid (duplicate qid -> hard error; coverage must be contiguous from min qid).
Metrics: parse_rate, R@0.3/0.5/0.7, mIoU (unparsed rows count as IoU 0, as in the checkpoint papers);
video-cluster bootstrap percentile CIs (B draws, seed) for R@0.5 and mIoU. Writes metrics + input sha16s.
--selftest runs the negative-control bench and exits 0 on all-pass.
"""
import argparse, json, hashlib, os, sys, random
import numpy as np


def iou_1d(a, b):
    if not a or not b or len(a) != 2 or len(b) != 2:
        return 0.0
    s1, e1 = float(a[0]), float(a[1]); s2, e2 = float(b[0]), float(b[1])
    if e1 < s1: s1, e1 = e1, s1
    inter = max(0.0, min(e1, e2) - max(s1, s2)); uni = max(e1, e2) - min(s1, s2)
    return inter / uni if uni > 0 else 0.0


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def load_rows(files):
    rows = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        rs = d["rows"] if isinstance(d, dict) and "rows" in d else d
        if not isinstance(rs, list) or not rs:
            raise ValueError(f"{f}: no rows")
        for r in rs:
            for k in ("qid", "video", "gsA", "geA"):
                if k not in r or r[k] is None:
                    raise ValueError(f"{f}: row missing {k}: {str(r)[:120]}")
            q = int(r["qid"])
            if q in rows:
                raise ValueError(f"duplicate qid {q} (file {f})")
            rows[q] = r
    qs = sorted(rows)
    if qs != list(range(qs[0], qs[0] + len(qs))):
        missing = sorted(set(range(qs[0], qs[-1] + 1)) - set(qs))
        raise ValueError(f"non-contiguous qid coverage: {len(missing)} missing, first {missing[:5]}")
    return [rows[q] for q in qs]


def metrics(rows, B=2000, seed=0):
    ious = np.array([iou_1d(r.get("pred"), (r["gsA"], r["geA"])) for r in rows])
    parsed = np.array([1 if r.get("pred") else 0 for r in rows])
    n = len(rows)
    out = {"n": n, "qid_min": int(rows[0]["qid"]), "qid_max": int(rows[-1]["qid"]), "parse_rate": float(parsed.mean()),
           "R@0.3": float((ious >= 0.3).mean()), "R@0.5": float((ious >= 0.5).mean()), "R@0.7": float((ious >= 0.7).mean()),
           "mIoU": float(ious.mean())}
    vids = sorted({r["video"] for r in rows}); vidx = {v: i for i, v in enumerate(vids)}
    groups = [[] for _ in vids]
    for i, r in enumerate(rows):
        groups[vidx[r["video"]]].append(i)
    rng = np.random.default_rng(seed); r5 = []; mi = []
    G = len(groups)
    for _ in range(B):
        pick = rng.integers(0, G, G); idx = np.concatenate([groups[g] for g in pick])
        r5.append((ious[idx] >= 0.5).mean()); mi.append(ious[idx].mean())
    out["R@0.5_ci95"] = [float(np.percentile(r5, 2.5)), float(np.percentile(r5, 97.5))]
    out["mIoU_ci95"] = [float(np.percentile(mi, 2.5)), float(np.percentile(mi, 97.5))]
    out["n_videos"] = G; out["bootstrap"] = {"B": B, "seed": seed, "cluster": "video"}
    return out


def selftest():
    ok = True
    def chk(c, m):
        nonlocal ok; ok &= bool(c); print(("  PASS " if c else "  FAIL ") + m)
    mk = lambda q, v, p, gs, ge: {"qid": q, "video": v, "pred": p, "gsA": gs, "geA": ge}
    rows = [mk(0, "a", [0, 10], 0, 10), mk(1, "a", [0, 5], 0, 10), mk(2, "b", None, 0, 10), mk(3, "b", [10, 0], 0, 10)]
    m = metrics(rows, B=50)
    chk(abs(m["mIoU"] - (1 + 0.5 + 0 + 1) / 4) < 1e-12, "mIoU exact on known rows (incl. reversed pred, unparsed=0)")
    chk(m["R@0.5"] == 0.75 and m["R@0.7"] == 0.5 and m["parse_rate"] == 0.75, "R@t and parse_rate exact")
    tmp = os.path.join(os.environ.get("TEMP", "/tmp"), "read_p4nat_selftest")
    os.makedirs(tmp, exist_ok=True)
    f1 = os.path.join(tmp, "s0.json"); f2 = os.path.join(tmp, "s1.json"); f3 = os.path.join(tmp, "dup.json"); f4 = os.path.join(tmp, "gap.json")
    json.dump({"rows": rows[:2]}, open(f1, "w")); json.dump(rows[2:], open(f2, "w")); json.dump(rows[1:3], open(f3, "w")); json.dump([rows[3]], open(f4, "w"))
    chk([r["qid"] for r in load_rows([f2, f1])] == [0, 1, 2, 3], "shards concatenate by qid regardless of file order")
    for fs, name in (([f1, f3], "duplicate qid rejected"), ([f1, f4], "qid gap rejected")):
        try: load_rows(fs); chk(False, name)
        except ValueError: chk(True, name)
    bad = dict(rows[0]); del bad["geA"]; json.dump([bad], open(f4, "w"))
    try: load_rows([f4]); chk(False, "missing GT field rejected")
    except ValueError: chk(True, "missing GT field rejected")
    m2 = metrics([mk(i, "v%d" % (i % 3), [0, 1], 0, 1) for i in range(30)], B=100)
    chk(m2["R@0.5_ci95"] == [1.0, 1.0] and m2["n_videos"] == 3, "bootstrap CI degenerate when all rows identical; clusters counted")
    print("[selftest]", "ALL PASS" if ok else "FAIL"); return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label"); ap.add_argument("--files", nargs="*", default=[]); ap.add_argument("--expect-n", type=int, default=None)
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    rows = load_rows(a.files)
    if a.expect_n is not None and len(rows) != a.expect_n:
        raise SystemExit(f"expected n={a.expect_n}, got {len(rows)}")
    m = metrics(rows, a.B, a.seed)
    m["label"] = a.label; m["inputs"] = [{"file": os.path.basename(f), "sha16": sha16(f)} for f in a.files]
    m["reader"] = "read_p4nat_v1"; m["reader_sha16"] = sha16(os.path.abspath(__file__))
    m["styles"] = sorted({str(r.get("style")) for r in rows}); m["fps"] = sorted({float(r.get("fps", 0) or 0) for r in rows})
    print(json.dumps({k: m[k] for k in ("label", "n", "parse_rate", "R@0.3", "R@0.5", "R@0.7", "mIoU", "R@0.5_ci95", "mIoU_ci95")}, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(m, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
