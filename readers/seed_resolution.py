#!/usr/bin/env python3
"""seed_resolution.py -- paired difference between two training seeds of one configuration (the audit's resolution).
Gamma(s1 - s0) on the questions both seeds localise, token-mean psi, cluster bootstrap over videos (B=2000, seed 0).
Usage: seed_resolution.py <audit_s0.json> <audit_s1.json> [out.json]"""
import json, sys, random
import numpy as np


def psi(r, op):
    return r[f"{op}_tokmean_drop_evid"] - r[f"{op}_tokmean_drop_rand"]


def main(a, b, out=None, B=2000):
    A = {r["qid"]: r for r in json.load(open(a)) if r.get("hit") and f"mask_tokmean_drop_evid" in r}
    Bm = {r["qid"]: r for r in json.load(open(b)) if r.get("hit") and f"mask_tokmean_drop_evid" in r}
    q = sorted(set(A) & set(Bm))
    res = {"s0": a, "s1": b, "n_cap": len(q)}
    for op in ("mask", "swap"):
        d = np.array([psi(Bm[i], op) - psi(A[i], op) for i in q])
        vids = sorted({A[i]["video"] for i in q}); by = {v: [k for k, i in enumerate(q) if A[i]["video"] == v] for v in vids}
        rng = random.Random(0); bs = []
        for _ in range(B):
            idx = [k for v in (rng.choice(vids) for _ in vids) for k in by[v]]
            bs.append(d[idx].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res[op] = {"diff": float(d.mean()), "ci": [float(lo), float(hi)], "abs_upper": float(max(abs(lo), abs(hi)))}
    print(json.dumps(res, indent=1))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
