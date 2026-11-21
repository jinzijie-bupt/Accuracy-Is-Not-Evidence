#!/usr/bin/env python3
"""read_r3_tokclass.py (2026-09-24): where in the answer string Gamma(mask) sits.

For every paired 14Ln r3 read (read_r3_<label>.json, which names its base and arm products), split the token-mean psi of
each question into three additive parts by token role, and report trained-minus-base on H-cap (both models localise the
question on the clean video), with video-cluster bootstrap intervals (B=2000, seed 0):
  onset : the first answer token after the opening tag -- where a model that can decline commits to a number or to a
          refusal (for the VideoChat-R1 template, which has no tag, the first token of the answer);
  ts    : digits and a '.' between two digits (the two timestamps);
  rest  : every other answer token (tags, separators, 'to').
Each part is (sum of per-token psi over its tokens) / answer length, so onset + ts + rest = the question's token-mean psi
and the three Gammas add up to Gamma. Also reports the base's and the trained model's clean-video probability of the
onset token actually written in the scored answer.
"""
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..")).replace("\\", "/")
import hashlib, json, os, re, sys
import numpy as np

R3 = _SUPP + "/results/judgment_20260924_r3"
RAW = _SUPP + "/results/review_r3_20260924/node3"
LABELS = ["node3_sft_tf5", "node3_tzpub", "node3_tr1pub", "node3_vcpub",
          "node3_pc_a05_tf5", "node3_pc_tz_a05", "node3_pc_tr1_a025", "node3_pc_vc_a05"]
PARTS = ("onset", "ts", "rest")


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def roles(toks):
    onset = next((i + 1 for i, t in enumerate(toks) if t.endswith(">")), 0)
    out = []
    for i, t in enumerate(toks):
        s = t.replace("\u0120", "")
        prev = toks[i - 1].replace("\u0120", "") if i else ""
        nxt = toks[i + 1].replace("\u0120", "") if i + 1 < len(toks) else ""
        if i == onset:
            out.append("onset")
        elif re.fullmatch(r"\d+", s) or (s == "." and prev.isdigit() and nxt.isdigit()):
            out.append("ts")
        else:
            out.append("rest")
    return out, onset


def parts(r):
    c, e, o = (np.array(r[k], float) for k in ("tok_clean", "mask_tok_evid", "mask_tok_rand"))
    psi = (c - e) - (c - o)
    ro, onset = roles(r["tok_str"])
    L = len(psi)
    d = {k: float(sum(psi[i] for i in range(L) if ro[i] == k) / L) for k in PARTS}
    d["p_onset"] = float(np.exp(c[onset]))
    return d


def boot(vals, vids, B=2000, seed=0):
    vals = np.asarray(vals, float); u = sorted(set(vids)); idx = {v: [i for i, x in enumerate(vids) if x == v] for v in u}
    rng = np.random.default_rng(seed); est = float(vals.mean()); bs = []
    for _ in range(B):
        take = rng.choice(len(u), len(u), replace=True)
        ii = [i for t in take for i in idx[u[t]]]
        bs.append(vals[ii].mean())
    return {"est": est, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def main():
    for lab in LABELS:
        rp = os.path.join(R3, f"read_r3_{lab}.json")
        inp = json.load(open(rp, encoding="utf-8"))["inputs"]
        bp, ap = os.path.join(RAW, inp["base"]["file"]), os.path.join(RAW, inp["arm"]["file"])
        assert sha16(bp) == inp["base"]["sha16"] and sha16(ap) == inp["arm"]["sha16"], f"{lab}: raw product differs from the one read_r3 read"
        B = {r["qid"]: r for r in json.load(open(bp, encoding="utf-8"))}
        A = {r["qid"]: r for r in json.load(open(ap, encoding="utf-8"))}
        q = sorted(k for k in B if k in A and B[k]["hit"] and A[k]["hit"])
        for k in q:
            assert B[k]["tok_str"] == A[k]["tok_str"], f"{lab} qid {k}: base and arm scored different answer strings"
        pb = {k: parts(B[k]) for k in q}; pa = {k: parts(A[k]) for k in q}
        vids = [B[k]["video"] for k in q]
        out = {"reader": "read_r3_tokclass", "reader_sha16": sha16(os.path.abspath(__file__)), "label": lab,
               "inputs": inp, "n_cap": len(q), "params": {"B": 2000, "seed": 0, "cluster": "video", "operator": "mask"},
               "paired": {f"Gamma_{k}": boot([pa[x][k] - pb[x][k] for x in q], vids) for k in PARTS},
               "p_onset_clean": {"base": float(np.mean([pb[x]["p_onset"] for x in q])),
                                 "trained": float(np.mean([pa[x]["p_onset"] for x in q]))}}
        out["Gamma_total"] = float(sum(out["paired"][f"Gamma_{k}"]["est"] for k in PARTS))
        op = os.path.join(R3, f"tokclass_{lab}.json")
        json.dump(out, open(op + ".tmp", "w", encoding="utf-8"), indent=1); os.replace(op + ".tmp", op)
        P = out["paired"]
        print(f"{lab:20s} n={len(q):3d} G={out['Gamma_total']:+.2f} onset {P['Gamma_onset']['est']:+.2f} ts {P['Gamma_ts']['est']:+.2f} "
              f"rest {P['Gamma_rest']['est']:+.2f} | p_onset {out['p_onset_clean']['base']:.3f}->{out['p_onset_clean']['trained']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
