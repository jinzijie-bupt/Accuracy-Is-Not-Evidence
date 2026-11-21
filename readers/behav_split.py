#!/usr/bin/env python3
"""behav_split.py (2026-09-24) -- where the behavioural evidence share comes from.

On the questions a model answers correctly with the placebo frames edited, split the answers lost when the evidence
frames are edited instead into (a) no interval returned (unparsed output or the refusal string) and (b) an interval
returned that misses (IoU < 0.5). Reported per model and as trained-minus-base differences on the questions both runs
share, with a 95% video-cluster bootstrap (B=2000, seed 0).

  lost        P(evid miss | placebo hit)
  no_answer   P(no interval under evid | placebo hit)
  wrong_ans   P(evid miss | placebo hit, interval returned under evid)
"""
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..")).replace("\\", "/")
import argparse, hashlib, json, os, random, time
import numpy as np

REFUSAL = "There are none"


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def load(p):
    return {r["qid"]: r for r in json.load(open(p, encoding="utf-8")) if not r.get("degenerate")}


def answered(r, cond):
    return bool(r[f"{cond}_parsed"]) and REFUSAL not in (r[f"{cond}_raw"] or "")


def stats(R, qs, op):
    s = [q for q in qs if R[q][f"{op}_rand_hit"]]
    ans = [q for q in s if answered(R[q], f"{op}_evid")]
    return np.array([sum(not R[q][f"{op}_evid_hit"] for q in s) / len(s),
                     sum(not answered(R[q], f"{op}_evid") for q in s) / len(s),
                     sum(not R[q][f"{op}_evid_hit"] for q in ans) / len(ans)])


def pair(B, A, op, n=2000, seed=0):
    qs = sorted(set(B) & set(A))
    vids = sorted({B[q]["video"] for q in qs}); by = {v: [q for q in qs if B[q]["video"] == v] for v in vids}
    sb, sa = stats(B, qs, op), stats(A, qs, op)
    rng = random.Random(seed); bs = []
    for _ in range(n):
        qq = [q for v in (rng.choice(vids) for _ in vids) for q in by[v]]
        bs.append(stats(A, qq, op) - stats(B, qq, op))
    lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
    names = ("lost", "no_answer", "wrong_ans")
    return {"n": len(qs), "base": dict(zip(names, map(float, sb))), "arm": dict(zip(names, map(float, sa))),
            "diff": {k: {"est": float(sa[i] - sb[i]), "ci95": [float(lo[i]), float(hi[i])]} for i, k in enumerate(names)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=_SUPP + "/results/native_p5n_20260920")
    ap.add_argument("--out", default=_SUPP + "/results/judgment_20260924_r3/behav_split.json")
    a = ap.parse_args()
    f = lambda n: os.path.join(a.dir, f"p5n6_{n}_node5_n1000_o0.json")
    pairs = [("pc Qwen a=0.5", "qwenbase_timer1_default", "natbase_a05_g1_s0_timer1_default"),
             ("SFT", "qwenbase_timer1_default", "natsft_b0_s0_timer1_default"),
             ("TimeZero public", "qwenbase_timer1_default", "tzbase_timer1_default_vf"),   # video-first run, like the Qwen base
             ("Time-R1 public", "qwenbase_timer1_default", "tr1base_timer1_default"),
             ("pc TimeZero a=0.5", "tzbase_timer1_default", "nattz_a05_g1_s0_timer1_default"),
             ("pc Time-R1 a=0.25", "tr1base_timer1_default", "nattr1_a025_g1_s0_timer1_default"),
             ("pc VideoChat-R1 a=0.5", "vcbase_vchat", "natvc_a05_g1_s0_vchat")]
    res = {"reader": "behav_split", "reader_sha16": sha16(__file__), "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "B": 2000, "seed": 0, "pairs": []}
    for lab, b, t in pairs:
        if not (os.path.exists(f(b)) and os.path.exists(f(t))):
            print(f"{lab:24s} pending: {os.path.basename(f(t))}"); continue
        B, A = load(f(b)), load(f(t))
        row = {"label": lab, "base": {"file": os.path.basename(f(b)), "sha16": sha16(f(b))},
               "arm": {"file": os.path.basename(f(t)), "sha16": sha16(f(t))}}
        for op in ("mask", "swap"):
            row[op] = pair(B, A, op)
        res["pairs"].append(row)
        d = row["mask"]["diff"]
        print(f"{lab:24s} n={row['mask']['n']}  " + "  ".join(f"{k} {100*v['est']:+.1f} [{100*v['ci95'][0]:.1f}, {100*v['ci95'][1]:.1f}]" for k, v in d.items()))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
