#!/usr/bin/env python3
"""read_r3.py (2026-09-24) -- reader for 14Ln_native_sentinel_probe_r3 products (review-round readouts).

Pair mode:  read_r3.py --pair BASE.json ARM.json --label L --out OUT.json   (both files from one machine, same qids)
Single:     read_r3.py --one RUN.json --label L --out OUT.json

Per question on the hit set (per model) and on H_cap (paired):
  psi_tok      token-mean psi (registered estimand; reproduces the r2 reader)
  psi_seq      sequence-sum psi
  psi_ts       psi on the timestamp tokens only (tokens containing a digit, and '.' between two digit tokens)
  psi_fmt      psi on the remaining (format) tokens
  psi_contig   token-mean psi against the contiguous placebo (blocks of |E*| consecutive frames outside E*)
  dO_contig    the contiguous placebo drop (token mean)
  ref_share    of the probability of y* lost under the evidence edit, the share that reappears on the refusal string:
               (p_ref(E) - p_ref(clean)) / (p_y(clean) - p_y(E)), sequence probabilities; reported as the mean of
               numerator and denominator over questions (ratio of means) to avoid division by tiny drops
  ref_lift     log p(refusal | x_E) - log p(refusal | x_O)   (sequence log-prob, placebo = mean over draws)
  reloc        relocation: the evidence block exchanged with a same-length block elsewhere.
               follow_lp = [lp(y_new|x_rel)-lp(y_new|x)]/n_new - [lp(y*|x_rel)-lp(y*|x)]/n_old   (token means)
               follow_iou = 1[IoU(pred on x_rel, y_new) >= 0.5]; stay_iou = 1[IoU(pred on x_rel, y*) >= 0.5]
Gamma-type quantities are trained-minus-base means on H_cap with 95% video-cluster bootstrap intervals (B=4000, seed 0);
single-model quantities are means over that model's hit set with the same bootstrap.
"""
import argparse, hashlib, json, os, random, re, time
import numpy as np

OPS = ("mask", "swap")
_DIG = re.compile(r"\d")


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def ts_mask(toks):
    """timestamp tokens: any token containing a digit, plus a '.' token between two digit-bearing tokens."""
    dig = [bool(_DIG.search(t)) for t in toks]
    m = list(dig)
    for i, t in enumerate(toks):
        if not dig[i] and t.strip("Ġ▁ ") == "." and 0 < i < len(toks) - 1 and dig[i - 1] and dig[i + 1]:
            m[i] = True
    return np.array(m, bool)


def per_q(r):
    """per-question readouts of one row (a hit with r3 fields); None where a field is absent."""
    out = {"video": r["video"]}
    for op in OPS:
        if f"{op}_tokmean_drop_evid" not in r:
            continue
        out[f"{op}_psi_tok"] = r[f"{op}_tokmean_drop_evid"] - r[f"{op}_tokmean_drop_rand"]
        out[f"{op}_psi_seq"] = r[f"{op}_drop_evid"] - r[f"{op}_drop_rand"]
        out[f"{op}_dE"] = r[f"{op}_tokmean_drop_evid"]; out[f"{op}_dO"] = r[f"{op}_tokmean_drop_rand"]
        if "tok_str" in r and f"{op}_tok_evid" in r:
            m = ts_mask(r["tok_str"]); c = np.array(r["tok_clean"]); e = np.array(r[f"{op}_tok_evid"]); o = np.array(r[f"{op}_tok_rand"])
            if len(m) == len(c) == len(e) == len(o) and m.any() and (~m).any():
                # psi restricted to a token set = mean over the set of (o - e)  [= (c-e) - (c-o)]
                out[f"{op}_psi_ts"] = float((o - e)[m].mean()); out[f"{op}_psi_fmt"] = float((o - e)[~m].mean())
                out[f"{op}_dE_ts"] = float((c - e)[m].mean()); out[f"{op}_dO_ts"] = float((c - o)[m].mean())
                out["n_ts"] = int(m.sum()); out["n_tok"] = int(len(m))
        if f"{op}_tokmean_cdrop_rand" in r:
            out[f"{op}_psi_contig"] = r[f"{op}_tokmean_drop_evid"] - r[f"{op}_tokmean_cdrop_rand"]
            out[f"{op}_dO_contig"] = r[f"{op}_tokmean_cdrop_rand"]
        if "ref_clean_sum" in r and f"{op}_ref_evid_sum" in r:
            py_c = np.exp(r["lp_clean_sum"]); py_e = np.exp(r["lp_clean_sum"] - r[f"{op}_drop_evid"])
            pr_c = np.exp(r["ref_clean_sum"]); pr_e = np.exp(r[f"{op}_ref_evid_sum"])
            out[f"{op}_ref_num"] = float(pr_e - pr_c); out[f"{op}_ref_den"] = float(py_c - py_e)
            out[f"{op}_ref_lift"] = r[f"{op}_ref_evid_sum"] - r[f"{op}_ref_rand_sum"]
            out[f"{op}_ref_lp_E"] = r[f"{op}_ref_evid_sum"]; out[f"{op}_ref_lp_c"] = r["ref_clean_sum"]
    if "reloc_lp_new_rel" in r:
        n_old = max(1, r["tgt_len"]); n_new = max(1, r["reloc_new_len"])
        out["reloc_follow_lp"] = (r["reloc_lp_new_rel"] - r["reloc_lp_new_orig"]) / n_new - (r["reloc_lp_old_rel"] - r["lp_clean_sum"]) / n_old
        out["reloc_follow_iou"] = float(r["reloc_iou_new"] >= 0.5); out["reloc_stay_iou"] = float(r["reloc_iou_old"] >= 0.5)
    return out


def load(p):
    rows = json.load(open(p))
    return {r["qid"]: per_q(r) for r in rows if r.get("hit") and "lp_clean_sum" in r}


def boot(vals, vids, B=4000, seed=0, ratio=None):
    """cluster bootstrap over videos of a mean (or ratio of means when ratio is a second array)."""
    vals = np.asarray(vals, float); vids = list(vids)
    uv = sorted(set(vids)); idx = {v: [i for i, x in enumerate(vids) if x == v] for v in uv}
    est = vals.mean() if ratio is None else vals.sum() / np.asarray(ratio).sum()
    rng = random.Random(seed); bs = []
    for _ in range(B):
        ii = [i for v in (rng.choice(uv) for _ in uv) for i in idx[v]]
        bs.append(vals[ii].mean() if ratio is None else vals[ii].sum() / np.asarray(ratio)[ii].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"est": float(est), "ci95": [float(lo), float(hi)], "n": int(len(vals)), "n_videos": len(uv)}


def summarise_one(Q):
    keys = sorted({k for q in Q.values() for k in q if k not in ("video", "n_ts", "n_tok") and not k.endswith(("_num", "_den"))})
    out = {"n_hits": len(Q)}
    for k in keys:
        qs = [q for q in Q.values() if k in q]
        if qs:
            out[k] = boot([q[k] for q in qs], [q["video"] for q in qs])
    for op in OPS:
        qs = [q for q in Q.values() if f"{op}_ref_num" in q]
        if qs:
            out[f"{op}_ref_share"] = boot([q[f"{op}_ref_num"] for q in qs], [q["video"] for q in qs], ratio=[q[f"{op}_ref_den"] for q in qs])
    ts = [q["n_ts"] for q in Q.values() if "n_ts" in q]
    if ts:
        out["tokens"] = {"ts_mean": float(np.mean(ts)), "all_mean": float(np.mean([q["n_tok"] for q in Q.values() if "n_tok" in q]))}
    return out


def summarise_pair(B, A):
    H = sorted(set(B) & set(A))
    out = {"n_cap": len(H)}
    keys = sorted({k for q in A.values() for k in q if k not in ("video", "n_ts", "n_tok") and not k.endswith(("_num", "_den"))})
    for k in keys:
        hh = [i for i in H if k in A[i] and k in B[i]]
        if hh:
            out["Gamma_" + k] = boot([A[i][k] - B[i][k] for i in hh], [A[i]["video"] for i in hh])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, metavar=("BASE", "ARM")); ap.add_argument("--one")
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"reader": "read_r3", "reader_sha16": sha16(__file__), "label": a.label,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.one:
        Q = load(a.one)
        res["inputs"] = {"run": {"file": os.path.basename(a.one), "sha16": sha16(a.one)}}
        res["model"] = summarise_one(Q)
    else:
        Bq, Aq = load(a.pair[0]), load(a.pair[1])
        res["inputs"] = {"base": {"file": os.path.basename(a.pair[0]), "sha16": sha16(a.pair[0])},
                         "arm": {"file": os.path.basename(a.pair[1]), "sha16": sha16(a.pair[1])}}
        res["base"] = summarise_one(Bq); res["arm"] = summarise_one(Aq); res["paired"] = summarise_pair(Bq, Aq)
    json.dump(res, open(a.out, "w"), indent=1)
    show = res.get("paired") or res.get("model")
    for k, v in show.items():
        if isinstance(v, dict) and "est" in v:
            print(f"  {k:28s} {v['est']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] n={v['n']}")


if __name__ == "__main__":
    main()
