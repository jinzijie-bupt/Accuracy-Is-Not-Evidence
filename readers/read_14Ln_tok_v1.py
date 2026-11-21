#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""read_14Ln_tok_v1.py (rebuttal 2026-11-06, R1 W2c) -- token-subset reader for 14Ln r10 --tok-lp products.

Estimator: read_14Ln_v1 (e18bf15596e3c3c1) unchanged, evaluated on three token subsets S of the GT answer:
  all   = every answer token                     (reproduces read_14Ln_v1 from the arrays; cross-checked)
  ts    = timestamp tokens (row ts_mask == 1: digits / decimal points of start and end, B3 trainer definition)
  nonts = every other answer token (tags, ' to', punctuation, spaces)
  per hit row i, operator op, subset S (n_S = |S|):
      lpS(x)   = sum_{t in S} tok_lp[x][t]
      dE_S     = lpS(clean) - lpS(<op>_evid)                    (seqsum, nats)
      dO_S     = lpS(clean) - mean_r lpS(<op>_rand[r])          (seqsum, nats)
      psi_S    = dE_S - dO_S;   tokmean = seqsum / n_S          (S = all: n_S = tgt_len, i.e. read_14Ln_v1 units)
  H(f)   = {qid : hit == 1 and every requested op scored}   (as read_14Ln_v1; a hit row without tok_lp is an ERROR)
  cohort = H(base) & H(arm)  (or --fixed-cohort intersected with it)
  Gamma_S = mean_cohort(psi_S,arm - psi_S,base); tokmean primary, seqsum alongside.
  CI: video-cluster bootstrap, B=2000, seed 0, exactly read_14Ln_v1's streams: one rng(seed) for tokmean and one
      rng(seed+10000) for seqsum, B draws of n_videos per op, ops in order.  The SAME draws are applied to all three
      subsets (paired), so the 'all' tokmean / seqsum CIs equal read_14Ln_v1's bit for bit on the same cohort.
Seqsum is additive (Gamma_all = Gamma_ts + Gamma_nonts); tokmean is not (different denominators).
Timestamp share (seqsum, the only additive unit):
  share_of_Gamma = Gamma_ts,seqsum / Gamma_all,seqsum with a paired bootstrap CI (ratio of resampled means);
  per question: ts share of the Gamma contribution, of psi and of the evidence drop dE (base and arm), in the TSV;
  summaries = median over questions whose |denominator| > --share-eps, plus ratio of cohort means;
  token_share = mean n_ts / tgt_len (the share if every token carried the same drop).
Inputs: --base and --arm each take one or more shards (r10 --n/--offset runs); rows are concatenated by qid
(a qid twice is an ERROR).  Each shard's .meta.json must say instrument *_r10 with args.tok_lp true; protocol args
must agree across shards and between base and arm.
Output: --out X.json (+ X.tsv per qid x op; '#' header lines carry reader and input sha16).
Exit codes: 0 status OK/WARN, 3 status ERROR (the JSON/TSV are still written).

usage:
  read_14Ln_tok_v1.py --base B0.json [B1.json ...] --arm A0.json [A1.json ...] [--ops mask,swap] [--expect-n N]
                      [--B 2000] [--seed 0] [--fixed-cohort qids.txt] [--strict-machine] [--label L] --out OUT.json
  read_14Ln_tok_v1.py --selftest [--sandbox DIR]
"""
import argparse, datetime, hashlib, json, math, os, sys
import numpy as np

READ_V1_SHA16 = "060842138b2f6d91"
MACHINE_TOKENS = ["node2", "node1", "node4", "node3", "tm"]
SUBSETS = ("all", "ts", "nonts")
SAME_ARGS = ("answer_style", "prompt_style", "R", "seed", "fps", "total_pixels", "min_pixels", "max_frames",
             "image_patch_size", "text_first", "jitter", "jitter_frac", "legacy_rng", "noise_sigma", "gblur_sigma",
             "pscram_grid", "occ_side", "thr", "max_new_tokens")
SUM_TOL = 1e-3


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def self_sha16():
    try:
        return sha16(os.path.abspath(__file__))
    except Exception:
        return None


def infer_machine(path):                                   # read_14Ln_v1
    b = "_" + os.path.basename(path)
    for m in MACHINE_TOKENS:
        if f"_{m}_" in b:
            return m
    return None


def read_cohort_file(p):                                   # read_14Ln_v1
    txt = open(p, encoding="utf-8").read().strip()
    if txt.startswith("["):
        return [int(x) for x in json.loads(txt)]
    out = []
    for line in txt.splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.append(int(line))
    return out


# ------------------------------------------------------------------ loading
def load_side(tag, paths, ops, errors, warns):
    rows, shards, metas, seen = [], [], [], {}
    for p in paths:
        ap_ = os.path.abspath(p)
        d = json.load(open(ap_, encoding="utf-8"))
        if not isinstance(d, list):
            raise SystemExit(f"not a row list: {ap_}")
        mp = ap_ + ".meta.json"
        meta = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else None
        sh = {"role": tag, "file": os.path.basename(ap_), "path": ap_.replace("\\", "/"), "sha16": sha16(ap_),
              "n_rows": len(d), "machine": infer_machine(ap_), "meta_sha16": sha16(mp) if meta is not None else None}
        if meta is None:
            warns.append(f"{tag} {sh['file']}: no .meta.json; instrument/tok_lp/protocol not verifiable from meta")
        else:
            sh.update({"instrument": meta.get("instrument"), "probe_sha16": meta.get("probe_sha16"),
                       "trainer_sha16": meta.get("trainer_sha16"),
                       "tsonly_sha16": (meta.get("tok_lp") or {}).get("tsonly_sha16"),
                       "offset": meta.get("args", {}).get("offset"), "n": meta.get("args", {}).get("n")})
            if not str(meta.get("instrument", "")).endswith("_r10"):
                errors.append(f"{tag} {sh['file']}: instrument {meta.get('instrument')!r} is not an r10 product")
            if meta.get("args", {}).get("tok_lp") is not True or "tok_lp" not in meta:
                errors.append(f"{tag} {sh['file']}: meta says the shard was not run with --tok-lp")
            miss = [op for op in ops if op not in (meta.get("ops") or [])]
            if miss:
                errors.append(f"{tag} {sh['file']}: ops {miss} not run (meta ops {meta.get('ops')})")
            metas.append((sh["file"], meta))
        for r in d:
            q = int(r["qid"])
            if q in seen:
                errors.append(f"{tag}: qid {q} occurs in {seen[q]} and {sh['file']} (duplicate across/within shards)")
                continue
            seen[q] = sh["file"]
            rows.append(r)
        shards.append(sh)
    return rows, shards, metas


def check_args(metas_b, metas_a, errors):
    allm = [("base", f, m) for f, m in metas_b] + [("arm", f, m) for f, m in metas_a]
    if not allm:
        return {}
    ref_tag, ref_f, ref = allm[0]
    out = {}
    for k in SAME_ARGS:
        v0 = ref.get("args", {}).get(k, "__absent__")
        out[k] = v0
        for tag, f, m in allm[1:]:
            v = m.get("args", {}).get(k, "__absent__")
            if v != v0:
                errors.append(f"protocol arg {k}: {ref_tag}/{ref_f}={v0!r} vs {tag}/{f}={v!r}")
    return out


# ------------------------------------------------------------------ per-row quantities
def row_tok_problem(r, ops):
    """None if the row's arrays are complete and reproduce its sum fields (|diff| <= SUM_TOL); else a message."""
    tl, ids, m = r.get("tok_lp"), r.get("tok_ids"), r.get("ts_mask")
    if tl is None or ids is None or m is None:
        return "tok_lp/tok_ids/ts_mask missing"
    n = int(r["tgt_len"])
    if len(ids) != n or len(m) != n or not set(m) <= {0, 1}:
        return f"tok_ids/ts_mask length or values inconsistent with tgt_len {n}"
    if sum(m) == 0 or sum(m) == n:
        return "ts_mask selects no token or every token"

    def bad(arr, s):
        return len(arr) != n or abs(math.fsum(arr) - float(s)) > SUM_TOL

    if "clean" not in tl or bad(tl["clean"], r["lp_clean_sum"]):
        return "tok_lp.clean does not sum to lp_clean_sum"
    for op in ops:
        ke, kr = f"{op}_evid", f"{op}_rand"
        if ke not in tl or bad(tl[ke], float(r["lp_clean_sum"]) - float(r[f"{op}_drop_evid"])):
            return f"tok_lp.{ke} does not sum to lp_clean_sum - {op}_drop_evid"
        rs = r.get(f"{op}_rand_sums") or []
        if kr not in tl or len(tl[kr]) != len(rs) or not rs or any(bad(a, s) for a, s in zip(tl[kr], rs)):
            return f"tok_lp.{kr} does not match {op}_rand_sums"
    return None


def subset_terms(r, op):
    """{S: (dE_seqsum, dO_seqsum, n_S)} for one row and op (math.fsum over the selected tokens)."""
    tl, m = r["tok_lp"], r["ts_mask"]
    sel = {"all": [True] * len(m), "ts": [x == 1 for x in m], "nonts": [x == 0 for x in m]}
    out = {}
    for S, w in sel.items():
        lp = lambda arr: math.fsum(v for v, k in zip(arr, w) if k)
        c = lp(tl["clean"]); e = lp(tl[f"{op}_evid"])
        rr = [lp(a) for a in tl[f"{op}_rand"]]
        out[S] = (c - e, c - float(np.mean(rr)), int(sum(w)))
    return out


def boot_picks(rng, nu, B):
    return [rng.integers(0, nu, nu) for _ in range(B)]


def boot_means(vals, vids, picks):
    """read_14Ln_v1.cluster_boot with the draws supplied (same concatenation order -> bit-identical means)."""
    vals = np.asarray(vals, float)
    uv = sorted(set(vids))
    gi = {v: [i for i, x in enumerate(vids) if x == v] for v in uv}
    return np.array([vals[np.concatenate([gi[uv[p]] for p in pick])].mean() for pick in picks])


def ci95(bs):
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def _share(num, den, eps):
    return (num / den) if abs(den) > eps else None


def _med(xs):
    xs = [x for x in xs if x is not None]
    return (float(np.median(xs)), len(xs)) if xs else (None, 0)


# ------------------------------------------------------------------ analysis
def analyse(a):
    errors, warns = [], []
    ops = [o for o in a.ops.split(",") if o]
    B_rows, sh_b, meta_b = load_side("base", a.base, ops, errors, warns)
    A_rows, sh_a, meta_a = load_side("arm", a.arm, ops, errors, warns)
    proto = check_args(meta_b, meta_a, errors)
    if a.expect_n:
        for tag, rows in (("base", B_rows), ("arm", A_rows)):
            if len(rows) != a.expect_n:
                errors.append(f"expect-n {a.expect_n} but {tag} has {len(rows)} rows (all shards)")
    mb = sorted({s["machine"] for s in sh_b}, key=str); ma = sorted({s["machine"] for s in sh_a}, key=str)
    if mb != ma or len(ma) != 1:
        msg = f"machine tokens base={mb} arm={ma} (only within-machine paired deltas are comparable)"
        (errors if a.strict_machine else warns).append(msg)
    sb = sorted({r.get("style") for r in B_rows if r.get("style")}); sa = sorted({r.get("style") for r in A_rows if r.get("style")})
    if sb != sa:
        errors.append(f"prompt style mismatch base={sb} arm={sa}")

    def hits(rows, tag):
        h, bad = {}, []
        for r in rows:
            if r.get("hit") == 1 and all(f"{op}_tokmean_drop_evid" in r and f"{op}_tokmean_drop_rand" in r for op in ops):
                p = row_tok_problem(r, ops)
                if p is None:
                    h[int(r["qid"])] = r
                else:
                    bad.append((int(r["qid"]), p))
        if bad:
            errors.append(f"{tag}: {len(bad)} hit row(s) without usable per-token arrays (never read as zero), "
                          f"e.g. qid {bad[0][0]}: {bad[0][1]}")
        return h, bad

    hb, bad_b = hits(B_rows, "base"); ha, bad_a = hits(A_rows, "arm")
    inter = sorted(set(hb) & set(ha))
    mism = []
    for q in inter:
        for k in ("n_estar", "nframes", "tgt_len", "video", "tok_ids", "ts_mask", "gt_target_text"):
            if hb[q].get(k) != ha[q].get(k):
                mism.append({"qid": q, "field": k})
    if mism:
        errors.append(f"per-qid structural mismatch on {len({m['qid'] for m in mism})} qid(s): "
                      + ", ".join(sorted({m['field'] for m in mism})))
    cohort, cohort_name, cohort_detail = inter, "H_int", None
    if a.fixed_cohort:
        want = read_cohort_file(a.fixed_cohort)
        cohort = [q for q in want if q in hb and q in ha]
        cohort_detail = {"requested": len(want), "kept": len(cohort),
                         "missing_in_base": len([q for q in want if q not in hb]),
                         "missing_in_arm": len([q for q in want if q not in ha])}
        cohort_name = "fixed:" + os.path.basename(a.fixed_cohort)
    if not cohort:
        errors.append("empty cohort")

    q2v = {int(r["qid"]): r.get("video") for r in A_rows}
    vids = [q2v[q] for q in cohort]
    nu = len(set(vids))
    rng = np.random.default_rng(a.seed)              # read_14Ln_v1 tokmean stream
    rng_s = np.random.default_rng(a.seed + 10000)    # read_14Ln_v1 seqsum stream
    eps = a.share_eps
    per_op, tsv_rows = {}, []
    for op in ops:
        tb = {q: subset_terms(hb[q], op) for q in cohort}
        tA = {q: subset_terms(ha[q], op) for q in cohort}
        picks = boot_picks(rng, nu, a.B) if cohort else []
        picks_s = boot_picks(rng_s, nu, a.B) if cohort else []
        res_op = {"n_cohort": len(cohort), "n_videos": nu, "subsets": {}}
        g = {}
        for S in SUBSETS:
            psiA = np.array([tA[q][S][0] - tA[q][S][1] for q in cohort]); psiB = np.array([tb[q][S][0] - tb[q][S][1] for q in cohort])
            nS = np.array([tA[q][S][2] for q in cohort], float)
            gs_ = psiA - psiB
            gt_ = psiA / nS - psiB / nS if len(cohort) else np.array([])
            g[S] = (gt_, gs_)
            d = {"n_tokens_mean": float(nS.mean()) if len(nS) else None}
            if len(cohort):
                bt, bs = boot_means(gt_, vids, picks), boot_means(gs_, vids, picks_s)
                d.update({"gamma": float(gt_.mean()), "gamma_ci95": ci95(bt), "pos_frac": float((gt_ > 0).mean()),
                          "gamma_seqsum": float(gs_.mean()), "gamma_seqsum_ci95": ci95(bs)})
                for side, T in (("base", tb), ("arm", tA)):
                    dE = np.array([T[q][S][0] for q in cohort]); dO = np.array([T[q][S][1] for q in cohort])
                    d[f"{side}_on_cohort"] = {"dE": float((dE / nS).mean()), "dO": float((dO / nS).mean()),
                                              "psi": float(((dE - dO) / nS).mean()), "dE_seqsum": float(dE.mean()),
                                              "dO_seqsum": float(dO.mean()), "psi_seqsum": float((dE - dO).mean())}
                g[S] = (gt_, gs_, bt, bs)
            res_op["subsets"][S] = d
        if cohort:
            # read_14Ln_v1 on the row fields, same draws -> its gamma/CI on this cohort, and the array cross-check
            gr = np.array([(float(ha[q][f"{op}_tokmean_drop_evid"]) - float(ha[q][f"{op}_tokmean_drop_rand"]))
                           - (float(hb[q][f"{op}_tokmean_drop_evid"]) - float(hb[q][f"{op}_tokmean_drop_rand"])) for q in cohort])
            grs = np.array([(float(ha[q][f"{op}_drop_evid"]) - float(ha[q][f"{op}_drop_rand"]))
                            - (float(hb[q][f"{op}_drop_evid"]) - float(hb[q][f"{op}_drop_rand"])) for q in cohort])
            dev_node5 = float(np.max(np.abs(gr - g["all"][0]))); dev_ss = float(np.max(np.abs(grs - g["all"][1])))
            res_op["rowfields_read_14Ln_v1"] = {
                "gamma": float(gr.mean()), "gamma_ci95": ci95(boot_means(gr, vids, picks)),
                "gamma_seqsum": float(grs.mean()), "gamma_seqsum_ci95": ci95(boot_means(grs, vids, picks_s)),
                "max_abs_dev_per_qid_vs_all_tokens": {"tokmean": dev_tm, "seqsum": dev_ss}}
            if dev_ss > 2 * SUM_TOL or dev_node5 > 2 * SUM_TOL:
                errors.append(f"{op}: per-token 'all' Gamma deviates from the row fields (seqsum {dev_ss:.3g}, tokmean {dev_tm:.3g})")
            add = float(np.max(np.abs(g["all"][1] - g["ts"][1] - g["nonts"][1])))
            if add > 1e-6:
                errors.append(f"{op}: seqsum additivity all = ts + nonts violated by {add:.3g}")
            # timestamp share
            ga, gt = g["all"][3], g["ts"][3]
            ratio_bs = gt / np.where(np.abs(ga) > 0, ga, np.nan)
            G_all, G_ts = float(g["all"][1].mean()), float(g["ts"][1].mean())
            per_q = []
            for i, q in enumerate(cohort):
                rec = {"qid": q, "video": q2v[q], "op": op, "tgt_len": int(hb[q]["tgt_len"]), "n_ts": tA[q]["ts"][2]}
                for S in SUBSETS:
                    rec[f"gamma_{S}"] = float(g[S][0][i]); rec[f"gamma_seqsum_{S}"] = float(g[S][1][i])
                for side, T in (("base", tb), ("arm", tA)):
                    for S in ("all", "ts"):
                        rec[f"{side}_dE_seqsum_{S}"] = T[q][S][0]; rec[f"{side}_psi_seqsum_{S}"] = T[q][S][0] - T[q][S][1]
                    rec[f"{side}_share_dE"] = _share(T[q]["ts"][0], T[q]["all"][0], eps)
                    rec[f"{side}_share_psi"] = _share(T[q]["ts"][0] - T[q]["ts"][1], T[q]["all"][0] - T[q]["all"][1], eps)
                rec["share_gamma"] = _share(float(g["ts"][1][i]), float(g["all"][1][i]), eps)
                per_q.append(rec)
            tsv_rows += per_q
            summ = {"share_of_Gamma_seqsum": _share(G_ts, G_all, 0.0),
                    "share_of_Gamma_seqsum_ci95": ([float(np.nanpercentile(ratio_bs, 2.5)), float(np.nanpercentile(ratio_bs, 97.5))]
                                                   if np.isfinite(ratio_bs).any() else [None, None]),
                    "token_share_mean": float(np.mean([tA[q]["ts"][2] / tA[q]["all"][2] for q in cohort])),
                    "share_eps": eps}
            for key in ("share_gamma", "base_share_dE", "arm_share_dE", "base_share_psi", "arm_share_psi"):
                m_, n_ = _med([r[key] for r in per_q])
                summ[f"median_{key}"] = m_; summ[f"n_defined_{key}"] = n_
            for side, T in (("base", tb), ("arm", tA)):
                summ[f"{side}_share_dE_of_means"] = _share(float(np.mean([T[q]["ts"][0] for q in cohort])),
                                                           float(np.mean([T[q]["all"][0] for q in cohort])), 0.0)
            res_op["timestamp_share"] = summ
        per_op[op] = res_op

    return {
        "reader": "read_14Ln_tok_v1", "reader_sha16": self_sha16(), "estimator_from": f"read_14Ln_v1 {READ_V1_SHA16}",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": a.label, "ops": ops, "status": "ERROR" if errors else ("WARN" if warns else "OK"),
        "errors": errors, "warnings": warns, "inputs": sh_b + sh_a, "protocol_args": proto, "prompt_style": sa,
        "machine_base": mb, "machine_arm": ma, "expect_n": a.expect_n,
        "n_rows_base": len(B_rows), "n_rows_arm": len(A_rows), "n_hits_base": len(hb), "n_hits_arm": len(ha),
        "n_hit_rows_without_tok_base": len(bad_b), "n_hit_rows_without_tok_arm": len(bad_a),
        "n_H_int": len(inter), "cohort": cohort_name, "n_cohort": len(cohort), "cohort_detail": cohort_detail,
        "bootstrap": {"B": a.B, "seed": a.seed, "cluster": "video",
                      "note": "read_14Ln_v1 streams: tokmean rng(seed), seqsum rng(seed+10000), ops in order; "
                              "the same draws serve all/ts/nonts (paired) and the share ratio"},
        "units": {"primary": "tokmean nats/token over the subset's tokens", "secondary": "seqsum nats (additive over subsets)"},
        "structural_mismatch": mism[:20], "n_structural_mismatch": len(mism),
        "per_op": per_op, "_tsv_rows": tsv_rows,
    }


TSV_COLS = (["qid", "video", "op", "tgt_len", "n_ts"] + [f"gamma_{S}" for S in SUBSETS] + [f"gamma_seqsum_{S}" for S in SUBSETS]
            + [f"{side}_{k}_seqsum_{S}" for side in ("base", "arm") for k in ("dE", "psi") for S in ("all", "ts")]
            + ["share_gamma", "base_share_dE", "arm_share_dE", "base_share_psi", "arm_share_psi"])


def write_outputs(res, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    rows = res.pop("_tsv_rows")
    tsv = (out[:-5] if out.endswith(".json") else out) + ".tsv"
    res["tsv"] = os.path.basename(tsv)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    def f(v):
        return "NA" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))
    with open(tsv, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# read_14Ln_tok_v1 reader_sha16={res['reader_sha16']} status={res['status']} json={os.path.basename(out)}\n")
        for s in res["inputs"]:
            fh.write(f"# input {s['role']} {s['file']} sha16={s['sha16']}\n")
        fh.write("\t".join(TSV_COLS) + "\n")
        for r in rows:
            fh.write("\t".join(f(r.get(c)) for c in TSV_COLS) + "\n")
    return tsv


# ------------------------------------------------------------------ selftest
MASK19 = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]   # '<answer> 12.54 to 17.83 </answer>' (Qwen2.5)


def _finish(r, ops):
    """derive every r9/r10 sum field from the arrays (as the probe writes them)."""
    tl = r["tok_lp"]; n = len(tl["clean"]); c = math.fsum(tl["clean"])
    r.update({"tgt_len": n, "lp_clean_sum": c, "lp_clean_mean": c / n})
    for op in ops:
        e = math.fsum(tl[f"{op}_evid"]); rs = [math.fsum(a) for a in tl[f"{op}_rand"]]
        r[f"{op}_drop_evid"] = c - e; r[f"{op}_drop_rand"] = c - float(np.mean(rs))
        r[f"{op}_tokmean_drop_evid"] = (c - e) / n; r[f"{op}_tokmean_drop_rand"] = (c - float(np.mean(rs))) / n
        r[f"{op}_rand_sums"] = rs; r[f"{op}_rand_idx"] = [[0]] * len(rs); r[f"{op}_donor"] = None


def _mk_rows(n, seed, ops=("mask", "swap"), R=3, nvid=25):
    rng = np.random.default_rng(seed); vr = np.random.default_rng(seed + 1)
    rows = []
    for q in range(n):
        hit = int(q % 3 != 2)
        r = {"qid": q, "video": f"v{int(vr.integers(0, nvid))}.mp4", "n_estar": 12, "nframes": 60, "fps": 2.0,
             "style": "timer1_default", "target_mode": "gt", "sentinel": "none", "dilate_k": 0,
             "clean_iou": 0.7 if hit else 0.1, "hit": hit, "clean_parsed": 1, "pred": [1.0, 2.0], "clean_text": "x"}
        if hit:
            L = len(MASK19)
            tl = {"clean": list(-rng.uniform(0.0, 1.5, L))}
            for op in ops:
                tl[f"{op}_evid"] = list(np.array(tl["clean"]) - rng.uniform(0.0, 0.6, L))
                tl[f"{op}_rand"] = [list(np.array(tl["clean"]) - rng.uniform(0.0, 0.3, L)) for _ in range(R)]
            r.update({"gt_target_text": "<answer> 12.54 to 17.83 </answer>", "tok_ids": list(range(100, 100 + L)),
                      "ts_mask": list(MASK19), "tok_lp": {k: ([list(map(float, a)) for a in v] if k.endswith("_rand") else list(map(float, v)))
                                                           for k, v in tl.items()}})
            _finish(r, ops)
        rows.append(r)
    return rows


def _meta(path, ops, instrument="14Ln_native_sentinel_probe_v1r2_tf_q3_r10", tok=True, **kw):
    args = {"answer_style": "timer1", "prompt_style": "timer1_default", "R": 3, "seed": 1000, "tok_lp": tok}
    args.update(kw)
    m = {"instrument": instrument, "ops": list(ops), "args": args, "probe_sha16": "selftest"}
    if tok:
        m["tok_lp"] = {"tsonly_sha16": "7153d2cdae11bbd8"}
    json.dump(m, open(path + ".meta.json", "w", encoding="utf-8"))


def _dump(rows, path, ops=("mask", "swap"), **kw):
    json.dump(rows, open(path, "w", encoding="utf-8")); _meta(path, ops, **kw)
    return path


def _run(base, arm, **kw):
    ns = argparse.Namespace(base=base if isinstance(base, list) else [base], arm=arm if isinstance(arm, list) else [arm],
                            ops=kw.get("ops", "mask,swap"), expect_n=kw.get("expect_n", 0), B=kw.get("B", 400), seed=0,
                            fixed_cohort=kw.get("fixed_cohort"), strict_machine=False, label=None,
                            share_eps=kw.get("share_eps", 1e-6))
    return analyse(ns)


def selftest(sandbox):
    import copy
    os.makedirs(sandbox, exist_ok=True)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    P = lambda nm: os.path.join(sandbox, nm)
    base = _mk_rows(150, 11)
    pb = _dump(base, P("14Ln_base_timer1_default_node2_nat_tok.json"))

    # PC1 identity
    r = _run(pb, pb)
    allz = all(r["per_op"][op]["subsets"][S][k] == 0.0 for op in ("mask", "swap") for S in SUBSETS for k in ("gamma", "gamma_seqsum"))
    cis = all(r["per_op"][op]["subsets"][S][k] == [0.0, 0.0] for op in ("mask", "swap") for S in SUBSETS
              for k in ("gamma_ci95", "gamma_seqsum_ci95"))
    chk("PC1 base == arm -> Gamma 0 and CI [0,0] on all / ts / nonts, both units", allz and cis and r["status"] == "OK",
        f"status={r['status']} n={r['n_cohort']} {r['errors']}")

    # PC2 the arm differs ONLY on timestamp tokens (E edit hurts the digits 0.25 nats more)
    def shift(rows, on_ts, delta, op_list=("mask", "swap")):
        out = copy.deepcopy(rows)
        for x in out:
            if x.get("hit") == 1:
                for op in op_list:
                    x["tok_lp"][f"{op}_evid"] = [v - (delta if (m == 1) == on_ts else 0.0) for v, m in zip(x["tok_lp"][f"{op}_evid"], x["ts_mask"])]
                _finish(x, ("mask", "swap"))
        return out
    pa2 = _dump(shift(base, True, 0.25), P("14Ln_armts_timer1_default_node2_nat_tok.json"))
    r2 = _run(pb, pa2)
    m = r2["per_op"]["mask"]["subsets"]; sh = r2["per_op"]["mask"]["timestamp_share"]
    chk("PC2 ts-only difference -> nonts Gamma exactly 0, CI [0,0] (both units)",
        m["nonts"]["gamma"] == 0.0 and m["nonts"]["gamma_seqsum"] == 0.0 and m["nonts"]["gamma_ci95"] == [0.0, 0.0]
        and m["nonts"]["gamma_seqsum_ci95"] == [0.0, 0.0], f"nonts g={m['nonts']['gamma']}")
    chk("PC2 ts Gamma == +0.25 nats/token, seqsum == 0.25 * 10, all-token seqsum == ts seqsum",
        abs(m["ts"]["gamma"] - 0.25) < 1e-9 and abs(m["ts"]["gamma_seqsum"] - 2.5) < 1e-9
        and abs(m["all"]["gamma_seqsum"] - 2.5) < 1e-9 and abs(m["all"]["gamma"] - 2.5 / 19) < 1e-9,
        f"ts={m['ts']['gamma']:.6f} ts_seq={m['ts']['gamma_seqsum']:.6f} all={m['all']['gamma']:.6f}")
    chk("PC2 share_of_Gamma == 1 (CI [1,1]) and every per-qid share_gamma == 1; token share 10/19",
        abs(sh["share_of_Gamma_seqsum"] - 1) < 1e-9 and all(abs(x - 1) < 1e-9 for x in sh["share_of_Gamma_seqsum_ci95"])
        and abs(sh["median_share_gamma"] - 1) < 1e-9 and abs(sh["token_share_mean"] - 10 / 19) < 1e-12, str(sh)[:160])

    # PC3 the arm differs ONLY on non-timestamp tokens -> ts Gamma exactly 0, share 0
    pa3 = _dump(shift(base, False, 0.4), P("14Ln_armnon_timer1_default_node2_nat_tok.json"))
    r3 = _run(pb, pa3)
    m3 = r3["per_op"]["swap"]["subsets"]
    chk("PC3 nonts-only difference -> ts Gamma exactly 0 and share 0, nonts Gamma == +0.4",
        m3["ts"]["gamma"] == 0.0 and m3["ts"]["gamma_ci95"] == [0.0, 0.0] and abs(m3["nonts"]["gamma"] - 0.4) < 1e-9
        and abs(r3["per_op"]["swap"]["timestamp_share"]["share_of_Gamma_seqsum"]) < 1e-12)

    # PC4 'all' == read_14Ln_v1 on the row fields (in-reader) and == read_14Ln_v1 itself (if found next to this file)
    pa4 = copy.deepcopy(base); rng = np.random.default_rng(3)
    for x in pa4:
        if x.get("hit") == 1:
            for op in ("mask", "swap"):
                x["tok_lp"][f"{op}_evid"] = [v - float(rng.normal(0.05, 0.1)) for v in x["tok_lp"][f"{op}_evid"]]
            _finish(x, ("mask", "swap"))
    pa4p = _dump(pa4, P("14Ln_armrand_timer1_default_node2_nat_tok.json"))
    r4 = _run(pb, pa4p, B=2000)
    o = r4["per_op"]["mask"]
    chk("PC4 all-token Gamma/CI == row-field (read_14Ln_v1) Gamma/CI on the same draws",
        abs(o["subsets"]["all"]["gamma"] - o["rowfields_read_14Ln_v1"]["gamma"]) < 1e-12
        and np.allclose(o["subsets"]["all"]["gamma_ci95"], o["rowfields_read_14Ln_v1"]["gamma_ci95"], atol=1e-12, rtol=0)
        and np.allclose(o["subsets"]["all"]["gamma_seqsum_ci95"], o["rowfields_read_14Ln_v1"]["gamma_seqsum_ci95"], atol=1e-12, rtol=0),
        f"g={o['subsets']['all']['gamma']:+.5f} ci={[round(v, 5) for v in o['subsets']['all']['gamma_ci95']]}")
    v1p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "read_14Ln_v1.py")
    if os.path.exists(v1p) and sha16(v1p) == READ_V1_SHA16:
        import importlib.util
        sp = importlib.util.spec_from_file_location("read_14Ln_v1_ref", v1p); R1 = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(R1)
        ns = argparse.Namespace(base=pb, arm=pa4p, out=None, expect_n=150, ops="mask,swap", B=2000, seed=0, machine=None,
                                transformers=None, text_first="auto", fixed_cohort=None, strict_machine=False, label=None)
        v1 = R1.analyse(ns)
        same = all(abs(v1["per_op"][op]["gamma"] - r4["per_op"][op]["subsets"]["all"]["gamma"]) < 1e-12
                   and np.allclose(v1["per_op"][op]["gamma_ci95"], r4["per_op"][op]["subsets"]["all"]["gamma_ci95"], atol=1e-12, rtol=0)
                   and np.allclose(v1["per_op"][op]["gamma_seqsum_ci95"], r4["per_op"][op]["subsets"]["all"]["gamma_seqsum_ci95"], atol=1e-12, rtol=0)
                   for op in ("mask", "swap"))
        chk("PC4b read_14Ln_v1 (e18bf155) itself gives the same all-token Gamma and CIs (B=2000, seed 0)", same and v1["n_cohort"] == r4["n_cohort"],
            f"v1 mask {v1['per_op']['mask']['gamma']:+.5f} {[round(v, 5) for v in v1['per_op']['mask']['gamma_ci95']]}")
    else:
        print("  [SKIP] PC4b read_14Ln_v1.py (e18bf155) not next to this file")

    # PC5 shards: base in 2 shards, arm in 3 -> identical numbers
    sb = [_dump(base[:70], P("14Ln_base_timer1_default_node2_nat_tok_s0.json")), _dump(base[70:], P("14Ln_base_timer1_default_node2_nat_tok_s1.json"))]
    sa = [_dump(pa4[i:j], P(f"14Ln_armrand_timer1_default_node2_nat_tok_s{k}.json")) for k, (i, j) in enumerate(((0, 40), (40, 100), (100, 150)))]
    r5 = _run(sb, sa, B=2000)
    chk("PC5 sharded inputs (2 + 3 shards) give identical per-op results", json.dumps(r5["per_op"], sort_keys=True) == json.dumps(r4["per_op"], sort_keys=True)
        and len(r5["inputs"]) == 5)

    # negative controls
    t1 = copy.deepcopy(pa4); h = next(x for x in t1 if x.get("hit") == 1); h["tok_lp"]["clean"][5] += 0.01
    r = _run(pb, _dump(t1, P("14Ln_bad1_timer1_default_node2_nat_tok.json")))
    chk("NC1 one token off by 0.01 (sum no longer = lp_clean_sum) -> ERROR", r["status"] == "ERROR" and r["n_hit_rows_without_tok_arm"] == 1, str(r["errors"])[:120])
    t2 = copy.deepcopy(pa4); h = next(x for x in t2 if x.get("hit") == 1); del h["tok_lp"]
    r = _run(pb, _dump(t2, P("14Ln_bad2_timer1_default_node2_nat_tok.json")))
    chk("NC2 hit row without tok_lp -> ERROR (never read as zero)", r["status"] == "ERROR" and r["n_cohort"] == r4["n_cohort"] - 1)
    r = _run(pb, [sa[0], sa[1], sa[1]])
    chk("NC3 a qid in two shards -> ERROR", r["status"] == "ERROR" and any("duplicate" in e for e in r["errors"]))
    t4 = copy.deepcopy(pa4); h = next(x for x in t4 if x.get("hit") == 1); h["tok_ids"][0] += 1
    r = _run(pb, _dump(t4, P("14Ln_bad4_timer1_default_node2_nat_tok.json")))
    chk("NC4 tok_ids differ between base and arm -> ERROR", r["status"] == "ERROR" and r["n_structural_mismatch"] == 1)
    p5 = _dump(pa4, P("14Ln_bad5_timer1_default_node2_nat_tok.json"), instrument="14Ln_native_sentinel_probe_v1r2_tf_q3_r9", tok=False)
    r = _run(pb, p5)
    chk("NC5 r9 / no --tok-lp meta -> ERROR", r["status"] == "ERROR" and any("r10" in e for e in r["errors"]))
    p6 = _dump(pa4, P("14Ln_bad6_timer1_default_node2_nat_tok.json"), answer_style="vchat")
    r = _run(pb, p6)
    chk("NC6 answer_style differs between base and arm -> ERROR", r["status"] == "ERROR" and any("answer_style" in e for e in r["errors"]))
    t7 = copy.deepcopy(pa4); h = next(x for x in t7 if x.get("hit") == 1); h["ts_mask"] = [1] * len(h["ts_mask"])
    r = _run(pb, _dump(t7, P("14Ln_bad7_timer1_default_node2_nat_tok.json")))
    chk("NC7 ts_mask selecting every token -> ERROR", r["status"] == "ERROR")
    out = P("out_selftest.json"); res = _run(pb, pa2); tsv = write_outputs(res, out)
    lines = open(tsv, encoding="utf-8").read().splitlines()
    chk("TSV written with sha16 header and one line per qid x op", lines[0].startswith("# read_14Ln_tok_v1") and
        sum(1 for l in lines if not l.startswith("#")) == 1 + 2 * res["n_cohort"], f"{len(lines)} lines")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+"); ap.add_argument("--arm", nargs="+"); ap.add_argument("--out")
    ap.add_argument("--expect-n", type=int, default=0, help="total rows per side over all shards (0 = no check)")
    ap.add_argument("--ops", default="mask,swap")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-cohort", dest="fixed_cohort")
    ap.add_argument("--strict-machine", action="store_true")
    ap.add_argument("--share-eps", type=float, default=1e-6, help="per-question shares are NA when |denominator| <= eps")
    ap.add_argument("--label")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sandbox", default="analysis/_selftest_tmp/read_14Ln_tok_v1")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest(a.sandbox))
    if not (a.base and a.arm and a.out):
        ap.error("--base, --arm and --out are required")
    res = analyse(a)
    tsv = write_outputs(res, a.out)

    def f(x, d=3):
        return "n/a" if x is None else f"{x:+.{d}f}"
    print(f"[{res['status']}] {res['label'] or os.path.basename(a.arm[0])} hits b/a {res['n_hits_base']}/{res['n_hits_arm']} "
          f"cohort {res['n_cohort']} -> {a.out} + {os.path.basename(tsv)}")
    for op, d in res["per_op"].items():
        for S in SUBSETS:
            s = d["subsets"][S]
            if s.get("gamma") is not None:
                print(f"  {op:6s} {S:5s}: Gamma {f(s['gamma'])} [{f(s['gamma_ci95'][0])},{f(s['gamma_ci95'][1])}] nats/tok | "
                      f"seqsum {f(s['gamma_seqsum'])} [{f(s['gamma_seqsum_ci95'][0])},{f(s['gamma_seqsum_ci95'][1])}]")
        t = d.get("timestamp_share")
        if t:
            print(f"  {op:6s} ts share of Gamma (seqsum) {f(t['share_of_Gamma_seqsum'])} "
                  f"[{f(t['share_of_Gamma_seqsum_ci95'][0])},{f(t['share_of_Gamma_seqsum_ci95'][1])}] token share {t['token_share_mean']:.3f}")
    for e in res["errors"]:
        print("  ERROR:", e)
    for w in res["warnings"]:
        print("  warn:", w)
    sys.exit(3 if res["status"] == "ERROR" else 0)


if __name__ == "__main__":
    main()
