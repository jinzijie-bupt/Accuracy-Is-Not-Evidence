#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""read_14Lqn_v2.py — reader for 14Lqn_native_query_swap_probe_r3.py products (124cfd516a39f427).

Two read-outs on the same rows (hit == 1 and has_partner == 1):

A) 2x2 cross design (the registered one; C9 upgrade A). Target y_i fixed, query in {q_i, q_j} x edit in {none, E*_i, E_j}:
     psi_own   = [lp(none) - lp(edit E*_i)]|q_i - [lp(none) - lp(edit E_j)]|q_i   (row field {op}_psi_own)
     psi_cross = the same under q_j                                                (row field {op}_psi_cross)
     Specificity = psi_own - psi_cross      Gamma_spec = Specificity(arm) - Specificity(base)
   Every control here is another annotated event of the SAME video, so nothing random enters the estimand.

B) the old (r2-comparable) read-out, on the r3 clean placebo pool T \\ (E*_i u E_j):
     psi_i = tokmean_drop_evid - tokmean_drop_rand (own query), psi_j = same under q_j,
     Gamma = psi_i(arm) - psi_i(base), Gamma_agnostic = psi_j(arm) - psi_j(base), Gamma_sem = Gamma - Gamma_agnostic.
   Rows with pool_short == 1 (|T \\ (E*_i u E_j)| < |E*_i|) are dropped here and the loss is reported explicitly
   (n_pool_short_rep / _base / n_dropped_pool_short); they are KEPT in read-out A, which does not use the pool.

Video-cluster bootstrap (B=2000, seed 0) everywhere; arm-vs-base always on the intersection of partnered hits.
usage: read_14Lqn_v2.py --rep F [--base B] [--ops mask,swap] [--B 2000] [--seed 0] [--out out.json]
       read_14Lqn_v2.py --selftest
"""
import argparse, json, hashlib, os, sys
import numpy as np


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def rows_spec(d, op):
    """Partnered hits that carry the 2x2 fields."""
    out = {}
    for r in d:
        if r.get("hit") == 1 and r.get("has_partner") and (op + "_psi_own") in r and (op + "_psi_cross") in r:
            out[int(r["qid"])] = r
    return out


def rows_old(d, op, drop_short=True):
    """Partnered hits that carry the four token-mean drops; pool_short rows dropped for the old read-out."""
    out = {}
    for r in d:
        if r.get("hit") != 1 or not r.get("has_partner"):
            continue
        if drop_short and int(r.get("pool_short") or 0):
            continue
        if all(k in r for k in (op + "_tokmean_drop_evid", op + "_tokmean_drop_rand",
                                op + "_q2_tokmean_drop_evid", op + "_q2_tokmean_drop_rand")):
            out[int(r["qid"])] = r
    return out


def psi_old(r, op, q2=False):
    k = (op + "_q2_tokmean") if q2 else (op + "_tokmean")
    return float(r[k + "_drop_evid"]) - float(r[k + "_drop_rand"])


def boot_mean(vals, vids, B, seed):
    vals = np.asarray(vals, float); vids = list(vids); uv = sorted(set(vids))
    if len(vals) == 0:
        return None, [None, None]
    idx = {v: [i for i, x in enumerate(vids) if x == v] for v in uv}
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        pick = rng.integers(0, len(uv), len(uv))
        ii = np.concatenate([idx[uv[p]] for p in pick])
        bs.append(vals[ii].mean())
    return float(vals.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def _put(out, name, vals, vids, B, seed):
    m, ci = boot_mean(vals, vids, B, seed); out[name] = m; out[name + "_ci"] = ci


def summarize_spec(d, op, B, seed):
    R = rows_spec(d, op); qq = sorted(R); vids = [R[q]["video"] for q in qq]
    own = [float(R[q][op + "_psi_own"]) for q in qq]
    cro = [float(R[q][op + "_psi_cross"]) for q in qq]
    spe = [a - b for a, b in zip(own, cro)]
    out = {"n": len(qq), "n_videos": len(set(vids)),
           "n_pool_short": sum(1 for q in qq if int(R[q].get("pool_short") or 0))}
    _put(out, "psi_own", own, vids, B, seed); _put(out, "psi_cross", cro, vids, B, seed)
    _put(out, "specificity", spe, vids, B, seed)
    out["pos_frac"] = float(np.mean([x > 0 for x in spe])) if spe else None
    return out


def compare_spec(base, rep, op, B, seed):
    Rb, Rr = rows_spec(base, op), rows_spec(rep, op); qq = sorted(set(Rb) & set(Rr))
    out = {"n": len(qq)}
    if not qq:
        return out
    vids = [Rr[q]["video"] for q in qq]
    d_own = [float(Rr[q][op + "_psi_own"]) - float(Rb[q][op + "_psi_own"]) for q in qq]
    d_cro = [float(Rr[q][op + "_psi_cross"]) - float(Rb[q][op + "_psi_cross"]) for q in qq]
    d_spe = [a - b for a, b in zip(d_own, d_cro)]
    out["n_videos"] = len(set(vids))
    _put(out, "d_psi_own", d_own, vids, B, seed); _put(out, "d_psi_cross", d_cro, vids, B, seed)
    _put(out, "gamma_spec", d_spe, vids, B, seed)
    out["pos_frac"] = float(np.mean([x > 0 for x in d_spe]))
    out["n_mismatch_partner"] = sum(1 for q in qq if Rr[q].get("partner_qid") != Rb[q].get("partner_qid"))
    out["n_mismatch_estar"] = sum(1 for q in qq if Rr[q].get("n_estar") != Rb[q].get("n_estar"))
    out["n_mismatch_nframes"] = sum(1 for q in qq if Rr[q].get("nframes") != Rb[q].get("nframes"))
    return out


def summarize_old(d, op, B, seed):
    R = rows_old(d, op); qq = sorted(R); vids = [R[q]["video"] for q in qq]
    pi = [psi_old(R[q], op) for q in qq]; pj = [psi_old(R[q], op, True) for q in qq]
    out = {"n": len(qq), "n_videos": len(set(vids)),
           "n_partnered_hits": len(rows_old(d, op, drop_short=False)),
           "n_pool_short": len(rows_old(d, op, drop_short=False)) - len(qq)}
    _put(out, "psi_i", pi, vids, B, seed); _put(out, "psi_j", pj, vids, B, seed)
    _put(out, "delta_sem", [a - b for a, b in zip(pi, pj)], vids, B, seed)
    return out


def compare_old(base, rep, op, B, seed):
    Rb, Rr = rows_old(base, op), rows_old(rep, op); qq = sorted(set(Rb) & set(Rr))
    out = {"n": len(qq)}
    if not qq:
        return out
    vids = [Rr[q]["video"] for q in qq]
    g = [psi_old(Rr[q], op) - psi_old(Rb[q], op) for q in qq]
    ga = [psi_old(Rr[q], op, True) - psi_old(Rb[q], op, True) for q in qq]
    gs = [a - b for a, b in zip(g, ga)]
    out["n_videos"] = len(set(vids))
    _put(out, "gamma", g, vids, B, seed); _put(out, "gamma_agnostic", ga, vids, B, seed); _put(out, "gamma_sem", gs, vids, B, seed)
    nb_all, nr_all = rows_old(base, op, drop_short=False), rows_old(rep, op, drop_short=False)
    out["n_pool_short_base"] = len(nb_all) - len(Rb); out["n_pool_short_rep"] = len(nr_all) - len(Rr)
    out["n_intersection_all_rows"] = len(set(nb_all) & set(nr_all))
    out["n_dropped_pool_short"] = out["n_intersection_all_rows"] - len(qq)
    out["sem_share"] = (out["gamma_sem"] / out["gamma"]) if out["gamma"] else None
    return out


def selftest():
    ok = True

    def chk(c, m):
        nonlocal ok
        ok &= bool(c)
        print(("  PASS " if c else "  FAIL ") + m)

    def row(q, v, own, cross, ei, oi, ej, oj, hit=1, hp=1, short=0, pq=9):
        return {"qid": q, "video": v, "hit": hit, "has_partner": hp, "pool_short": short, "partner_qid": pq,
                "n_estar": 4, "nframes": 20,
                "mask_psi_own": own, "mask_psi_cross": cross,
                "mask_tokmean_drop_evid": ei, "mask_tokmean_drop_rand": oi,
                "mask_q2_tokmean_drop_evid": ej, "mask_q2_tokmean_drop_rand": oj}

    d = [row(0, "a", 1.0, 0.2, 1.0, 0.2, 0.5, 0.1), row(1, "a", 0.6, 0.6, 0.8, 0.0, 0.0, 0.0),
         row(2, "b", 0.4, 0.1, 0.3, 0.3, 0.3, 0.3), row(3, "b", 9, 9, 9, 9, 9, 9, hit=0),
         row(4, "c", 9, 9, 9, 9, 9, 9, hp=0), row(5, "c", 0.2, 0.0, 5.0, 0.0, 5.0, 0.0, short=1)]
    s = summarize_spec(d, "mask", 50, 0)
    chk(s["n"] == 4 and s["n_videos"] == 3 and s["n_pool_short"] == 1,
        "read-out A keeps every partnered hit, pool_short included (the 2x2 term never touches the pool)")
    chk(abs(s["specificity"] - (0.8 + 0.0 + 0.3 + 0.2) / 4) < 1e-12 and s["pos_frac"] == 0.75,
        "Specificity = psi_own - psi_cross, exact mean and pos_frac")
    so = summarize_old(d, "mask", 50, 0)
    chk(so["n"] == 3 and so["n_pool_short"] == 1 and so["n_partnered_hits"] == 4,
        "read-out B drops pool_short rows and reports the loss (absence is a value)")
    chk(abs(so["psi_i"] - (0.8 + 0.8 + 0.0) / 3) < 1e-12 and abs(so["psi_j"] - (0.4 + 0.0 + 0.0) / 3) < 1e-12,
        "psi_i / psi_j exact on the clean-pool read-out")

    b = [row(0, "a", 0.5, 0.1, 0.5, 0.1, 0.5, 0.1), row(1, "a", 0.3, 0.2, 0.1, 0.0, 0.0, 0.0), row(7, "z", 1, 1, 1, 1, 1, 1)]
    c = compare_spec(b, d, "mask", 50, 0)
    chk(c["n"] == 2, "arm-vs-base uses the intersection of partnered hits")
    chk(abs(c["gamma_spec"] - (((1.0 - 0.2) - (0.5 - 0.1)) + ((0.6 - 0.6) - (0.3 - 0.2))) / 2) < 1e-12,
        "Gamma_spec = Spec(arm) - Spec(base), exact")
    chk(abs(c["gamma_spec"] - (c["d_psi_own"] - c["d_psi_cross"])) < 1e-12,
        "Gamma_spec = dGamma(psi_own) - dGamma(psi_cross) (the identity the 2x2 rests on)")
    co = compare_old(b, d, "mask", 50, 0)
    chk(abs(co["gamma"] - (co["gamma_agnostic"] + co["gamma_sem"])) < 1e-12, "Gamma = Gamma_agnostic + Gamma_sem")
    chk(co["n_dropped_pool_short"] == 0 and co["n_pool_short_rep"] == 1,
        "the old read-out reports its own n loss from pool_short")

    same = compare_spec(d, d, "mask", 50, 0)
    chk(same["n"] == 4 and same["gamma_spec"] == 0.0 and same["d_psi_own"] == 0.0 and same["gamma_spec_ci"] == [0.0, 0.0],
        "NEGATIVE: arm == base -> Gamma_spec is identically 0 (point estimate and bootstrap CI)")
    deg = [row(0, "a", 0.0, 0.0, 1.0, 0.2, 1.0, 0.2), row(1, "b", 0.0, 0.0, 0.5, 0.5, 0.5, 0.5)]
    sd = summarize_spec(deg, "mask", 50, 0)
    chk(sd["specificity"] == 0.0 and sd["psi_own"] == 0.0,
        "NEGATIVE: E_j := E*_i (edit identical, psi_own = psi_cross = 0) -> Specificity degenerates to 0")
    chk(summarize_spec([], "mask", 10, 0)["n"] == 0 and compare_spec([], d, "mask", 10, 0)["n"] == 0
        and summarize_old([], "mask", 10, 0)["n"] == 0 and compare_old([], d, "mask", 10, 0)["n"] == 0,
        "empty inputs -> n=0, no crash")
    mism = compare_spec([row(0, "a", 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, pq=3)], d, "mask", 50, 0)
    chk(mism["n"] == 1 and mism["n_mismatch_partner"] == 1,
        "pairing precondition is checked, not assumed (partner_qid mismatch is counted)")
    print("[selftest]", "ALL PASS" if ok else "FAIL")
    return ok


def fmt(v, nd=3):
    return "n/a" if v is None else ("%+.*f" % (nd, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep"); ap.add_argument("--base", default=None); ap.add_argument("--ops", default="mask,swap")
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    rep = json.load(open(a.rep, encoding="utf-8"))
    base = json.load(open(a.base, encoding="utf-8")) if a.base else None
    out = {"reader": "read_14Lqn_v2", "reader_sha16": sha16(os.path.abspath(__file__)), "B": a.B, "seed": a.seed,
           "inputs": [{"role": "rep", "file": os.path.basename(a.rep), "path": a.rep.replace("\\", "/"), "sha16": sha16(a.rep)}]}
    if base is not None:
        out["inputs"].append({"role": "base", "file": os.path.basename(a.base), "path": a.base.replace("\\", "/"), "sha16": sha16(a.base)})
    for nm, d in (("rep", rep), ("base", base)):
        if d is None:
            continue
        out[nm + "_counts"] = {"n_rows": len(d), "n_hits": sum(1 for r in d if r.get("hit") == 1),
                               "n_partnered_hits": sum(1 for r in d if r.get("hit") == 1 and r.get("has_partner")),
                               "n_pool_short": sum(1 for r in d if int(r.get("pool_short") or 0)),
                               "instrument": (d[0].get("prov", {}) or {}).get("instrument") if d else None}
    for op in [o for o in a.ops.split(",") if o]:
        blk = {"spec": {"rep": summarize_spec(rep, op, a.B, a.seed)}, "old": {"rep": summarize_old(rep, op, a.B, a.seed)}}
        if base is not None:
            blk["spec"]["base"] = summarize_spec(base, op, a.B, a.seed)
            blk["spec"]["vs_base"] = compare_spec(base, rep, op, a.B, a.seed)
            blk["old"]["base"] = summarize_old(base, op, a.B, a.seed)
            blk["old"]["vs_base"] = compare_old(base, rep, op, a.B, a.seed)
        out[op] = blk
        s = blk["spec"]["rep"]
        line = (f"{op}: [2x2] n={s['n']} psi_own {fmt(s['psi_own'])} psi_cross {fmt(s['psi_cross'])} "
                f"Spec {fmt(s['specificity'])} {[None if x is None else round(x, 3) for x in s['specificity_ci']]}")
        if base is not None and blk["spec"]["vs_base"].get("n"):
            c = blk["spec"]["vs_base"]
            line += (f" | vs base n={c['n']} Gamma_spec {fmt(c['gamma_spec'])} "
                     f"{[None if x is None else round(x, 3) for x in c['gamma_spec_ci']]} "
                     f"(dpsi_own {fmt(c['d_psi_own'])} dpsi_cross {fmt(c['d_psi_cross'])}; "
                     f"mismatch partner/estar/nframes {c['n_mismatch_partner']}/{c['n_mismatch_estar']}/{c['n_mismatch_nframes']})")
        print(line)
        o = blk["old"]["rep"]
        line = f"{op}: [old] n={o['n']} (pool_short dropped {o['n_pool_short']}) psi_i {fmt(o['psi_i'])} psi_j {fmt(o['psi_j'])}"
        if base is not None and blk["old"]["vs_base"].get("n"):
            c = blk["old"]["vs_base"]
            line += (f" | vs base n={c['n']} Gamma {fmt(c['gamma'])} agnostic {fmt(c['gamma_agnostic'])} "
                     f"{[None if x is None else round(x, 3) for x in c['gamma_agnostic_ci']]} sem {fmt(c['gamma_sem'])} "
                     f"(n lost to pool_short {c['n_dropped_pool_short']})")
        print(line)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("->", a.out)


if __name__ == "__main__":
    main()
