#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""read_14Lpn_v1.py -- registered reader for
14Lpn_native_paraphrase_probe_v1_tf_r1 products (probe ec9a10f8d534288c).

Construction: partner_kind == "sibling" is a second human-written caption of the
SAME annotated interval (ej_idx == estar_idx by construction, so {op}_psi_own == 0
identically; that field is a construction check here, not a readout).

Estimator (native analogue of the controlled-protocol paraphrase control, P3a):
  per row with hit == 1 and has_partner == 1, per op in {mask, swap}:
      psi_q1 = {op}_tokmean_drop_evid    - {op}_tokmean_drop_rand      (original query)
      psi_q2 = {op}_q2_tokmean_drop_evid - {op}_q2_tokmean_drop_rand   (sibling paraphrase)
  cohort  = qids hit+partnered in BOTH files (base and arm)
  per model:  mean psi_q1, mean psi_q2, delta = mean(psi_q2 - psi_q1)   [paraphrase difference]
  paired:     Gamma_q1 = mean(psi_q1_arm - psi_q1_base), Gamma_q2 likewise,
              retention = Gamma_q2 / Gamma_q1
  CIs: video-cluster bootstrap, B=2000, seed 0. Units: nats/token (tokmean).

Guards: psi_own must be identically 0 (|.| <= 1e-9) on every used row; both files must
carry the same probe_sha16; cohort n printed; status ERROR trips exit 3.

usage: read_14Lpn_v1.py --base BASE.json --arm ARM.json [--ops mask,swap]
                        [--B 2000] [--seed 0] [--label L] --out OUT.json
"""
import argparse, hashlib, json, os, sys
import numpy as np


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def load(path, ops):
    rows = json.load(open(path, encoding="utf-8"))
    out = {}
    bad_own = 0
    for r in rows:
        if r.get("hit") != 1 or r.get("has_partner") != 1:
            continue
        ok = True
        vals = {}
        for op in ops:
            for k in (f"{op}_tokmean_drop_evid", f"{op}_tokmean_drop_rand",
                      f"{op}_q2_tokmean_drop_evid", f"{op}_q2_tokmean_drop_rand"):
                if r.get(k) is None:
                    ok = False
            if not ok:
                break
            own = r.get(f"{op}_psi_own")
            if own is None or abs(own) > 1e-9:
                bad_own += 1
            vals[f"{op}_q1"] = r[f"{op}_tokmean_drop_evid"] - r[f"{op}_tokmean_drop_rand"]
            vals[f"{op}_q2"] = r[f"{op}_q2_tokmean_drop_evid"] - r[f"{op}_q2_tokmean_drop_rand"]
        if ok:
            vals["video"] = r["video"]
            out[r["qid"]] = vals
    prov = rows[0].get("prov", {}) if rows else {}
    return out, prov, bad_own, len(rows)


def cluster_ci(qids, videos, fn, B, seed):
    """fn(idx_subset)->stat; resample videos with replacement."""
    rng = np.random.default_rng(seed)
    vids = sorted(set(videos))
    byv = {}
    for i, v in enumerate(videos):
        byv.setdefault(v, []).append(i)
    stats = []
    for _ in range(B):
        pick = rng.choice(len(vids), size=len(vids), replace=True)
        idx = [i for p in pick for i in byv[vids[p]]]
        stats.append(fn(idx))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ops", default="mask,swap")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ops = a.ops.split(",")

    base, pb, bo_b, nb = load(a.base, ops)
    arm, pa, bo_a, na = load(a.arm, ops)
    status, notes = "OK", []
    if bo_b or bo_a:
        status = "ERROR"
        notes.append(f"psi_own nonzero on {bo_b}+{bo_a} rows (construction broken)")
    if pb.get("probe_sha16") != pa.get("probe_sha16"):
        status = "ERROR"
        notes.append("probe_sha16 mismatch between base and arm")
    cohort = sorted(set(base) & set(arm))
    if len(cohort) < 30:
        status = "ERROR"
        notes.append(f"cohort too small: {len(cohort)}")

    videos = [base[q]["video"] for q in cohort]
    res = {}
    for op in ops:
        b1 = np.array([base[q][f"{op}_q1"] for q in cohort])
        b2 = np.array([base[q][f"{op}_q2"] for q in cohort])
        a1 = np.array([arm[q][f"{op}_q1"] for q in cohort])
        a2 = np.array([arm[q][f"{op}_q2"] for q in cohort])
        g1, g2 = a1 - b1, a2 - b2
        db, da = b2 - b1, a2 - a1
        r = {
            "n": len(cohort),
            "psi_base_q1": float(b1.mean()), "psi_base_q2": float(b2.mean()),
            "psi_arm_q1": float(a1.mean()), "psi_arm_q2": float(a2.mean()),
            "gamma_q1": float(g1.mean()), "gamma_q2": float(g2.mean()),
            "retention": float(g2.mean() / g1.mean()) if abs(g1.mean()) > 1e-12 else None,
            "delta_base": float(db.mean()), "delta_arm": float(da.mean()),
        }
        r["gamma_q1_ci"] = cluster_ci(cohort, videos, lambda i: g1[i].mean(), a.B, a.seed)
        r["gamma_q2_ci"] = cluster_ci(cohort, videos, lambda i: g2[i].mean(), a.B, a.seed)
        r["retention_ci"] = cluster_ci(
            cohort, videos,
            lambda i: g2[i].mean() / g1[i].mean() if abs(g1[i].mean()) > 1e-12 else np.nan,
            a.B, a.seed)
        r["delta_base_ci"] = cluster_ci(cohort, videos, lambda i: db[i].mean(), a.B, a.seed)
        r["delta_arm_ci"] = cluster_ci(cohort, videos, lambda i: da[i].mean(), a.B, a.seed)
        res[op] = r

    out = {
        "reader": os.path.basename(__file__), "reader_sha16": sha16(__file__),
        "estimator": "psi_q = tokmean_drop_evid - tokmean_drop_rand under q1 (original) and q2 (sibling paraphrase); Gamma paired on hit cohort; video-cluster bootstrap",
        "label": a.label, "status": status, "notes": notes,
        "inputs": {"base": {"path": a.base, "sha16": sha16(a.base), "rows": nb},
                   "arm": {"path": a.arm, "sha16": sha16(a.arm), "rows": na}},
        "probe_prov": pa, "B": a.B, "seed": a.seed,
        "cohort_n": len(cohort), "results": res,
    }
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"[{status}] wrote {a.out} cohort={len(cohort)}")
    for op in ops:
        r = res[op]
        print(f"  {op}: G_q1={r['gamma_q1']:+.3f} {r['gamma_q1_ci']} G_q2={r['gamma_q2']:+.3f} {r['gamma_q2_ci']} "
              f"retention={r['retention']:.3f} {r['retention_ci']} d_base={r['delta_base']:+.4f} {r['delta_base_ci']} d_arm={r['delta_arm']:+.4f} {r['delta_arm_ci']}")
    sys.exit(0 if status == "OK" else 3)


if __name__ == "__main__":
    main()
