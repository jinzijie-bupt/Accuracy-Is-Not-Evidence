#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""read_14Ln_v1.py -- registered reader for
14Ln_native_sentinel_probe_r3 products (instrument 14Ln_native_sentinel_probe_v1r2[_tf]).

Replaces the print-only analysis/pair_14Ln_tmp.py (975e0ecab2283480): same estimator,
same video-cluster bootstrap stream, but with provenance (input sha16, reader_sha16,
machine, transformers, protocol), a dual-unit cross-check, per-question dumps, a fixed
cohort mode and a --selftest bench.

Estimator (unchanged from pair_14Ln_tmp.py):
  per hit row i and operator op in {mask, swap}
      dE_i  = {op}_tokmean_drop_evid   = (lp_clean_sum - lp(op applied to E*))     / tgt_len
      dO_i  = {op}_tokmean_drop_rand   = (lp_clean_sum - mean_R lp(op on rand nonev)) / tgt_len
      psi_i = dE_i - dO_i
  H(f)      = {qid : hit == 1 and all requested ops scored}
  cohort    = H(base) & H(arm)            (or the given --fixed-cohort intersected with it)
  Gamma     = mean_{cohort} ( psi_arm - psi_base )
  CI        = video-cluster bootstrap over the cohort's videos, B=2000, seed 0.

Units: tokmean (nats/token) is primary, seqsum ({op}_drop_*) is reported alongside;
per question seqsum/tokmean == tgt_len exactly, which is checked (memory: units-seqsum-vs-tokmean).

usage:
  read_14Ln_v1.py --base BASE.json --arm ARM.json [--expect-n 200] [--ops mask,swap]
                  [--B 2000] [--seed 0] [--machine M] [--transformers V]
                  [--text-first auto|true|false] [--fixed-cohort qids.txt]
                  [--strict-machine] [--label L] --out OUT.json
  read_14Ln_v1.py --selftest [--sandbox DIR]

Exit codes: 0 ok (status OK/WARN), 3 status ERROR (guard tripped; the JSON is still written).
"""
import argparse, json, hashlib, os, sys, datetime
import numpy as np

MACHINE_TOKENS = ["node2", "node1", "node4", "node3", "node5"]
TF_BY_MACHINE = {"node2": "5.13", "node1": "5.12", "node3": "5.12", "node5": "5.12", "node4": "4.51"}
# 14Ln_base_timer1_default_{node1,node3} are byte-identical -> one environment (memory: C10 risk 5)
SHARED_BASE_SHA = "370af855b3e3ff91"
EQUIV_MACHINES = {"node1", "node3"}


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


def infer_machine(path):
    b = "_" + os.path.basename(path)
    for m in MACHINE_TOKENS:
        if f"_{m}_" in b:
            return m
    return None


def infer_instrument(path):
    b = os.path.basename(path)
    for pfx in ("14Lqn", "14Ln"):
        if b.startswith(pfx + "_"):
            return pfx
    return "unknown"


def infer_text_first(path):
    """filename token wins; else .meta.json; else unknown. Probe does bool(args.text_first)."""
    b = os.path.basename(path)
    if "textfirst" in b or "_tf_" in b:
        return True, "filename"
    mp = path + ".meta.json"
    if os.path.exists(mp):
        try:
            m = json.load(open(mp, encoding="utf-8"))
            v = m.get("args", {}).get("text_first", "__absent__")
            if v == "__absent__":
                return "unknown", "meta(no key)"
            return bool(v), "meta" + ("(None->false)" if v is None else "")
        except Exception:
            return "unknown", "meta(unreadable)"
    return "unknown", "none"


def load_rows(path):
    d = json.load(open(path, encoding="utf-8"))
    if not isinstance(d, list):
        raise SystemExit(f"not a row list: {path}")
    return d


def hitset(rows, ops):
    out = {}
    for r in rows:
        if r.get("hit") == 1 and all(f"{op}_tokmean_drop_evid" in r and f"{op}_tokmean_drop_rand" in r for op in ops):
            out[int(r["qid"])] = r
    return out


def hitset_first_op_only(rows, ops):
    out = set()
    for r in rows:
        if r.get("hit") == 1 and f"{ops[0]}_tokmean_drop_evid" in r:
            out.add(int(r["qid"]))
    return out


def side_stats(hits, op, qids=None):
    keys = sorted(hits) if qids is None else [q for q in qids if q in hits]
    if not keys:
        return {"n": 0, "dE": None, "dO": None, "psi": None,
                "dE_seqsum": None, "dO_seqsum": None, "psi_seqsum": None, "tgt_len_mean": None}
    e = np.array([float(hits[q][f"{op}_tokmean_drop_evid"]) for q in keys])
    o = np.array([float(hits[q][f"{op}_tokmean_drop_rand"]) for q in keys])
    es = np.array([float(hits[q][f"{op}_drop_evid"]) for q in keys])
    os_ = np.array([float(hits[q][f"{op}_drop_rand"]) for q in keys])
    tl = np.array([float(hits[q]["tgt_len"]) for q in keys])
    return {"n": len(keys), "dE": float(e.mean()), "dO": float(o.mean()), "psi": float((e - o).mean()),
            "dE_seqsum": float(es.mean()), "dO_seqsum": float(os_.mean()), "psi_seqsum": float((es - os_).mean()),
            "tgt_len_mean": float(tl.mean())}


def cluster_boot(vals, vids, rng, B):
    vals = np.asarray(vals, float)
    uv = sorted(set(vids))
    gi = {v: [i for i, x in enumerate(vids) if x == v] for v in uv}
    bs = []
    for _ in range(B):
        pick = rng.integers(0, len(uv), len(uv))
        idx = np.concatenate([gi[uv[p]] for p in pick])
        bs.append(vals[idx].mean())
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], len(uv)


def read_cohort_file(p):
    txt = open(p, encoding="utf-8").read().strip()
    if txt.startswith("["):
        return [int(x) for x in json.loads(txt)]
    out = []
    for line in txt.splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.append(int(line))
    return out


def analyse(a):
    errors, warns = [], []
    ops = [o for o in a.ops.split(",") if o]
    bpath, apath = os.path.abspath(a.base), os.path.abspath(a.arm)
    B_rows, A_rows = load_rows(bpath), load_rows(apath)
    b_sha, a_sha = sha16(bpath), sha16(apath)

    if a.expect_n:
        for tag, rows in (("base", B_rows), ("arm", A_rows)):
            if len(rows) != a.expect_n:
                errors.append(f"expect-n {a.expect_n} but {tag} has {len(rows)} rows")

    mb, ma = infer_machine(bpath), infer_machine(apath)
    equiv = (mb != ma and {mb, ma} <= EQUIV_MACHINES and b_sha == SHARED_BASE_SHA)
    if mb != ma:
        if equiv:
            warns.append(f"cross-machine pairing allowed: base sha16={SHARED_BASE_SHA} is the byte-identical node1/node3 base")
        else:
            msg = f"machine mismatch base={mb} arm={ma} (base sha16={b_sha}, not the shared node1=node3 base)"
            (errors if a.strict_machine else warns).append(msg)
    machine = a.machine or ("node1≡node3" if (equiv or (b_sha == SHARED_BASE_SHA and ma in EQUIV_MACHINES)) else (ma or "unknown"))
    tf = a.transformers or TF_BY_MACHINE.get(ma, "unknown")

    ib, ia = infer_instrument(bpath), infer_instrument(apath)
    if ib != ia:
        errors.append(f"instrument mismatch base={ib} arm={ia}")

    if a.text_first == "auto":
        tfb, srcb = infer_text_first(bpath)
        tfa, srca = infer_text_first(apath)
        if tfb != tfa:
            errors.append(f"text_first mismatch base={tfb}({srcb}) arm={tfa}({srca})")
        text_first, tf_src = tfa, srca
    else:
        text_first, tf_src = (a.text_first == "true"), "cli"

    sb = sorted({r.get("style") for r in B_rows if r.get("style")})
    sa = sorted({r.get("style") for r in A_rows if r.get("style")})
    if sb != sa:
        errors.append(f"prompt style mismatch base={sb} arm={sa}")
    order = ("text-first" if text_first is True else
             ("video-first" if text_first is False else "order-unknown(probe default video-first)"))
    protocol = f"{'/'.join(sa) or 'unknown'} · {order}"

    hb, ha = hitset(B_rows, ops), hitset(A_rows, ops)
    for tag, rows, hs in (("base", B_rows, hb), ("arm", A_rows, ha)):
        only = hitset_first_op_only(rows, ops)
        if len(only) != len(hs):
            warns.append(f"{tag}: {len(only)-len(hs)} hit rows have {ops[0]} but not every op; dropped")
    inter = sorted(set(hb) & set(ha))

    # per-question structural agreement on the cohort -- never silent (memory: absence-as-a-value)
    mism = []
    for q in inter:
        for k in ("n_estar", "nframes", "tgt_len", "video"):
            if hb[q].get(k) != ha[q].get(k):
                mism.append({"qid": q, "field": k, "base": hb[q].get(k), "arm": ha[q].get(k)})
    if mism:
        errors.append(f"per-qid structural mismatch on {len({m['qid'] for m in mism})} qid(s): "
                      + ", ".join(sorted({m['field'] for m in mism})))

    cohort_name, cohort_missing = "H_int", None
    cohort = inter
    if a.fixed_cohort:
        want = read_cohort_file(a.fixed_cohort)
        cohort = [q for q in want if q in hb and q in ha]
        cohort_missing = {"requested": len(want), "kept": len(cohort),
                          "missing_in_base": len([q for q in want if q not in hb]),
                          "missing_in_arm": len([q for q in want if q not in ha])}
        cohort_name = "fixed:" + os.path.basename(a.fixed_cohort)
    if not cohort:
        errors.append("empty cohort")

    q2v = {int(r["qid"]): r.get("video") for r in A_rows}
    rng = np.random.default_rng(a.seed)            # tokmean stream: byte-identical to pair_14Ln_tmp.py
    rng_s = np.random.default_rng(a.seed + 10000)  # separate stream for the seqsum side-channel
    out_ops = {}
    for op in ops:
        base_all = side_stats(hb, op)
        arm_all = side_stats(ha, op)
        base_c = side_stats(hb, op, cohort)
        arm_c = side_stats(ha, op, cohort)
        g = np.array([(float(ha[q][f"{op}_tokmean_drop_evid"]) - float(ha[q][f"{op}_tokmean_drop_rand"]))
                      - (float(hb[q][f"{op}_tokmean_drop_evid"]) - float(hb[q][f"{op}_tokmean_drop_rand"]))
                      for q in cohort]) if cohort else np.array([])
        gs = np.array([(float(ha[q][f"{op}_drop_evid"]) - float(ha[q][f"{op}_drop_rand"]))
                       - (float(hb[q][f"{op}_drop_evid"]) - float(hb[q][f"{op}_drop_rand"]))
                       for q in cohort]) if cohort else np.array([])
        vids = [q2v[q] for q in cohort]
        if len(g):
            ci, nv = cluster_boot(g, vids, rng, a.B)
            ci_s, _ = cluster_boot(gs, vids, rng_s, a.B)
            gamma, gamma_s, pos = float(g.mean()), float(gs.mean()), float((g > 0).mean())
        else:
            ci = ci_s = [None, None]; nv = 0; gamma = gamma_s = pos = None
        # unit cross-check: per question seqsum/tokmean must equal tgt_len exactly
        dev = 0.0
        for q in cohort:
            for side in (hb[q], ha[q]):
                tl = float(side["tgt_len"])
                for kind in ("evid", "rand"):
                    node5 = float(side[f"{op}_tokmean_drop_{kind}"])
                    ss = float(side[f"{op}_drop_{kind}"])
                    if abs(node5) > 1e-9:
                        dev = max(dev, abs(ss / node5 - tl))
        if dev > 1e-4:
            errors.append(f"{op}: seqsum/tokmean ratio deviates from tgt_len by {dev:.3g}")
        out_ops[op] = {
            "gamma": gamma, "gamma_ci95": ci, "pos_frac": pos,
            "gamma_seqsum": gamma_s, "gamma_seqsum_ci95": ci_s,
            "n_cohort": len(cohort), "n_videos": nv,
            "base_all_hits": base_all, "arm_all_hits": arm_all,
            "base_on_cohort": base_c, "arm_on_cohort": arm_c,
            "units": {"primary": "tokmean nats/token", "secondary": "seqsum nats",
                      "tgt_len_mean_cohort": arm_c["tgt_len_mean"],
                      "seqsum_over_tokmean_max_abs_dev_from_tgt_len": dev,
                      "gamma_seqsum_over_gamma": (gamma_s / gamma) if (gamma not in (None, 0.0)) else None},
        }

    per_qid = [{"qid": q, "video": q2v[q],
                **{f"gamma_{op}": float((float(ha[q][f'{op}_tokmean_drop_evid']) - float(ha[q][f'{op}_tokmean_drop_rand']))
                                        - (float(hb[q][f'{op}_tokmean_drop_evid']) - float(hb[q][f'{op}_tokmean_drop_rand'])))
                   for op in ops}} for q in cohort]

    res = {
        "reader": "read_14Ln_v1", "reader_sha16": self_sha16(),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": a.label, "instrument": ia, "ops": ops,
        "status": "ERROR" if errors else ("WARN" if warns else "OK"),
        "errors": errors, "warnings": warns,
        "inputs": [{"role": "base", "file": os.path.basename(bpath), "path": bpath.replace("\\", "/"), "sha16": b_sha,
                    "machine": mb, "n_rows": len(B_rows)},
                   {"role": "arm", "file": os.path.basename(apath), "path": apath.replace("\\", "/"), "sha16": a_sha,
                    "machine": ma, "n_rows": len(A_rows)}],
        "paired_base_path": bpath.replace("\\", "/"),
        "machine": machine, "machine_base": mb, "machine_arm": ma,
        "transformers_version": tf, "transformers_source": ("cli" if a.transformers else "machine-map"),
        "text_first": text_first, "text_first_source": tf_src, "protocol": protocol,
        "prompt_style": sa, "expect_n": a.expect_n,
        "n_rows_base": len(B_rows), "n_rows_arm": len(A_rows),
        "n_hits_base": len(hb), "n_hits_arm": len(ha), "n_H_int": len(inter),
        "cohort": cohort_name, "n_cohort": len(cohort), "cohort_detail": cohort_missing,
        "clean_R50_base": float(np.mean([r.get("hit", 0) for r in B_rows])),
        "clean_R50_arm": float(np.mean([r.get("hit", 0) for r in A_rows])),
        "base_parsed": int(sum(r.get("clean_parsed", 0) for r in B_rows)),
        "arm_parsed": int(sum(r.get("clean_parsed", 0) for r in A_rows)),
        "bootstrap": {"B": a.B, "seed": a.seed, "cluster": "video",
                      "note": "tokmean CIs replicate pair_14Ln_tmp.py bit-for-bit (one rng, ops in order); seqsum CIs use seed+10000"},
        "structural_mismatch": mism[:20], "n_structural_mismatch": len(mism),
        "per_op": out_ops,
        "per_qid_gamma": per_qid,
    }
    return res


# ------------------------------------------------------------------ selftest

def _mk_rows(n, seed, tgt_len=18, hit_every=2, psi_mask=0.20, psi_swap=0.10, jitter=0.0, nvid=25):
    rng = np.random.default_rng(seed)
    vrng = np.random.default_rng(seed + 1)          # videos assigned at random: non-degenerate clusters
    vassign = [int(vrng.integers(0, nvid)) for _ in range(n)]
    rows = []
    for q in range(n):
        v = f"v{vassign[q]}.mp4"
        hit = 1 if q % hit_every == 0 else 0
        r = {"qid": q, "video": v, "n_estar": 12, "nframes": 60, "fps": 2.0,
             "style": "timer1_default", "target_mode": "gt", "sentinel": "none", "dilate_k": 0,
             "clean_iou": 0.7 if hit else 0.1, "hit": hit, "clean_parsed": 1, "pred": [1.0, 2.0],
             "clean_text": "x"}
        if hit:
            tl = tgt_len
            r["tgt_len"] = tl
            r["lp_clean_sum"] = -float(rng.uniform(20, 40))
            r["lp_clean_mean"] = r["lp_clean_sum"] / tl
            for op, psi in (("mask", psi_mask), ("swap", psi_swap)):
                dO = float(rng.normal(0.1, 0.05))
                dE = dO + psi + (float(rng.normal(0.0, jitter)) if jitter else 0.0)
                r[f"{op}_tokmean_drop_evid"] = dE
                r[f"{op}_tokmean_drop_rand"] = dO
                r[f"{op}_drop_evid"] = dE * tl
                r[f"{op}_drop_rand"] = dO * tl
                r[f"{op}_rand_sums"] = [0.0, 0.0, 0.0]
                r[f"{op}_rand_idx"] = [[0], [1], [2]]
                r[f"{op}_donor"] = "d.mp4"
        rows.append(r)
    return rows


def _run(base, arm, out, **kw):
    ns = argparse.Namespace(base=base, arm=arm, out=out, expect_n=kw.get("expect_n", 0),
                            ops=kw.get("ops", "mask,swap"), B=kw.get("B", 400), seed=0,
                            machine=kw.get("machine"), transformers=kw.get("transformers"),
                            text_first=kw.get("text_first", "auto"), fixed_cohort=kw.get("fixed_cohort"),
                            strict_machine=kw.get("strict_machine", False), label=kw.get("label"))
    return analyse(ns)


def selftest(sandbox):
    os.makedirs(sandbox, exist_ok=True)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    base = _mk_rows(200, 11)
    pb = os.path.join(sandbox, "14Ln_base_timer1_default_node2_nat_o0n200R3.json")
    json.dump(base, open(pb, "w", encoding="utf-8"))

    # PC1 identity: arm == base -> Gamma exactly 0, CI [0,0], pos_frac 0 (all per-qid diffs are exactly 0)
    r = _run(pb, pb, None, expect_n=200)
    m = r["per_op"]["mask"]
    chk("PC1 identity gamma==0", m["gamma"] == 0.0 and m["gamma_ci95"] == [0.0, 0.0] and m["pos_frac"] == 0.0,
        f"g={m['gamma']} ci={m['gamma_ci95']} pos={m['pos_frac']}")
    chk("PC1 identity n_H_int == n_hits", r["n_H_int"] == r["n_hits_base"] == r["n_hits_arm"], f"n={r['n_H_int']}")
    chk("PC1 status OK", r["status"] == "OK", str(r["errors"]))

    # PC2 exactly-mean-zero jitter null: Gamma ~ 0, CI straddles 0, pos_frac ~ 0.5
    rng = np.random.default_rng(7)
    arm = json.loads(json.dumps(base))
    hitrows = [r0 for r0 in arm if r0.get("hit") == 1]
    for op in ("mask", "swap"):
        eps = rng.normal(0.0, 0.15, len(hitrows))
        eps = eps - eps.mean()                       # null by construction
        for r0, e in zip(hitrows, eps):
            r0[f"{op}_tokmean_drop_evid"] += float(e)
            r0[f"{op}_drop_evid"] = r0[f"{op}_tokmean_drop_evid"] * r0["tgt_len"]
    pa = os.path.join(sandbox, "14Ln_natnull_s0_timer1_default_node2_nat_o0n200R3.json")
    json.dump(arm, open(pa, "w", encoding="utf-8"))
    r = _run(pb, pa, None, expect_n=200, B=2000)
    m = r["per_op"]["mask"]
    chk("PC2 null gamma ~ 0 and CI contains 0",
        abs(m["gamma"]) < 1e-9 and m["gamma_ci95"][0] < 0 < m["gamma_ci95"][1],
        f"g={m['gamma']:+.4f} ci={[round(x,4) for x in m['gamma_ci95']]}")
    chk("PC2 pos_frac ~ 0.5", 0.35 <= m["pos_frac"] <= 0.65, f"pos={m['pos_frac']:.2f}")

    # PC3 known shift: arm psi = base psi + 0.5 -> Gamma == +0.5 exactly, pos_frac == 1
    arm3 = json.loads(json.dumps(base))
    for r0 in arm3:
        if r0.get("hit") == 1:
            for op in ("mask", "swap"):
                r0[f"{op}_tokmean_drop_evid"] += 0.5
                r0[f"{op}_drop_evid"] = r0[f"{op}_tokmean_drop_evid"] * r0["tgt_len"]
    p3 = os.path.join(sandbox, "14Ln_natshift_s0_timer1_default_node2_nat_o0n200R3.json")
    json.dump(arm3, open(p3, "w", encoding="utf-8"))
    r = _run(pb, p3, None, expect_n=200)
    m = r["per_op"]["swap"]
    chk("PC3 known +0.5 shift recovered", abs(m["gamma"] - 0.5) < 1e-9 and m["pos_frac"] == 1.0,
        f"g={m['gamma']:.6f}")
    chk("PC3 seqsum unit ratio == tgt_len", abs(m["units"]["gamma_seqsum_over_gamma"] - 18.0) < 1e-6,
        f"ratio={m['units']['gamma_seqsum_over_gamma']:.4f}")

    # NC1 cross-machine base with --strict-machine must ERROR (and must WARN without it)
    pb_node1 = os.path.join(sandbox, "14Ln_base_timer1_default_node1_nat_o0n200R3.json")
    json.dump(base, open(pb_node1, "w", encoding="utf-8"))
    r_strict = _run(pb_node1, p3, None, expect_n=200, strict_machine=True)
    r_loose = _run(pb_node1, p3, None, expect_n=200)
    chk("NC1 cross-machine ERROR under --strict-machine",
        r_strict["status"] == "ERROR" and any("machine mismatch" in e for e in r_strict["errors"]))
    chk("NC1 cross-machine WARN without it (mechanism removed -> no error)",
        r_loose["status"] == "WARN" and not r_loose["errors"])

    # NC2 per-qid n_estar/nframes mismatch must ERROR, not pass silently
    arm4 = json.loads(json.dumps(arm3))
    for r0 in arm4:
        if r0.get("hit") == 1:
            r0["n_estar"] = 99
            break
    p4 = os.path.join(sandbox, "14Ln_natbad_s0_timer1_default_node2_nat_o0n200R3.json")
    json.dump(arm4, open(p4, "w", encoding="utf-8"))
    r = _run(pb, p4, None, expect_n=200)
    chk("NC2 n_estar mismatch -> ERROR", r["status"] == "ERROR" and r["n_structural_mismatch"] == 1,
        str(r["errors"]))

    # NC3 broken unit consistency (seqsum no longer tgt_len * tokmean) must ERROR
    arm5 = json.loads(json.dumps(arm3))
    for r0 in arm5:
        if r0.get("hit") == 1:
            r0["mask_drop_evid"] = r0["mask_drop_evid"] * 1.7
            break
    p5 = os.path.join(sandbox, "14Ln_natunit_s0_timer1_default_node2_nat_o0n200R3.json")
    json.dump(arm5, open(p5, "w", encoding="utf-8"))
    r = _run(pb, p5, None, expect_n=200)
    chk("NC3 unit inconsistency -> ERROR", r["status"] == "ERROR" and any("seqsum/tokmean" in e for e in r["errors"]),
        str(r["errors"]))

    # NC4 wrong --expect-n must ERROR
    r = _run(pb, p3, None, expect_n=1541)
    chk("NC4 expect-n guard", r["status"] == "ERROR" and any("expect-n" in e for e in r["errors"]))

    # NC5 fixed cohort restricts the sample and reports what it dropped
    cf = os.path.join(sandbox, "cohort.txt")
    open(cf, "w", encoding="utf-8").write("\n".join(str(q) for q in [0, 2, 4, 6, 8, 1]))
    r = _run(pb, p3, None, expect_n=200, fixed_cohort=cf)
    chk("NC5 fixed cohort kept 5/6 (qid 1 is not a hit)",
        r["n_cohort"] == 5 and r["cohort_detail"]["missing_in_base"] == 1, str(r["cohort_detail"]))

    # NC6 text_first mismatch between the two ends must ERROR
    p6 = os.path.join(sandbox, "14Ln_natshift_s0_timer1_default_textfirst_node2_nat_o0n200R3.json")
    json.dump(arm3, open(p6, "w", encoding="utf-8"))
    r = _run(pb, p6, None, expect_n=200)
    chk("NC6 text_first mismatch -> ERROR", r["status"] == "ERROR" and any("text_first" in e for e in r["errors"]))

    # PC4 replication against the frozen print-only script on real products, if present
    rb = "results/native_audit_20260917/14Ln_base_timer1_default_node1_nat_o0n200R3.json"
    ra = "results/native_audit_20260917/14Ln_natbase_a05_abl_single_s0_timer1_default_node1_nat_o0n200R3.json"
    if os.path.exists(rb) and os.path.exists(ra):
        r = _run(rb, ra, None, expect_n=200, B=2000)
        m, s = r["per_op"]["mask"], r["per_op"]["swap"]
        chk("PC4 abl_single replicates C10 (mask +0.536 [+0.341,+0.711], swap +1.088)",
            abs(m["gamma"] - 0.536) < 0.0005 and abs(m["gamma_ci95"][0] - 0.341) < 0.0005
            and abs(m["gamma_ci95"][1] - 0.711) < 0.0005 and abs(s["gamma"] - 1.088) < 0.0005,
            f"mask {m['gamma']:+.4f} {[round(x,4) for x in m['gamma_ci95']]} swap {s['gamma']:+.4f}")
    else:
        print("  [SKIP] PC4 replication (products not on this host)")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"); ap.add_argument("--arm"); ap.add_argument("--out")
    ap.add_argument("--expect-n", type=int, default=200)
    ap.add_argument("--ops", default="mask,swap")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--machine"); ap.add_argument("--transformers")
    ap.add_argument("--text-first", dest="text_first", default="auto", choices=["auto", "true", "false"])
    ap.add_argument("--fixed-cohort", dest="fixed_cohort")
    ap.add_argument("--strict-machine", action="store_true")
    ap.add_argument("--label")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sandbox", default="analysis/_selftest_tmp/read_14Ln_v1")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest(a.sandbox))
    if not (a.base and a.arm):
        ap.error("--base and --arm are required")
    res = analyse(a)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    m = res["per_op"].get("mask", {}); s = res["per_op"].get("swap", {})

    def f(x, d=3):
        return "n/a" if x is None else f"{x:+.{d}f}"
    print(f"[{res['status']}] {res['label'] or os.path.basename(a.arm)} machine={res['machine']} tf={res['transformers_version']} "
          f"hits b/a {res['n_hits_base']}/{res['n_hits_arm']} H_int {res['n_H_int']} cohort {res['n_cohort']}")
    for nm, d in (("mask", m), ("swap", s)):
        if d:
            print(f"  {nm}: Gamma {f(d['gamma'])} [{f(d['gamma_ci95'][0])},{f(d['gamma_ci95'][1])}] pos {d['pos_frac']} "
                  f"| arm dE {f(d['arm_all_hits']['dE'])} dO {f(d['arm_all_hits']['dO'])} psi {f(d['arm_all_hits']['psi'])}")
    for e in res["errors"]:
        print("  ERROR:", e)
    for w in res["warnings"]:
        print("  warn:", w)
    sys.exit(3 if res["status"] == "ERROR" else 0)


if __name__ == "__main__":
    main()
