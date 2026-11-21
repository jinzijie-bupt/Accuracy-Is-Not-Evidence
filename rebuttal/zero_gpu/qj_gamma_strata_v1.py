#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qj_gamma_strata_v1.py (revision, zero-GPU, 2026-11-06)

Q-J for the likelihood audit: split the per-question Gamma (psi_arm - psi_base on the cohort H_int, nats/token,
as written per question by the registered reader analysis/read_14Ln_v1.py into gamma_*.json -> per_qid_gamma)
by prior-solvability of the question, and report Gamma per stratum + the interaction (prior - not prior).

Strata are fixed attributes of a Charades-STA test question (qid = line index of charades_sta_test.txt):
  const : the constant (0,10) s window has IoU>=0.5 with the annotation (no model)
  blind : >=1 of six BASE products (tm, 12p5n, op=black) hits IoU>=0.5 on the all-black video
          (labels read from qc_qj_p5n_v1.py's inputs; no trained model involved)
Labels exist for qid 0..999; the audit uses qid 0..199.

Guards: (1) mean(per_qid_gamma) must reproduce the reader's reported Gamma for both ops;
        (2) each per-qid video must equal the annotation video of that qid;
        (3) the raw 14Ln base/arm products named in the gamma json must exist and match the recorded sha16.
Video-cluster bootstrap B=2000 seed 0.  Output: qj_gamma_strata_v1.json + .tsv
"""
import os, json, hashlib, datetime
import numpy as np
LOCAL_ROOT = os.environ.get("LOCAL_ROOT", ".")  # analysis directory holding the pulled result files (results/, rebuttal/)

HERE = os.path.dirname(os.path.abspath(__file__))
G = (LOCAL_ROOT + "/results/judgment_20260921/gamma/")
B, SEED = 2000, 0

ROWS = [  # (label, file, kind, paper)
    ("Qwen7B a=0.5 (node4)", "gamma_14Ln_natbase_a05_g1_s0_timer1_default_node4__vs__14Ln_base_timer1_node4.json", "placebo-controlled", "Table 3 row 1 (+1.63/+1.41)"),
    ("TimeZero a=0.5 (node2)", "gamma_14Ln_nattz_a05_g1_s0_timer1_default_node2__vs__14Ln_tzbase_timer1_default_node2.json", "placebo-controlled", "Table 3 row 2 (+1.40/+1.31)"),
    ("Time-R1 a=0.5 (node3)", "gamma_14Ln_nattr1_a05_g1_s0_timer1_default_node3__vs__14Ln_tr1base_timer1_default_node3.json", "placebo-controlled", "Table 3 row 3 (+1.46/+1.20)"),
    ("VideoChat-R1 a=0.5 (node4)", "gamma_14Ln_natvc_a05_g1_s0_vchat_node4__vs__14Ln_vcbase_vchat_node4.json", "placebo-controlled", "Table 3 row 4 (+1.43/+0.89)"),
    ("TimeLens-8B a=0.5 (node3)", "gamma_14Ln_nattimelens8b_a05_g1_s0_timelens_node3__vs__14Ln_timelens8b_base_timelens_node3.json", "placebo-controlled", "Table 3 row 5 (+1.36/+0.88)"),
    ("Qwen7B SFT (node4)", "gamma_14Ln_natsft_b0_s0_timer1_default_node4__vs__14Ln_base_timer1_node4.json", "accuracy (SFT)", "Table 1 / S1"),
    ("Qwen7B mDPO-style (node4)", "gamma_14Ln_natmdpo_a05_s0_timer1_default_node4__vs__14Ln_base_timer1_node4.json", "accuracy (mDPO)", "Table 1 / S1"),
    ("Qwen7B IoU-GRPO (node1)", "gamma_14Ln_x0s0iou700_timer1_default_node1__vs__14Ln_base_timer1_default_node1.json", "accuracy (IoU-GRPO)", "Table 1 / S1"),
]
P5N_RESULT = os.path.join(HERE, "qc_qj_p5n_v1.json")


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def labels():
    """recompute the two stratum labels from the same product files qc_qj_p5n_v1.py used (sha-checked)."""
    res = json.load(open(P5N_RESULT, encoding="utf-8"))
    inp = res["inputs"]
    ref = {int(r["qid"]): r for r in json.load(open(inp["node5_qwenbase"]["file"], encoding="utf-8"))}
    assert sha16(inp["node5_qwenbase"]["file"]) == inp["node5_qwenbase"]["sha16"]
    const, video = {}, {}
    for q, r in ref.items():
        gs, ge = float(r["gsA"]), float(r["geA"])
        inter = max(0.0, min(10.0, ge) - max(0.0, gs)); uni = max(10.0, ge) - min(0.0, gs)
        const[q] = int((inter / uni if uni > 0 else 0) >= 0.5); video[q] = r["video"]
    blind = {q: 0 for q in ref}
    used = []
    for k in ["node5_qwenbase", "node5_qwenbase_vchat", "node5_tzbase_tf", "node5_tzbase_vf", "node5_tr1base", "node5_vcbase_tf"]:
        f = inp[k]["file"]; assert sha16(f) == inp[k]["sha16"], k
        used.append({"key": k, "file": f, "sha16": inp[k]["sha16"]})
        for r in json.load(open(f, encoding="utf-8")):
            if r.get("black_hit"):
                blind[int(r["qid"])] = 1
    return const, blind, video, used, {"file": P5N_RESULT, "sha16": sha16(P5N_RESULT)}


def groups_of(v):
    ix = {}
    for i, x in enumerate(v):
        ix.setdefault(x, []).append(i)
    return [np.array(g) for g in ix.values()]


def boot(groups, B=B, seed=SEED):
    rng = np.random.default_rng(seed); Gn = len(groups)
    return [np.concatenate([groups[k] for k in rng.integers(0, Gn, Gn)]) for _ in range(B)]


def summ(pt, d):
    d = np.asarray([x for x in d if x is not None], float)
    return {"est": float(pt), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def main():
    const, blind, video, used, p5n_src = labels()
    out = {"script": os.path.basename(__file__), "script_sha16": sha16(os.path.abspath(__file__)),
           "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "params": {"B": B, "seed": SEED, "cluster": "video", "unit": "nats/token (tokmean), as Table 3"},
           "label_sources": used, "label_source_result": p5n_src, "rows": []}
    tsv = ["row\tkind\tpaper\top\tstratum_def\tn_cohort\tGamma_all\tn_prior\tGamma_prior\tn_notprior\tGamma_notprior\tinteraction(prior-not)"]
    for lab, fn, kind, paper in ROWS:
        p = G + fn
        d = json.load(open(p, encoding="utf-8"))
        raw_chk = []
        for inp in d["inputs"]:
            ok = os.path.exists(inp["path"]) and sha16(inp["path"]) == inp["sha16"]
            raw_chk.append({"role": inp["role"], "file": inp["path"], "sha16_recorded": inp["sha16"], "exists_and_matches": ok})
        pq = d["per_qid_gamma"]
        qs = [int(x["qid"]) for x in pq]
        for x in pq:
            if video[int(x["qid"])] != x["video"]:
                raise SystemExit("video mismatch %s qid %s" % (fn, x["qid"]))
        rec = {"row": lab, "kind": kind, "paper": paper, "gamma_file": p, "gamma_file_sha16": sha16(p),
               "raw_inputs": raw_chk, "n_cohort": len(qs), "machine": d.get("machine"), "ops": {}}
        vids = [x["video"] for x in pq]
        Gr = groups_of(vids); D = boot(Gr)
        for op in ("mask", "swap"):
            g = np.array([float(x["gamma_%s" % op]) for x in pq])
            rep = d["per_op"][op]["gamma"]
            if abs(g.mean() - rep) > 1e-9:
                raise SystemExit("per-qid mean %.6f != reported %.6f (%s %s)" % (g.mean(), rep, fn, op))
            orec = {"Gamma_all": summ(g.mean(), [g[ix].mean() for ix in D]), "reported_Gamma": rep, "strata": {}}
            for sname, L in (("const", const), ("blind", blind)):
                f = np.array([L[q] for q in qs])
                s = {"n_prior": int(f.sum()), "n_notprior": int((1 - f).sum())}
                for lev, nm in ((1, "prior_solvable"), (0, "not_prior_solvable")):
                    m = f == lev
                    if m.sum() < 15:
                        s[nm] = {"note": "fewer than 15 cohort questions: not estimated (absent, not 0)", "n": int(m.sum())}; continue
                    sg = [v for v, mm in zip(vids, m) if mm]; gm = g[m]
                    Dm = boot(groups_of(sg))
                    s[nm] = summ(gm.mean(), [gm[ix].mean() for ix in Dm]); s[nm]["n"] = int(m.sum())

                def inter(ix):
                    a, b = ix[f[ix] == 1], ix[f[ix] == 0]
                    return None if (not a.size or not b.size) else g[a].mean() - g[b].mean()
                if "est" in s.get("prior_solvable", {}) and "est" in s.get("not_prior_solvable", {}):
                    s["interaction_prior_minus_not"] = summ(inter(np.arange(len(qs))), [inter(ix) for ix in D])
                orec["strata"][sname] = s
                fmt = lambda e: "NA" if not e or "est" not in e else "%+.2f [%+.2f, %+.2f]" % (e["est"], e["ci95"][0], e["ci95"][1])
                tsv.append("\t".join([lab, kind, paper, op, sname, str(len(qs)), fmt(orec["Gamma_all"]), str(s["n_prior"]),
                                      fmt(s.get("prior_solvable")), str(s["n_notprior"]), fmt(s.get("not_prior_solvable")),
                                      fmt(s.get("interaction_prior_minus_not"))]))
            rec["ops"][op] = orec
        out["rows"].append(rec)
        print("done", lab, [r["exists_and_matches"] for r in raw_chk])
    json.dump(out, open(os.path.join(HERE, "qj_gamma_strata_v1.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    open(os.path.join(HERE, "qj_gamma_strata_v1.tsv"), "w", encoding="utf-8").write("\n".join(tsv) + "\n")


if __name__ == "__main__":
    main()
