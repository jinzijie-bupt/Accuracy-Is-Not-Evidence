# -*- coding: utf-8 -*-
"""A2: evidence dependence on newly-correct questions.
NC = {qid in 0..199 : trained model hit (scored) and base miss}.  H_int = both hit.  Same-environment pairs only.
Per op: psi_arm(NC) mean + CI; psi_arm(H_int) mean; psi_base(H_base) mean + CI;
unpaired diff psi_arm(NC) - psi_base(H_base) and psi_arm(H_arm) - psi_base(H_base) with a joint video-cluster
bootstrap over the 200-question universe (B=2000, seed 2); paired Gamma on H_int for reference."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *

CFG = [  # (label, env, base_stem, arm_stem)
    ("Qwen7B pure SFT", "A", "14Ln_base_timer1_default_node2", "14Ln_natsft_b0_s0_tf5_timer1_default_node2"),
    ("Qwen7B pure SFT", "B", "14Ln_base_timer1_default_node1", "14Ln_natsft_b0_s0_tf5_timer1_default_node1"),
    ("Qwen7B pure SFT", "C", "14Ln_base_timer1_node4", "14Ln_natsft_b0_s0_timer1_default_node4"),
    ("Qwen7B IoU-GRPO", "B", "14Ln_base_timer1_default_node1", "14Ln_x0s0iou700_timer1_default_node1"),
    ("Qwen7B grouped a=0.25", "B", "14Ln_base_timer1_default_node1", "14Ln_natbase_a025_g1_s0_timer1_default_node1"),
    ("Qwen7B grouped a=0.5", "A", "14Ln_base_timer1_default_node2", "14Ln_natbase_a05_g1_s0_tf5_timer1_default_node2"),
    ("Qwen7B grouped a=0.5", "C", "14Ln_base_timer1_node4", "14Ln_natbase_a05_g1_s0_timer1_default_node4"),
    ("Qwen7B grouped a=1.0 s0", "A", "14Ln_base_timer1_default_node2", "14Ln_natbase_a10_g1_s0_timer1_node2"),
    ("Qwen7B grouped a=1.0 s1", "A", "14Ln_base_timer1_default_node2", "14Ln_natbase_a10_g1_s1_timer1_default_node2"),
    ("TimeZero (public RL) vs Qwen", "A", "14Ln_base_timer1_default_node2", "14Ln_tzbase_timer1_default_node2"),
    ("TimeZero (public RL) vs Qwen", "B", "14Ln_base_timer1_default_node3", "14Ln_tzbase_timer1_default_node3"),
    ("Time-R1 (public RL) vs Qwen", "B", "14Ln_base_timer1_default_node3", "14Ln_tr1base_timer1_default_node3"),
    ("Time-R1 (public RL) vs Qwen", "C", "14Ln_base_timer1_node4", "14Ln_tr1base_timer1_default_node4"),
]

res, inputs = [], []
for lab, env, bs, as_ in CFG:
    bp, ap = audit_path(bs), audit_path(as_)
    assert env_of(bp) == env_of(ap) == env, (bp, ap)
    inputs += [bp, ap]
    Br, Ar = load(bp), load(ap)
    assert [int(r["qid"]) for r in Br] == [int(r["qid"]) for r in Ar] == list(range(200))
    assert [r["video"] for r in Br] == [r["video"] for r in Ar]
    hb, ha = hitset(Br), hitset(Ar)
    qs = list(range(200)); vids = [r["video"] for r in Ar]
    inB = np.array([q in hb for q in qs]); inA = np.array([q in ha for q in qs])
    NC, NL, HI = inA & ~inB, inB & ~inA, inA & inB
    bt = Boot(vids, B=B_DEFAULT, seed=2)
    row = {"label": lab, "env": env, "base": bp, "arm": ap, "n_base_hits": int(inB.sum()), "n_arm_hits": int(inA.sum()),
           "n_NC": int(NC.sum()), "n_NL": int(NL.sum()), "n_H_int": int(HI.sum())}
    for op in OPS:
        pa = np.array([psi(ha[q], op) if q in ha else np.nan for q in qs])
        pb = np.array([psi(hb[q], op) if q in hb else np.nan for q in qs])
        d = {}
        d["psi_arm_NC"] = float(pa[NC].mean()) if NC.any() else None
        d["psi_arm_NC_ci95"] = bt.ci(masked_mean(pa, NC)) if NC.sum() >= 2 else [None, None]
        d["psi_arm_NC_frac_pos"] = float((pa[NC] > 0).mean()) if NC.any() else None
        d["psi_arm_Hint"] = float(pa[HI].mean())
        d["psi_arm_allhits"] = float(pa[inA].mean())
        d["psi_base_Hint"] = float(pb[HI].mean())
        d["psi_base_basehits"] = float(pb[inB].mean())
        d["psi_base_basehits_ci95"] = bt.ci(masked_mean(pb, inB))
        d["psi_base_NL"] = float(pb[NL].mean()) if NL.any() else None
        d["gamma_Hint"] = float((pa[HI] - pb[HI]).mean())

        def diff_nc(ix, pa=pa, pb=pb):
            m1, m2 = NC[ix], inB[ix]
            return float(pa[ix][m1].mean() - pb[ix][m2].mean()) if (m1.any() and m2.any()) else np.nan

        def diff_all(ix, pa=pa, pb=pb):
            m1, m2 = inA[ix], inB[ix]
            return float(pa[ix][m1].mean() - pb[ix][m2].mean())
        d["diff_NC_minus_basehits"] = (d["psi_arm_NC"] - d["psi_base_basehits"]) if NC.any() else None
        d["diff_NC_minus_basehits_ci95"] = bt.ci(diff_nc) if NC.any() else [None, None]
        d["diff_allhits_unpaired"] = d["psi_arm_allhits"] - d["psi_base_basehits"]
        d["diff_allhits_unpaired_ci95"] = bt.ci(diff_all)
        row[op] = d
    res.append(row)

out = write_out("a2_newly_correct", __file__, {"configs": res, "bootstrap": {"B": B_DEFAULT, "seed": 2, "cluster": "video",
                "universe": "200 audit qids; NC/base-hit membership recomputed inside each draw"}}, inputs)
for r in res:
    m, s = r["mask"], r["swap"]
    print(f'{r["label"]:30s} {r["env"]} |NC| {r["n_NC"]:3d} |NL| {r["n_NL"]:3d} H {r["n_H_int"]:3d} | mask psiNC {f3(m["psi_arm_NC"])} {fci(m["psi_arm_NC_ci95"])} '
          f'psiA(H) {f3(m["psi_arm_Hint"])} psiB(Hb) {f3(m["psi_base_basehits"])} {fci(m["psi_base_basehits_ci95"])} '
          f'diff {f3(m["diff_NC_minus_basehits"])} {fci(m["diff_NC_minus_basehits_ci95"])} all {f3(m["diff_allhits_unpaired"])} {fci(m["diff_allhits_unpaired_ci95"])}')
    print(f'{"":33s} swap psiNC {f3(s["psi_arm_NC"])} {fci(s["psi_arm_NC_ci95"])} psiA(H) {f3(s["psi_arm_Hint"])} psiB(Hb) {f3(s["psi_base_basehits"])} '
          f'diff {f3(s["diff_NC_minus_basehits"])} {fci(s["diff_NC_minus_basehits_ci95"])} all {f3(s["diff_allhits_unpaired"])} {fci(s["diff_allhits_unpaired_ci95"])}')
print(out)
