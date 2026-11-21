# -*- coding: utf-8 -*-
"""A4: per-question agreement between the likelihood reading (14Ln, psi) and generation behaviour (12p5n).
Model-by-model, same environment, same message order, same adapter (verified from prov + clean-prediction agreement).
Cohort: qid 0..199, likelihood hit (scored) AND behaviour clean hit (none_hit==1) AND not degenerate.
Behavioural contrast per question b_op = {op}_rand_hit - {op}_evid_hit  (= evidence-failure - placebo-failure).
Reported per op: Spearman(psi_op, b_op) and Pearson (point-biserial generalisation, b is ternary),
Spearman(psi_op, evidence failure), and the median split: mean b (psi high) - mean b (psi low).
All CIs are video-cluster bootstrap (B=2000, seed 3) over the cohort."""
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..", "..")).replace("\\", "/")
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *
from scipy.stats import spearmanr, pearsonr

P5 = RES + "/native_p5n_20260920/"
PROV = _SUPP + "/results/run_order/"
MATCH = [  # (label, env, behaviour file, likelihood stem)
    ("Qwen7B base", "B(node5|node1)", "p5n6_qwenbase_timer1_default_node5_n1000_o0.json", "14Ln_base_timer1_default_node1"),
    ("Qwen7B pure SFT (tf5 adapter)", "B(node5|node1)", "p5n6_natsft_b0_s0_timer1_default_node5_n1000_o0.json", "14Ln_natsft_b0_s0_tf5_timer1_default_node1"),
    ("Time-R1 base", "B(node5|node3)", "p5n6_tr1base_timer1_default_node5_n1000_o0.json", "14Ln_tr1base_timer1_default_node3"),
    ("Time-R1 grouped a=0.1", "B(node3)", "p5n_nattr1_a01_g1_s0_timer1_default_node3.json", "14Ln_nattr1_a01_g1_s0_timer1_default_node3"),
    ("TimeZero grouped a=0.5 b=0.03", "B(node3)", "p5n_nattz_a05_b003_g1_s0_timer1_default_node3.json", "14Ln_nattz_a05_b003_g1_s0_timer1_default_node3"),
    ("Qwen7B grouped a=0.5", "C(node4)", "p5n_natbase_a05_g1_s0_timer1_default_node4_n1000_o0.json", "14Ln_natbase_a05_g1_s0_timer1_default_node4"),
    ("Qwen7B pure SFT", "C(node4)", "p5n_natsft_b0_s0_timer1_default_node4_n1000_o0.json", "14Ln_natsft_b0_s0_timer1_default_node4"),
    ("Time-R1 base", "C(node4)", "p5n_tr1base_timer1_default_node4_n1000_o0.json", "14Ln_tr1base_timer1_default_node4"),
    ("Time-R1 grouped a=0.25", "C(node4)", "p5n_nattr1_a025_g1_s0_timer1_default_node4_n1000_o0.json", "14Ln_nattr1_a025_g1_s0_timer1_default_node4"),
    ("VideoChat-R1 base (vchat)", "C(node4)", "p5n_vcbase_vchat_node4_n1000_o0.json", "14Ln_vcbase_vchat_node4"),
    ("VideoChat-R1 grouped a=0.5 (vchat)", "C(node4)", "p5n_natvc_a05_g1_s0_vchat_node4_n1000_o0.json", "14Ln_natvc_a05_g1_s0_vchat_node4"),
]
EXCLUDED = [
    "p5n6_natbase_a05 (node5, tf5 adapter, env B): the only likelihood audits of this adapter are env A (node2) -> cross-env, excluded",
    "p5n6_nattr1_a025 (node5, tf5 adapter): likelihood audit exists only for the tf4 (node4) adapter -> different adapter, excluded",
    "p5n6_tzbase / p5n6_nattz_a05 / p5n6_vcbase / p5n6_natvc_a05 (node5): run TEXT-FIRST (prov text_first=True); no env-B text-first "
    "likelihood audit of the same checkpoint -> excluded (NB: appendix table S6 uses these node5 text-first products)",
]


def spear(x, y):
    if len(set(x)) < 2 or len(set(y)) < 2:
        return np.nan
    return float(spearmanr(x, y)[0])


res, inputs = [], []
for lab, env, bf, stem in MATCH:
    bpath = P5 + bf; lpath = audit_path(stem)
    inputs += [bpath, lpath]
    pv = PROV + bf + ".prov.json"
    tf = None
    if os.path.exists(pv):
        inputs.append(pv); tf = load(pv).get("text_first")
    Bh = {int(r["qid"]): r for r in load(bpath)}
    L = load(lpath); hl = hitset(L)
    agree = sum(1 for r in L if int(r["qid"]) in Bh and Bh[int(r["qid"])].get("none_pred") == r.get("pred"))
    coh = [q for q in range(200) if q in hl and Bh[q].get("none_hit") == 1 and not Bh[q].get("degenerate")]
    nmis = sum(1 for q in coh if Bh[q].get("n_estar") != hl[q].get("n_estar") or Bh[q].get("video") != hl[q].get("video"))
    vids = [hl[q]["video"] for q in coh]
    bt = Boot(vids, B=B_DEFAULT, seed=3)
    row = {"label": lab, "env": env, "behaviour": bpath, "likelihood": lpath, "behaviour_text_first": tf,
           "clean_pred_agreement_of_200": agree, "n_lik_hits": len(hl), "n_cohort": len(coh), "n_videos": bt.nv,
           "n_struct_mismatch": nmis}
    for op in OPS:
        if not all(f"{op}_evid_hit" in Bh[q] for q in coh):
            row[op] = None      # node4 behaviour products carry the mask operator only
            continue
        ps = np.array([psi(hl[q], op) for q in coh])
        eh = np.array([int(Bh[q][f"{op}_evid_hit"]) for q in coh]); rh = np.array([int(Bh[q][f"{op}_rand_hit"]) for q in coh])
        b = rh - eh; ef = 1 - eh
        med = float(np.median(ps)); hi = ps > med
        d = {"evid_fail_rate": float(ef.mean()), "placebo_fail_rate": float(1 - rh.mean()), "b_mean": float(b.mean()),
             "spearman_psi_b": spear(ps, b), "pearson_psi_b": float(pearsonr(ps, b)[0]) if len(set(b)) > 1 else None,
             "spearman_psi_evidfail": spear(ps, ef), "psi_median": med, "psi_mean": float(ps.mean()),
             "b_high": float(b[hi].mean()), "b_low": float(b[~hi].mean()),
             "evidfail_high": float(ef[hi].mean()), "evidfail_low": float(ef[~hi].mean())}
        d["b_high_minus_low"] = d["b_high"] - d["b_low"]
        d["spearman_psi_b_ci95"] = bt.ci(lambda ix: spear(ps[ix], b[ix]))
        d["spearman_psi_evidfail_ci95"] = bt.ci(lambda ix: spear(ps[ix], ef[ix]))

        def split(ix):
            h = hi[ix]
            return float(b[ix][h].mean() - b[ix][~h].mean()) if (h.any() and (~h).any()) else np.nan
        d["b_high_minus_low_ci95"] = bt.ci(split)
        row[op] = d
    res.append(row)

ml = [(r["mask"]["psi_mean"], r["mask"]["b_mean"], r["mask"]["evid_fail_rate"]) for r in res]
model_level = {"n_models": len(ml), "spearman_mean_psi_vs_mean_b": float(spearmanr([a for a, _, _ in ml], [b for _, b, _ in ml])[0]),
               "spearman_mean_psi_vs_evid_fail": float(spearmanr([a for a, _, _ in ml], [c for _, _, c in ml])[0])}
print("MODEL-LEVEL", model_level)
out = write_out("a4_lik_vs_behav", __file__, {"models": res, "model_level_mask": model_level, "excluded": EXCLUDED,
                "bootstrap": {"B": B_DEFAULT, "seed": 3, "cluster": "video"},
                "median_split": "psi > median(psi) within model = high"}, inputs)
for r in res:
    print(f'{r["label"][:34]:34s} {r["env"]:12s} agree {r["clean_pred_agreement_of_200"]:3d} n {r["n_cohort"]:3d} tf {r["behaviour_text_first"]}')
    for op in OPS:
        d = r[op]
        if d is None:
            continue
        print(f'    {op}: evidFail {d["evid_fail_rate"]:.2f} plcFail {d["placebo_fail_rate"]:.2f} rho(psi,b) {f3(d["spearman_psi_b"])} {fci(d["spearman_psi_b_ci95"])} '
              f'r {f3(d["pearson_psi_b"])} rho(psi,ef) {f3(d["spearman_psi_evidfail"])} {fci(d["spearman_psi_evidfail_ci95"])} '
              f'b hi/lo {d["b_high"]:.2f}/{d["b_low"]:.2f} diff {f3(d["b_high_minus_low"])} {fci(d["b_high_minus_low_ci95"])}')
print(out)
