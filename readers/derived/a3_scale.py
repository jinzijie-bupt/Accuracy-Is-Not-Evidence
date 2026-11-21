# -*- coding: utf-8 -*-
"""A3: is Gamma a scale artefact?  For every trained-vs-base pair with 14Ln audits in the same environment:
clean NLL (-lp_clean_mean, nats/token) of arm and base on H_int; scale-free variants
 (a) sign fraction  P(psi_arm > psi_base) on H_int, Wilson 95% CI (iid) and video-cluster CI;
 (b) normalised contrast c = (dE - dO)/(|dE| + |dO| + 0.05); Gamma_c = mean(c_arm - c_base), video-cluster CI.
Then across configs: Spearman(Gamma_mask, Delta NLL); sign/rank agreement of Gamma, sign fraction, Gamma_c.
Pair list = every 14Ln gamma record in paper_numbers_v1.json with a mask value (Charades + chtl + qvh), recomputed
from the raw audits (and checked bit-exact against the registry), plus the public-RL-vs-Qwen pairs of A1."""
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..", "..")).replace("\\", "/")
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *
from scipy.stats import spearmanr

REGF = os.path.join(_SUPP, "tables_and_figures", "tables", "paper_numbers_v1.json")
reg = load(REGF)
pairs = []
for r in reg["records"]:
    if r["kind"] != "gamma" or r.get("gamma_mask") is None:
        continue
    ins = {i["role"]: i for i in (r.get("inputs") or [])}
    if not ins or not ins["arm"]["file"].startswith("14Ln_"):
        continue                                   # 14Lqn / 14Ln7 records have no local 14Ln raw file pair here
    pairs.append({"label": r["name"], "family": r["family"], "objective": r["objective"], "alpha": r.get("alpha"),
                  "beta": r.get("beta"), "env": r["env"], "dataset": r["dataset"],
                  "base": ins["base"]["path"], "arm": ins["arm"]["path"], "reg_gamma_mask": r["gamma_mask"],
                  "reg_gamma_mask_ci": r.get("gamma_mask_ci"), "reg_sha": (ins["base"]["sha16"], ins["arm"]["sha16"])})
EXTRA = [
    ("TimeZero(public RL) vs Qwen", "TimeZero-7B vs Qwen", "public-RL", "A", "14Ln_base_timer1_default_node2", "14Ln_tzbase_timer1_default_node2"),
    ("TimeZero(public RL) vs Qwen", "TimeZero-7B vs Qwen", "public-RL", "B", "14Ln_base_timer1_default_node3", "14Ln_tzbase_timer1_default_node3"),
    ("Time-R1(public RL) vs Qwen", "Time-R1-7B vs Qwen", "public-RL", "B", "14Ln_base_timer1_default_node3", "14Ln_tr1base_timer1_default_node3"),
    ("Time-R1(public RL) vs Qwen", "Time-R1-7B vs Qwen", "public-RL", "C", "14Ln_base_timer1_node4", "14Ln_tr1base_timer1_default_node4"),
    ("natsft_b0_s0_tf5_timer1_default_node2", "Qwen2.5-VL-7B", "sft", "A", "14Ln_base_timer1_default_node2", "14Ln_natsft_b0_s0_tf5_timer1_default_node2"),
    ("natsftshrink069_b0_s0_timer1_default_node1", "Qwen2.5-VL-7B", "sft-shrink0.69", "B", "14Ln_base_timer1_default_node1", "14Ln_natsftshrink069_b0_s0_timer1_default_node1"),
    # 09-25: TimeLens-8B against its own base Qwen3-VL-8B (TimeLens template); = judgment_20260921/gamma/gamma_tl8b_vs_q3vl8b_node3.json
    ("TimeLens-8B(public) vs Qwen3-VL-8B", "TimeLens-8B vs Qwen3-VL-8B", "public-RL", "B", "14Ln_q3vl8b_base_timelens_node3", "14Ln_timelens8b_base_timelens_node3"),
]
for lab, fam, obj, env, bs, as_ in EXTRA:
    pairs.append({"label": lab, "family": fam, "objective": obj, "alpha": None, "beta": None, "env": env,
                  "dataset": "charades-sta", "base": audit_path(bs), "arm": audit_path(as_), "reg_gamma_mask": None})
# 09-25: TimeLens-label audits of the two node4-trained Qwen2.5-VL-7B models, read on their training machine
for lab, obj, a, b, as_ in [("chtl_natbase_a05_g1_s0_timer1_default_node4", "grouped", 0.5, 0.1, "14Ln_chtl_natbase_a05_g1_s0_timer1_default_node4"),
                            ("chtl_natsft_b0_s0_timer1_default_node4", "sft", None, 0.0, "14Ln_chtl_natsft_b0_s0_timer1_default_node4")]:
    pairs.append({"label": lab, "family": "Qwen2.5-VL-7B", "objective": obj, "alpha": a, "beta": b, "env": "C",
                  "dataset": "charades-timelens", "base": audit_path("14Ln_chtl_qwenbase_timer1_default_node4"),
                  "arm": audit_path(as_), "reg_gamma_mask": None})
pairs = [p for p in pairs if p["family"] != "Qwen2.5-VL-3B"]

rows, inputs, reg_check = [], [REGF], []
for p in pairs:
    mb, ma = machine_of(p["base"]), machine_of(p["arm"])
    same_env = ENV_BY_MACHINE.get(mb) == ENV_BY_MACHINE.get(ma)
    assert same_env, p
    g = gamma_pair(p["base"], p["arm"], extra=True)
    inputs += [p["base"], p["arm"]]
    row = {k: p[k] for k in ("label", "family", "objective", "alpha", "beta", "env", "dataset", "base", "arm")}
    row.update(g)
    if p["reg_gamma_mask"] is not None:
        ok = (g["mask"]["gamma"] == p["reg_gamma_mask"] and g["mask"]["gamma_ci95"] == p["reg_gamma_mask_ci"])
        row["registry_bit_exact"] = ok
        reg_check.append(ok)
    rows.append(row)


def sgn(x, tol=0.0):
    return 0 if abs(x) <= tol else (1 if x > 0 else -1)


def sig(ci, null=0.0):
    return 0 if (ci[0] <= null <= ci[1]) else (1 if ci[0] > null else -1)


summ = {}
for scope, sel in (("charades", lambda r: r["dataset"] == "charades-sta"), ("all", lambda r: True)):
    R = [r for r in rows if sel(r)]
    G = np.array([r["mask"]["gamma"] for r in R]); DN = np.array([r["d_nll"] for r in R])
    S = np.array([r["mask"]["sign_frac"] for r in R]); GC = np.array([r["mask"]["gamma_c"] for r in R])
    GS = np.array([r["swap"]["gamma"] for r in R])
    rho, p = spearmanr(G, DN)
    rho_s, p_s = spearmanr(GS, DN)
    rho_gs, _ = spearmanr(G, S); rho_gc, _ = spearmanr(G, GC)
    dis_point, dis_sig = [], []
    for op in OPS:
        for r in R:
            d = r[op]
            a = (sgn(d["gamma"]), sgn(d["sign_frac"] - 0.5), sgn(d["gamma_c"]))
            b = (sig(d["gamma_ci95"]), sig(d["sign_frac_cluster_ci95"], 0.5), sig(d["gamma_c_ci95"]))
            if len(set(a)) > 1:
                dis_point.append({"op": op, "label": r["label"], "env": r["env"], "dataset": r["dataset"],
                                  "gamma": d["gamma"], "sign_frac": d["sign_frac"], "gamma_c": d["gamma_c"]})
            if len(set(b)) > 1:
                dis_sig.append({"op": op, "label": r["label"], "env": r["env"], "dataset": r["dataset"],
                                "sig(gamma,signfrac,gamma_c)": b, "gamma_ci": d["gamma_ci95"],
                                "signfrac_ci": d["sign_frac_cluster_ci95"], "gamma_c_ci": d["gamma_c_ci95"]})
    summ[scope] = {"n_configs": len(R), "spearman_gamma_mask_vs_dNLL": [float(rho), float(p)],
                   "spearman_gamma_swap_vs_dNLL": [float(rho_s), float(p_s)],
                   "spearman_gamma_mask_vs_signfrac": float(rho_gs), "spearman_gamma_mask_vs_gamma_c": float(rho_gc),
                   "disagree_point_sign": dis_point, "disagree_ci_significance": dis_sig}

out = write_out("a3_scale", __file__, {"pairs": rows, "summary": summ,
                "registry_bit_exact": {"n_checked": len(reg_check), "n_exact": int(sum(reg_check))},
                "notes": ["c uses eps=0.05 nats/token in the denominator",
                          "sign_frac Wilson CI treats questions as iid; the cluster CI resamples videos (B=2000, seed 1)",
                          "Delta NLL = mean over H_int of (-lp_clean_mean)_arm - (-lp_clean_mean)_base (nats/token)",
                          "excluded: 14Ln_natmdpo_a05_s0_timer1_default_node4 (no provenance meta: base model unverifiable)"]},
                inputs)
print("registry bit-exact", sum(reg_check), "/", len(reg_check))
for r in rows:
    m = r["mask"]
    print(f'{r["label"][:44]:44s} {r["env"]} {r["dataset"][:8]:8s} n{r["n_H_int"]:4d} NLL b/a {r["nll_base"]:.3f}/{r["nll_arm"]:.3f} dNLL {f3(r["d_nll"])} '
          f'G {f3(m["gamma"])} {fci(m["gamma_ci95"])} sf {m["sign_frac"]:.2f} Gc {f3(m["gamma_c"])} {fci(m["gamma_c_ci95"])} | swap G {f3(r["swap"]["gamma"])} sf {r["swap"]["sign_frac"]:.2f} Gc {f3(r["swap"]["gamma_c"])}')
for k, v in summ.items():
    print(k, {kk: vv for kk, vv in v.items() if not kk.startswith("disagree")})
    print("  point-sign disagreements:", [(d["op"], d["label"], d["env"]) for d in v["disagree_point_sign"]])
    print("  CI-significance disagreements:", [(d["op"], d["label"], d["env"], d["sig(gamma,signfrac,gamma_c)"]) for d in v["disagree_ci_significance"]])
print(out)
