# -*- coding: utf-8 -*-
"""A1: public RL checkpoints (TimeZero, Time-R1) vs their common base Qwen2.5-VL-7B-Instruct.
(i)  paired Gamma on the 200-question likelihood audit, same environment, same protocol (timer1_default, video-first)
(ii) paired Delta R@0.5 on the full Charades-STA test set (3720), same environment/protocol, via the registered
     reader read_p4nat_selective_v1.read_pair (video-cluster bootstrap B=4000 seed 0)."""
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..", "..")).replace("\\", "/")
import os, sys, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *

SEL = _SUPP + "/readers/read_p4nat_selective_v1.py"
spec = importlib.util.spec_from_file_location("sel", SEL); sel = importlib.util.module_from_spec(spec); spec.loader.exec_module(sel)

inputs = [SEL, _SUPP + "/readers/read_p4nat_v1.py"]

# ---------------- sanity: registry value grouped a05 vs Qwen base, env A
sb, sa = audit_path("14Ln_base_timer1_default_node2"), audit_path("14Ln_natbase_a05_g1_s0_tf5_timer1_default_node2")
s = gamma_pair(sb, sa, extra=False)
REG = RES + "/judgment_20260921/gamma/gamma_14Ln_natbase_a05_g1_s0_tf5_timer1_default_node2__vs__14Ln_base_timer1_default_node2.json"
reg = load(REG)["per_op"]
sanity = {"base": sb, "arm": sa, "gamma_mask": s["mask"]["gamma"], "ci": s["mask"]["gamma_ci95"],
          "gamma_swap": s["swap"]["gamma"], "ci_swap": s["swap"]["gamma_ci95"],
          "brief_expected": [1.729, 1.434, 2.026],
          "registered_reader_product": REG,
          "registered": [reg["mask"]["gamma"]] + reg["mask"]["gamma_ci95"] + [reg["swap"]["gamma"]] + reg["swap"]["gamma_ci95"],
          "note": "point estimate matches the brief (1.729). The CI matches the registered read_14Ln_v1 product bit-for-bit "
                  "([1.4374, 2.0055]); the brief's [1.434, 2.026] is not reproduced by that reader's stream -- it is within "
                  "Monte-Carlo variation of the same bootstrap (B=10000 gives [1.433, 2.021]; video order-of-appearance gives "
                  "[1.426, 2.028])."}
sanity["pass"] = (abs(s["mask"]["gamma"] - 1.729) < 5e-4 and s["mask"]["gamma"] == reg["mask"]["gamma"]
                  and s["mask"]["gamma_ci95"] == reg["mask"]["gamma_ci95"] and s["swap"]["gamma_ci95"] == reg["swap"]["gamma_ci95"])
inputs += [sb, sa, REG]
print("SANITY", sanity)
assert sanity["pass"]

PAIRS = [  # (label, env, base_stem, arm_stem)
    ("TimeZero vs Qwen", "A", "14Ln_base_timer1_default_node2", "14Ln_tzbase_timer1_default_node2"),
    ("TimeZero vs Qwen", "B", "14Ln_base_timer1_default_node3", "14Ln_tzbase_timer1_default_node3"),
    ("Time-R1 vs Qwen", "B", "14Ln_base_timer1_default_node3", "14Ln_tr1base_timer1_default_node3"),
    ("Time-R1 vs Qwen", "C", "14Ln_base_timer1_node4", "14Ln_tr1base_timer1_default_node4"),
]
gam = []
for lab, env, bs, as_ in PAIRS:
    bp, ap = audit_path(bs), audit_path(as_)
    assert env_of(bp) == env_of(ap) == env, (bp, ap)
    r = gamma_pair(bp, ap, extra=True)
    r.update({"label": lab, "env": env, "base": bp, "arm": ap})
    gam.append(r); inputs += [bp, ap]

# ---------------- full test set, same env & protocol
FULL = [
    ("TimeZero vs Qwen", "A", "video-first", RES + "/native_full_20260918/p4nat_qwenbase_timer1_default_tf5meta_node2_full.json",
     RES + "/native_full_20260918/p4nat_tzbase_timer1_default_tf5meta_node2_full.json"),
    ("TimeZero(official dir) vs Qwen", "A", "video-first", RES + "/native_full_20260918/p4nat_qwenbase_timer1_default_tf5meta_node2_full.json",
     RES + "/native_full_20260918/p4nat_TimeZero-Charades-7B_timer1_default_tf5meta_node2_full.json"),
    ("TimeZero vs Qwen", "B", "video-first", RES + "/native_eval_20260918_node3/p4nat_qwenbase_timer1_default_tf5meta_node3_full.json",
     RES + "/native_eval_20260918_node3/p4nat_tzbase_timer1_default_tf5meta_node3_full.json"),
    ("Time-R1 vs Qwen", "B", "video-first", RES + "/native_eval_20260918_node3/p4nat_qwenbase_timer1_default_tf5meta_node3_full.json",
     RES + "/native_eval_20260918_node3/p4nat_tr1base_timer1_default_tf5meta_node3_full.json"),
    ("TimeZero vs Qwen", "B", "text-first", RES + "/native_eval_20260918_node3/p4nat_qwenbase_timer1_default_textfirst_tf5meta_node3_full.json",
     RES + "/native_eval_20260918_node3/p4nat_tzbase_timer1_default_textfirst_tf5meta_node3_full.json"),
    ("Time-R1 vs Qwen", "B", "text-first", RES + "/native_eval_20260918_node3/p4nat_qwenbase_timer1_default_textfirst_tf5meta_node3_full.json",
     RES + "/native_eval_20260918_node3/p4nat_tr1base_timer1_default_textfirst_tf5meta_node3_full.json"),
]
full = []
for lab, env, order, rp, ap in FULL:
    R, A = load(rp), load(ap)
    st = sorted({r.get("style") for r in R} | {r.get("style") for r in A})
    res = sel.read_pair(R, A, B=4000, seed=0)
    full.append({"label": lab, "env": env, "order": order, "ref": rp, "arm": ap, "styles": st, "n": res["n"],
                 "ref_R50": res["point"]["ref_marginal_R@0.5"], "arm_R50": res["point"]["arm_marginal_R@0.5"],
                 "delta_R50": res["point"]["delta_marginal"], "delta_R50_ci95": res["delta_marginal_boot"]["ci95"],
                 "coverage_ref": res["coverage_ref"], "coverage_arm": res["coverage_arm"], "n_videos": res["n_videos"],
                 "same_pred_rows": int(sum(a.get("pred") == b.get("pred") for a, b in zip(R, A)))})
    inputs += [rp, ap]

missing = [
    "Time-R1 vs Qwen, env C (node4, tf4.51), full test set: no Qwen2.5-VL-7B base full-set file exists on node4 "
    "(p4nat_Time-R1-7B_timer1_default_tf451_node4_full.json exists, the base counterpart does not) -> not computable.",
    "VideoChat-R1 vs Qwen (vchat protocol): Qwen base vchat likelihood audit is still running on GPU -> left blank.",
    "TimeZero vs Qwen, env C: no TimeZero audit on node4.",
]
out = write_out("a1_public_rl_vs_qwen", __file__, {"sanity": sanity, "gamma_pairs": gam, "full_set_pairs": full,
                                                   "not_computable": missing}, inputs)
for g in gam:
    print(g["label"], g["env"], "n", g["n_H_int"], "mask", f3(g["mask"]["gamma"]), fci(g["mask"]["gamma_ci95"]),
          "pos", round(g["mask"]["pos_frac"], 3), "swap", f3(g["swap"]["gamma"]), fci(g["swap"]["gamma_ci95"]),
          "pos", round(g["swap"]["pos_frac"], 3), "tmis", g["n_target_text_mismatch"], "smis", g["n_structural_mismatch"],
          "R50", g["R50_audit_base"], g["R50_audit_arm"])
    for op in OPS:
        d = g[op]; print("   ", op, "arm dE/dO", f3(d["dE_arm"]), f3(d["dO_arm"]), "base dE/dO", f3(d["dE_base"]), f3(d["dO_base"]))
for f in full:
    print(f["label"], f["env"], f["order"], f["ref_R50"], f["arm_R50"], f3(f["delta_R50"]), fci(f["delta_R50_ci95"]), f["same_pred_rows"])
print(out)
