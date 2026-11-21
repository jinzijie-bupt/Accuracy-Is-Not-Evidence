#!/usr/bin/env python3
"""a6_vc_vs_qwen.py -- paired Gamma of VideoChat-R1 against its base Qwen2.5-VL-7B-Instruct,
both audited under VideoChat-R1's 'vchat' template with the 14Ln v1 instrument on the tf5.12 / node1 stack.
Base: node5 run of 2026-09-24. Arm: the node5 VideoChat-R1 run when present (same node), else the
node3 run (same frozen stack, node1 architecture). Same estimator as read_14Ln_v1 via _common.gamma_pair."""
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..", "..")).replace("\\", "/")
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import _common as C

RES = _SUPP + "/results"
BASE = RES + "/native_audit_20260924/14Ln_qwenbase_vchat_node5_nat_o0n200R3.json"
ARMS = [("node5", RES + "/native_audit_20260924/14Ln_vcbase_vchat_node5_nat_o0n200R3.json"),
        ("node3", RES + "/native_audit_20260917/14Ln_vcbase_vchat_node3_nat_o0n200R3.json")]

pairs, ins = [], [BASE]
for node, arm in ARMS:
    if not os.path.exists(arm):
        continue
    g = C.gamma_pair(BASE, arm)
    g.update(label=f"VideoChat-R1 ({node}) vs Qwen2.5-VL-7B (node5), vchat template", base=BASE, arm=arm)
    pairs.append(g); ins.append(arm)
    m, s = g["mask"], g["swap"]
    print(f"{node}: n_cap={g['n_H_int']} mismatch={g['n_structural_mismatch']}/{g['n_target_text_mismatch']} "
          f"Gm={m['gamma']:+.3f} [{m['gamma_ci95'][0]:+.3f},{m['gamma_ci95'][1]:+.3f}] "
          f"Gs={s['gamma']:+.3f} [{s['gamma_ci95'][0]:+.3f},{s['gamma_ci95'][1]:+.3f}] "
          f"dE/dO arm {m['dE_arm']:.3f}/{m['dO_arm']:.3f} base {m['dE_base']:.3f}/{m['dO_base']:.3f}")
print(C.write_out("a6_vc_vs_qwen", __file__, {"pairs": pairs}, ins))
