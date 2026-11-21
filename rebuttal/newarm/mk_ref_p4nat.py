"""mk_ref_p4nat.py -> newarm/ref_p4nat_jobs.jsonl: same-machine full-3720 accuracy for the EXISTING paper arms that the new
arms are compared with (tf451: ours a0.5 s0, SFT, mDPO; tf5: l4800 arm, IoU-GRPO 16f), same evaluators as mk_newarm_jobs.py."""
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
R = WORK_ROOT; P = f"{R}/probe3_selector_evidence"; D1 = (DATASETS_ROOT + "/ds-1")
FA = [f"{R}/setup/attn_launch.py", "flash_attention_2"]
CFG = [("ref_qwena05_s0", "tf451", f"{D1}/local__nat_base_a05_g1_s0"), ("ref_qwensft", "tf451", f"{D1}/local__nat_sft_b0_s0"),
       ("ref_qwenmdpo", "tf451", f"{D1}/local__adapter_nat_mdpo_a05_s0"),
       ("ref_l4800", "tf5", f"{D1}/node3__adapter_nat_base_a05_g1_l4800_s0"), ("ref_iougrpo16f", "tf5", f"{D1}/local__x0_s0_iou700")]
jobs = []
for tag, env, ad in CFG:
    O = f"{R}/results/newarm/{tag}"; sfx = "tf451fa2" if env == "tf451" else "tf5"
    p4 = "12p4n_native_acc_eval_tf451_r5.py" if env == "tf451" else "12p4n_native_acc_eval_tf5meta_r8.py"
    for off in range(0, 3720, 465):
        out = f"{O}/p4nat_{tag}_{sfx}_o{off}n465.json"
        jobs.append({"name": f"na__{tag}__p4nat_o{off}", "env": env, "cwd": P, "prio": 14,
                     "argv": (FA if env == "tf451" else []) + [p4, "--model", f"{R}/models/Qwen2.5-VL-7B-Instruct", "--adapter", ad,
                                                              "--prompt-style", "timer1_default", "--n", "465", "--offset", str(off), "--out", out],
                     "out": out, "mem_gb": 30.0, "mkdir": O})
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_p4nat_jobs.jsonl"), "w", encoding="utf-8", newline="\n") as f:
    for j in jobs: f.write(json.dumps(j) + "\n")
print(len(jobs))
