"""mk_a6_jobs.py -> a6_jobs.jsonl. A6 (R3 W2/Q3): RA-RFT vs ITS base Time-R1-7B (RA-RFT model card: base_model
Boshenxx/Time-R1-7B; trained with transformers 4.51.1) -> both on the tf451 stack, FlashAttention-2 via attn_launch.py,
Time-R1 native prompt (timer1_default / timer1) for both. Instruments = the paper's (sha16 in comments).
  14Ln r8 0794417ae4d45d30  first 200 (mask,swap) + independent 600 (indep600_qids.json)
  12p4n tf451 r5 a91c1c9ac14f87bb  full 3720
  12p5n r3 abf8ec93628ba41a  1000, ops none,mask_evid,mask_rand,swap_evid,swap_rand,black
  14Lqn r3 124cfd516a39f427  first 200 (mask,swap)
"""
import json, os
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)

HERE = os.path.dirname(os.path.abspath(__file__))
R = WORK_ROOT; P = f"{R}/probe3_selector_evidence"; O = f"{R}/results/a6"
L = [f"{R}/setup/attn_launch.py", "flash_attention_2"]
MODELS = {"rarft": "RA-RFT-Qwen2.5-VL-7B", "tr1base": "Time-R1-7B"}
STY = ["--prompt-style", "timer1_default"]
jobs = []


def add(name, argv, out, prio=10, mem=30):
    jobs.append({"name": name, "env": "tf451", "cwd": P, "prio": prio, "argv": L + argv + ["--out", out], "out": out,
                 "mem_gb": mem, "gpus": [0, 1, 2, 3]})


for tag, md in MODELS.items():
    M = ["--model", f"{R}/models/{md}"]
    a14 = ["--answer-style", "timer1", "--ops", "mask,swap", "--R", "3", "--gpu", "0"]
    for off in (0, 100):
        add(f"a6__14Ln_first200_{tag}__o{off}", ["14Ln_native_sentinel_probe_r8.py"] + M + STY + a14 + ["--n", "100", "--offset", str(off)],
            f"{O}/14Ln_{tag}_timer1_default_tf451fa2_o{off}n100R3.json", prio=8)
    for off in range(0, 600, 100):
        add(f"a6__14Ln_indep600_{tag}__o{off}", ["14Ln_native_sentinel_probe_r8.py"] + M + STY + a14 +
            ["--qid-file", f"{P}/indep600_qids.json", "--n", "100", "--offset", str(off)],
            f"{O}/14Ln_indep600_{tag}_timer1_default_tf451fa2_o{off}n100R3.json")
    for off in range(0, 3720, 465):
        add(f"a6__p4nat_{tag}__o{off}", ["12p4n_native_acc_eval_tf451_r5.py"] + M + STY + ["--n", "465", "--offset", str(off)],
            f"{O}/p4nat_{tag}_timer1_default_tf451fa2_o{off}n465.json")
    for off in range(0, 1000, 250):
        add(f"a6__p5n_{tag}__o{off}", ["12p5n_native_perturb_eval_r3.py"] + M + STY +
            ["--ops", "none,mask_evid,mask_rand,swap_evid,swap_rand,black", "--seed", "1000", "--gpu", "0", "--n", "250", "--offset", str(off)],
            f"{O}/p5n_{tag}_timer1_default_tf451fa2_o{off}n250.json")
    for off in (0, 100):
        add(f"a6__14Lqn_{tag}__o{off}", ["14Lqn_native_query_swap_probe_r3.py"] + M + STY + a14 + ["--n", "100", "--offset", str(off)],
            f"{O}/14Lqn_{tag}_timer1_default_tf451fa2_o{off}n100R3.json")
with open(os.path.join(HERE, "a6_jobs.jsonl"), "w", encoding="utf-8", newline="\n") as f:
    for j in jobs:
        f.write(json.dumps(j) + "\n")
print(len(jobs), "jobs")
