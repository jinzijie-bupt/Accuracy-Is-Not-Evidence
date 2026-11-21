"""mk_a7_jobs.py -> a7_jobs.jsonl. A7: behavioural eval (12p5n r3 abf8ec93628ba41a, paper instrument) for the Qwen family on
its TRAINING stack (tf451+FA2), mask AND swap (the old node4 run had mask only; tm used tf5-converted adapters):
base / ours a0.5 s0 / SFT / mDPO, 1000 q as 4x250, ops none,mask_evid,mask_rand,swap_evid,swap_rand,black, seed 1000."""
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
R = WORK_ROOT; P = f"{R}/probe3_selector_evidence"; O = f"{R}/results/a7"; D1 = (DATASETS_ROOT + "/ds-1")
L = [f"{R}/setup/attn_launch.py", "flash_attention_2"]
CFG = [("qwenbase", None), ("qwena05", f"{D1}/local__nat_base_a05_g1_s0"), ("qwensft", f"{D1}/local__nat_sft_b0_s0"),
       ("qwenmdpo", f"{D1}/local__adapter_nat_mdpo_a05_s0")]
jobs = []
for tag, ad in CFG:
    for off in range(0, 1000, 250):
        out = f"{O}/p5n_{tag}_timer1_default_tf451fa2_o{off}n250.json"
        argv = L + ["12p5n_native_perturb_eval_r3.py", "--model", f"{R}/models/Qwen2.5-VL-7B-Instruct"] + (["--adapter", ad] if ad else []) + \
               ["--prompt-style", "timer1_default", "--ops", "none,mask_evid,mask_rand,swap_evid,swap_rand,black", "--seed", "1000",
                "--gpu", "0", "--n", "250", "--offset", str(off), "--out", out]
        jobs.append({"name": f"a7__p5n_{tag}__o{off}", "env": "tf451", "cwd": P, "prio": 21, "argv": argv, "out": out, "mem_gb": 30.0, "mkdir": O})
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "a7_jobs.jsonl"), "w", encoding="utf-8", newline="\n") as f:
    for j in jobs: f.write(json.dumps(j) + "\n")
print(len(jobs))
