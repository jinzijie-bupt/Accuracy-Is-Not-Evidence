"""mk_a5_accept_r10.py -> a5_accept_r10.jsonl. r10 (8a10ee350096c4e4) acceptance, same machine as A1 (r8):
 (1) tf5, defaults, Qwen base, indep600 o0 n8 -> rows == A1-07 o0n100 rows (all fields)
 (2) tf5, --tok-lp --cands --base-pred-file A1-07 o0n100, IoU-GRPO adapter, n8 -> non-tok fields == A1-08 rows; xcheck in meta
 (3) tf451+FA2, --tok-lp, Qwen base, indep600 o0 n4 -> non-tok fields == A1-01 o0n100 rows; xcheck (tf4fps path)"""
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
R = WORK_ROOT; P = f"{R}/probe3_selector_evidence"; O = f"{R}/results/a5/accept"
L = [f"{R}/setup/attn_launch.py", "flash_attention_2"]
C = ["--model", f"{R}/models/Qwen2.5-VL-7B-Instruct", "--prompt-style", "timer1_default", "--answer-style", "timer1",
     "--ops", "mask,swap", "--R", "3", "--thr", "0.5", "--seed", "1000", "--fps", "2.0", "--total-pixels", "2809856",
     "--min-pixels", "12544", "--max-frames", "768", "--max-new-tokens", "128", "--qid-file", f"{P}/indep600_qids.json",
     "--gpu", "0", "--offset", "0"]
S = ["14Ln_native_sentinel_probe_r10.py"]
jobs = []
def add(name, env, argv, out):
    jobs.append({"name": name, "env": env, "cwd": P, "prio": 4, "argv": argv + ["--out", out], "out": out, "mem_gb": 30.0, "mkdir": O})
add("a5acc__r10_default_base_tf5", "tf5", S + C + ["--n", "8"], f"{O}/r10_default_base_tf5_o0n8.json")
add("a5acc__r10_tok_cands_iougrpo_tf5", "tf5", S + C + ["--n", "8", "--adapter", (DATASETS_ROOT + "/ds-1/local__x0_s0_iou700"),
    "--tok-lp", "--cands", "--base-pred-file", f"{R}/results/a1/A1-07_qwen_base_t1d_tf5/A1-07_qwen_base_t1d_tf5__o0n100.json"],
    f"{O}/r10_tok_cands_iougrpo_tf5_o0n8.json")
add("a5acc__r10_tok_base_tf451", "tf451", L + S + C + ["--n", "4", "--tok-lp"], f"{O}/r10_tok_base_tf451fa2_o0n4.json")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "a5_accept_r10.jsonl"), "w", encoding="utf-8", newline="\n") as f:
    for j in jobs: f.write(json.dumps(j) + "\n")
print(len(jobs))
