"""mk_a5_accept.py -> a5_accept.jsonl. r9 acceptance on the tf5 stack, same machine as A1-07/A1-08 (r8):
 (1) r9 defaults, base, indep600 o0 n8 -> must equal A1-07 o0n100 rows 0..7 field-by-field (except meta)
 (2) r9 --cands --base-pred-file A1-07 o0n100, IoU-GRPO adapter (A1-08), n8 -> cand_lp sanity; non-cand fields = A1-08 rows."""
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
R = WORK_ROOT; P = f"{R}/probe3_selector_evidence"; O = f"{R}/results/a5/accept"
COMMON = ["--model", f"{R}/models/Qwen2.5-VL-7B-Instruct", "--prompt-style", "timer1_default", "--answer-style", "timer1",
          "--ops", "mask,swap", "--R", "3", "--thr", "0.5", "--seed", "1000", "--fps", "2.0", "--total-pixels", "2809856",
          "--min-pixels", "12544", "--max-frames", "768", "--max-new-tokens", "128", "--qid-file", f"{P}/indep600_qids.json",
          "--gpu", "0", "--offset", "0", "--n", "8"]
jobs = []
def add(name, extra, out):
    jobs.append({"name": name, "env": "tf5", "cwd": P, "prio": 5, "argv": ["14Ln_native_sentinel_probe_r9.py"] + COMMON + extra + ["--out", out],
                 "out": out, "mem_gb": 30.0, "mkdir": O})
add("a5acc__r9_default_base", [], f"{O}/r9_default_base_o0n8.json")
add("a5acc__r9_cands_iougrpo", ["--adapter", (DATASETS_ROOT + "/ds-1/local__x0_s0_iou700"), "--cands",
    "--base-pred-file", f"{R}/results/a1/A1-07_qwen_base_t1d_tf5/A1-07_qwen_base_t1d_tf5__o0n100.json"], f"{O}/r9_cands_iougrpo_o0n8.json")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "a5_accept.jsonl"), "w", encoding="utf-8", newline="\n") as f:
    for j in jobs: f.write(json.dumps(j) + "\n")
print(len(jobs))
