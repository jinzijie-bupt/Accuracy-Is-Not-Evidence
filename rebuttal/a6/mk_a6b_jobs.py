"""mk_a6b_jobs.py -> a6b_jobs.jsonl. A6b: RA-RFT native prompt (system "You are a video analysis expert." + RA-RFT
QUESTION_TEMPLATE, 512 new tokens, refusal = no interval inside <answer>) on the full 3720, for RA-RFT and its base Time-R1-7B.
Instrument 12p4n_native_acc_eval_tf451_r6.py 3910c888fe96f3ed (generator mk_12p4n_tf451_r6.py b2ecd6bdb69773f3), tf451+FA2."""
import json, os
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
R = WORK_ROOT; P = f"{R}/probe3_selector_evidence"; O = f"{R}/results/a6"
L = [f"{R}/setup/attn_launch.py", "flash_attention_2"]
jobs = []
for tag, md in {"rarft": "RA-RFT-Qwen2.5-VL-7B", "tr1base": "Time-R1-7B"}.items():
    for off in range(0, 3720, 465):
        out = f"{O}/p4nat_{tag}_rarft_native_tf451fa2_o{off}n465.json"
        jobs.append({"name": f"a6b__p4nat_rarftnative_{tag}__o{off}", "env": "tf451", "cwd": P, "prio": 9,
                     "argv": L + ["12p4n_native_acc_eval_tf451_r6.py", "--model", f"{R}/models/{md}", "--prompt-style", "rarft_native",
                                  "--n", "465", "--offset", str(off), "--out", out], "out": out, "mem_gb": 30.0, "mkdir": O})
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "a6b_jobs.jsonl"), "w", encoding="utf-8", newline="\n") as f:
    for j in jobs: f.write(json.dumps(j) + "\n")
print(len(jobs))
