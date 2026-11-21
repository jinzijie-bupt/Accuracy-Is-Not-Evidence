"""mk_a3f_jobs.py -> a3f_jobs.jsonl. A3(2) forced-answer decoding (R2 W3): 12p5n forced 087ab4f48db0c0ee, --forced prefix,
1000 q (4x250), ops none,mask_evid,mask_rand,swap_evid,swap_rand,black, seed 1000. Base and arm on the same stack.
Plus an off-mode equivalence check vs r3 abf8ec93628ba41a (n4, tf5 Qwen base) at prio 5."""
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
R = WORK_ROOT; P = f"{R}/probe3_selector_evidence"; O = f"{R}/results/a3f"; D1 = (DATASETS_ROOT + "/ds-1")
L = [f"{R}/setup/attn_launch.py", "flash_attention_2"]
CFG = [  # tag, stack, model, adapter
    ("qwenbase", "tf451", "Qwen2.5-VL-7B-Instruct", None),
    ("qwena05", "tf451", "Qwen2.5-VL-7B-Instruct", f"{D1}/local__nat_base_a05_g1_s0"),
    ("qwensft", "tf451", "Qwen2.5-VL-7B-Instruct", f"{D1}/local__nat_sft_b0_s0"),
    ("tr1base", "tf451", "Time-R1-7B", None),
    ("rarft", "tf451", "RA-RFT-Qwen2.5-VL-7B", None),
    ("tzbase", "tf5", "TimeZero-Charades-7B", None),
    ("tza05", "tf5", "TimeZero-Charades-7B", f"{D1}/local__nat_tz_a05_g1_s0"),
]
OPS = ["--ops", "none,mask_evid,mask_rand,swap_evid,swap_rand,black", "--seed", "1000", "--gpu", "0", "--prompt-style", "timer1_default"]
jobs = []
def add(name, env, argv, out, prio):
    pre = L if env == "tf451" else []
    jobs.append({"name": name, "env": env, "cwd": P, "prio": prio, "argv": pre + argv + ["--out", out], "out": out, "mem_gb": 30.0, "mkdir": O})
B = ["--model", f"{R}/models/Qwen2.5-VL-7B-Instruct"] + OPS + ["--n", "4", "--offset", "0"]
add("a3f__eq_r3_n4", "tf5", ["12p5n_native_perturb_eval_r3.py"] + B, f"{O}/eq/r3_qwenbase_tf5_o0n4.json", 5)
add("a3f__eq_forcedoff_n4", "tf5", ["12p5n_native_perturb_eval_forced.py"] + B + ["--forced", "off"], f"{O}/eq/forcedoff_qwenbase_tf5_o0n4.json", 5)
for tag, env, md, ad in CFG:
    M = ["--model", f"{R}/models/{md}"] + (["--adapter", ad] if ad else [])
    sfx = "tf451fa2" if env == "tf451" else "tf5"
    for off in range(0, 1000, 250):
        add(f"a3f__{tag}__o{off}", env, ["12p5n_native_perturb_eval_forced.py"] + M + OPS + ["--forced", "prefix", "--n", "250", "--offset", str(off)],
            f"{O}/p5nF_{tag}_timer1_default_{sfx}_o{off}n250.json", 20)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "a3f_jobs.jsonl"), "w", encoding="utf-8", newline="\n") as f:
    for j in jobs: f.write(json.dumps(j) + "\n")
print(len(jobs))
