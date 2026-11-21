"""mk_newarm_jobs.py -- the standard evaluation suite for a NEW rebuttal arm (trained on node7), as q.py jobs.
usage: mk_newarm_jobs.py TAG ENV ADAPTER_DIR [--model M] [--prio P] -> rebuttal/newarm/<TAG>_jobs.jsonl
Suite (instrument versions = the ones already used on node7 for the base partners, same machine & stack):
  1. 14Ln r10 --tok-lp --cands, first 200 (2x100), --base-pred-file = a5 base product   [pairs with a5 base qwenbase451 / qwenbase5]
  2. 14Ln r8, indep600 (6x100)                                                          [pairs with A1-01 (tf451) / A1-07 (tf5)]
  3. 12p4n full 3720 (8x465): tf451 r5 a91c1c9a / tf5meta r8                              [+ base job for the same evaluator]
  4. 12p5n r3, 1000 (4x250), ops none,mask_evid,mask_rand,swap_evid,swap_rand,black       [pairs with a7 qwenbase (tf451) / + base job (tf5)]
Base jobs carry fixed names, so q.py add skips them if already queued/done."""
import json, os, sys
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)

TAG, ENV, AD = sys.argv[1], sys.argv[2], sys.argv[3]
MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "Qwen2.5-VL-7B-Instruct"
PRIO = int(sys.argv[sys.argv.index("--prio") + 1]) if "--prio" in sys.argv else 12
assert ENV in ("tf451", "tf5")
assert AD.startswith("/") and ":" not in AD, f"ADAPTER_DIR looks MSYS-mangled: {AD!r} (run with MSYS_NO_PATHCONV=1)"
R = WORK_ROOT; P = f"{R}/probe3_selector_evidence"; O = f"{R}/results/newarm/{TAG}"
FA = [f"{R}/setup/attn_launch.py", "flash_attention_2"] if ENV == "tf451" else []
W = f"{R}/setup/wait_launch.py"
sfx = "tf451fa2" if ENV == "tf451" else "tf5"
M = ["--model", f"{R}/models/{MODEL}", "--adapter", AD]
MB = ["--model", f"{R}/models/{MODEL}"]
STY = ["--prompt-style", "timer1_default", "--answer-style", "timer1"]
C14 = ["--ops", "mask,swap", "--R", "3", "--thr", "0.5", "--seed", "1000", "--fps", "2.0", "--total-pixels", "2809856",
       "--min-pixels", "12544", "--max-frames", "768", "--max-new-tokens", "128", "--gpu", "0"]
jobs = []


def add(name, argv, out, prio, mkdir):
    jobs.append({"name": name, "env": ENV, "cwd": P, "prio": prio, "argv": argv + ["--out", out], "out": out, "mem_gb": 30.0, "mkdir": mkdir})


btag = "qwenbase451" if ENV == "tf451" else "qwenbase5"
assert MODEL == "Qwen2.5-VL-7B-Instruct", "base partners below are Qwen-only; extend before using another backbone"
for off in (0, 100):  # 1
    bo = f"{R}/results/a5/{btag}/14Ln_{btag}_{ENV}{'fa2' if ENV == 'tf451' else ''}_tokc_o{off}n100R3.json"
    add(f"na__{TAG}__14Ln_tokc_o{off}", [W, bo, "14400"] + FA + ["14Ln_native_sentinel_probe_r10.py"] + M + STY + C14 +
        ["--tok-lp", "--cands", "--base-pred-file", bo, "--n", "100", "--offset", str(off)],
        f"{O}/14Ln_{TAG}_{sfx}_tokc_o{off}n100R3.json", PRIO, O)
for off in range(0, 600, 100):  # 2
    add(f"na__{TAG}__14Ln_indep600_o{off}", FA + ["14Ln_native_sentinel_probe_r8.py"] + M + STY + C14 +
        ["--qid-file", f"{P}/indep600_qids.json", "--n", "100", "--offset", str(off)],
        f"{O}/14Ln_indep600_{TAG}_{sfx}_o{off}n100R3.json", PRIO + 1, O)
p4 = "12p4n_native_acc_eval_tf451_r5.py" if ENV == "tf451" else "12p4n_native_acc_eval_tf5meta_r8.py"
BO = f"{R}/results/newarm/_base_{ENV}"
for off in range(0, 3720, 465):  # 3
    add(f"na__{TAG}__p4nat_o{off}", FA + [p4] + M + ["--prompt-style", "timer1_default", "--n", "465", "--offset", str(off)],
        f"{O}/p4nat_{TAG}_{sfx}_o{off}n465.json", PRIO + 2, O)
    add(f"na__base_{ENV}__p4nat_o{off}", FA + [p4] + MB + ["--prompt-style", "timer1_default", "--n", "465", "--offset", str(off)],
        f"{BO}/p4nat_qwenbase_{sfx}_o{off}n465.json", PRIO + 2, BO)
P5 = ["--prompt-style", "timer1_default", "--ops", "none,mask_evid,mask_rand,swap_evid,swap_rand,black", "--seed", "1000", "--gpu", "0"]
for off in range(0, 1000, 250):  # 4
    add(f"na__{TAG}__p5n_o{off}", FA + ["12p5n_native_perturb_eval_r3.py"] + M + P5 + ["--n", "250", "--offset", str(off)],
        f"{O}/p5n_{TAG}_{sfx}_o{off}n250.json", PRIO + 3, O)
    if ENV == "tf5":
        add(f"na__base_tf5__p5n_o{off}", ["12p5n_native_perturb_eval_r3.py"] + MB + P5 + ["--n", "250", "--offset", str(off)],
            f"{BO}/p5n_qwenbase_tf5_o{off}n250.json", PRIO + 3, BO)
od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "newarm")
os.makedirs(od, exist_ok=True)
with open(os.path.join(od, f"{TAG}_jobs.jsonl"), "w", encoding="utf-8", newline="\n") as f:
    for j in jobs:
        f.write(json.dumps(j) + "\n")
print(TAG, len(jobs), "jobs")
