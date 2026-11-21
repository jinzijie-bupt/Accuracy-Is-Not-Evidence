"""mk_a5_jobs.py -> a5_jobs.jsonl. A5 (R3 W1/Q2 competing intervals) + R1 W2c (timestamp-only Gamma), one pass:
14Ln r10 8a10ee350096c4e4 with --tok-lp --cands on the paper's first 200 (2 x 100), ops mask,swap, R 3, seed 1000.
Arm jobs take --base-pred-file = the same-stack base r10 product of the same shard (wait_launch.py waits for it).
Base jobs prio p, arm jobs prio p+1, so bases are claimed first. Args per cell follow rebuttal/a1/a1_configs.tsv."""
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)

R = WORK_ROOT; P = f"{R}/probe3_selector_evidence"; O = f"{R}/results/a5"
D1 = (DATASETS_ROOT + "/ds-1"); D2 = (DATASETS_ROOT + "/ds-2")
FA = [f"{R}/setup/attn_launch.py", "flash_attention_2"]
W = f"{R}/setup/wait_launch.py"
SC = "14Ln_native_sentinel_probe_r10.py"
COM = ["--ops", "mask,swap", "--R", "3", "--thr", "0.5", "--seed", "1000", "--fps", "2.0", "--total-pixels", "2809856",
       "--min-pixels", "12544", "--max-frames", "768", "--max-new-tokens", "128", "--gpu", "0", "--tok-lp", "--cands"]
STY = {"t1": ["--prompt-style", "timer1_default", "--answer-style", "timer1"],
       "vc": ["--prompt-style", "vchat", "--answer-style", "vchat"],
       "tl": ["--prompt-style", "timelens", "--answer-style", "timelens", "--image-patch-size", "16"]}
# group: (env, model, style, prio, base_tag, [(arm_tag, adapter_or_model)])
G = [
    ("tf451", "Qwen2.5-VL-7B-Instruct", "t1", 15, "qwenbase451", [
        ("qwena05", f"{D1}/local__nat_base_a05_g1_s0"), ("qwensft", f"{D1}/local__nat_sft_b0_s0"),
        ("qwenmdpo", f"{D1}/local__adapter_nat_mdpo_a05_s0")]),
    ("tf5", "Qwen2.5-VL-7B-Instruct", "t1", 15, "qwenbase5", [
        ("iougrpo16f", f"{D1}/local__x0_s0_iou700"), ("l4800", f"{D1}/node3__adapter_nat_base_a05_g1_l4800_s0"),
        ("qwena025", f"{D2}/node1__adapter_nat_base_a025_g1_s0"), ("qwena10", f"{D2}/node2__adapter_nat_base_a10_g1_s0")]),
    ("tf5", "TimeZero-Charades-7B", "t1", 15, "tzbase", [("tza05", f"{D1}/local__nat_tz_a05_g1_s0")]),
    ("tf5", "Time-R1-7B", "t1", 15, "tr1base5", [("tr1a05", f"{D1}/node3__adapter_nat_tr1_a05_g1_s0")]),
    ("tf451", "Time-R1-7B", "t1", 17, "tr1base451", [("rarft", "MODEL:RA-RFT-Qwen2.5-VL-7B")]),
    ("tf451", "VideoChat-R1_7B", "vc", 25, "vcbase451", [("vca05", f"{D1}/local__nat_vc_a05_g1_s0")]),
    ("tf5", "TimeLens-8B", "tl", 25, "tlbase", [("tla05", f"{D1}/node3__adapter_nat_timelens8b_a05_g1_s0")]),
]
jobs = []


def out_of(tag, env, off):
    return f"{O}/{tag}/14Ln_{tag}_{env}{'fa2' if env == 'tf451' else ''}_tokc_o{off}n100R3.json"


for env, model, sty, prio, btag, arms in G:
    pre = FA if env == "tf451" else []
    for off in (0, 100):
        bo = out_of(btag, env, off)
        argv = pre + [SC, "--model", f"{R}/models/{model}"] + STY[sty] + COM + ["--n", "100", "--offset", str(off), "--out", bo]
        jobs.append({"name": f"a5__{btag}__o{off}", "env": env, "cwd": P, "prio": prio, "argv": argv, "out": bo,
                     "mem_gb": 30.0, "mkdir": os.path.dirname(bo)})
        for atag, ad in arms:
            ao = out_of(atag, env, off)
            m = ["--model", f"{R}/models/{ad[6:]}"] if ad.startswith("MODEL:") else ["--model", f"{R}/models/{model}", "--adapter", ad]
            argv = [W, bo, "14400"] + pre + [SC] + m + STY[sty] + COM + ["--base-pred-file", bo, "--n", "100", "--offset", str(off), "--out", ao]
            jobs.append({"name": f"a5__{atag}__o{off}", "env": env, "cwd": P, "prio": prio + 1, "argv": argv, "out": ao,
                         "mem_gb": 30.0, "mkdir": os.path.dirname(ao)})
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "a5_jobs.jsonl"), "w", encoding="utf-8", newline="\n") as f:
    for j in jobs:
        f.write(json.dumps(j) + "\n")
print(len(jobs), "jobs")
