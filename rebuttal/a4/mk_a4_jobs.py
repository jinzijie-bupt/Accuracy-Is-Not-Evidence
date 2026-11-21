"""mk_a4_jobs.py -> a4_jobs.jsonl. A4 audit robustness (R2 Q2), first-200 questions (paired with the paper's audit):
pairs: Qwen base vs alpha=0.5 (tf451, A1-01/A1-04 args), TimeZero base vs alpha=0.5 (tf5, A1-09/A1-13 args)
conditions: --jitter shrink|expand|shiftl|shiftr (--jitter-frac 0.10, ops mask,swap), and ops freeze,occlude (no jitter).
Sharded 2 x 100 questions."""
import csv, json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)

HERE = os.path.dirname(os.path.abspath(__file__))
R = WORK_ROOT; P = f"{R}/probe3_selector_evidence"
cfg = {r["cell_id"].split("_")[0]: r for r in csv.DictReader(open(os.path.join(HERE, "..", "a1", "a1_configs.tsv"),
                                                                  encoding="utf-8"), delimiter="\t")}
CELLS = {"qwen_base": "A1-01", "qwen_a05": "A1-04", "tz_base": "A1-09", "tz_a05": "A1-13"}
COND = [("jit_shrink", ["--jitter", "shrink", "--jitter-frac", "0.10"], None),
        ("jit_expand", ["--jitter", "expand", "--jitter-frac", "0.10"], None),
        ("jit_shiftl", ["--jitter", "shiftl", "--jitter-frac", "0.10"], None),
        ("jit_shiftr", ["--jitter", "shiftr", "--jitter-frac", "0.10"], None),
        ("natocc", [], "freeze,occlude")]
out = []
for tag, cid in CELLS.items():
    r = cfg[cid]
    ad = r["adapter_staged_name"].strip()
    base_argv = ["14Ln_native_sentinel_probe_r8.py", "--model", f"{R}/models/{r['model_dir']}"]
    if r["stack"] == "tf451":  # 11-06: tf451 jobs run FA2 via attn_launch.py (same as A5/A6/A7/newarm)
        base_argv = [f"{R}/setup/attn_launch.py", "flash_attention_2"] + base_argv
    if ad and ad != "-":
        base_argv += ["--adapter", f"{DATASETS_ROOT}/ds-1/{ad}"]
    a = r["args_string"].split()
    for cname, extra, ops in COND:
        aa = list(a)
        if ops:
            aa[aa.index("--ops") + 1] = ops
        for off in (0, 100):
            name = f"a4__{tag}__{cname}__o{off}"
            o = f"{R}/results/a4/{tag}__{cname}__{r['stack']}__o{off}n100.json"
            out.append({"name": name, "env": r["stack"], "cwd": P, "prio": 30,
                        "argv": base_argv + aa + extra + ["--gpu", "0", "--offset", str(off), "--n", "100", "--out", o],
                        "out": o, "mem_gb": 30})
os.makedirs(HERE, exist_ok=True)
with open(os.path.join(HERE, "a4_jobs.jsonl"), "w", encoding="utf-8", newline="\n") as f:
    for j in out:
        f.write(json.dumps(j) + "\n")
print(len(out), "jobs")
