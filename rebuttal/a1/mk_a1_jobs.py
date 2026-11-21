"""mk_a1_jobs.py -> a1_jobs.jsonl (q.py job lines) for A1: 21 cells x 600 independent qids, sharded.
Usage: python mk_a1_jobs.py [SHARD=100] [MEM_GB=30]
Each shard = r8 probe on --qid-file indep600_qids.json --offset k*SHARD --n SHARD; output
$R/results/a1/<cell>/<cell>__o{off}n{SHARD}.json. Merging + Gamma reading is done later from the shards."""
import csv, json, os, sys, hashlib
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
LOCAL_ROOT = os.environ.get("LOCAL_ROOT", ".")  # analysis directory holding the pulled result files (results/, rebuttal/)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)

HERE = os.path.dirname(os.path.abspath(__file__))
SHARD = int(sys.argv[1]) if len(sys.argv) > 1 else 100
MEM = float(sys.argv[2]) if len(sys.argv) > 2 else 30
R = WORK_ROOT; P = f"{R}/probe3_selector_evidence"
QF = f"{P}/indep600_qids.json"
qsha = hashlib.sha256(open((LOCAL_ROOT + "/rebuttal/indep600_qids.json"), "rb").read()).hexdigest()[:16]
assert qsha == "e218239c86f321ce", qsha
NQ = 600
rows = list(csv.DictReader(open(os.path.join(HERE, "a1_configs.tsv"), encoding="utf-8"), delimiter="\t"))
out = []
for r in rows:
    cell = r["cell_id"]
    ad = r["adapter_staged_name"].strip()
    if ad.startswith("NOT_STAGED:"):
        nm = ad.split(":", 1)[1].split()[0]                        # adapter_nat_base_a025_g1_s0 ...
        src = "node1__" if "a025" in nm else "node2__"
        adp = f"{DATASETS_ROOT}/ds-2/{src}{nm}"
    elif ad and ad != "-":
        adp = f"{DATASETS_ROOT}/ds-1/{ad}"
    else:
        adp = None
    argv = ["14Ln_native_sentinel_probe_r8.py", "--model", f"{R}/models/{r['model_dir']}"]
    if r["stack"] == "tf451":  # 11-06: tf451 jobs run FA2 via attn_launch.py (equivalence-tested vs node4; same as A5/A6/A7/newarm)
        argv = [f"{R}/setup/attn_launch.py", "flash_attention_2"] + argv
    if adp:
        argv += ["--adapter", adp]
    argv += r["args_string"].split() + ["--qid-file", QF, "--gpu", "0"]
    assert "--n" not in argv and "--offset" not in argv and "--out" not in argv
    for off in range(0, NQ, SHARD):
        o = f"{R}/results/a1/{cell}/{cell}__o{off}n{SHARD}.json"
        out.append({"name": f"a1__{cell}__o{off}", "env": r["stack"], "cwd": P, "prio": 20,
                    "argv": argv + ["--offset", str(off), "--n", str(SHARD), "--out", o],
                    "out": o, "mem_gb": MEM, "mkdir": os.path.dirname(o)})
with open(os.path.join(HERE, "a1_jobs.jsonl"), "w", encoding="utf-8", newline="\n") as f:
    for j in out:
        f.write(json.dumps(j) + "\n")
print(len(rows), "cells", len(out), "jobs ->", os.path.join(HERE, "a1_jobs.jsonl"))
