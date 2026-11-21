"""mk_a8_jobs.py -> a8/a8_jobs.jsonl: A8 self-audit selective prediction (R3 Q1) as q.py jobs.

For each model: 14Ln r11 --anno-override on the full Charades-STA test (3720 = 8 x 465), E window and scored target =
the model's OWN 12p4n prediction from the same machine & stack; ops mask only, R=3, seed 1000, all other probe
parameters = mk_newarm_jobs.py C14.  r11 shard o{off}n465 needs exactly the 12p4n shard o{off}n465 (same offsets),
so every job waits (wait_launch.py, file present and size stable 10 s) for its own override shard only; r11 itself
re-checks that every qid of its slice is in the override with the same video/caption/GT.
Override sources (out paths of the generators that produce them):
  qwenbase        tf451  mk_newarm_jobs.py  na__base_tf451__p4nat_o{off} -> results/newarm/_base_tf451/p4nat_qwenbase_tf451fa2_o{off}n465.json
  ref_*           tf451/tf5  mk_ref_p4nat.py  na__{tag}__p4nat_o{off}  -> results/newarm/{tag}/p4nat_{tag}_{sfx}_o{off}n465.json
  qwenbase5 (opt-in --with-base-tf5) tf5 mk_newarm_jobs.py na__base_tf5__p4nat_o{off} -> results/newarm/_base_tf5/p4nat_qwenbase_tf5_o{off}n465.json
All paths are built in this file from WORK_ROOT / DATASETS_ROOT; the only accepted command-line argument is --with-base-tf5 (some Windows
shells rewrite POSIX path arguments).  Every path is asserted absolute POSIX with no ':'.
Deploy 14Ln_native_sentinel_probe_r11.py (and its trainer r4 a2bf612cc95bde4e, as for r10) into probe3_selector_evidence.
usage: python mk_a8_jobs.py [--with-base-tf5]
"""
import json, os, sys
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)

extra = sys.argv[1:]
assert all(x == "--with-base-tf5" for x in extra), f"only --with-base-tf5 is accepted (paths are literals here), got {extra!r}"
WITH_BASE_TF5 = "--with-base-tf5" in extra

R = WORK_ROOT; P = f"{R}/probe3_selector_evidence"; D1 = (DATASETS_ROOT + "/ds-1")
W = f"{R}/setup/wait_launch.py"; WAIT_S = "172800"                 # 48 h; exit 3 -> q.py FAIL, requeue
FA = [f"{R}/setup/attn_launch.py", "flash_attention_2"]
PROBE = "14Ln_native_sentinel_probe_r11.py"
PRIO = 22
MB = ["--model", f"{R}/models/Qwen2.5-VL-7B-Instruct"]
STY = ["--prompt-style", "timer1_default", "--answer-style", "timer1"]
C14 = ["--ops", "mask", "--R", "3", "--thr", "0.5", "--seed", "1000", "--fps", "2.0", "--total-pixels", "2809856",
       "--min-pixels", "12544", "--max-frames", "768", "--max-new-tokens", "128", "--gpu", "0"]   # mk_newarm_jobs C14, ops mask
N, SHARD = 3720, 465

# (tag, env, adapter or None, override dir, override file stem) -- adapters as in mk_ref_p4nat.py CFG
CFG = [("qwenbase", "tf451", None, f"{R}/results/newarm/_base_tf451", "p4nat_qwenbase_tf451fa2"),
       ("ref_qwena05_s0", "tf451", f"{D1}/local__nat_base_a05_g1_s0", f"{R}/results/newarm/ref_qwena05_s0", "p4nat_ref_qwena05_s0_tf451fa2"),
       ("ref_qwensft", "tf451", f"{D1}/local__nat_sft_b0_s0", f"{R}/results/newarm/ref_qwensft", "p4nat_ref_qwensft_tf451fa2"),
       ("ref_qwenmdpo", "tf451", f"{D1}/local__adapter_nat_mdpo_a05_s0", f"{R}/results/newarm/ref_qwenmdpo", "p4nat_ref_qwenmdpo_tf451fa2"),
       ("ref_l4800", "tf5", f"{D1}/node3__adapter_nat_base_a05_g1_l4800_s0", f"{R}/results/newarm/ref_l4800", "p4nat_ref_l4800_tf5"),
       ("ref_iougrpo16f", "tf5", f"{D1}/local__x0_s0_iou700", f"{R}/results/newarm/ref_iougrpo16f", "p4nat_ref_iougrpo16f_tf5")]
if WITH_BASE_TF5:
    CFG.append(("qwenbase5", "tf5", None, f"{R}/results/newarm/_base_tf5", "p4nat_qwenbase_tf5"))

jobs = []
for tag, env, ad, odir, stem in CFG:
    assert env in ("tf451", "tf5")
    sfx = "tf451fa2" if env == "tf451" else "tf5"
    assert stem.endswith(sfx)
    O = f"{R}/results/a8/{tag}"
    M = MB + (["--adapter", ad] if ad else [])
    for off in range(0, N, SHARD):
        ovr = f"{odir}/{stem}_o{off}n{SHARD}.json"
        out = f"{O}/14Lself_{tag}_{sfx}_mask_o{off}n{SHARD}R3.json"
        argv = ([W, ovr, WAIT_S] + (FA if env == "tf451" else []) + [PROBE] + M + STY + C14
                + ["--all-rows", "--anno-override", ovr, "--n", str(SHARD), "--offset", str(off), "--out", out])
        jobs.append({"name": f"a8__{tag}__self_o{off}", "env": env, "cwd": P, "prio": PRIO, "argv": argv, "out": out,
                     "mem_gb": 30.0, "mkdir": O})


# ------------------------------------------------------------------ guards (MSYS mangling, coverage, uniqueness)
def _strings(j):
    yield j["cwd"]; yield j["out"]; yield j["mkdir"]
    yield from j["argv"]


for j in jobs:
    for s in _strings(j):
        assert isinstance(s, str) and ":" not in s, f"{j['name']}: ':' in {s!r} (MSYS-mangled or drive path)"
        if "/" in s:
            assert s.startswith("/") and "//" not in s, f"{j['name']}: path not absolute POSIX: {s!r}"
            assert s.startswith(((WORK_ROOT + "/"), (DATASETS_ROOT + "/ds-1/"))), f"{j['name']}: unexpected root {s!r}"
    a = j["argv"]
    assert a[0] == W and a[2] == WAIT_S and a.index(PROBE) == (5 if j["env"] == "tf451" else 3), j["name"]
    assert a[a.index("--anno-override") + 1] == a[1], f"{j['name']}: waits for a file other than its override"
    assert a[a.index("--out") + 1] == j["out"] and a.count("--ops") == 1 and a[a.index("--ops") + 1] == "mask"
    assert (FA[0] in a) == (j["env"] == "tf451")
assert len({j["name"] for j in jobs}) == len(jobs) and len({j["out"] for j in jobs}) == len(jobs)
for tag, *_ in CFG:
    offs = sorted(int(j["argv"][j["argv"].index("--offset") + 1]) for j in jobs if j["name"].startswith(f"a8__{tag}__"))
    assert offs == list(range(0, N, SHARD)) and len(offs) * SHARD == N, (tag, offs)

od = os.path.dirname(os.path.abspath(__file__))
fn = os.path.join(od, "a8_jobs.jsonl")
with open(fn, "w", encoding="utf-8", newline="\n") as f:
    for j in jobs:
        f.write(json.dumps(j) + "\n")
print(fn, len(jobs), "jobs,", len(CFG), "models:", ", ".join(c[0] for c in CFG))
