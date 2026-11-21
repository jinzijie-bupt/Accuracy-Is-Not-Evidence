#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""readout_all.py (revision, 2026-11-06) -- local one-shot readout of the node7 (cluster) products.

Local only: reads results/node7_pull/ (filled by podpull.py), writes results/node7_merged/ and rebuttal/<exp>/read/,
and rebuttal/READOUT_STATUS.md.  Never touches the remote machine or any existing file
(merge_shards.py, the readers and the mk_*_jobs.py scripts are called / read, never edited).

Pipeline
  1. Job lines from the mk_*_jobs.py outputs (JOBFILES below) -> each job's `out` (remote path under
     $WORK_ROOT/results/) -> local results/node7_pull/<same rel path>.  Jobs are grouped into CELLS by
     stripping the shard suffix (_o{off}n{n}[R3].json); the expected shards of a cell are exactly its job lines
     (deduplicated by `out`; base jobs shared by several newarm jsonl files count once).
  2. A shard counts only if it parses as a JSON row list with exactly --n rows and (when the .meta.json exists)
     meta.args.out == the job's out.  A cell whose shards are all present is merged with merge_shards.py into
     results/node7_merged/<rel dir>/<14Ln_|14Lqn_ prefix><stem>_node7_o{off0}n{N}[R3].json (written to a temp name,
     then renamed).  Merged names are checked to carry none of read_14Ln_v1's machine tokens
     (_node2_ _node1_ _node4_ _node3_ _node5_) nor its text-first tokens (textfirst, _tf_); the machine is passed as
     --machine node7.
  3. PAIRS (explicit table below, each row with its source line) -> registered reader as a subprocess; exit code,
     reader status, stdout/stderr tails, reader sha16 and input sha16s go to <readout>.stamp.json.
     A readout is re-run only when its argv, reader sha16 or input sha16s changed (or --force-read).
  4. rebuttal/READOUT_STATUS.md: per experiment a cell table (shards k/n, merged sha16) and a pair table
     (status, key numbers from the reader json, paper value from a1_configs.tsv where applicable, readout sha16,
     input sha16s).

Readers (analysis/):  R14 read_14Ln_v1 (merged files, --machine node7 --transformers 4.51|5.12)
                      TOK read_14Ln_tok_v1 (raw r10 shards: it checks every shard's meta for r10 + --tok-lp)
                      P4  read_p4nat_v1 (one readout per cell, merged file, --expect-n 3720; the pair row shows
                          arm - base of the two point estimates -- unpaired, no CI)
                      P5  read_p5n_v2 --pair (paired reader for 12p5n r3 products; v1 is per-product only)
                      F   read_forced_v1 (merged files)
                      QN  read_14Lqn_v2 --rep ARM --base BASE (reader of 14Lqn r3 124cfd516a39f427)
                      EQ  builtin field-by-field comparison (A3f off-mode equivalence r3 vs forced --forced off)

usage:  python rebuttal/readout_all.py [--dry-run] [--only a1,a6,...] [--force-read] [--no-md]
        --dry-run: no merge, no reader, no file written; prints the status report to stdout.
"""
import argparse, csv, datetime, glob, hashlib, json, os, re, subprocess, sys
LOCAL_ROOT = os.environ.get("LOCAL_ROOT", ".")  # analysis directory holding the pulled result files (results/, rebuttal/)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)

ROOT = LOCAL_ROOT
PULL = os.path.join(ROOT, "results", "node7_pull")
MERGED = os.path.join(ROOT, "results", "node7_merged")
REB = os.path.join(ROOT, "rebuttal")
AN = os.path.join(ROOT, "analysis")
MERGE_PY = os.path.join(REB, "merge_shards.py")
REMOTE = (WORK_ROOT + "/results/")
MD_OUT = os.path.join(REB, "READOUT_STATUS.md")
TSV_A1 = os.path.join(REB, "a1", "a1_configs.tsv")
MACHINE = "node7"
TF_OF = {"tf451": "4.51", "tf5": "5.12"}       # node7 stacks: tf4.51 / tf5.12.1 (A1_NOTES sec. 3.7)
MACHINE_TOKENS = ["node2", "node1", "node4", "node3", "tm"]   # read_14Ln_v1.MACHINE_TOKENS
SHARD_RE = re.compile(r"^(?P<stem>.*?)_+o(?P<off>\d+)n(?P<n>\d+)(?P<R>R\d+)?\.json$")
INSTR_RE = re.compile(r"^(14Ln|14Lqn|12p4n|12p5n)\S*\.py$")

EXPS = ["a1", "a4", "a5", "a6", "a3f", "a7", "newarm"]
EXP_TITLE = {
    "a1": "A1 independent 600-question audit (21 cells, r8, --qid-file indep600_qids.json)",
    "a4": "A4 audit robustness: span jitter ±10% and natural occlusion (first 200 questions)",
    "a5": "A5 / R1 W2c: r10 --tok-lp --cands, first 200 questions, timestamp token subset",
    "a6": "A6 RA-RFT vs its base model Time-R1-7B (tf451+FA2)",
    "a3f": "A3(2) forced-answer decoding (12p5n forced, 1000 questions)",
    "a7": "A7 behavioral eval of the Qwen family on training stack tf451 (12p5n r3, mask+swap)",
    "newarm": "standard eval suite for newly trained runs + old runs p4nat on the same machine",
}
JOBFILES = {
    "a1": ["a1/a1_jobs.jsonl"], "a4": ["a4/a4_jobs.jsonl", "a4/a4s_jobs.jsonl"], "a5": ["a5/a5_jobs.jsonl"],
    "a6": ["a6/a6_jobs.jsonl", "a6/a6b_jobs.jsonl"], "a3f": ["a3f/a3f_jobs.jsonl"], "a7": ["a7/a7_jobs.jsonl"],
    "newarm": None,   # = every rebuttal/newarm/*.jsonl (checked against NEWARM / NEWARM_REF below)
}
READER = {"R14": "read_14Ln_v1.py", "TOK": "read_14Ln_tok_v1.py", "P4": "read_p4nat_v1.py",
          "P5": "read_p5n_v2.py", "F": "read_forced_v1.py", "QN": "read_14Lqn_v2.py", "EQ": None}

# =====================================================================================================================
# PAIR TABLE (explicit).  Cell ids = remote out path relative to results/, shard suffix stripped.
# Row: (pair_id, kind, base_cell, arm_cell, source, opts)   opts: stack (tf451|tf5), ops, paper=(tsv line | literal)
# =====================================================================================================================
def _a1(name):
    return f"a1/{name}/{name}"


A1 = {  # tsv line -> cell_id   (a1_configs.tsv: header = line 1)
    2: "A1-01_qwen_base_t1d_tf451", 3: "A1-02_qwen_sft_tf451", 4: "A1-03_qwen_mdpo_tf451", 5: "A1-04_qwen_a05_tf451",
    6: "A1-05_vc_base_vchat_tf451", 7: "A1-06_vc_a05_tf451", 8: "A1-07_qwen_base_t1d_tf5", 9: "A1-08_qwen_iougrpo_tf5",
    10: "A1-09_tz_base_t1d_tf5", 11: "A1-10_tr1_base_t1d_tf5", 12: "A1-11_qwen_base_vchat_tf5",
    13: "A1-12_vc_base_vchat_tf5", 14: "A1-13_tz_a05_tf5", 15: "A1-14_tr1_a05_tf5", 16: "A1-15_qwen_a025_tf5",
    17: "A1-16_qwen_a10_tf5", 18: "A1-17_q3vl8b_base_timelens_tf5", 19: "A1-18_timelens8b_base_tf5",
    20: "A1-19_timelens8b_a05_tf5", 21: "A1-20_timelens8b_a025_tf5", 22: "A1-21_timelens8b_a10_tf5",
}
L = {v.split("_")[0]: k for k, v in A1.items()}     # "A1-04" -> tsv line 5
SRC_A1 = "a1_configs.tsv: {} notes/paper_ref; A1_NOTES.md sec. 2 pair table"

PAIRS = []


def P(pid, kind, base, arm, src, **opts):
    PAIRS.append({"exp": opts.pop("exp"), "pid": pid, "kind": kind, "base": base, "arm": arm, "src": src, **opts})


# ---- A1: 16 pairs (arm vs base), 600 independent questions, read_14Ln_v1 --expect-n 600
for arm, base, stack in [
        ("A1-02", "A1-01", "tf451"), ("A1-03", "A1-01", "tf451"), ("A1-04", "A1-01", "tf451"),
        ("A1-06", "A1-05", "tf451"),
        ("A1-08", "A1-07", "tf5"), ("A1-09", "A1-07", "tf5"), ("A1-10", "A1-07", "tf5"),
        ("A1-12", "A1-11", "tf5"),
        ("A1-13", "A1-09", "tf5"), ("A1-14", "A1-10", "tf5"),
        ("A1-15", "A1-07", "tf5"), ("A1-16", "A1-07", "tf5"),
        ("A1-18", "A1-17", "tf5"), ("A1-19", "A1-18", "tf5"), ("A1-20", "A1-18", "tf5"), ("A1-21", "A1-18", "tf5")]:
    P(f"{arm}_vs_{base}", "R14", _a1(A1[L[base]]), _a1(A1[L[arm]]), SRC_A1.format(L[arm]),
      exp="a1", stack=stack, paper=("tsv", L[arm]))

# ---- A4: mk_a4_jobs.py:11-16 (CELLS x COND), first 200; natocc runs --ops freeze,occlude
for cond, ops in [("jit_shrink", "mask,swap"), ("jit_expand", "mask,swap"), ("jit_shiftl", "mask,swap"),
                  ("jit_shiftr", "mask,swap"), ("natocc", "freeze,occlude")]:
    P(f"A4_qwensft_{cond}", "R14", f"a4/qwen_base__{cond}__tf451", f"a4/qwen_sft__{cond}__tf451",
      "a4/a4s_jobs.jsonl (11-10; qwen_sft = A1-02 args)", exp="a4", stack="tf451", ops=ops, paper=("tsv", L["A1-02"]))
    P(f"A4_qwen_{cond}", "R14", f"a4/qwen_base__{cond}__tf451", f"a4/qwen_a05__{cond}__tf451",
      "mk_a4_jobs.py:11-16 (qwen_base=A1-01 args, qwen_a05=A1-04 args)", exp="a4", stack="tf451", ops=ops,
      paper=("tsv", L["A1-04"]))
    P(f"A4_tz_{cond}", "R14", f"a4/tz_base__{cond}__tf5", f"a4/tz_a05__{cond}__tf5",
      "mk_a4_jobs.py:11-16 (tz_base=A1-09 args, tz_a05=A1-13 args)", exp="a4", stack="tf5", ops=ops,
      paper=("tsv", L["A1-13"]))


# ---- A5: mk_a5_jobs.py:18-30 (G); arm --base-pred-file = same-stack base of the same shard
def _a5(tag, env):
    return f"a5/{tag}/14Ln_{tag}_{env}{'fa2' if env == 'tf451' else ''}_tokc"


for btag, env, atag, paper, line in [
        ("qwenbase451", "tf451", "qwena05", ("tsv", L["A1-04"]), 19), ("qwenbase451", "tf451", "qwensft", ("tsv", L["A1-02"]), 19),
        ("qwenbase451", "tf451", "qwenmdpo", ("tsv", L["A1-03"]), 19),
        ("qwenbase5", "tf5", "iougrpo16f", ("tsv", L["A1-08"]), 22), ("qwenbase5", "tf5", "l4800",
                                                                       ("lit", "Γ +2.78 (A1_NOTES sec. 3.11; judgment_20260921 gamma file)"), 22),
        ("qwenbase5", "tf5", "qwena025", ("tsv", L["A1-15"]), 22), ("qwenbase5", "tf5", "qwena10", ("tsv", L["A1-16"]), 22),
        ("tzbase", "tf5", "tza05", ("tsv", L["A1-13"]), 25), ("tr1base5", "tf5", "tr1a05", ("tsv", L["A1-14"]), 26),
        ("tr1base451", "tf451", "rarft", None, 27), ("vcbase451", "tf451", "vca05", ("tsv", L["A1-06"]), 28),
        ("tlbase", "tf5", "tla05", ("tsv", L["A1-19"]), 29)]:
    P(f"A5_{atag}_vs_{btag}", "TOK", _a5(btag, env), _a5(atag, env), f"mk_a5_jobs.py:{line}", exp="a5", stack=env,
      paper=paper)

# ---- A6: mk_a6_jobs.py:24-43 (MODELS rarft / tr1base), mk_a6b_jobs.py:8-13
P("A6_14Ln_first200", "R14", "a6/14Ln_tr1base_timer1_default_tf451fa2", "a6/14Ln_rarft_timer1_default_tf451fa2",
  "mk_a6_jobs.py:27-29", exp="a6", stack="tf451")
P("A6_14Ln_indep600", "R14", "a6/14Ln_indep600_tr1base_timer1_default_tf451fa2",
  "a6/14Ln_indep600_rarft_timer1_default_tf451fa2", "mk_a6_jobs.py:30-33", exp="a6", stack="tf451")
P("A6_p4nat_timer1", "P4", "a6/p4nat_tr1base_timer1_default_tf451fa2", "a6/p4nat_rarft_timer1_default_tf451fa2",
  "mk_a6_jobs.py:34-36", exp="a6", stack="tf451")
P("A6b_p4nat_rarftnative", "P4", "a6/p4nat_tr1base_rarft_native_tf451fa2", "a6/p4nat_rarft_rarft_native_tf451fa2",
  "mk_a6b_jobs.py:8-13", exp="a6", stack="tf451")
P("A6_p5n", "P5", "a6/p5n_tr1base_timer1_default_tf451fa2", "a6/p5n_rarft_timer1_default_tf451fa2",
  "mk_a6_jobs.py:37-40", exp="a6", stack="tf451")
P("A6_14Lqn", "QN", "a6/14Lqn_tr1base_timer1_default_tf451fa2", "a6/14Lqn_rarft_timer1_default_tf451fa2",
  "mk_a6_jobs.py:41-43", exp="a6", stack="tf451")

# ---- A3f: mk_a3f_jobs.py:7-15 (CFG), 21-23 (eq)
for arm, base, sfx, line in [("qwena05", "qwenbase", "tf451fa2", 9), ("qwensft", "qwenbase", "tf451fa2", 10),
                             ("rarft", "tr1base", "tf451fa2", 12), ("tza05", "tzbase", "tf5", 14)]:
    P(f"A3F_{arm}_vs_{base}", "F", f"a3f/p5nF_{base}_timer1_default_{sfx}", f"a3f/p5nF_{arm}_timer1_default_{sfx}",
      f"mk_a3f_jobs.py:{line}", exp="a3f", stack="tf451" if sfx != "tf5" else "tf5")
P("A3F_eq_r3_vs_forcedoff", "EQ", "a3f/eq/r3_qwenbase_tf5", "a3f/eq/forcedoff_qwenbase_tf5", "mk_a3f_jobs.py:21-23",
  exp="a3f", stack="tf5")

# ---- A7: mk_a7_jobs.py:7-8 (CFG)
for arm in ("qwena05", "qwensft", "qwenmdpo"):
    P(f"A7_{arm}_vs_qwenbase", "P5", "a7/p5n_qwenbase_timer1_default_tf451fa2", f"a7/p5n_{arm}_timer1_default_tf451fa2",
      "mk_a7_jobs.py:7-8", exp="a7", stack="tf451")

# ---- newarm: one row per registered TAG (its jsonl is mk_newarm_jobs.py TAG ENV ...); the four partners per TAG are
#      the ones mk_newarm_jobs.py docstring lines 4-7 / code lines 31-55 name.
NEWARM = [("A2_grpo2fps_s0", "tf451"), ("B1a_qwen_a05_s2", "tf451"), ("B3_tsonly_a05_s0", "tf451"),
          ("B4_sft_l4800_s0", "tf5"), ("B5_grpo_placebo_lam001_s0", "tf451"),
          ("B5s1_grpo_placebo_lam001_s1", "tf451"),
          ("B2a_bound100_005_s0", "tf451"), ("B2b_bound150_020_s0", "tf451"),
          ("A2_grpo2fps_s1", "tf451"), ("A2_grpo2fps_s1c300", "tf451"),   # 11-11: IoU-GRPO seed 1 + its step-300 ckpt
          ("B6_tzgrpo_placebo_lam001_s0", "tf451tz")]   # 11-11: B5 recipe from TimeZero, paired with TimeZero tf451 base  # 11-10: bound-constant arms (R2 Q2)  # 11-10: B5 seed 1 (trained + evaluated on node9, copied into node7_pull)
NA_BASE = {  # env -> (tok base cell, indep600 base cell, p4nat base cell, p5n base cell)
    "tf451": (_a5("qwenbase451", "tf451"), _a1(A1[L["A1-01"]]), "newarm/_base_tf451/p4nat_qwenbase_tf451fa2",
              "a7/p5n_qwenbase_timer1_default_tf451fa2"),
    "tf5": (_a5("qwenbase5", "tf5"), _a1(A1[L["A1-07"]]), "newarm/_base_tf5/p4nat_qwenbase_tf5",
            "newarm/_base_tf5/p5n_qwenbase_tf5"),
    "tf451tz": (_a5("tzbase451", "tf451"), _a1("TZ451_tz_base_t1d_tf451"), "newarm/_base_tz_tf451/p4nat_tzbase_tf451fa2",
                "newarm/_base_tz_tf451/p5n_tzbase_timer1_default_tf451fa2"),
}
for tag, bkey in NEWARM:
    env = "tf5" if bkey == "tf5" else "tf451"
    sfx = "tf451fa2" if env == "tf451" else "tf5"
    tb, ib, pb, qb = NA_BASE[bkey]
    src = f"newarm/{tag}_jobs.jsonl; mk_newarm_jobs.py:"
    P(f"NA_{tag}_tokc_first200", "TOK", tb, f"newarm/{tag}/14Ln_{tag}_{sfx}_tokc", src + "33-37", exp="newarm", stack=env)
    P(f"NA_{tag}_14Ln_indep600", "R14", ib, f"newarm/{tag}/14Ln_indep600_{tag}_{sfx}", src + "38-41", exp="newarm",
      stack=env)
    P(f"NA_{tag}_p4nat", "P4", pb, f"newarm/{tag}/p4nat_{tag}_{sfx}", src + "42-48", exp="newarm", stack=env)
    P(f"NA_{tag}_p5n", "P5", qb, f"newarm/{tag}/p5n_{tag}_{sfx}", src + "49-55", exp="newarm", stack=env)
NEWARM_REF = [("ref_qwena05_s0", "tf451"), ("ref_qwensft", "tf451"), ("ref_qwenmdpo", "tf451"),
              ("ref_l4800", "tf5"), ("ref_iougrpo16f", "tf5")]          # mk_ref_p4nat.py:6-8
for tag, env in NEWARM_REF:
    sfx = "tf451fa2" if env == "tf451" else "tf5"
    P(f"NA_{tag}_p4nat", "P4", NA_BASE[env][2], f"newarm/{tag}/p4nat_{tag}_{sfx}", "mk_ref_p4nat.py:6-8", exp="newarm",
      stack=env)
NEWARM_JSONL = {f"{t}_jobs.jsonl" for t, _ in NEWARM} | {"ref_p4nat_jobs.jsonl"}

KNOWN_CAVEATS = {
    "newarm": ["indep600 for new tf451 runs uses attn_launch.py flash_attention_2, while the a1 job line for base model A1-01 has no attn_launch "
               "(different attention implementation); treat this as a known difference when reading, and add an FA2 version of A1-01 if needed."],
    "a6": ["p4nat is read once per side (read_p4nat_v1 is a single-file reader); the R@0.5 difference in the table is a difference of two point estimates, unpaired, no CI."],
    "newarm_p4": [],
}


# =====================================================================================================================
def sha16(p, _c={}):
    st = os.stat(p)
    k = (p, st.st_size, st.st_mtime_ns)
    if k not in _c:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        _c[k] = h.hexdigest()[:16]
    return _c[k]


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def argval(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def bad_name_tokens(name):
    b = "_" + name
    hit = [m for m in MACHINE_TOKENS if f"_{m}_" in b]
    if "textfirst" in name or "_tf_" in name:
        hit.append("text-first token")
    return hit


# ------------------------------------------------------------------------------------------------------------- cells
def load_jobs():
    files = {}
    for exp in EXPS:
        if JOBFILES[exp] is None:
            files[exp] = sorted(glob.glob(os.path.join(REB, "newarm", "*.jsonl")))
        else:
            files[exp] = [os.path.join(REB, *f.split("/")) for f in JOBFILES[exp]]
    cells, problems, jf_info = {}, [], []
    for exp, fl in files.items():
        for f in fl:
            if not os.path.exists(f):
                problems.append(f"job file does not exist: {rel(f)}")
                continue
            jf_info.append((exp, rel(f), sha16(f)))
            if exp == "newarm" and os.path.basename(f) not in NEWARM_JSONL:
                problems.append(f"unregistered newarm jsonl (not in pair table, skipped): {rel(f)}")
            for ln, line in enumerate(open(f, encoding="utf-8"), 1):
                if not line.strip():
                    continue
                j = json.loads(line)
                out = j["out"]
                assert out.startswith(REMOTE), out
                rp = out[len(REMOTE):]
                m = SHARD_RE.match(os.path.basename(rp))
                if not m:
                    problems.append(f"{rel(f)}:{ln} out is not a shard name: {rp}")
                    continue
                cid = os.path.dirname(rp) + "/" + m["stem"]
                argv = j["argv"]
                instr = next((a for a in argv if INSTR_RE.match(os.path.basename(a))), None)
                n = int(argval(argv, "--n", m["n"])); off = int(argval(argv, "--offset", m["off"]))
                if n != int(m["n"]) or off != int(m["off"]):
                    problems.append(f"{rel(f)}:{ln} argv --n/--offset does not match file name: {rp}")
                c = cells.setdefault(cid, {"id": cid, "exp": rp.split("/")[0], "instr": instr, "R": m["R"] or "",
                                           "env": j.get("env"), "shards": {}, "src": set()})
                c["src"].add(f"{rel(f)}")
                if c["instr"] != instr or c["R"] != (m["R"] or ""):
                    problems.append(f"cell {cid} inconsistent instrument/R within: {c['instr']} vs {instr}")
                sh = c["shards"].get(off)
                if sh and sh["remote"] != out:
                    problems.append(f"cell {cid} offset {off} has two different outs")
                if sh and sh["argv"] != argv:
                    problems.append(f"cell {cid} offset {off} argv differs across multiple jsonl ({rel(f)}:{ln})")
                if not sh:
                    c["shards"][off] = {"off": off, "n": n, "remote": out, "argv": argv, "job": j.get("name"),
                                        "local": os.path.join(PULL, *rp.split("/"))}
    return cells, problems, jf_info


def check_shard(sh):
    p = sh["local"]
    if not os.path.exists(p):
        return "missing", None
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        return f"bad: unparseable (in transit?) {type(e).__name__}", None
    if not isinstance(d, list):
        return "bad: not a list of rows", None
    if len(d) != sh["n"]:
        return f"bad: row count {len(d)} != --n {sh['n']}", None
    mp = p + ".meta.json"
    if os.path.exists(mp):
        try:
            m = json.load(open(mp, encoding="utf-8"))
            mo = (m.get("args") or {}).get("out")
            if mo is not None and mo != sh["remote"]:
                return f"bad: meta.args.out={mo} is not the out of this job", None
        except Exception as e:
            return f"bad: meta unparseable {type(e).__name__}", None
        return "ok", sha16(p)
    return "ok(no meta)", sha16(p)


def merged_path(c):
    offs = sorted(c["shards"])
    total = sum(s["n"] for s in c["shards"].values())
    d, stem = c["id"].rsplit("/", 1)
    ib = os.path.basename(c["instr"] or "")
    pre = "14Lqn" if ib.startswith("14Lqn") else ("14Ln" if ib.startswith("14Ln") else "")
    if pre and not stem.startswith(pre + "_"):
        stem = f"{pre}_{stem}"
    name = f"{stem}_{MACHINE}_o{offs[0]}n{total}{c['R']}.json"
    bad = bad_name_tokens(name)
    if bad:
        raise SystemExit(f"merged name {name} carries reader tokens {bad}; fix the naming before running")
    return os.path.join(MERGED, *d.split("/"), name), total


def process_cell(c, dry):
    offs = sorted(c["shards"])
    c["n_shards"] = len(offs)
    c["merged"], c["total"] = merged_path(c)
    exp_off = 0
    for o in offs:                      # contiguity of the job plan itself
        if o != exp_off:
            c.setdefault("plan_warn", []).append(f"offset not contiguous: expected {exp_off} got {o}")
        exp_off = o + c["shards"][o]["n"]
    for o in offs:
        st, s16 = check_shard(c["shards"][o])
        c["shards"][o]["status"], c["shards"][o]["sha16"] = st, s16
    good = [o for o in offs if c["shards"][o]["status"].startswith("ok")]
    c["k_ok"] = len(good)
    c["missing"] = [o for o in offs if c["shards"][o]["status"] == "missing"]
    c["bad"] = [(o, c["shards"][o]["status"]) for o in offs if c["shards"][o]["status"].startswith("bad")]
    c["merged_sha16"] = None
    if c["k_ok"] < len(offs):
        c["state"] = "INCOMPLETE"
        return
    want = [(os.path.basename(c["shards"][o]["local"]), c["shards"][o]["sha16"], c["shards"][o]["n"]) for o in offs]
    mp, mm = c["merged"], c["merged"] + ".meta.json"
    if os.path.exists(mp) and os.path.exists(mm):
        try:
            have = [(x["file"], x["sha16"], x["n_rows"]) for x in json.load(open(mm, encoding="utf-8")).get("merged_from", [])]
        except Exception:
            have = None
        if have == want:
            c["state"], c["merged_sha16"] = "MERGED", sha16(mp)
            return
        if have is None:
            c["state"], c["merge_err"] = "MERGE_ERROR", "existing merged file has unreadable meta, refusing to overwrite"
            return
    if dry:
        c["state"] = "READY_TO_MERGE"
        return
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    tmp = mp[:-5] + ".partial.json"
    r = subprocess.run([sys.executable, MERGE_PY, tmp] + [c["shards"][o]["local"] for o in offs], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        c["state"], c["merge_err"] = "MERGE_ERROR", (r.stderr or r.stdout).strip().splitlines()[-1:]
        for t in (tmp, tmp + ".meta.json"):
            if os.path.exists(t):
                os.remove(t)
        return
    os.replace(tmp + ".meta.json", mm)
    os.replace(tmp, mp)
    c["state"], c["merged_sha16"] = "MERGED", sha16(mp)


# ---------------------------------------------------------------------------------------------------------- readouts
READ_ROOT = REB          # --sandbox redirects this (and PULL / MERGED / MD_OUT) into a scratch directory


def read_dir(exp):
    return os.path.join(READ_ROOT, exp, "read")


def slug(cid):
    return cid.replace("/", "__")


def reader_sha(kind):
    return sha16(os.path.join(AN, READER[kind])) if READER[kind] else sha16(os.path.abspath(__file__))


def build_task(kind, pid, exp, cells_in, opts):
    """-> dict(out, argv, inputs) or None. cells_in: list of cells (base, arm) or [cell] for P4 cell readouts."""
    rd = read_dir(exp)
    py = sys.executable
    if kind == "R14":
        b, a = cells_in
        out = os.path.join(rd, f"{pid}.json")
        argv = [py, os.path.join(AN, READER[kind]), "--base", b["merged"], "--arm", a["merged"],
                "--expect-n", str(a["total"]), "--ops", opts.get("ops", "mask,swap"), "--machine", MACHINE,
                "--transformers", TF_OF[opts["stack"]], "--label", pid, "--out", out]
        inputs = [b["merged"], a["merged"]]
    elif kind == "TOK":
        b, a = cells_in
        out = os.path.join(rd, f"{pid}.json")
        bs = [b["shards"][o]["local"] for o in sorted(b["shards"])]
        as_ = [a["shards"][o]["local"] for o in sorted(a["shards"])]
        argv = [py, os.path.join(AN, READER[kind]), "--base"] + bs + ["--arm"] + as_ + \
               ["--expect-n", str(a["total"]), "--label", pid, "--out", out]
        inputs = bs + as_
    elif kind == "P4":
        (c,) = cells_in
        out = os.path.join(rd, f"p4nat_cell__{slug(c['id'])}.json")
        argv = [py, os.path.join(AN, READER[kind]), "--label", c["id"], "--files", c["merged"], "--expect-n",
                str(c["total"]), "--out", out]
        inputs = [c["merged"]]
    elif kind == "P5":
        b, a = cells_in
        out = os.path.join(rd, f"{pid}.json")
        argv = [py, os.path.join(AN, READER[kind]), "--pair", b["merged"], a["merged"], "--label", pid, "--out", out]
        inputs = [b["merged"], a["merged"]]
    elif kind == "F":
        b, a = cells_in
        pre = os.path.join(rd, pid)
        out = pre + ".json"
        argv = [py, os.path.join(AN, READER[kind]), "--base", b["merged"], "--arm", a["merged"], "--label", pid,
                "--out", pre]
        inputs = [b["merged"], a["merged"]]
    elif kind == "QN":
        b, a = cells_in
        out = os.path.join(rd, f"{pid}.json")
        argv = [py, os.path.join(AN, READER[kind]), "--rep", a["merged"], "--base", b["merged"], "--out", out]
        inputs = [b["merged"], a["merged"]]
    elif kind == "EQ":
        b, a = cells_in
        out = os.path.join(rd, f"{pid}.json")
        argv = ["<builtin eq_compare>", b["merged"], a["merged"], out]
        inputs = [b["merged"], a["merged"]]
    else:
        raise ValueError(kind)
    return {"kind": kind, "out": out, "argv": argv, "inputs": inputs, "stamp": out + ".stamp.json"}


EQ_FIELD = re.compile(r"^(none|mask_evid|mask_rand|swap_evid|swap_rand|black)_(hit|parsed|iou)$")
EQ_KEYS = ("n_estar", "nframes", "swap_donor", "mask_rand_idx", "swap_rand_idx", "degenerate", "video")


def eq_compare(bp, ap, out):
    B = {int(r["qid"]): r for r in json.load(open(bp, encoding="utf-8"))}
    A = {int(r["qid"]): r for r in json.load(open(ap, encoding="utf-8"))}
    qs = sorted(set(B) & set(A))
    mism, nf = [], 0
    for q in qs:
        keys = sorted({k for k in set(B[q]) | set(A[q]) if EQ_FIELD.match(k)} | {k for k in EQ_KEYS if k in B[q] or k in A[q]})
        for k in keys:
            nf += 1
            if B[q].get(k, "<absent>") != A[q].get(k, "<absent>"):
                mism.append({"qid": q, "field": k, "r3": B[q].get(k, "<absent>"), "forced_off": A[q].get(k, "<absent>")})
    res = {"reader": "readout_all.eq_compare", "reader_sha16": sha16(os.path.abspath(__file__)),
           "generated_utc": now(), "status": "OK" if (qs and not mism and set(A) == set(B)) else "DIFF",
           "definition": "field-by-field equality of {cond}_{hit,parsed,iou} and pairing keys, r3 product vs forced --forced off",
           "inputs": [{"role": "r3", "file": os.path.basename(bp), "sha16": sha16(bp), "n_rows": len(B)},
                      {"role": "forced_off", "file": os.path.basename(ap), "sha16": sha16(ap), "n_rows": len(A)}],
           "n_qid_shared": len(qs), "qid_only_one_side": sorted(set(A) ^ set(B)), "n_fields_compared": nf,
           "n_mismatch": len(mism), "mismatch_first20": mism[:20]}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return 0 if res["status"] == "OK" else 4


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_of(t):
    return {"argv": [rel(x) if os.path.isabs(x) and x.startswith(ROOT) else (os.path.basename(x) if x == sys.executable else x)
                     for x in t["argv"]],
            "reader": READER[t["kind"]] or "readout_all.eq_compare", "reader_sha16": reader_sha(t["kind"]),
            "inputs": [{"path": rel(p), "sha16": sha16(p)} for p in t["inputs"]]}


def run_task(t, dry, force):
    """returns the task with state/rc/status/out_sha16 filled."""
    want = stamp_of(t)
    st = None
    if os.path.exists(t["stamp"]):
        try:
            st = json.load(open(t["stamp"], encoding="utf-8"))
        except Exception:
            st = None
    fresh = (st is not None and all(st.get(k) == want[k] for k in ("argv", "reader_sha16", "inputs"))
             and (st.get("out_sha16") is None or (os.path.exists(t["out"]) and sha16(t["out"]) == st.get("out_sha16"))))
    if fresh and not force:
        t.update({k: v for k, v in st.items() if k not in ("out", "argv", "inputs")}, state="CACHED")
        return t
    if os.path.exists(t["out"]) and st is None:
        t.update(state="CONFLICT", status="CONFLICT", rc=None,
                 note="reader output path holds a file not produced by this script (no stamp), refusing to overwrite")
        return t
    if dry:
        t.update(state="WOULD_RUN", status="to read", rc=None)
        return t
    os.makedirs(os.path.dirname(t["out"]), exist_ok=True)
    for p in (t["out"], t["out"][:-5] + ".tsv"):
        if os.path.exists(p) and st is not None:      # our own previous output (it has a stamp)
            os.remove(p)
    if t["kind"] == "EQ":
        try:
            rc, so, se = eq_compare(t["inputs"][0], t["inputs"][1], t["out"]), "", ""
        except Exception as e:
            rc, so, se = 1, "", f"{type(e).__name__}: {e}"
    else:
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        r = subprocess.run(t["argv"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        rc, so, se = r.returncode, r.stdout, r.stderr
    status = None
    out_sha = None
    if os.path.exists(t["out"]):
        out_sha = sha16(t["out"])
        try:
            status = json.load(open(t["out"], encoding="utf-8")).get("status")
        except Exception:
            status = "UNREADABLE"
    if status is None:
        status = "OK" if (rc == 0 and out_sha) else "ERROR"
    rec = dict(want, rc=rc, status=status, out_rel=rel(t["out"]), out_sha16=out_sha, generated_utc=now(),
               stdout_tail=so.strip().splitlines()[-12:], stderr_tail=se.strip().splitlines()[-8:])
    json.dump(rec, open(t["stamp"], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    t.update({k: v for k, v in rec.items() if k not in ("argv", "inputs")}, state="RAN")
    return t


# ------------------------------------------------------------------------------------------------------- formatting
def f3(x, d=3):
    return "n/a" if x is None else f"{x:+.{d}f}"


def fci(ci, d=3):
    if not ci or ci[0] is None:
        return "[n/a]"
    return f"[{ci[0]:+.{d}f},{ci[1]:+.{d}f}]"


def jload(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def in_ci(v, ci):
    if v is None or not ci or ci[0] is None:
        return None
    return ci[0] <= v <= ci[1]


PAPER_G = re.compile(r"\bG([ms])\s*([+-]?\d+\.\d+)")


def paper_gm_gs(s):
    out = {}
    for k, v in PAPER_G.findall(s or ""):
        out.setdefault({"m": "mask", "s": "swap"}[k], float(v))
    return out


def key_numbers(kind, d, pair=None, ops=("mask", "swap")):
    """-> (key text, {op: (gamma, ci)} for replication check)"""
    if d is None:
        return "reader output unreadable", {}
    if kind == "R14":
        po = d.get("per_op", {})
        parts, g = [], {}
        for op in ops:
            x = po.get(op) or {}
            parts.append(f"Γ{op} {f3(x.get('gamma'))} {fci(x.get('gamma_ci95'))}")
            g[op] = (x.get("gamma"), x.get("gamma_ci95"))
        b, a = d.get("clean_R50_base"), d.get("clean_R50_arm")
        parts.append(f"n={d.get('n_cohort')}; R@.5 b/a {b:.3f}/{a:.3f} (Δ{a - b:+.3f})" if b is not None and a is not None else "")
        if d.get("errors"):
            parts.append("ERR: " + "; ".join(d["errors"])[:200])
        return "; ".join(p for p in parts if p), g
    if kind == "TOK":
        po = d.get("per_op", {})
        parts, g = [], {}
        for op in ops:
            x = po.get(op) or {}
            s = x.get("subsets", {})
            al, ts = s.get("all", {}), s.get("ts", {})
            sh = (x.get("timestamp_share") or {})
            parts.append(f"Γ{op} all {f3(al.get('gamma'))} {fci(al.get('gamma_ci95'))}, ts {f3(ts.get('gamma'))} "
                         f"{fci(ts.get('gamma_ci95'))}, ts share (seqsum) {f3(sh.get('share_of_Gamma_seqsum'), 2)} "
                         f"{fci(sh.get('share_of_Gamma_seqsum_ci95'), 2)}")
            g[op] = (al.get("gamma"), al.get("gamma_ci95"))
        parts.append(f"n={d.get('n_cohort')}")
        if d.get("errors"):
            parts.append("ERR: " + "; ".join(d["errors"])[:200])
        return "; ".join(parts), g
    if kind == "P5":
        aq = d.get("all_questions") or {}
        pr = aq.get("paired", {})
        parts = []
        for ax in ("mask", "swap"):
            e = (pr.get(ax) or {}).get("DiD_contrast_arm_minus_base") or {}
            parts.append(f"DiD {ax} {f3(e.get('est'))} {fci(e.get('ci95'))} p={e.get('p_two_sided')}")
        nb = (((aq.get("base") or {}).get("conditions") or {}).get("none") or {}).get("R@0.5", {}).get("est")
        na = (((aq.get("arm") or {}).get("conditions") or {}).get("none") or {}).get("R@0.5", {}).get("est")
        if nb is not None and na is not None:
            parts.append(f"clean R@.5 b/a {nb:.3f}/{na:.3f}")
        sub = d.get("all_conditions_parsed_subset") or {}
        if sub.get("n"):
            parts.append("fully parsed subset n=%d DiD swap %s mask %s" % (
                sub["n"], f3(((sub.get("paired") or {}).get("swap") or {}).get("DiD_contrast_arm_minus_base", {}).get("est")),
                f3(((sub.get("paired") or {}).get("mask") or {}).get("DiD_contrast_arm_minus_base", {}).get("est"))))
        parts.append(f"n={(d.get('pairing') or {}).get('n_paired')}")
        return "; ".join(parts), {}
    if kind == "F":
        R = d.get("results") or {}
        dr = (R.get("arm_minus_base") or {}).get("drops", {})
        parts = []
        for ax in ("mask", "swap"):
            e = ((dr.get(ax) or {}).get("O_minus_E") or {}).get("R@0.5") or {}
            parts.append(f"E−O DiD(R@.5) {ax} {f3(e.get('est'))} {fci(e.get('ci95'))} p={e.get('p_two_sided')}")
        parts.append(f"n={R.get('n')}")
        return "; ".join(parts), {}
    if kind == "QN":
        parts = []
        for op in ("mask", "swap"):
            x = d.get(op) or {}
            sp = (x.get("spec") or {}).get("vs_base") or {}
            ol = (x.get("old") or {}).get("vs_base") or {}
            parts.append(f"{op}: Γspec {f3(sp.get('gamma_spec'))} {fci(sp.get('gamma_spec_ci'))} (n={sp.get('n')}), "
                         f"Γ(old) {f3(ol.get('gamma'))} {fci(ol.get('gamma_ci'))}, Γsem {f3(ol.get('gamma_sem'))}")
        return "; ".join(parts), {}
    if kind == "EQ":
        return f"n_qid={d.get('n_qid_shared')} field {d.get('n_fields_compared')} inconsistent {d.get('n_mismatch')}", {}
    return "", {}


# ------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of experiments (a1,a4,a5,a6,a3f,a7,newarm)")
    ap.add_argument("--force-read", action="store_true", help="re-run readers even if the stamp is fresh")
    ap.add_argument("--no-md", action="store_true", help="do not write READOUT_STATUS.md (print it instead)")
    ap.add_argument("--sandbox", default=None, help="test mode: read DIR/node7_pull, write DIR/node7_merged, DIR/<exp>/read, "
                                                    "DIR/READOUT_STATUS.md (job files, readers, tsv stay the real ones)")
    a = ap.parse_args()
    if a.sandbox:
        global PULL, MERGED, READ_ROOT, MD_OUT
        sb = os.path.abspath(a.sandbox)
        PULL, MERGED, READ_ROOT, MD_OUT = (os.path.join(sb, "node7_pull"), os.path.join(sb, "node7_merged"), sb,
                                           os.path.join(sb, "READOUT_STATUS.md"))
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    only = set(a.only.split(",")) if a.only else set(EXPS)
    bad_only = only - set(EXPS)
    if bad_only:
        ap.error(f"unknown experiment(s) {sorted(bad_only)}")

    tsv = {i: r for i, r in enumerate(csv.DictReader(open(TSV_A1, encoding="utf-8"), delimiter="\t"), 2)}
    for ln, cid in A1.items():            # guard: the line numbers cited in the pair table still point at these cells
        assert tsv[ln]["cell_id"] == cid, (ln, tsv[ln]["cell_id"], cid)

    cells, problems, jf_info = load_jobs()
    pair_cells = set()
    for p in PAIRS:
        for side in ("base", "arm"):
            if p[side] not in cells:
                problems.append(f"pair {p['pid']} references cell {p[side]} which is not in any job file")
            pair_cells.add(p[side])
    unpaired = sorted(c for c in cells if c not in pair_cells)

    need = {c for p in PAIRS if p["exp"] in only for c in (p["base"], p["arm"]) if c in cells}
    for cid in sorted(need):
        process_cell(cells[cid], a.dry_run)

    # ---- readouts
    p4_tasks = {}
    for p in PAIRS:
        if p["exp"] not in only:
            continue
        b, ar = cells.get(p["base"]), cells.get(p["arm"])
        p["state"], p["task"], p["tasks"] = None, None, []
        if b is None or ar is None:
            p["state"] = "TABLE_ERROR"
            continue
        if p["kind"] == "P4":
            for c in (b, ar):
                if c["state"] == "MERGED" and c["id"] not in p4_tasks:
                    p4_tasks[c["id"]] = run_task(build_task("P4", None, c["exp"], [c], p), a.dry_run, a.force_read)
            p["tasks"] = [p4_tasks.get(b["id"]), p4_tasks.get(ar["id"])]
        if b["state"] == "INCOMPLETE" or ar["state"] == "INCOMPLETE":
            p["state"] = "MISSING"
        elif "MERGE_ERROR" in (b["state"], ar["state"]):
            p["state"] = "MERGE_ERROR"
        elif "READY_TO_MERGE" in (b["state"], ar["state"]):
            p["state"] = "READY"
        elif p["kind"] == "P4":
            p["state"] = "READ"
        else:
            if p["kind"] in ("R14", "TOK") and b["total"] != ar["total"]:
                p["state"] = "TABLE_ERROR"; p["note"] = f"base n {b['total']} != arm n {ar['total']}"
                continue
            p["task"] = run_task(build_task(p["kind"], p["pid"], p["exp"], [b, ar], p), a.dry_run, a.force_read)
            p["state"] = "READ"

    md = render(cells, problems, jf_info, unpaired, tsv, only, a.dry_run)
    if a.dry_run or a.no_md:
        print(md)
    else:
        tmp = MD_OUT + ".partial"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(md)
        os.replace(tmp, MD_OUT)
    # ---- terminal summary
    for exp in EXPS:
        if exp not in only:
            continue
        ps = [p for p in PAIRS if p["exp"] == exp]
        cnt = {}
        for p in ps:
            k = pair_status(p)[0]
            cnt[k] = cnt.get(k, 0) + 1
        print(f"{exp:7s} pairs {len(ps):3d}  " + "  ".join(f"{k}:{v}" for k, v in sorted(cnt.items())))
    for p in PAIRS:
        if p["exp"] in only and p.get("task") and p["task"].get("state") == "RAN":
            print(f"  ran {p['pid']}: rc={p['task'].get('rc')} status={p['task'].get('status')}")
    if not (a.dry_run or a.no_md):
        print("->", MD_OUT, "sha16", sha16(MD_OUT))


def pair_status(p):
    s = p.get("state")
    if s == "MISSING":
        return "missing shards", None
    if s in ("TABLE_ERROR", "MERGE_ERROR"):
        return s, None
    if s == "READY":
        return "to merge/to read", None
    if s == "READ":
        ts = [p["task"]] if p["task"] else [t for t in p["tasks"] if t]
        if not ts:
            return "to read", None
        sts = [t.get("status") for t in ts]
        if any(x == "to read" for x in sts):
            return "to read", None
        if any(x in ("ERROR", "UNREADABLE", "CONFLICT", "DIFF") for x in sts):
            bad = [f"{x}(rc={t.get('rc')})" for x, t in zip(sts, ts) if x not in ("OK", "WARN")]
            return "reader " + ",".join(bad), None
        return "read " + "/".join(sorted(set(sts))), None
    return str(s), None


def shard_cell_txt(c):
    if c is None:
        return "?"
    n = c["n_shards"] if "n_shards" in c else len(c["shards"])
    k = c.get("k_ok", "?")
    t = f"{k}/{n}"
    if c.get("bad"):
        t += " bad:" + ",".join(f"o{o}" for o, _ in c["bad"])
    return t


def render(cells, problems, jf_info, unpaired, tsv, only, dry):
    L_ = []
    w = L_.append
    w("# READOUT_STATUS (revision, local readout of node7 outputs)")
    w("")
    w(f"- Generated by: `rebuttal/readout_all.py` (sha16 `{sha16(os.path.abspath(__file__))}`),{now()} UTC"
      + (", **dry-run** (not merged, readers not run)" if dry else ""))
    w(f"- Local pull root: `results/node7_pull/`; merged files: `results/node7_merged/`; merge script `rebuttal/merge_shards.py` sha16 `{sha16(MERGE_PY)}`")
    w("- reader sha16:" + ",".join(f"`{v}` `{sha16(os.path.join(AN, v))}`" for k, v in READER.items() if v))
    w(f"- Machine token: merged file names do not contain `_node2_ _node1_ _node4_ _node3_ _node5_`; read_14Ln_v1 is always passed `--machine {MACHINE}`;"
      "`--transformers` by stack: tf451→4.51, tf5→5.12 (A1_NOTES sec. 3.7).")
    w("- Status: **missing shards** = at least one side has incomplete shards (k/n listed); **read** = reader exited and json status OK/WARN; **reader ERROR** = exit code≠0 or json status ERROR.")
    w("- Every number comes from the reader output json in the table (path + sha16), which records input sha16; the stamp sidecar `<reader output>.stamp.json` records argv, exit code, reader sha16.")
    w("")
    w("job files:" + ";".join(f"`{p}` `{s}`" for _, p, s in jf_info))
    w("")
    if problems:
        w("**Table/plan-level issues**:")
        for x in problems:
            w(f"- {x}")
        w("")
    if unpaired:
        w("Cells not referenced by any pair (status only, not read):" + ", ".join(f"`{c}`" for c in unpaired))
        w("")
    for exp in EXPS:
        if exp not in only:
            continue
        ps = [p for p in PAIRS if p["exp"] == exp]
        w(f"## {exp}:{EXP_TITLE[exp]}")
        w("")
        for cv in KNOWN_CAVEATS.get(exp, []):
            w(f"> Note: {cv}")
            w("")
        # cells
        cids = []
        for p in ps:
            for cid in (p["base"], p["arm"]):
                if cid not in cids:
                    cids.append(cid)
        w("| cell | instrument | shards k/n | missing offsets | merged file | sha16 |")
        w("|---|---|---|---|---|---|")
        for cid in cids:
            c = cells.get(cid)
            if c is None:
                w(f"| `{cid}` | — | not in job file | | | |")
                continue
            miss = ",".join(str(o) for o in c.get("missing", [])) or "—"
            st = c.get("state", "?")
            mtxt = rel(c["merged"]) if c.get("merged_sha16") else (st if st != "INCOMPLETE" else "—")
            if c.get("merge_err"):
                mtxt = f"MERGE_ERROR: {c['merge_err']}"
            w(f"| `{cid}` | {os.path.basename(c['instr'] or '?')} | {shard_cell_txt(c)} | {miss} | {mtxt} | "
              f"{('`' + c['merged_sha16'] + '`') if c.get('merged_sha16') else ''} |")
        w("")
        # pairs
        has_paper = any(p.get("paper") for p in ps)
        hdr = "| pair | source | base shards | arm shards | status | key numbers |"
        sep = "|---|---|---|---|---|---|"
        if has_paper:
            hdr += " paper value | paper point value inside new CI (mask/swap) |"
            sep += "---|---|"
        hdr += " reader output (sha16) | input sha16 (base / arm) |"
        sep += "---|---|"
        w(hdr)
        w(sep)
        for p in ps:
            b, ar = cells.get(p["base"]), cells.get(p["arm"])
            st, _ = pair_status(p)
            key, gam, rfile, ins = "", {}, "", ""
            if p.get("kind") == "P4" and p.get("state") == "READ":
                tb, ta = (p.get("tasks") or [None, None])
                db = jload(tb["out"]) if tb and tb.get("out_sha16") else None
                da = jload(ta["out"]) if ta and ta.get("out_sha16") else None
                if db and da:
                    key = (f"R@.5 base {db['R@0.5']:.3f} {fci(db['R@0.5_ci95'])} / arm {da['R@0.5']:.3f} {fci(da['R@0.5_ci95'])}; "
                           f"Δ(arm−base) {da['R@0.5'] - db['R@0.5']:+.3f}(unpaired point-estimate difference); mIoU Δ {da['mIoU'] - db['mIoU']:+.3f}; "
                           f"parse b/a {db['parse_rate']:.3f}/{da['parse_rate']:.3f}; n {db['n']}/{da['n']}")
                    ins = f"`{db['inputs'][0]['sha16']}` / `{da['inputs'][0]['sha16']}`"
                rfile = "<br>".join(f"`{rel(t['out'])}` (`{t.get('out_sha16')}`)" for t in (tb, ta) if t and t.get("out_sha16"))
                if not rfile:
                    rfile = "<br>".join(f"{t.get('status')} rc={t.get('rc')}: {' '.join(t.get('stderr_tail') or [])[:160]}"
                                        for t in (tb, ta) if t)
            elif p.get("task"):
                t = p["task"]
                if t.get("out_sha16"):
                    d = jload(t["out"])
                    key, gam = key_numbers(p["kind"], d, p, tuple((p.get("ops") or "mask,swap").split(",")))
                    rfile = f"`{rel(t['out'])}` (`{t['out_sha16']}`)"
                    ins = input_shas(p["kind"], d)
                    if t.get("rc") not in (0, None):
                        key += f"(rc={t.get('rc')})"
                else:
                    key = f"{t.get('status')} rc={t.get('rc')}: " + " ".join((t.get("stderr_tail") or [])[-2:])[:240]
                    key = key.replace("|", "/")
            elif p.get("state") == "MISSING":
                key = "missing " + "; ".join(f"{side} {shard_cell_txt(c)}" for side, c in (("base", b), ("arm", ar))
                                       if c and c.get("k_ok", 0) < len(c["shards"]))
            row = f"| {p['pid']} | {p['src']} | {shard_cell_txt(b)} | {shard_cell_txt(ar)} | {st} | {key} |"
            if has_paper:
                pv, rep = "—", ""
                if p.get("paper"):
                    if p["paper"][0] == "tsv":
                        r = tsv[p["paper"][1]]
                        pv = f"{r['paper_value']}(tsv:{p['paper'][1]} {r['cell_id'].split('_')[0]})"
                    else:
                        pv = p["paper"][1]
                    pg = paper_gm_gs(pv) if p["paper"][0] == "tsv" else {}
                    if gam and pg and not st.startswith("read"):
                        rep = "— (reader not OK/WARN, not compared)"
                    elif gam and pg:
                        rr = []
                        for op in ("mask", "swap"):
                            if op in pg and op in gam:
                                v = in_ci(pg[op], gam[op][1])
                                rr.append(f"{op} {pg[op]:+.2f}:" + ("—" if v is None else ("yes" if v else "no")))
                        rep = "; ".join(rr)
                row += f" {pv.replace('|', '/')} | {rep} |"
            row += f" {rfile} | {ins} |"
            w(row)
        w("")
    return "\n".join(L_) + "\n"


def input_shas(kind, d):
    if not d:
        return ""
    if kind in ("R14",):
        i = {x["role"]: x["sha16"] for x in d.get("inputs", [])}
        return f"`{i.get('base')}` / `{i.get('arm')}`"
    if kind == "TOK":
        ins = d.get("inputs", [])
        return " / ".join(s + " " + ",".join(f"`{x.get('sha16')}`" for x in ins if x.get("role") == s) for s in ("base", "arm"))
    if kind == "P5":
        i = d.get("inputs", {})
        return f"`{i.get('base', {}).get('sha16')}` / `{i.get('arm', {}).get('sha16')}`"
    if kind == "F":
        i = d.get("inputs", {})
        return " / ".join(",".join(f"`{x['sha16']}`" for x in i.get(s, [])) for s in ("base", "arm"))
    if kind == "QN":
        i = {x["role"]: x["sha16"] for x in d.get("inputs", [])}
        return f"`{i.get('base')}` / `{i.get('rep')}`"
    if kind == "EQ":
        return " / ".join(f"`{x['sha16']}`" for x in d.get("inputs", []))
    return ""


if __name__ == "__main__":
    main()
