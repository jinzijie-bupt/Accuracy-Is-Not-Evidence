#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""canonical_v2.py -- one canonical evaluation environment per model configuration (2026-09-23).

Rules (approved by the author, enforced here and nowhere else):
  R1  every model configuration (family, objective, alpha, beta, seed, trained message order,
      evaluated message order) has exactly ONE canonical environment; every absolute number of that
      configuration in every table (main text and appendix) comes from that environment;
  R2  prefer the transformers-5 stacks (A = node2 tf5.13, B = node1/node3/node5 tf5.12); use C (node4, tf4.51)
      only when neither A nor B holds ANY table-relevant product of the configuration.  If both A and B
      hold products, pick the one covering more of the configuration's table cells; on a tie, the one
      whose products pair more completely with the configuration's own base inside that stack;
  R3  paired quantities (Delta R@0.5, Gamma, behavioural DiD, query-swap contrasts) close inside the
      trained model's canonical stack: the base run is taken from THAT stack.  A missing base run in that
      stack renders \\pending -- never a cross-stack borrow;
  R4  a base row shows the base's own canonical stack; the base runs used inside paired cells of trained
      models may come from other stacks (listed in the manifest as base_pairing_envs);
  R5  no per-cell cherry-picking: within one (kind, configuration, stack, dataset) at most one record may
      match; the few documented duplicates are resolved by PRIMARY_RUN (fixed, model-independent reasons),
      anything else raises.

Cell types that count toward coverage (R2) are the table-relevant (kind, dataset) pairs of the
registry (acc, black, 14Ln gamma, base-referenced pair_acc, tneg, labelswap, behav) plus the table
products that live outside the registry (S11 query-swap qswap3_*, S12 newly-correct keys, S13 message
order runs, S14 Charades-CD runs), each counted once per (kind, dataset).

Usage as a module:
    from canonical_v2 import Canon, M
    C = Canon()                          # reads tables/paper_numbers_v2.json
    C.ctx("T1", "Qwen / grouped a=0.5")
    acc = C.pick("acc", M(Q7, "grouped", 0.5, 0.1, 0), column="R50")
Usage as a script:  python canonical_v2.py   -> writes tables/CANONICAL_MANIFEST_v2.json
"""
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..", "..")).replace("\\", "/")
import glob, hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = _SUPP + "/results"
REG_V2 = os.path.join(ROOT, "tables", "paper_numbers_v2.json")
OVERRIDES = os.path.join(ROOT, "tables", "CANONICAL_OVERRIDES_v2.json")   # optional, {"<key>": "A|B|C", ...}
MANIFEST = os.path.join(ROOT, "tables", "CANONICAL_MANIFEST_v2.json")

Q7, Q3 = "Qwen2.5-VL-7B", "Qwen2.5-VL-3B"
EXCLUDE_FAMILIES = {Q3}
TZ, TR, VC, TL = "TimeZero-7B", "Time-R1-7B", "VideoChat-R1-7B", "TimeLens-8B"
TF5 = ("A", "B")
ENV_OF_MACHINE = {"node2": "A", "node1": "B", "node3": "B", "node5": "B", "node5": "B", "node4": "C"}
TABLE_KINDS = ("acc", "black", "gamma", "pair_acc", "tneg", "labelswap", "behav")
PEND = "\\pending"


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------- model keys
def M(family, objective, alpha=None, beta=None, seed=None, train="v", eval="v"):
    """model-configuration key.  base: alpha/beta/seed/train are None/'v' by construction."""
    if objective == "base":
        alpha = beta = seed = None
        train = "v"
    return (family, objective, alpha, beta, seed, train, eval)


def key_str(k):
    f, o, a, b, s, tr, ev = k
    parts = [f, o]
    if a is not None:
        parts.append(f"a={a:g}")
    if b is not None:
        parts.append(f"b={b:g}")
    if s is not None:
        parts.append(f"s={s}")
    parts.append(f"train={tr}")
    parts.append(f"eval={ev}")
    return "|".join(parts)


def base_of(k):
    return M(k[0], "base", eval=k[6])


def _train_tf(name):
    return bool(re.search(r"_g1_tf_s\d|sft_tf_b\d|sft2g_(tz|vc)_tf_s\d|nat_tz_a05_g1_tf_s\d", name or ""))


def _eval_tf_from_name(name):
    n = name or ""
    return bool("textfirst" in n or re.search(r"_tf_(node2|node1|node3|node4|node5)$", n) or re.search(r"_tf$", n))


def rec_name(r):
    return r.get("name") or r.get("label") or ""


# ---------------------------------------------------------------------------- evaluated message order
# Order of evidence (, after the S6 finding that four node5 behavioural runs of TimeZero /
# VideoChat-R1 were evaluated text-first although their names carry no 'textfirst'):
#   1. the run's prov sidecar (ops/mon/prov/<run>.prov.json) when its text_first_basis is meta/cli;
#   2. the reader's own text_first field (black / gamma / tneg readers);
#   3. the run name ('textfirst', trailing '_tf_<machine>').
# A disagreement between 1 and 2 raises.
PROV_DIR = _SUPP + "/results/run_order"
_PROV = None


def prov_order(fname):
    """True/False from the prov sidecar of a run file, or None when absent / basis unknown."""
    global _PROV
    if _PROV is None:
        _PROV = {}
        for p in glob.glob(os.path.join(PROV_DIR, "*.prov.json")):
            try:
                d = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            if d.get("text_first") in (True, False) and d.get("text_first_basis") not in (None, "unknown"):
                _PROV[os.path.basename(p)[:-len(".prov.json")]] = (bool(d["text_first"]), d.get("text_first_basis"))
    return _PROV.get(fname)


def _input_files(r):
    ins = r.get("inputs")
    if isinstance(ins, dict):
        ins = [x for v in ins.values() for x in (v if isinstance(v, list) else [v])]
    return [i["file"] for i in (ins or []) if isinstance(i, dict) and i.get("file")]


def eval_order(r, files=None):
    """('t'|'v', basis) for a single-run record; `files` restricts the prov lookup (labelswap: STA run only)."""
    files = _input_files(r) if files is None else files
    pv = {prov_order(f) for f in files} - {None}
    pvals = {v for v, _ in pv}
    rd = r.get("text_first_eval")
    if len(pvals) > 1:
        raise SystemExit(f"canonical_v2: inputs of {rec_name(r)} disagree on message order: {pv}")
    if pvals:
        v = pvals.pop()
        if rd is not None and bool(rd) != v:
            raise SystemExit(f"canonical_v2: prov and reader disagree on message order for {rec_name(r)}")
        return ("t" if v else "v"), "prov:" + ",".join(sorted(b for _, b in pv))
    if rd is not None:
        return ("t" if rd else "v"), "reader"
    return ("t" if _eval_tf_from_name(rec_name(r)) else "v"), "name"


def rec_key(r, files=None):
    """model key of a single-run record (acc/black/gamma/tneg/labelswap/behav)."""
    fam, obj = r.get("family"), r.get("objective")
    if fam is None or obj is None:
        return None
    n = rec_name(r)
    ev, basis = eval_order(r, files)
    r["_order_basis"] = basis
    return M(fam, obj, r.get("alpha"), r.get("beta"), r.get("seed"),
             "t" if _train_tf(n) else "v", ev)


def side_key(sub, files=()):
    """model key of the arm/ref side of a pair_acc record (order from the side's run prov, else name)."""
    fam, obj = sub.get("family"), sub.get("objective")
    if fam is None or obj is None:
        return None
    n = sub.get("name") or ""
    pv = {prov_order(f) for f in files if n and n in f} - {None}
    if len({v for v, _ in pv}) == 1:
        ev = "t" if next(iter(pv))[0] else "v"
    else:
        ev = "t" if _eval_tf_from_name(n) else "v"
    return M(fam, obj, sub.get("alpha"), sub.get("beta"), sub.get("seed"),
             "t" if _train_tf(n) else "v", ev)


# ---------------------------------------------------------------------------- documented duplicates
# (kind, family, objective, eval, env, dataset) -> (substring the primary run's name must contain, reason)
PRIMARY_RUN = {
    ("acc", TZ, "base", "v", "A", "charades-sta"): (
        "tzbase_timer1_default_tf5meta",
        "tzbase read is the paper-wide base model read; TimeZero-Charades-7B_ is a second read of the same public weights (same as v1 selector)"),
    ("acc", TZ, "base", "t", "A", "charades-sta"): (
        "tzbase_timer1_default_textfirst",
        "timer1_default prompt; tz_official is a different prompt protocol"),
    ("acc", TZ, "grouped", "t", "A", "charades-sta"): (
        "timer1_default_textfirst_tf5meta_node2_full",
        "full 3720 questions; tz_official n200 is a 200-question subset with a different prompt protocol"),
    ("acc", TR, "base", "v", "B", "charades-sta"): (
        "tr1base_timer1_default_full",
        "timer1_default prompt; r1_full_h2s0/h2s1 are two samplings of the r1 prompt protocol (same as v1 selector)"),
    ("black", Q7, "base", "v", "B", "charades-sta"): (
        "node1", "node1 and node3 base model all-black readouts are bit-identical (same environment, two machines); same as v1 selector"),
    ("black", TR, "base", "t", "B", "charades-sta"): (
        "node3", "node3 and node5 readouts are bit-identical (ans .762 / R@0.5 .131); take node3 for consistency with the other B-stack base models"),
    ("acc", Q7, "base", "v", "B", "charades-timelens"): (
        "timer1_default", "same timer1_default prompt as the trained run (the other one uses the timelens prompt protocol)"),
}


# ---------------------------------------------------------------------------- products outside the registry
def external_inventory():
    """[(kind, dataset, key, env, file, extra)] for S11/S12/S13/S14 products."""
    inv = []
    # S11 query-swap r3 (tag -> key, env by machine suffix)
    QS = {
        "natbase_a025_vs_base_node1": M(Q7, "grouped", 0.25, 0.1, 0),
        "natbase_a05_vs_base_node4": M(Q7, "grouped", 0.5, 0.1, 0),
        "natbase_a05_tf5_vs_base_node2": M(Q7, "grouped", 0.5, 0.1, 0),   # 2026-09-24: canonical env A adapter (_tf5)
        "natbase_a05_b003_vs_base_node2": M(Q7, "grouped", 0.5, 0.03, 0),
        "natbase_a10_s0_vs_base_node2": M(Q7, "grouped", 1.0, 0.1, 0),
        "natbase_a10_s1_vs_base_node2": M(Q7, "grouped", 1.0, 0.1, 1),
        "natsft2g_a05_vs_base_node1": M(Q7, "two-stage", 0.5, 0.1, 0),
        "natsft2g_b003_vs_base_node1": M(Q7, "two-stage", None, 0.03, 0),
        "abl_single_vs_base_node1": M(Q7, "single-margin", 0.5, 0.1, 0),
        "natsft_b0_tf5_vs_base_node1": M(Q7, "sft", None, 0.0, 0),
        "nattz_a025_vs_tzbase_node2": M(TZ, "grouped", 0.25, 0.1, 0),
        "nattz_a05_vs_tzbase_node2": M(TZ, "grouped", 0.5, 0.1, 0),
        "nattz_a05_tf_vs_tzbase_tf_node5": M(TZ, "grouped", 0.5, 0.1, 0, eval="t"),
        "nattr1_a025_vs_tr1base_node4": M(TR, "grouped", 0.25, 0.1, 0),
        "natvc_a025_vs_vcbase_node4": M(VC, "grouped", 0.25, 0.1, 0),
        "natvc_a05_vs_vcbase_node4": M(VC, "grouped", 0.5, 0.1, 0),
        "natvc_a05_tf_vs_vcbase_tf_node5": M(VC, "grouped", 0.5, 0.1, 0, eval="t"),
    }
    for p in sorted(glob.glob(os.path.join(RESULTS, "judgment_20260922", "qswap3_*.json"))):
        tag = re.sub(r"^qswap3_|\.json$", "", os.path.basename(p))
        if tag not in QS:
            raise SystemExit(f"canonical_v2: unmapped qswap3 product {tag}")
        env = ENV_OF_MACHINE[tag.rsplit("_", 1)[1]]
        for i in json.load(open(p, encoding="utf-8")).get("inputs", []):
            po = prov_order(i.get("file", ""))
            if po is not None and po[0] != (QS[tag][6] == "t"):
                raise SystemExit(f"canonical_v2: qswap3 {tag} input {i.get('file')} order disagrees with the mapped key")
        inv.append(("qswap", "charades-sta", QS[tag], env, p, {"tag": tag}))
    # S12 newly-correct keys
    NC = {"iou_grpo_envB": (M(Q7, "iou-grpo"), "B"), "grouped_a05_envA": (M(Q7, "grouped", 0.5, 0.1, 0), "A"),
          "grouped_a10_envA": (M(Q7, "grouped", 1.0, 0.1, 0), "A"), "sft_envC_subset994": (M(Q7, "sft", None, 0.0, 0), "C")}
    pnc = os.path.join(RESULTS, "judgment_20260923", "newly_correct_v1.json")
    if os.path.exists(pnc):
        d = json.load(open(pnc, encoding="utf-8"))
        for k in d:
            if k in ("reader", "B", "inputs"):
                continue
            if k not in NC:
                raise SystemExit(f"canonical_v2: unmapped newly-correct key {k}")
            inv.append(("newly", "charades-sta", NC[k][0], NC[k][1], pnc, {"key": k}))
    # S13 message-order runs (node3, all four bases, both orders)
    OB = [(Q7, "qwenbase", "timer1_default"), (TZ, "tzbase", "timer1_default"),
          (TR, "tr1base", "timer1_default"), (VC, "vcbase", "vchat")]
    for fam, b, style in OB:
        for ev, sfx in (("v", ""), ("t", "_textfirst")):
            fc = os.path.join(RESULTS, "native_eval_20260918_node3", f"p4nat_{b}_{style}{sfx}_tf5meta_node3_full.json")
            fb = os.path.join(RESULTS, "native_eval_20260918_node3", f"p4nat_{b}_{style}{sfx}_black_node3_full.json")
            for f_ in (fc, fb):
                po = prov_order(os.path.basename(f_))
                if po is not None and po[0] != (ev == "t"):
                    raise SystemExit(f"canonical_v2: S13 run {os.path.basename(f_)} prov order disagrees with its name")
            if os.path.exists(fc) and os.path.exists(fb):
                inv.append(("order", "charades-sta", M(fam, "base", eval=ev), "B", fc, {"black": fb, "base": b}))
    # S14 Charades-CD runs
    CD = {"qwenbase": M(Q7, "base"), "qwensft": M(Q7, "sft", None, 0.0, 0), "grpo": M(Q7, "iou-grpo"),
          "qwena05": M(Q7, "grouped", 0.5, 0.1, 0), "tzbase": M(TZ, "base"), "tzbasetf": M(TZ, "base", eval="t"),
          "tzsfttf": M(TZ, "sft", None, 0.0, 0, train="t", eval="t"), "tza05": M(TZ, "grouped", 0.5, 0.1, 0),
          "tr1base": M(TR, "base"), "tr1a05": M(TR, "grouped", 0.5, 0.1, 0), "tr1a01": M(TR, "grouped", 0.1, 0.1, 0),
          "vcbase": M(VC, "base"), "vcbasetf": M(VC, "base", eval="t"),
          "vcsfttf": M(VC, "sft", None, 0.0, 0, train="t", eval="t"), "vca10": M(VC, "grouped", 1.0, 0.1, 0),
          "tlbase": M(TL, "base"), "tlsft": M(TL, "sft", None, 0.0, 0), "tla05": M(TL, "grouped", 0.5, 0.1, 0)}
    for p in sorted(glob.glob(os.path.join(RESULTS, "cd_20260923", "p4cd_*.json"))):
        m = re.match(r"p4cd_(.+)_(iid|ood)_([a-z0-9]+)\.json$", os.path.basename(p))
        if not m or m.group(1) not in CD:
            raise SystemExit(f"canonical_v2: unmapped CD product {os.path.basename(p)}")
        tfs = {int(bool(row.get("text_first"))) for row in json.load(open(p, encoding="utf-8"))}
        if tfs != {1 if CD[m.group(1)][6] == "t" else 0}:     # message order recorded per row must match the key
            raise SystemExit(f"canonical_v2: CD product {os.path.basename(p)} order {tfs} != key {key_str(CD[m.group(1)])}")
        inv.append(("cd", "charades-cd-" + m.group(2), CD[m.group(1)], ENV_OF_MACHINE[m.group(3)], p,
                    {"tag": m.group(1), "split": m.group(2)}))
    return inv


# ---------------------------------------------------------------------------- Canon
class Canon:
    def __init__(self, reg_path=REG_V2, overrides_path=OVERRIDES, externals=True):
        with open(reg_path, encoding="utf-8") as f:
            reg = json.load(f)
        self.reg_path, self.reg_sha16 = reg_path, sha16(reg_path)
        self.R = []
        for r in reg["records"]:
            r = dict(r)
            if r.get("kind") == "pair_acc":
                fs = _input_files(r)
                r["_key"] = side_key(r.get("arm") or {}, fs)
                r["_ref_key"] = side_key(r.get("ref") or {}, fs)
                r["_env"] = (r.get("arm") or {}).get("env")
                r["_ref_env"] = (r.get("ref") or {}).get("env")
            elif r.get("kind") == "oracle":
                lab = r.get("label") or ""
                r["_key"] = M(TL, "grouped", 0.5, 0.1, 0) if lab.startswith("tl8b_a05") else (
                    M(TL, "base") if lab.startswith("tl8b_base") else None)
                ins = json.dumps(r.get("inputs"))
                r["_env"] = "B" if ("node3" in ins or "node1" in ins) else ("A" if "node2" in ins else ("C" if "node4" in ins else None))
            elif r.get("kind") == "labelswap":
                # the record joins two runs: the STA run (defines the configuration) and a TL run
                fs = _input_files(r)
                sta = [f for f in fs if "chtl" not in f]
                tl = [f for f in fs if "chtl" in f]
                r["_key"] = rec_key(r, sta)
                r["_env"] = r.get("env")
                if len(tl) == 1:
                    po = prov_order(tl[0])
                    r["_tl_file"] = tl[0]
                    r["_tl_env"] = ENV_OF_MACHINE.get(re.search(r"_([a-z0-9]+)\.json$", tl[0]).group(1))
                    r["_tl_order"] = ("t" if po[0] else "v") if po else ("t" if _eval_tf_from_name(tl[0]) else "v")
                    r["_tl_order_basis"] = "prov:" + po[1] if po else "name"
            else:
                r["_key"] = rec_key(r)
                r["_env"] = r.get("env")
                if r.get("kind") == "gamma":
                    envs = {ENV_OF_MACHINE.get(i.get("machine")) for i in (r.get("inputs") or []) if isinstance(i, dict) and i.get("machine")}
                    if len(envs) > 1:
                        raise SystemExit(f"canonical_v2: gamma {rec_name(r)} pairs runs from different environments {envs}")
            if r.get("_key") and r["_key"][0] in EXCLUDE_FAMILIES:
                continue
            self.R.append(r)
        self.ext = external_inventory() if externals else []
        self.overrides = {}
        if overrides_path and os.path.exists(overrides_path):
            self.overrides = json.load(open(overrides_path, encoding="utf-8"))
        self.sources, self.pending, self._ctx = [], [], ("?", "?")
        self._decide()

    # -- coverage ---------------------------------------------------------------
    def _cells(self):
        """{key: {env: {(kind, dataset): [src descriptors]}}} over table-relevant products."""
        cov = {}

        def add(k, env, cell, desc):
            if k is None or env is None:
                return
            cov.setdefault(k, {}).setdefault(env, {}).setdefault(cell, []).append(desc)

        for r in self.R:
            kind = r.get("kind")
            if kind not in TABLE_KINDS:
                continue
            if kind == "gamma" and r.get("instrument") != "14Ln":
                continue
            if kind == "pair_acc":
                rk = r.get("_ref_key")
                if not (rk and rk[1] == "base" and rk[0] == r["_key"][0] and r["_ref_env"] == r["_env"]):
                    continue
            src = r.get("src") or {}
            add(r["_key"], r["_env"], (kind, r.get("dataset")), {"file": src.get("file"), "sha16": src.get("sha16")})
        for kind, ds, k, env, p, extra in self.ext:
            add(k, env, (kind, ds), {"file": os.path.basename(p), "sha16": sha16(p), **extra})
        return cov

    def _pairing(self, k, env, cells):
        """how many of the configuration's cells in env have their base counterpart in the same env."""
        if k[1] == "base":
            return 0
        bk = base_of(k)
        bcells = self.cov.get(bk, {}).get(env, {})
        n = 0
        for (kind, ds) in cells:
            if kind in ("gamma", "pair_acc", "behav", "qswap", "newly"):
                n += 1                              # the product carries its own same-stack base
            elif kind in ("acc", "black", "tneg", "labelswap", "cd"):
                n += int((kind, ds) in bcells)
        return n

    def _decide(self):
        self.cov = self._cells()
        self.decisions = {}
        for k, envs in sorted(self.cov.items(), key=lambda kv: key_str(kv[0])):
            cand = sorted(envs)
            covn = {e: len(envs[e]) for e in cand}
            pair = {e: self._pairing(k, e, envs[e]) for e in cand}
            tf5 = [e for e in cand if e in TF5]
            if not tf5:
                env, why = "C", "R2: neither A nor B has any table output for this config, can only take C"
            elif len(tf5) == 1:
                env = tf5[0]
                why = f"R2: in the tf5 stack only {env} has this config" + (" (C also has it, not taken per R2)" if "C" in cand else "")
            else:
                a, b = (covn["A"], pair["A"]), (covn["B"], pair["B"])
                if a[0] != b[0]:
                    env = "A" if a[0] > b[0] else "B"
                    why = f"R2: both A and B have it, covered cells A={a[0]} B={b[0]}, take {env}"
                elif a[1] != b[1]:
                    env = "A" if a[1] > b[1] else "B"
                    why = f"R2: A and B have equal coverage {a[0]} cells, base model pair completeness A={a[1]} B={b[1]}, take {env}"
                else:
                    env = "A"
                    why = f"R2: A and B tie on coverage and pairing ({a[0]}/{a[1]}), take A by fixed order (listed as pending in report)"
            ks = key_str(k)
            if ks in self.overrides:
                env, why = self.overrides[ks], f"OVERRIDE (CANONICAL_OVERRIDES_v2.json); rule decision was {env}: {why}"
            self.decisions[k] = {"env": env, "candidates": cand, "coverage": covn, "base_pairing": pair, "reason": why,
                                 "cells": {e: {f"{c[0]}@{c[1]}": v for c, v in sorted(envs[e].items())} for e in cand}}

    def env(self, k):
        d = self.decisions.get(k)
        return d["env"] if d else None

    # -- selection ----------------------------------------------------------------
    def ctx(self, table, row):
        self._ctx = (table, row)

    def _match(self, r, kind, k, env, dataset, extra):
        if r.get("kind") != kind or r.get("_key") != k:
            return False
        if env is not None and r.get("_env") != env:
            return False
        if dataset is not None and r.get("dataset") != dataset:
            return False
        for kk, vv in extra.items():
            if kk == "name_has":
                if vv not in rec_name(r):
                    return False
            elif kk == "name_not":
                if vv in rec_name(r):
                    return False
            elif r.get(kk) != vv:
                return False
        return True

    def _resolve(self, out, kind, k, env, dataset):
        if len(out) <= 1:
            return out
        pr = PRIMARY_RUN.get((kind, k[0], k[1], k[6], env, dataset))
        if pr:
            out2 = [r for r in out if pr[0] in rec_name(r)]
            if len(out2) == 1:
                out2[0]["_primary_reason"] = pr[1]
                return out2
        names = [rec_name(o) for o in out]
        raise SystemExit(f"canonical_v2: selector {kind} {key_str(k)} env={env} ds={dataset} matched {len(out)}: {names}")

    def note(self, rec, column, env_rule):
        src = rec.get("src") or {}
        self.sources.append({"table": self._ctx[0], "row": self._ctx[1], "column": column,
                             "kind": rec.get("kind"), "key": key_str(rec["_key"]) if rec.get("_key") else None,
                             "env": rec.get("_env"), "env_rule": env_rule, "name": rec_name(rec),
                             "dataset": rec.get("dataset"), "src_file": src.get("file"), "sha16": src.get("sha16"),
                             "reader": rec.get("reader"), "reader_sha16": rec.get("reader_sha16"),
                             "primary_reason": rec.get("_primary_reason")})

    def log_pending(self, column, kind, k, dataset, env, why=None, extra=None):
        extra = extra or {}
        other = sorted({r["_env"] for r in self.R if self._match(r, kind, k, None, dataset, extra) and r["_env"] != env})
        cls = "other_env" if other else "absent"
        self.pending.append({"table": self._ctx[0], "row": self._ctx[1], "column": column, "kind": kind,
                             "key": key_str(k) if k else None, "dataset": dataset, "canonical_env": env,
                             "class": cls, "other_envs": other, "why": why})

    def pick(self, kind, k, dataset="charades-sta", column=None, env=None, pending=True, **extra):
        """exactly-one record of `kind` for configuration k in its canonical env (or the env given,
        which callers use ONLY for the base side of a paired cell, rule R3)."""
        rule = "canonical" if env is None else "paired-stack"
        env = env or self.env(k)
        if env is None:
            if pending:
                self.log_pending(column, kind, k, dataset, None, "this config has no table output in any environment", extra)
            return None
        out = [r for r in self.R if self._match(r, kind, k, env, dataset, extra)]
        out = self._resolve(out, kind, k, env, dataset)
        if out:
            self.note(out[0], column, rule)
            return out[0]
        if pending:
            self.log_pending(column, kind, k, dataset, env, None, extra)
        return None

    def pick_pair(self, k, dataset="charades-sta", column="delta_R50"):
        """paired Delta R@0.5 of k against its own base, both inside k's canonical stack."""
        env = self.env(k)
        out = [r for r in self.R if r.get("kind") == "pair_acc" and r["_key"] == k and r["_env"] == env
               and r["_ref_env"] == env and r.get("dataset") == dataset and r.get("_ref_key") == base_of(k)]
        if len(out) > 1:
            raise SystemExit(f"canonical_v2: pair selector {key_str(k)} matched {len(out)}")
        if out:
            self.note(out[0], column, "canonical")
            return out[0]
        return None

    def delta(self, k, dataset="charades-sta", column="delta_R50"):
        """(pair_rec or None, arm_acc, base_acc_same_stack, text) -- paired reader if present, else the
        difference of two marginals INSIDE the arm's canonical stack (dagger), else \\pending."""
        pr = self.pick_pair(k, dataset, column)
        acc = self.pick("acc", k, dataset, column=column + ":arm", pending=False)
        benv = self.env(k)
        bacc = self.pick("acc", base_of(k), dataset, column=column + ":base", env=benv, pending=False)
        return pr, acc, bacc

    def rows_for(self, kind, dataset=None, **extra):
        """exhaustive tables: one entry per configuration holding `kind` (in any env):
        (key, record in the canonical env or None, envs where the kind exists)."""
        keys = []
        for r in self.R:
            if r.get("kind") == kind and r.get("_key") and (dataset is None or r.get("dataset") == dataset) \
                    and all((extra[kk] in rec_name(r)) if kk == "name_has" else r.get(kk) == vv for kk, vv in extra.items()):
                t = (r["_key"], r.get("dataset"))
                if t not in keys:
                    keys.append(t)
        out = []
        for k, ds in keys:
            env = self.env(k)
            recs = [r for r in self.R if self._match(r, kind, k, env, ds, extra)]
            recs = self._resolve(recs, kind, k, env, ds)
            envs = sorted({r["_env"] for r in self.R if self._match(r, kind, k, None, ds, extra)})
            out.append((k, ds, recs[0] if recs else None, envs))
        return out

    def ext_pick(self, kind, k, dataset=None, column=None, **extra):
        """S11-S14 products: the one in k's canonical env, else None (+pending)."""
        env = self.env(k)
        hits = [x for x in self.ext if x[0] == kind and x[2] == k and x[3] == env and (dataset is None or x[1] == dataset)
                and all(x[5].get(kk) == vv for kk, vv in extra.items())]
        if len(hits) > 1:
            raise SystemExit(f"canonical_v2: external {kind} {key_str(k)} env={env} matched {len(hits)}")
        if hits:
            x = hits[0]
            self.sources.append({"table": self._ctx[0], "row": self._ctx[1], "column": column, "kind": kind,
                                 "key": key_str(k), "env": env, "env_rule": "canonical", "name": os.path.basename(x[4]),
                                 "dataset": x[1], "src_file": os.path.basename(x[4]), "sha16": sha16(x[4]), "extra": x[5]})
            return x
        other = sorted({x[3] for x in self.ext if x[0] == kind and x[2] == k and (dataset is None or x[1] == dataset)
                        and all(x[5].get(kk) == vv for kk, vv in extra.items())} - {env})
        self.pending.append({"table": self._ctx[0], "row": self._ctx[1], "column": column, "kind": kind,
                             "key": key_str(k), "dataset": dataset, "canonical_env": env,
                             "class": "other_env" if other else "absent", "other_envs": other, "why": None})
        return None

    # -- manifest -------------------------------------------------------------------
    def manifest(self):
        base_pair_envs = {}
        for k in self.decisions:
            if k[1] != "base":
                bk = base_of(k)
                base_pair_envs.setdefault(bk, set()).add(self.decisions[k]["env"])
        out = {"generated_by": "canonical_v2.py", "registry": os.path.basename(self.reg_path),
               "registry_sha16": self.reg_sha16, "overrides": self.overrides,
               "rules": __doc__.split("Usage as a module")[0].strip(),
               "primary_run": {"|".join(map(str, k)): {"must_contain": v[0], "reason": v[1]} for k, v in PRIMARY_RUN.items()},
               "configurations": {}}
        for k, d in self.decisions.items():
            e = dict(d)
            if k[1] == "base":
                pe = sorted(base_pair_envs.get(k, set()))
                e["base_pairing_envs"] = pe
                e["rule4_multi_env"] = any(x != d["env"] for x in pe)
            out["configurations"][key_str(k)] = e
        return out


if __name__ == "__main__":
    C = Canon()
    man = C.manifest()
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    for ks, d in man["configurations"].items():
        print(f"{d['env']}  {ks:60s} cand={d['candidates']} cov={d['coverage']} pair={d['base_pairing']}"
              + ("  [R4 multi-env base]" if d.get("rule4_multi_env") else ""))
    print("->", MANIFEST)
