#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_numbers_v2.py -- paper number registry v2 (2026-09-23; copy of build_numbers_v1.py).

v2 changes against v1 (minimal, additive):
  * mDPO-style corrupted-video preference baseline (natmdpo_*) is parsed: family Qwen2.5-VL-7B,
    objective "mdpo" (v1: family/objective None, and its gamma file postdates the v1 run);
  * new collector collect_scale: results/judgment_20260921/scale/p4nat_natmdpo_*_clean_*.json
    (reader read_p4nat_v1, full-split clean accuracy; v1 never globbed scale/);
  * two-stage adapters trained on TimeZero / VideoChat-R1 (natsft2g_tz_*, natsft2g_vc_*) were
    parsed as Qwen2.5-VL-7B in v1 (token "natsft" matched first); now TimeZero-7B / VideoChat-R1-7B;
  * additive field text_first_eval = the message order the reader recorded for the run
    (black: text_first list, gamma: text_first, tneg: foreign_prov.text_first), None if absent.
    v1 field text_first is kept unchanged (it conflates trained-order and evaluated-order);
  * default --out is the rebuild copy, never the original paper directory.

v1 docstring follows.

build_numbers_v1.py -- paper number registry.

Reads ONLY registered reader outputs under results/judgment_20260921/ and
normalises every arm into one record with (family, objective, alpha, beta,
env, protocol, dataset) plus the numbers the paper uses.  Each number keeps
its source file + sha16 so tables/figures can print provenance footnotes.

Usage:  python build_numbers_v1.py [--root <project_root>] [--out paper_numbers_v1.json]

No numbers are typed here.  If a field is absent the record carries None
(absent, not zero).  Files with status != OK are skipped and listed.
"""
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..", "..")).replace("\\", "/")
import argparse, glob, hashlib, json, os, re, sys

# ----------------------------------------------------------------------------
# arm-name parsing
# ----------------------------------------------------------------------------
FAMILY_BY_TOKEN = [
    ("natsft2g_tz", "TimeZero-7B"), ("natsft2g_vc", "VideoChat-R1-7B"), ("natmdpo", "Qwen2.5-VL-7B"),  # v2
    ("nattimelens8b", "TimeLens-8B"), ("timelens8b", "TimeLens-8B"),
    ("nattr1", "Time-R1-7B"), ("tr1base", "Time-R1-7B"), ("Time-R1", "Time-R1-7B"),
    ("nattz", "TimeZero-7B"), ("tzbase", "TimeZero-7B"), ("TimeZero", "TimeZero-7B"),
    ("natvc", "VideoChat-R1-7B"), ("vcbase", "VideoChat-R1-7B"), ("VideoChat", "VideoChat-R1-7B"),
    ("natq3b", "Qwen2.5-VL-3B"), ("qwen3b", "Qwen2.5-VL-3B"), ("q3b", "Qwen2.5-VL-3B"),
    ("natbase", "Qwen2.5-VL-7B"), ("natsft", "Qwen2.5-VL-7B"), ("qwenbase", "Qwen2.5-VL-7B"),
    ("x0s0iou700", "Qwen2.5-VL-7B"), ("e1full", "Qwen2.5-VL-7B"), ("k16sel", "Qwen2.5-VL-7B"),
    ("base_timer1", "Qwen2.5-VL-7B"),
]
ENV_BY_MACHINE = {"node2": "A", "node1": "B", "node3": "B", "node5": "B", "node5": "B", "node4": "C"}
TF_BY_MACHINE = {"node2": "5.13", "node1": "5.12", "node3": "5.12", "node5": "5.12", "node5": "5.12", "node4": "4.51"}


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def parse_arm(name):
    """name = product basename without extension. Returns dict."""
    n = name
    fam = None
    for tok, f in FAMILY_BY_TOKEN:
        if tok in n:
            fam = f
            break
    is_base = any(t in n for t in ("qwenbase", "tzbase", "tr1base", "vcbase", "qwen3b", "base_timer1", "timelens8b_")) \
        and not any(t in n for t in ("natbase", "nattz", "nattr1", "natvc", "nattimelens8b", "natq3b"))
    if re.search(r"(^|_)timelens8b_(timelens|base)", n) and "nattimelens8b" not in n:
        is_base = True
    if re.match(r"^(Time-R1-7B|TimeZero-Charades-7B|VideoChat-R1_7B)_", n):   # public checkpoints read as bases
        is_base = True
    obj = None
    if is_base:
        obj = "base"
    elif "natmdpo" in n:          # v2
        obj = "mdpo"
    elif "sft2g" in n:
        obj = "two-stage"
    elif "natsft" in n or "_sft_b0" in n or "tzsft" in n or "sft_tf" in n:
        obj = "sft"
    elif "x0s0iou700" in n:
        obj = "iou-grpo"
    elif "abl_single" in n:
        obj = "single-margin"
    elif "abl_uniform_nohinge" in n:
        obj = "grouped-uniform-nohinge"
    elif "abl_nohinge" in n:
        obj = "grouped-nohinge"
    elif "abl_nodro" in n:
        obj = "grouped-nodro"
    elif "e1full" in n:
        obj = "grouped-full-budget"
    elif "l4800" in n:            # 2026-09-24: native-protocol run on 4,800 pairs (same trainer/settings, --limit 4800)
        obj = "grouped-4800"
    elif "k16sel" in n:
        obj = "grouped-k16"
    elif "sftshrink" in n:
        obj = "sft-shrink"
    elif re.search(r"nat(base|tz|tr1|vc|timelens8b|q3b)_a", n):
        obj = "grouped"
    m = re.search(r"_a(0\d{1,2}|10)(?=_)", n)   # _a01/_a025/_a05/_a10 ; never _node1
    alpha = None
    if m:
        s = m.group(1)
        alpha = float("0." + s[1:]) if s.startswith("0") else float(s[0] + "." + s[1:])
        if s in ("10",):
            alpha = 1.0
        if s in ("01",):
            alpha = 0.1
        if s in ("025",):
            alpha = 0.25
        if s in ("05",):
            alpha = 0.5
    beta = None
    if obj and obj.startswith("grouped") or obj in ("single-margin", "two-stage"):
        beta = 0.03 if "_b003" in n else 0.1
    if obj == "sft":
        beta = 0.0
    seed = None
    m = re.search(r"_s(\d)(?=_|$)", n)
    if m:
        seed = int(m.group(1))
    # message order: '_textfirst' / '_tf_node2' / '_tf_s0' (trained text-first) / trailing '_tf' -- NOT 'tf5'/'tf451'
    text_first = bool(re.search(r"textfirst|_tf(_|$)(?!5|451)", n))
    machine = norm_machine(n.replace("_", " "))
    dataset = "charades-sta"
    if "_qvh_" in n or "p6n_qvh" in n:
        dataset = "qvhighlights"
    if "chtl" in n or "_timelens_" in n and "nattimelens8b" not in n and "timelens8b_timelens" not in n:
        dataset = "charades-timelens" if "chtl" in n else dataset
    if "chtl" in n:
        dataset = "charades-timelens"
    protocol = "vchat" if "vchat" in n else ("timelens" if "_timelens" in n else "timer1")
    tf = None
    if "tf451" in n:
        tf = "4.51"
    elif machine:
        tf = TF_BY_MACHINE.get(machine)
    return dict(name=name, family=fam, objective=obj, alpha=alpha, beta=beta, seed=seed,
                text_first=text_first, machine=machine, env=ENV_BY_MACHINE.get(machine) if machine else None,
                transformers=tf, dataset=dataset, protocol=protocol, is_base=is_base)


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def norm_machine(m):
    """'node1=node3' / 'node2 (Env A)' / ['node2'] -> canonical token or None."""
    if isinstance(m, list):
        m = m[0] if m else None
    if not isinstance(m, str):
        return None
    m = m.strip().lower()
    for tok in ("node2", "node1", "node3", "node4", "node5", "node5"):
        if re.search(r"(^|[^a-z0-9])" + tok + r"([^a-z0-9]|$)", m):
            return tok
    return None


def norm_bool(v, default=None):
    """readers store text_first as bool / 'False' / '0' / ['1'] / 'video-first'."""
    if isinstance(v, list):
        v = v[0] if v else None
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "text-first", "textfirst", "yes"):
        return True
    if s in ("0", "false", "f", "video-first", "videofirst", "no", "none", "unknown", ""):
        return False if s not in ("none", "unknown", "") else default
    return default


LABEL_OVERRIDE = {  # gamma files whose inputs lack an arm role: label -> (family, objective, alpha, beta, text_first)
    "qvh_a05tf5_node2": ("Qwen2.5-VL-7B", "grouped", 0.5, 0.1, False),
    "qvh_a10_node2": ("Qwen2.5-VL-7B", "grouped", 1.0, 0.1, False),
    "qvh_sfttf5_node2": ("Qwen2.5-VL-7B", "sft", None, 0.0, False),
    "qvh_tz_a05tf_textfirst_node2": ("TimeZero-7B", "grouped", 0.5, 0.1, True),
    "qvh_a05_b003_node2": ("Qwen2.5-VL-7B", "grouped", 0.5, 0.03, False),   # 2026-09-24
    "qvh_tz_a025_node2": ("TimeZero-7B", "grouped", 0.25, 0.1, False),
    "qvh_tz_a05_node2": ("TimeZero-7B", "grouped", 0.5, 0.1, False),
}


def rec_src(p):
    return {"file": os.path.basename(p), "sha16": sha16(p)}


# ----------------------------------------------------------------------------
# collectors
# ----------------------------------------------------------------------------
def collect_gamma(root, out, skipped):
    for p in sorted(glob.glob(os.path.join(root, "gamma", "*.json"))):
        b = os.path.basename(p)
        if b.startswith(("GAMMA", "SEED", "UNPAIRED")):
            continue
        d = load(p)
        if d.get("status") not in (None, "OK"):
            skipped.append((b, d.get("status"), d.get("errors")))
            continue
        arm_file = None
        for i in d.get("inputs", []):
            if i.get("role") == "arm":
                arm_file = i.get("file")
        if not arm_file:
            m = re.match(r"gamma_(.+?)__vs__", b)
            arm_file = m.group(1) if m else b
        arm_name = re.sub(r"_nat_o0n200R3\.json$|\.json$", "", arm_file)
        arm_name = re.sub(r"^14Lq?n_", "", arm_name)
        instrument = d.get("instrument") or ("14Lqn" if "14Lqn" in b else "14Ln")
        po = d.get("per_op", {})
        rec = parse_arm(arm_name)
        lab = d.get("label") or ""
        if lab in LABEL_OVERRIDE:
            fam, obj, al, be, tfirst = LABEL_OVERRIDE[lab]
            rec.update(family=fam, objective=obj, alpha=al, beta=be, text_first=tfirst, is_base=False)
        rec.update({
            "kind": "gamma", "instrument": instrument, "label": d.get("label"),
            "machine": norm_machine(d.get("machine")) or norm_machine(d.get("machine_arm")) or rec["machine"],
            "text_first": norm_bool(d.get("text_first"), default=rec["text_first"]),
            "dataset": "charades-timelens" if "chtl" in b else ("qvhighlights" if "qvh" in b else rec["dataset"]),
            "n_H_int": d.get("n_H_int"), "clean_R50_base_audit": d.get("clean_R50_base"),
            "clean_R50_arm_audit": d.get("clean_R50_arm"),
            "gamma_mask": po.get("mask", {}).get("gamma"), "gamma_mask_ci": po.get("mask", {}).get("gamma_ci95"),
            "pos_frac_mask": po.get("mask", {}).get("pos_frac"),
            "gamma_swap": po.get("swap", {}).get("gamma"), "gamma_swap_ci": po.get("swap", {}).get("gamma_ci95"),
            "pos_frac_swap": po.get("swap", {}).get("pos_frac"),
            "dE_mask_arm": (po.get("mask", {}).get("arm_on_cohort") or {}).get("dE"),
            "dO_mask_arm": (po.get("mask", {}).get("arm_on_cohort") or {}).get("dO"),
            "dE_mask_base": (po.get("mask", {}).get("base_on_cohort") or {}).get("dE"),
            "dO_mask_base": (po.get("mask", {}).get("base_on_cohort") or {}).get("dO"),
            "dE_swap_arm": (po.get("swap", {}).get("arm_on_cohort") or {}).get("dE"),
            "dO_swap_arm": (po.get("swap", {}).get("arm_on_cohort") or {}).get("dO"),
            "dE_swap_base": (po.get("swap", {}).get("base_on_cohort") or {}).get("dE"),
            "dO_swap_base": (po.get("swap", {}).get("base_on_cohort") or {}).get("dO"),
            "reader": d.get("reader"), "reader_sha16": d.get("reader_sha16"),
            "src": rec_src(p), "inputs": d.get("inputs"),
            "text_first_eval": norm_bool(d.get("text_first"), default=None),   # v2
        })
        rec["env"] = ENV_BY_MACHINE.get(rec["machine"]) if rec["machine"] else None
        rec["transformers"] = d.get("transformers_version") or rec["transformers"]
        out.append(rec)


def collect_selective(root, out, skipped):
    for p in sorted(glob.glob(os.path.join(root, "selective", "*.json"))):
        b = os.path.basename(p)
        d = load(p)
        mode = d.get("mode")
        if mode == "single":
            name = re.sub(r"\.json$", "", d["inputs"][0]["file"])
            name = re.sub(r"^p4nat_|^p6n_", "", name)
            rec = parse_arm(name)
            mg, cd, ab = d.get("marginal", {}), d.get("conditional", {}), d.get("abstain", {})
            n = d.get("n")
            rec.update({
                "kind": "acc", "label": d.get("label"), "n": n, "n_videos": d.get("n_videos"),
                "R50": mg.get("R@0.5"), "R50_ci": mg.get("R@0.5_ci95"), "R30": mg.get("R@0.3"), "R70": mg.get("R@0.7"),
                "mIoU": mg.get("mIoU"), "parse_rate": mg.get("parse_rate"), "coverage": d.get("coverage"),
                "R50_answered": cd.get("R@0.5"), "R50_answered_ci": cd.get("R@0.5_ci95"),
                "n_abstained": ab.get("n_abstained"),
                "abstain_strict_rate": (sum(v.get("n", 0) for v in (ab.get("strict") or {}).values()) / n) if n and ab.get("strict") is not None else None,
                "abstain_wide_rate": (sum(v.get("n", 0) for v in (ab.get("wide") or {}).values()) / n) if n and ab.get("wide") is not None else None,
                "reader": d.get("reader"), "reader_sha16": d.get("reader_sha16"),
                "src": rec_src(p), "inputs": d.get("inputs"),
            })
            if "p6n_qvh" in b:
                rec["dataset"] = "qvhighlights"
            if "p6n_chtl" in b:
                rec["dataset"] = "charades-timelens"
            out.append(rec)
        elif mode == "pair":
            ref = re.sub(r"\.json$", "", d.get("ref_file", "")); arm = re.sub(r"\.json$", "", d.get("arm_file", ""))
            ra, aa = parse_arm(re.sub(r"^p4nat_|^p6n_", "", ref)), parse_arm(re.sub(r"^p4nat_|^p6n_", "", arm))
            pt, bt = d.get("point", {}), d.get("delta_marginal_boot", {})
            rec = dict(kind="pair_acc", ref=ra, arm=aa, label=d.get("label"), n=d.get("n"),
                       ref_R50=pt.get("ref_marginal_R@0.5"), arm_R50=pt.get("arm_marginal_R@0.5"),
                       delta_R50=pt.get("delta_marginal"), delta_R50_ci=bt.get("ci95"), delta_p=bt.get("p_two_sided"),
                       coverage_ref=d.get("coverage_ref"), coverage_arm=d.get("coverage_arm"),
                       arm_R50_answered=pt.get("arm_conditional_R@0.5"),
                       ref_on_arm_answered_R50=pt.get("ref_on_arm_answered_R@0.5"),
                       ref_on_arm_abstained_R50=pt.get("ref_on_arm_abstained_R@0.5"),
                       delta_conditional_paired=pt.get("delta_conditional_paired"),
                       delta_conditional_paired_ci=(d.get("delta_conditional_paired_boot") or {}).get("ci95"),
                       dataset=("qvhighlights" if "p6n_qvh" in b else ("charades-timelens" if "p6n_chtl" in b else "charades-sta")),
                       reader=d.get("reader"), reader_sha16=d.get("reader_sha16"), src=rec_src(p), inputs=d.get("inputs"))
            out.append(rec)
        else:
            skipped.append((b, "unknown mode", mode))


def collect_black(root, out, skipped):
    for p in sorted(glob.glob(os.path.join(root, "black", "read_black_*.json"))):
        b = os.path.basename(p)
        d = load(p)
        blk = None
        for i in d.get("inputs", []):
            if "black" in i.get("file", ""):
                blk = i["file"]
        name = re.sub(r"\.json$", "", blk or b)
        name = re.sub(r"^p4nat_|^p6n_", "", name)
        rec = parse_arm(name)
        ret = d.get("retention") or {}
        rec.update({
            "kind": "black", "label": d.get("label"), "n": d.get("n"),
            "dataset": "qvhighlights" if "qvh" in b else rec["dataset"],
            "black_answer_rate": d.get("parse_rate"), "black_R50": d.get("R@0.5"), "black_mIoU": d.get("mIoU"),
            "black_hard_refusal_wide": (d.get("wide") or {}).get("hard_refusal"),
            "black_any_refusal_wide": (d.get("wide") or {}).get("any_refusal"),
            "black_hard_refusal_strict": (d.get("strict") or {}).get("hard_refusal"),
            "clean_R50_for_retention": ret.get("clean_R@0.5"), "retention_R50": ret.get("retention_R@0.5"),
            "black_prior_component": (d.get("decomposition") or {}).get("prior_component"),
            "black_query_component": (d.get("decomposition") or {}).get("query_component"),
            "reader": d.get("reader"), "reader_sha16": d.get("reader_sha16"), "src": rec_src(p), "inputs": d.get("inputs"),
        })
        tfv = d.get("text_first")
        if isinstance(tfv, list) and tfv and tfv[0] in ("1", 1, True):
            rec["text_first"] = True
        rec["text_first_eval"] = norm_bool(tfv, default=None)   # v2 ('None' -> None)
        out.append(rec)


def collect_tneg(root, out, skipped):
    for p in sorted(glob.glob(os.path.join(root, "tneg", "*.json"))):
        d = load(p)
        clean_file = None
        for i in d.get("inputs", []):
            if i.get("role") == "clean":
                clean_file = i.get("file")
        name = re.sub(r"\.json$", "", clean_file or os.path.basename(p))
        name = re.sub(r"^p4nat_", "", name)
        rec = parse_arm(name)
        de = d.get("delta", {})
        rec.update({
            "kind": "tneg", "label": d.get("label"), "n_pairs": d.get("n_pairs"),
            "clean_abstain_strict": (d.get("clean") or {}).get("abstain_strict_rate"),
            "foreign_abstain_strict": (d.get("foreign") or {}).get("abstain_strict_rate"),
            "clean_abstain_wide": (d.get("clean") or {}).get("abstain_wide_rate"),
            "foreign_abstain_wide": (d.get("foreign") or {}).get("abstain_wide_rate"),
            "clean_parse": (d.get("clean") or {}).get("parse_rate"), "foreign_parse": (d.get("foreign") or {}).get("parse_rate"),
            "clean_R50": (d.get("clean") or {}).get("R@0.5"), "foreign_R50": (d.get("foreign") or {}).get("R@0.5"),
            "delta_abstain_strict": (de.get("abstain_strict") or {}).get("point"),
            "delta_abstain_strict_ci": (de.get("abstain_strict") or {}).get("ci95"),
            "delta_abstain_wide": (de.get("abstain_wide") or {}).get("point"),
            "delta_abstain_wide_ci": (de.get("abstain_wide") or {}).get("ci95"),
            "delta_R50": (de.get("R@0.5") or {}).get("point"), "delta_R50_ci": (de.get("R@0.5") or {}).get("ci95"),
            "text_first": bool((d.get("foreign_prov") or {}).get("text_first")) if (d.get("foreign_prov") or {}).get("text_first") is not None else rec["text_first"],
            "text_first_eval": norm_bool((d.get("foreign_prov") or {}).get("text_first"), default=None),   # v2
            "reader": d.get("reader"), "reader_sha16": d.get("reader_sha16"), "src": rec_src(p), "inputs": d.get("inputs"),
        })
        out.append(rec)


def collect_labelswap(root, out, skipped):
    for p in sorted(glob.glob(os.path.join(root, "labelswap", "read_tl_labelswap_*.json"))):
        b = os.path.basename(p)
        if b.endswith("SUMMARY.json"):
            continue
        d = load(p)
        sta_in = (d.get("inputs") or {}).get("sta") or []
        name = re.sub(r"\.json$", "", sta_in[0]["file"]) if sta_in else d.get("label")
        name = re.sub(r"^p4nat_", "", name)
        rec = parse_arm(name)
        full, common = d.get("full") or {}, d.get("common") or {}
        mat = common.get("matrix") or {}
        rec.update({
            "kind": "labelswap", "label": d.get("label"),
            "STA_R50_full": (full.get("sta_run_vs_STA_labels") or {}).get("R@0.5"),
            "STA_n_full": (full.get("sta_run_vs_STA_labels") or {}).get("n"),
            "TL_R50_full": (full.get("tl_run_vs_TL_labels") or {}).get("R@0.5"),
            "TL_n_full": (full.get("tl_run_vs_TL_labels") or {}).get("n"),
            "n_common": (d.get("match") or {}).get("n_common"),
            "pred_identical_frac": (common.get("pred_identity_sta_run_vs_tl_run") or {}).get("frac_identical"),
            "matrix_870": {k: (v.get("R@0.5") if isinstance(v, dict) else None) for k, v in mat.items()},
            "same_env": (d.get("machines") or {}).get("same_env"),
            "machine": (d.get("machines") or {}).get("sta"),
            "reader": d.get("reader"), "reader_sha16": d.get("reader_sha16"), "src": rec_src(p), "inputs": d.get("inputs"),
        })
        rec["env"] = ENV_BY_MACHINE.get(rec["machine"]) if rec["machine"] else None
        out.append(rec)


def collect_oracle(root, out, skipped):
    for p in sorted(glob.glob(os.path.join(root, "oracle", "*.json"))):
        d = load(p)
        rec = dict(kind="oracle", label=d.get("label"), mode=(d.get("params") or {}).get("mode"),
                   window_len_s=(d.get("params") or {}).get("window_len_s"),
                   original_R50=((d.get("results") or {}).get("original") or {}).get("R@0.5"),
                   oracle_R50=((d.get("results") or {}).get("oracle") or {}).get("R@0.5"),
                   original_mIoU=((d.get("results") or {}).get("original") or {}).get("mIoU"),
                   oracle_mIoU=((d.get("results") or {}).get("oracle") or {}).get("mIoU"),
                   reader=d.get("reader"), reader_sha16=d.get("reader_sha16"), src=rec_src(p), inputs=d.get("inputs"))
        out.append(rec)


def collect_p5n(root, out, skipped):
    for p in sorted(glob.glob(os.path.join(root, "p5n_v2", "read_p5n_v2_*.json"))):
        b = os.path.basename(p)
        if b.endswith("SUMMARY.json"):
            continue
        d = load(p)
        arm_file = ((d.get("inputs") or {}).get("arm") or {}).get("file", "")
        name = re.sub(r"\.json$", "", arm_file)
        name = re.sub(r"^p5n6?_", "", name)
        rec = parse_arm(name)
        aq = d.get("all_questions") or {}
        paired = aq.get("paired") or {}
        rec.update({
            "kind": "behav", "label": d.get("label"), "n": aq.get("n"),
            "did_swap": ((paired.get("swap") or {}).get("DiD_contrast_arm_minus_base") or {}).get("est"),
            "did_swap_ci": ((paired.get("swap") or {}).get("DiD_contrast_arm_minus_base") or {}).get("ci95"),
            "did_swap_p": ((paired.get("swap") or {}).get("DiD_contrast_arm_minus_base") or {}).get("p"),
            "did_mask": ((paired.get("mask") or {}).get("DiD_contrast_arm_minus_base") or {}).get("est"),
            "did_mask_ci": ((paired.get("mask") or {}).get("DiD_contrast_arm_minus_base") or {}).get("ci95"),
            #: evidence share (normalised contrast), trained minus base -- the estimand of T1 and Fig. 5c
            "nshare_mask": ((paired.get("mask") or {}).get("normalised_contrast_diff") or {}).get("est"),
            "nshare_mask_ci": ((paired.get("mask") or {}).get("normalised_contrast_diff") or {}).get("ci95"),
            "nshare_swap": ((paired.get("swap") or {}).get("normalised_contrast_diff") or {}).get("est"),
            "nshare_swap_ci": ((paired.get("swap") or {}).get("normalised_contrast_diff") or {}).get("ci95"),
            "base_conditions": ((aq.get("base") or {}).get("conditions")),
            "arm_conditions": ((aq.get("arm") or {}).get("conditions")),
            "reader": d.get("reader"), "reader_sha16": d.get("reader_sha16"), "src": rec_src(p), "inputs": d.get("inputs"),
        })
        out.append(rec)


def collect_scale(root, out, skipped):
    """v2: full-split clean accuracy of the mDPO-style baseline (reader read_p4nat_v1; no abstention fields)."""
    for p in sorted(glob.glob(os.path.join(root, "scale", "p4nat_natmdpo_*_clean_*.json"))):
        d = load(p)
        ins = d.get("inputs") or []
        name = re.sub(r"\.json$", "", ins[0]["file"]) if ins else os.path.basename(p)
        name = re.sub(r"^p4nat_", "", name)
        # a selective read of the same input carries these fields plus abstention; it supersedes this record
        sel = os.path.join(root, "selective", "read_selective_p4nat_%s.json" % name)
        if ins and os.path.exists(sel) and [i.get("sha16") for i in load(sel).get("inputs") or []] == [ins[0].get("sha16")]:
            continue
        rec = parse_arm(name)
        rec.update({
            "kind": "acc", "label": d.get("label"), "n": d.get("n"), "n_videos": d.get("n_videos"),
            "R50": d.get("R@0.5"), "R50_ci": d.get("R@0.5_ci95"), "R30": d.get("R@0.3"), "R70": d.get("R@0.7"),
            "mIoU": d.get("mIoU"), "parse_rate": d.get("parse_rate"), "coverage": None,
            "R50_answered": None, "R50_answered_ci": None, "n_abstained": None,
            "abstain_strict_rate": None, "abstain_wide_rate": None,
            "reader": d.get("reader"), "reader_sha16": d.get("reader_sha16"),
            "src": rec_src(p), "inputs": ins,
        })
        out.append(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=_SUPP)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    jroot = os.path.join(a.root, "results", "judgment_20260921")
    out, skipped = [], []
    for fn in (collect_gamma, collect_selective, collect_black, collect_tneg, collect_labelswap, collect_oracle, collect_p5n,
               collect_scale):
        try:
            fn(jroot, out, skipped)
        except Exception as e:  # keep going, report
            skipped.append((fn.__name__, "EXC", repr(e)))
    reg = {"generated_by": "build_numbers_v2.py", "n_records": len(out),
           "kinds": {k: sum(1 for r in out if r.get("kind") == k) for k in sorted({r.get("kind") for r in out})},
           "skipped": skipped, "records": out}
    dst = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables", "paper_numbers_v2.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=1)
    print(f"records={len(out)} kinds={reg['kinds']} skipped={len(skipped)} -> {dst}")
    for s in skipped:
        print("  SKIP", s)


if __name__ == "__main__":
    main()
