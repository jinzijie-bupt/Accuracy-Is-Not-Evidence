#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qn_evidnec_halluc_v2.py (revision, zero-GPU, 2026-11-06)

Answers three questions for R2-W2 ("raises dependence but is less accurate than SFT") and R3-W1/Q1 ("what is the
practical benefit of raising Gamma"), from products already on disk.  Reuses the reading / pairing / stratum /
bootstrap code of qc_qj_p5n_v1.py by importing it (the v1 file is not modified; its sha16 is recorded).

  QN1  accuracy on evidence-necessary questions.  Evidence-necessary = NOT prior-solvable, with the two prior
       strata of v1:  const (constant (0,10)s window IoU>=0.5 with the annotation, no model) and blind (>=1 of six
       base products hits IoU>=0.5 on the all-black video).  Layers: EN_const, EN_blind, EN_both (= neither),
       and for contrast PS_const, PS_blind, PS_any (= const or blind).  Clean-video R@0.5 / mIoU of every model,
       paired differences (arm - base, ours - SFT, ours - IoU-GRPO ...) with video-cluster bootstrap, and the
       interaction  Delta(EN) - Delta(PS)  from one joint bootstrap on all questions.
       Sources: 12p5n (tm/node1/node2/node4, 1,000 questions) + a mask-only 12p5n replication on node4 (older instrument)
       + a full-test-set (3,720 questions) replication of the const layer from 12p4n clean products.
  QN2  behaviour when the evidence is missing.  Under each condition every question falls in exactly one of
       {no interval, interval with IoU<t (confident wrong), interval with IoU>=t (correct)}, t in {0.3, 0.5}.
       Rates per model and condition; arm - base under E; E - O per model; DiD (arm-base) x (E-O);
       ours - SFT / ours - IoU-GRPO directly.  On all questions and on EN_both.
  QN3  accuracy-matched comparison.  Search result for 12p5n products of early SFT checkpoints (none exist) and,
       as the closest available substitute, 12p4n clean accuracy (qid 0..999) + 14Ln Gamma of early SFT
       checkpoints (node4, seeds 1/2, every 50 steps) vs our alpha=0.5 arms, all paired to the same-machine base.

Video-cluster bootstrap B=2000 seed 0 (same RNG stream as v1: groups in first-appearance order, one
rng.integers(0,G,G) per draw); two-sided p = 2*min(P(d<=0), P(d>=0)).  Every input file is recorded with sha16.
Outputs: qn_evidnec_halluc_v2.json + qn1_evidnec_acc.tsv + qn2_halluc_stacked.tsv + qn3_accmatched.tsv
"""
import os, sys, json, hashlib, importlib.util, datetime, glob
import numpy as np
LOCAL_ROOT = os.environ.get("LOCAL_ROOT", ".")  # analysis directory holding the pulled result files (results/, rebuttal/)

HERE = os.path.dirname(os.path.abspath(__file__))
R = (LOCAL_ROOT + "/results/")
V1_PATH = os.path.join(HERE, "qc_qj_p5n_v1.py")
P4_PATH = (LOCAL_ROOT + "/analysis/read_p4nat_v1.py")
B, SEED = 2000, 0
THRS = (0.3, 0.5)


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


V1 = _imp("qc_qj_p5n_v1", V1_PATH)      # reading, pairing, strata, refusal taxonomy, bootstrap summary
P4 = _imp("read_p4nat_v1", P4_PATH)      # frozen p4nat reader: iou_1d, load_rows

# extra 12p5n products: the same three Qwen models on node4, older instrument (ops none/mask_evid/mask_rand only)
V1.P.update({
    "node4_qwenbase": (R + "native_audit_20260917/p5n_qwenbase_timer1_default_node4_n1000_o0.json", "91d8f311c59efef5"),
    "node4_a05": (R + "native_p5n_20260920/p5n_natbase_a05_g1_s0_timer1_default_node4_n1000_o0.json", "7a0bfe02714b6e4f"),
    "node4_sft": (R + "native_p5n_20260920/p5n_natsft_b0_s0_timer1_default_node4_n1000_o0.json", "5686786d6b291d76"),
})

# ------------------------------------------------------------------ model sets
# Qwen2.5-VL-7B family, 12p5n v1_tf.  tm and node1 base products are byte-identical (same sha16), so one base row.
QW = {"base": "node5_qwenbase", "ours_a05": "node5_natbase_a05", "SFT": "node5_natsft", "mDPO": "node5_natmdpo",
      "IoU-GRPO": "node1_iougrpo", "abl_single": "node1_abl_single"}
QW_MACH = {"base": "tm=node1", "ours_a05": "tm", "SFT": "tm", "mDPO": "tm", "IoU-GRPO": "node1", "abl_single": "node1"}
QW_CMP = [("ours_a05", "base"), ("SFT", "base"), ("mDPO", "base"), ("IoU-GRPO", "base"), ("abl_single", "base"),
          ("ours_a05", "SFT"), ("ours_a05", "IoU-GRPO"), ("ours_a05", "mDPO")]
# other backbones: (label, base key, arm key) -- placebo-controlled arms only; no SFT/IoU-GRPO p5n products exist
OTHER = [(pid, bk, ak) for pid, mach, bk, ak, kind, t19 in V1.PAIRS if not pid.startswith("Qwen7B")]
# node4 replication (mask only)
SX = {"base": "node4_qwenbase", "ours_a05": "node4_a05", "SFT": "node4_sft"}
SX_CMP = [("ours_a05", "base"), ("SFT", "base"), ("ours_a05", "SFT")]

LAYERS = ("all", "EN_both", "EN_const", "EN_blind", "PS_any", "PS_const", "PS_blind")
INTER = (("EN_const", "PS_const"), ("EN_blind", "PS_blind"), ("EN_both", "PS_any"))


# ------------------------------------------------------------------ fast cluster bootstrap (weights matrix)
class Boot:
    """B x n weight matrix: W[b, i] = number of times question i's video was drawn in draw b.  Identical draws to
    V1.draws_idx (same group order, same rng calls), so means equal the v1 concatenation approach."""

    def __init__(self, vids, B=B, seed=SEED):
        groups = V1.groups_of(vids)
        cl = np.empty(len(vids), int)
        for g, ix in enumerate(groups):
            cl[ix] = g
        rng = np.random.default_rng(seed); G = len(groups)
        C = np.zeros((B, G))
        for b in range(B):
            C[b] = np.bincount(rng.integers(0, G, G), minlength=G)
        self.W = C[:, cl]; self.n = len(vids); self.G = G

    def mean(self, x, mask=None):
        x = np.asarray(x, float)
        if mask is None:
            return float(x.mean()), (self.W @ x) / self.W.sum(1)
        m = np.asarray(mask, bool)
        if not m.any():
            return None, None
        Wm = self.W[:, m]; den = Wm.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = (Wm @ x[m]) / den
        return float(x[m].mean()), d


def S(pt, d, scale=1.0):
    if pt is None:
        return {"est": None, "ci95": [None, None], "p": None}
    return V1.summ(scale * pt, list(scale * np.asarray(d)))


# ------------------------------------------------------------------ helpers
def check_annotation(data, keys, ref):
    for k in keys:
        for q in range(1000):
            a, b = data[k][q], ref[q]
            if (a["video"], a["gsA"], a["geA"]) != (b["video"], b["gsA"], b["geA"]):
                raise SystemExit("annotation mismatch %s qid %d" % (k, q))


def check_hit_consistency(rows, ops):
    """hit must equal parsed & iou>=0.5 and unparsed rows must carry iou 0 -- else the 3-way split is not exact"""
    bad = 0
    for q in range(1000):
        r = rows[q]
        for op in ops:
            p, i, h = r["%s_parsed" % op], float(r["%s_iou" % op]), r["%s_hit" % op]
            if (not p and (i != 0.0 or h)) or (p and int(i >= 0.5) != int(h)):
                bad += 1
    return bad


def layers_from(const, blind):
    c, b = np.asarray(const, bool), np.asarray(blind, bool)
    return {"all": np.ones_like(c), "EN_const": ~c, "EN_blind": ~b, "EN_both": ~c & ~b,
            "PS_const": c, "PS_blind": b, "PS_any": c | b}


# ------------------------------------------------------------------ QN1
def qn1(models, data, cmps, qs, L, vids, ops_needed=("none",)):
    """clean R@0.5 / mIoU per model per layer + paired differences + EN-PS interaction (joint bootstrap)."""
    X = {m: {"hit": V1.vec(data[k], qs, "none", "hit"), "iou": V1.vec(data[k], qs, "none", "iou"),
             "parsed": V1.vec(data[k], qs, "none", "parsed")} for m, k in models.items()}
    Bt = {lay: Boot([v for v, f in zip(vids, L[lay]) if f]) for lay in LAYERS}
    Bfull = Bt["all"]
    out = {"n": {lay: int(L[lay].sum()) for lay in LAYERS},
           "n_videos": {lay: Bt[lay].G for lay in LAYERS}, "per_model": {}, "paired": {}, "interaction_EN_minus_PS": {}}
    for m in models:
        out["per_model"][m] = {}
        for lay in LAYERS:
            msk = L[lay]
            r = {}
            for met, key, sc in (("R@0.5", "hit", 1), ("mIoU", "iou", 1), ("parse_rate", "parsed", 1)):
                x = X[m][key][msk]
                pt, d = Bt[lay].mean(x)
                r[met] = S(pt, d)
            out["per_model"][m][lay] = r
    for a, b in cmps:
        nm = "%s - %s" % (a, b)
        out["paired"][nm] = {}
        for lay in LAYERS:
            msk = L[lay]; r = {}
            for met, key in (("R@0.5_pts", "hit"), ("mIoU", "iou")):
                sc = 100.0 if met.endswith("pts") else 1.0
                pt, d = Bt[lay].mean(X[a][key][msk] - X[b][key][msk])
                r[met] = S(pt, d, sc)
            out["paired"][nm][lay] = r
        # interaction: Delta(EN) - Delta(PS), one joint bootstrap over all questions
        out["interaction_EN_minus_PS"][nm] = {}
        for en, ps in INTER:
            r = {}
            for met, key in (("R@0.5_pts", "hit"), ("mIoU", "iou")):
                sc = 100.0 if met.endswith("pts") else 1.0
                dx = X[a][key] - X[b][key]
                pe, de = Bfull.mean(dx, L[en]); pp, dp = Bfull.mean(dx, L[ps])
                r[met] = S(pe - pp, de - dp, sc)
            out["interaction_EN_minus_PS"][nm]["%s - %s" % (en, ps)] = r
        # where does the clean gain come from: share of the all-question gain contributed by EN_both questions
        dx = X[a]["hit"] - X[b]["hit"]
        tot = dx.sum()
        out["paired"][nm]["gain_share_from_EN_both"] = (None if abs(tot) < 1e-9 else float(dx[L["EN_both"]].sum() / tot))
        out["paired"][nm]["question_share_EN_both"] = float(L["EN_both"].mean())
    return out


# ------------------------------------------------------------------ QN2
def qn2(models, data, cmps, qs, L, vids, ops, layers=("all", "EN_both")):
    """3-way split {no interval, confident wrong (parsed & IoU<t), correct (IoU>=t)} per model x condition x t."""
    out = {"definition": {"no_interval": "no parseable interval (refusal or other unparsed output)",
                          "conf_wrong@t": "parseable interval with IoU < t",
                          "correct@t": "parseable interval with IoU >= t",
                          "note": "the three add to 1 for each model, condition and t"},
           "layers": {}}
    V = {}
    for m, k in models.items():
        for op in ops:
            par = V1.vec(data[k], qs, op, "parsed"); iou = V1.vec(data[k], qs, op, "iou")
            s, w = V1.refusal_flags(data[k], qs, op)
            V[(m, op, "no_interval")] = 1 - par
            V[(m, op, "explicit_refusal_wide")] = w
            for t in THRS:
                V[(m, op, "conf_wrong@%.1f" % t)] = par * (iou < t)
                V[(m, op, "correct@%.1f" % t)] = par * (iou >= t)
    cats = ["no_interval", "explicit_refusal_wide"] + ["%s@%.1f" % (c, t) for t in THRS for c in ("conf_wrong", "correct")]
    axes = [ax for ax in ("mask", "swap") if "%s_evid" % ax in ops]
    for lay in layers:
        msk = L[lay]; Bt = Boot([v for v, f in zip(vids, msk) if f])
        blk = {"n": int(msk.sum()), "n_videos": Bt.G, "rates": {}, "contrasts": {}}
        for m in models:
            blk["rates"][m] = {}
            for op in ops:
                blk["rates"][m][op] = {c: S(*Bt.mean(V[(m, op, c)][msk])) for c in cats}
                for t in THRS:   # exactness guard
                    tot = V[(m, op, "no_interval")][msk] + V[(m, op, "conf_wrong@%.1f" % t)][msk] + V[(m, op, "correct@%.1f" % t)][msk]
                    assert np.allclose(tot, 1.0), (m, op, t)
            # E - O per model (pts)
            blk["rates"][m]["E_minus_O_pts"] = {}
            for ax in axes:
                e, o = "%s_evid" % ax, "%s_rand" % ax
                blk["rates"][m]["E_minus_O_pts"][ax] = {c: S(*Bt.mean(V[(m, e, c)][msk] - V[(m, o, c)][msk]), 100.0) for c in cats}
        for a, b in cmps:
            nm = "%s - %s" % (a, b); r = {}
            for ax in axes:
                e, o = "%s_evid" % ax, "%s_rand" % ax
                r[ax] = {}
                for c in cats:
                    dE = V[(a, e, c)][msk] - V[(b, e, c)][msk]
                    dO = V[(a, o, c)][msk] - V[(b, o, c)][msk]
                    r[ax][c] = {"diff_under_E_pts": S(*Bt.mean(dE), 100.0),
                                "diff_under_O_pts": S(*Bt.mean(dO), 100.0),
                                "DiD_E_minus_O_pts": S(*Bt.mean(dE - dO), 100.0)}
            # clean reference
            r["clean"] = {c: {"diff_pts": S(*Bt.mean(V[(a, "none", c)][msk] - V[(b, "none", c)][msk]), 100.0)} for c in cats}
            blk["contrasts"][nm] = r
        out["layers"][lay] = blk
    return out


# ------------------------------------------------------------------ QN1 full-test-set replication (const layer, 12p4n)
P4FULL = {
    "node4": {"base": R + "native_eval_20260918_node4/p4nat_qwenbase_timer1_default_tf451_node4_full.json",
            "ours_a05_s0": R + "native_full_20260918/p4nat_natbase_a05_g1_s0_timer1_default_tf451_node4_full.json",
            "ours_a05_s1": R + "native_20260925_node4/p4nat_natbase_a05_g1_s1_timer1_default_tf451_node4_full.json",
            "SFT": R + "native_full_20260918/p4nat_natsft_b0_s0_timer1_default_tf451_node4_full.json"},
    "node1": {"base": R + "native_full_20260918/p4nat_base_timer1_default_tf5meta_node1_full.json",
             "SFT": R + "native_audit_20260917/p4nat_natsft_b0_s0_tf5_timer1_default_tf5meta_node1_full.json",
             "IoU-GRPO": [R + "native_eval_20260917/p4nat_x0s0iou700_timer1_default_tf5meta_node1_n1000_o0.json",
                          R + "native_eval_20260917/p4nat_x0s0iou700_timer1_default_tf5meta_node1_n2720_o1000.json"],
             "abl_single": R + "native_full_20260918/p4nat_natbase_a05_abl_single_s0_timer1_default_tf5meta_node1_full.json"},
}
P4FULL_CMP = {"node4": [("ours_a05_s0", "base"), ("ours_a05_s1", "base"), ("SFT", "base"), ("ours_a05_s0", "SFT"), ("ours_a05_s1", "SFT")],
              "node1": [("SFT", "base"), ("IoU-GRPO", "base"), ("abl_single", "base")]}


def p4_load(files):
    files = files if isinstance(files, list) else [files]
    rows = P4.load_rows(files)
    for r in rows:
        if r.get("video_op") not in (None, "none"):
            raise SystemExit("non-clean row in %s" % files)
    return rows, [{"file": f, "sha16": sha16(f)} for f in files]


def p4_vecs(rows):
    iou = np.array([P4.iou_1d(r.get("pred"), (r["gsA"], r["geA"])) for r in rows])
    return {"hit": (iou >= 0.5).astype(float), "iou": iou, "parsed": np.array([1.0 if r.get("pred") else 0.0 for r in rows])}


def qn1_full_const():
    out = {"definition": "12p4n clean products, all 3,720 Charades-STA test questions; only the const stratum is "
                         "defined here (blind needs the all-black 12p5n run, which exists for qid 0..999 only)",
           "machines": {}, "inputs": {}}
    for mach, files in P4FULL.items():
        rows, X, ann = {}, {}, None
        for m, f in files.items():
            rows[m], out["inputs"]["%s/%s" % (mach, m)] = p4_load(f)
            a = [(r["qid"], r["video"], r["gsA"], r["geA"]) for r in rows[m]]
            if ann is None:
                ann = a
            elif a != ann:
                raise SystemExit("p4nat annotation mismatch %s %s" % (mach, m))
            X[m] = p4_vecs(rows[m])
        ref = rows["base"]; n = len(ref)
        const = np.array([1 if P4.iou_1d([0.0, 10.0], (r["gsA"], r["geA"])) >= 0.5 else 0 for r in ref], bool)
        vids = [r["video"] for r in ref]
        L = {"all": np.ones(n, bool), "EN_const": ~const, "PS_const": const}
        Bt = {k: Boot([v for v, f in zip(vids, m) if f]) for k, m in L.items()}
        blk = {"n": {k: int(m.sum()) for k, m in L.items()}, "per_model": {}, "paired": {}}
        for m in files:
            blk["per_model"][m] = {k: {"R@0.5": S(*Bt[k].mean(X[m]["hit"][L[k]])), "mIoU": S(*Bt[k].mean(X[m]["iou"][L[k]])),
                                       "parse_rate": float(X[m]["parsed"][L[k]].mean())} for k in L}
        for a, b in P4FULL_CMP[mach]:
            nm = "%s - %s" % (a, b); r = {}
            for k in L:
                r[k] = {"R@0.5_pts": S(*Bt[k].mean(X[a]["hit"][L[k]] - X[b]["hit"][L[k]]), 100.0),
                        "mIoU": S(*Bt[k].mean(X[a]["iou"][L[k]] - X[b]["iou"][L[k]]))}
            dx = X[a]["hit"] - X[b]["hit"]
            pe, de = Bt["all"].mean(dx, L["EN_const"]); pp, dp = Bt["all"].mean(dx, L["PS_const"])
            r["interaction_EN_const_minus_PS_const_R@0.5_pts"] = S(pe - pp, de - dp, 100.0)
            blk["paired"][nm] = r
        out["machines"][mach] = blk
    return out


# ------------------------------------------------------------------ QN3 accuracy-matched
GJ = R + "judgment_20260921/gamma/"
GN = R + "judgment_20260925_native/"
DYN_node4 = R + "dyn_20260924_node4/"
DYN_node2 = R + "dyn_20260924_node2/"
node2_NE = R + "pull/node2/native_eval_20260918/"
node1_NE = R + "pull/node1/native_eval_20260918/"


def qn3_rows():
    """(machine, label, kind, step, p4nat file(s) or None, gamma json or None)"""
    rows = [("node4", "base", "base", None, P4FULL["node4"]["base"], None),
            ("node4", "ours a=0.5 seed0 (Table 3 row 1)", "ours", 300, P4FULL["node4"]["ours_a05_s0"],
             GJ + "gamma_14Ln_natbase_a05_g1_s0_timer1_default_node4__vs__14Ln_base_timer1_node4.json"),
            ("node4", "ours a=0.5 seed1", "ours", 300, P4FULL["node4"]["ours_a05_s1"],
             GN + "gamma_14Ln_natbase_a05_g1_s1_timer1_default_node4_nat_o0n200R3__vs__14Ln_base_timer1_node4_nat_o0n200R3.json"),
            ("node4", "SFT seed0 final", "SFT", 300, P4FULL["node4"]["SFT"],
             GJ + "gamma_14Ln_natsft_b0_s0_timer1_default_node4__vs__14Ln_base_timer1_node4.json")]
    for s in (1, 2):
        for st in (50, 100, 150, 200, 250, 300):
            p4 = DYN_node4 + "p4nat_dyn_sft_s%d_step%d_node4_n1000.json" % (s, st)
            g = glob.glob(GN + "dyn/gamma_14Ln_dyn_sft_s%d_step%d_node4_nat_o0n200R3*.json" % (s, st))
            rows.append(("node4", "SFT seed%d step%d" % (s, st), "SFT", st, p4 if os.path.exists(p4) else None, g[0] if g else None))
    rows.append(("node2", "base", "base", None, node2_NE + "p4nat_qwenbase_timer1_default_tf5meta_node2_full.json", None))
    for st in (50, 100, 150, 200, 250, 300):
        p4 = DYN_node2 + "p4nat_dyn_pc_a05_s1_step%d_node2_n1000.json" % st
        g = GN + "dyn/gamma_14Ln_dyn_pc_a05_s1_step%d_node2_nat_o0n200R3.json" % st
        rows.append(("node2", "ours a=0.5 seed1 step%d" % st, "ours", st, p4 if os.path.exists(p4) else None, g if os.path.exists(g) else None))
    rows += [("node1", "base", "base", None, P4FULL["node1"]["base"], None),
             ("node1", "SFT seed0 final (tf5)", "SFT", 300, P4FULL["node1"]["SFT"],
              GJ + "gamma_14Ln_natsft_b0_s0_tf5_timer1_default_node1__vs__14Ln_base_timer1_default_node1.json"),
             ("node1", "two-stage SFT->ours a=0.5", "two-stage", None,
              R + "native_eval_20260918_node1/p4nat_natsft2g_a05_g1_s0_timer1_default_tf5meta_node1_full.json",
              GJ + "gamma_14Ln_natsft2g_a05_g1_s0_timer1_default_node1__vs__14Ln_base_timer1_default_node1.json"),
             ("node1", "two-stage SFT->ours a=1.0,b=0.03", "two-stage", None,
              R + "native_eval_20260918_node1/p4nat_natsft2g_b003_g1_s0_timer1_default_tf5meta_node1_full.json",
              GJ + "gamma_14Ln_natsft2g_b003_g1_s0_timer1_default_node1__vs__14Ln_base_timer1_default_node1.json")]
    return rows


def qn3(L=None, ann_ref=None):
    out = {"p5n_search": {}, "rows": [], "checks": {}, "inputs": {}}
    # (a) are there 12p5n products of early / small-budget SFT checkpoints?  scan the v1 inventory + a fresh glob
    inv = json.load(open(os.path.join(HERE, "inventory_p5n_v1.json"), encoding="utf-8"))
    sft_like = []
    for g in inv["products"]:
        names = " ".join(os.path.basename(c) for c in g["copies"]).lower()
        if "sft" in names:
            sft_like.append({"sha16": g["sha16"], "copies": [os.path.basename(c) for c in g["copies"]], "ops": g["ops"]})
    fresh = []
    for dp, dns, fns in os.walk(R):
        dns[:] = [d for d in dns if d != "_superseded"]
        for fn in fns:
            if fn.startswith("p5n") and fn.endswith(".json") and not fn.endswith(".meta.json") and \
                    any(s in fn.lower() for s in ("sft", "step", "dyn", "lim")):
                fresh.append(os.path.join(dp, fn).replace("\\", "/").replace(R, ""))
    out["p5n_search"] = {"inventory_file": {"file": os.path.join(HERE, "inventory_p5n_v1.json"),
                                            "sha16": sha16(os.path.join(HERE, "inventory_p5n_v1.json"))},
                         "sft_like_p5n_products": sft_like, "fresh_glob_p5n_with_sft_step_dyn_lim": sorted(fresh),
                         "conclusion": "only final SFT (natsft_b0_s0) has 12p5n products (tm, node4); no early-step or "
                                       "small-limit SFT checkpoint has a 12p5n product"}
    # (b) substitute: 12p4n clean accuracy on qid 0..999 + 14Ln Gamma
    base_vec, base_rows = {}, {}
    rows = qn3_rows()
    for mach, lab, kind, st, p4, g in rows:
        if kind == "base":
            rr, inf = p4_load(p4)
            rr = [r for r in rr if int(r["qid"]) < 1000]
            base_rows[mach] = rr; base_vec[mach] = p4_vecs(rr); out["inputs"]["%s/base" % mach] = inf
    BT = {m: Boot([r["video"] for r in base_rows[m]]) for m in base_rows}
    ours_ref = {}
    for mach, lab, kind, st, p4, g in rows:
        rec = {"machine": mach, "label": lab, "kind": kind, "step": st}
        if p4:
            rr, inf = p4_load(p4)
            rr = [r for r in rr if int(r["qid"]) < 1000]
            if [(r["qid"], r["video"], r["gsA"], r["geA"]) for r in rr] != [(r["qid"], r["video"], r["gsA"], r["geA"]) for r in base_rows[mach]]:
                raise SystemExit("qid/annotation mismatch %s" % lab)
            X = p4_vecs(rr); Xb = base_vec[mach]
            rec["p4nat"] = inf
            rec["clean_R@0.5_q0_999"] = float(X["hit"].mean()); rec["clean_mIoU_q0_999"] = float(X["iou"].mean())
            rec["parse_rate_q0_999"] = float(X["parsed"].mean())
            rec["clean_R@0.5_q0_199"] = float(X["hit"][:200].mean())
            if kind != "base":
                rec["dR@0.5_vs_base_pts"] = S(*BT[mach].mean(X["hit"] - Xb["hit"]), 100.0)
                rec["dmIoU_vs_base"] = S(*BT[mach].mean(X["iou"] - Xb["iou"]))
            if lab.startswith("ours a=0.5 seed0"):
                ours_ref[mach] = X
            rec["_X"] = X
        if g:
            d = json.load(open(g, encoding="utf-8"))
            for inp in d["inputs"]:
                if os.path.exists(inp["path"]) and sha16(inp["path"]) != inp["sha16"]:
                    raise SystemExit("gamma raw input sha mismatch %s" % inp["path"])
            rec["gamma_file"] = {"file": g, "sha16": sha16(g), "raw_inputs": [{"role": i["role"], "sha16": i["sha16"], "exists": os.path.exists(i["path"])} for i in d["inputs"]]}
            rec["Gamma_mask"] = {"est": d["per_op"]["mask"]["gamma"], "ci95": d["per_op"]["mask"]["gamma_ci95"]}
            rec["Gamma_swap"] = {"est": d["per_op"]["swap"]["gamma"], "ci95": d["per_op"]["swap"]["gamma_ci95"]}
            rec["14Ln_clean_R50_q0_199"] = {"base": d.get("clean_R50_base"), "arm": d.get("clean_R50_arm")}
            rec["14Ln_n_cohort"] = d.get("n_cohort")
        out["rows"].append(rec)
    # paired vs our seed-0 arm (same machine) for SFT checkpoints on node4; and, because 12p4n covers qid 0..999,
    # the same evidence-necessary / prior-solvable layers as QN1 (labels from the tm products, annotation asserted equal)
    if L is not None:
        for mach in base_rows:
            if [(int(r["qid"]), r["video"], float(r["gsA"]), float(r["geA"])) for r in base_rows[mach]] != ann_ref:
                raise SystemExit("12p4n annotation differs from the 12p5n strata source on %s" % mach)
        BL = {m: {lay: Boot([r["video"] for r, f in zip(base_rows[m], L[lay]) if f]) for lay in LAYERS} for m in base_rows}
    for rec in out["rows"]:
        X = rec.pop("_X", None)
        if X is None:
            continue
        mach = rec["machine"]
        if mach in ours_ref and rec["kind"] == "SFT":
            rec["dR@0.5_vs_ours_s0_pts"] = S(*BT[mach].mean(X["hit"] - ours_ref[mach]["hit"]), 100.0)
        if L is not None:
            rec["layers_R@0.5"] = {lay: float(X["hit"][L[lay]].mean()) for lay in LAYERS}
            if rec["kind"] != "base":
                rec["layers_dR@0.5_vs_base_pts"] = {lay: S(*BL[mach][lay].mean(X["hit"][L[lay]] - base_vec[mach]["hit"][L[lay]]), 100.0) for lay in LAYERS}
            if mach in ours_ref and rec["kind"] == "SFT":
                rec["layers_dR@0.5_ours_s0_minus_this_pts"] = {
                    lay: S(*BL[mach][lay].mean(ours_ref[mach]["hit"][L[lay]] - X["hit"][L[lay]]), 100.0) for lay in LAYERS}
                dx = ours_ref[mach]["hit"] - X["hit"]
                rec["interaction_ours_s0_minus_this_ENboth_minus_PSany_pts"] = S(
                    *(lambda a, b: (a[0] - b[0], a[1] - b[1]))(BT[mach].mean(dx, L["EN_both"]), BT[mach].mean(dx, L["PS_any"])), 100.0)
    # checks: (1) 12p4n clean preds == 12p5n 'none' preds on node4 for base / ours / SFT (instrument consistency)
    chk = {}
    for k, f in (("base", P4FULL["node4"]["base"]), ("ours_a05", P4FULL["node4"]["ours_a05_s0"]), ("SFT", P4FULL["node4"]["SFT"])):
        p5 = {int(r["qid"]): r for r in json.load(open(V1.P[SX[k]][0], encoding="utf-8"))}
        p4 = {int(r["qid"]): r for r in json.load(open(f, encoding="utf-8"))}
        chk["node4_%s_p4nat_pred_equals_p5n_none_pred" % k] = int(sum(1 for q in range(1000) if p5[q]["none_pred"] == p4[q]["pred"]))
    # (2) 12p4n R@0.5 on qid 0..199 vs the 14Ln generation R@0.5 on the same 200 questions
    chk["p4nat_q0_199_vs_14Ln_clean_R50"] = [{"label": r["label"], "machine": r["machine"], "p4nat": r.get("clean_R@0.5_q0_199"),
                                               "14Ln": (r.get("14Ln_clean_R50_q0_199") or {}).get("arm")} for r in out["rows"] if r["kind"] != "base"]
    out["checks"] = chk
    # (3) matching: SFT checkpoints whose accuracy gain over base is within 2 pts of our seed-0 arm (same machine)
    match = {}
    for mach in ("node4",):
        ref = [r for r in out["rows"] if r["machine"] == mach and r["label"].startswith("ours a=0.5 seed0")][0]
        d0 = ref["dR@0.5_vs_base_pts"]["est"]
        cand = [r for r in out["rows"] if r["machine"] == mach and r["kind"] == "SFT" and "dR@0.5_vs_base_pts" in r]
        match[mach] = {"ours_seed0_dR@0.5_pts": d0, "tolerance_pts": 2.0,
                       "matched": [{"label": r["label"], "dR@0.5_vs_base_pts": r["dR@0.5_vs_base_pts"]["est"],
                                    "dR@0.5_vs_ours_s0_pts": r.get("dR@0.5_vs_ours_s0_pts"),
                                    "Gamma_mask": r.get("Gamma_mask"), "Gamma_swap": r.get("Gamma_swap")}
                                   for r in cand if abs(r["dR@0.5_vs_base_pts"]["est"] - d0) <= 2.0],
                       "ours": {"Gamma_mask": ref.get("Gamma_mask"), "Gamma_swap": ref.get("Gamma_swap")}}
    out["matching"] = match
    out["runs_needed_for_behavioural_match"] = [
        "SFT early checkpoints were saved only on node4 (<node4>/dyn_20260924/ckpt_sft_s{1,2}/step{50,100,150}; copier "
        "logs in results/dyn_20260924_node4/copier_sft_s*.log give their adapter sha16) and were not pulled; node4 is retired.",
        "Re-train SFT seed 1 with --save-every 50 (dyn_node4.conf TRAIN_ARGS, trainer 27_papo_train_qcdro_v3seln.py, "
        "--limit 2400 --objective mean --alpha 1.0 --beta 1e-9 --rand-hinge 0) on the cluster A100 node, keep step50/step100/step150, "
        "then run 12p5n (ops none,mask_evid,mask_rand,swap_evid,swap_rand,black; n=1000 offset 0; timer1_default) on each, "
        "plus the base and our a=0.5 seed0 adapter on the SAME node (cross-machine parse rates are not portable).",
        "Our own intermediate checkpoints already exist locally: results/pull/node2/.../dyn_20260924/"
        "ckpt_pc_a05_s1/step{50..300} (node2, tf5 key tree) -- 12p5n on step100-150 gives an our-side accuracy ladder too.",
        "Cheaper alternative inside B4 (SFT 4,800 pairs, already queued): add --save-every 50 so its early steps can be "
        "behaviour-evaluated without a separate run."]
    return out


# ------------------------------------------------------------------ main
def main():
    data, info = {}, {}
    keys = sorted(set(QW.values()) | {k for _, bk, ak in OTHER for k in (bk, ak)} | set(SX.values()) | set(V1.BLIND_REF))
    for k in keys:
        data[k], info[k] = V1.load(k)
    if info["node5_qwenbase"]["sha16"] != V1.P["node1_qwenbase"][1]:
        raise SystemExit("tm and node1 base products are no longer byte-identical; split the base row")
    qs = list(range(1000))
    ref = data["node5_qwenbase"]
    check_annotation(data, keys, ref)
    const = V1.const_window_hit(ref, qs)
    blind = (np.vstack([V1.vec(data[k], qs, "black", "hit") for k in V1.BLIND_REF]).sum(0) > 0).astype(float)
    L = layers_from(const, blind)
    vids = [ref[q]["video"] for q in qs]
    res = {"script": os.path.basename(__file__), "script_sha16": sha16(os.path.abspath(__file__)),
           "imports": {"qc_qj_p5n_v1.py": sha16(V1_PATH), "read_p4nat_v1.py": sha16(P4_PATH),
                       "read_p4nat_selective_v1.py": sha16(V1._SELP)},
           "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "params": {"B": B, "seed": SEED, "cluster": "video", "p": "2*min(P(d<=0),P(d>=0))", "thr_hit": 0.5, "thr_split": list(THRS)},
           "inputs": info,
           "strata": {"const_n": int(const.sum()), "blind_n": int(blind.sum()), "layer_n": {k: int(v.sum()) for k, v in L.items()},
                      "note": "identical definitions and source products as qc_qj_p5n_v1.py (const from annotation; blind from "
                              "all-black hits of %s)" % V1.BLIND_REF},
           "checks": {}}
    # pairing + hit-consistency guards
    for m, k in QW.items():
        if m != "base":
            bad = V1.assert_pairing(data[QW["base"]], data[k], qs)
            if bad: raise SystemExit("pairing failed %s %s" % (m, bad[:5]))
    for _, bk, ak in OTHER:
        bad = V1.assert_pairing(data[bk], data[ak], qs)
        if bad: raise SystemExit("pairing failed %s %s" % (ak, bad[:5]))
    for m in ("ours_a05", "SFT"):
        bad = [(q, kk) for q in qs for kk in ("n_estar", "nframes", "mask_rand_idx")
               if data[SX["base"]][q].get(kk) != data[SX[m]][q].get(kk)]
        if bad: raise SystemExit("pairing failed node4 %s %s" % (m, bad[:5]))
    for k in keys:
        if any(data[k][q].get("degenerate") for q in qs):
            raise SystemExit("degenerate rows in %s (v1 found none)" % k)
    OPS6 = V1.OPS; OPS3 = ("none", "mask_evid", "mask_rand")
    res["checks"]["hit_vs_iou_parsed_inconsistent_cells"] = {k: check_hit_consistency(data[k], OPS3 if k.startswith("node4") else OPS6) for k in keys}
    if any(res["checks"]["hit_vs_iou_parsed_inconsistent_cells"].values()):
        raise SystemExit("hit/iou/parsed inconsistent: %s" % res["checks"]["hit_vs_iou_parsed_inconsistent_cells"])

    # QN1
    res["QN1"] = {"qwen_p5n": qn1(QW, data, QW_CMP, qs, L, vids), "machines_qwen": QW_MACH,
                  "other_backbones": {}, "node4_p5n_v1_instrument": qn1(SX, data, SX_CMP, qs, L, vids)}
    for pid, bk, ak in OTHER:
        res["QN1"]["other_backbones"][pid] = qn1({"base": bk, "arm": ak}, data, [("arm", "base")], qs, L, vids)
    print("QN1 p5n done", flush=True)
    res["QN1"]["full_test_const_p4nat"] = qn1_full_const()
    print("QN1 full done", flush=True)
    # QN2
    ops5 = ("none", "mask_evid", "mask_rand", "swap_evid", "swap_rand")
    res["QN2"] = {"qwen_p5n": qn2(QW, data, QW_CMP, qs, L, vids, ops5),
                  "node4_p5n_v1_instrument": qn2(SX, data, SX_CMP, qs, L, vids, OPS3), "other_backbones": {}}
    for pid, bk, ak in OTHER:
        res["QN2"]["other_backbones"][pid] = qn2({"base": bk, "arm": ak}, data, [("arm", "base")], qs, L, vids, ops5, layers=("all",))
    print("QN2 done", flush=True)
    # QN3
    res["QN3"] = qn3(L, [(q, ref[q]["video"], float(ref[q]["gsA"]), float(ref[q]["geA"])) for q in qs])
    print("QN3 done", flush=True)
    out = os.path.join(HERE, "qn_evidnec_halluc_v2.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    write_tsv(res)
    print("wrote", out, sha16(out))


# ------------------------------------------------------------------ tsv
def fe(s, fmt="%+.1f"):
    if not s or s.get("est") is None:
        return "NA"
    lo, hi = s["ci95"]
    if lo is None:
        return fmt % s["est"]
    return (fmt + " [" + fmt + ", " + fmt + "]") % (s["est"], lo, hi)


def f3(s):
    return "NA" if not s or s.get("est") is None else "%.3f" % s["est"]


def write_tsv(res):
    with open(os.path.join(HERE, "qn1_evidnec_acc.tsv"), "w", encoding="utf-8") as f:
        f.write("source\tcomparison\tlayer\tn\tR@0.5_a\tR@0.5_b\tdR@0.5(pts)\tdmIoU\tinteraction_EN-PS_dR(pts)\n")
        blocks = [("p5n_qwen(tm/node1)", res["QN1"]["qwen_p5n"]), ("p5n_v1_node4(mask-only instr.)", res["QN1"]["node4_p5n_v1_instrument"])] + \
                 [("p5n_" + pid, b) for pid, b in res["QN1"]["other_backbones"].items()]
        for src, blk in blocks:
            for nm, pr in blk["paired"].items():
                a, b = nm.split(" - ")
                for lay in LAYERS:
                    inter = ""
                    for en, ps in INTER:
                        if lay == en:
                            inter = fe(blk["interaction_EN_minus_PS"][nm]["%s - %s" % (en, ps)]["R@0.5_pts"])
                    f.write("\t".join([src, nm, lay, str(blk["n"][lay]), f3(blk["per_model"][a][lay]["R@0.5"]),
                                       f3(blk["per_model"][b][lay]["R@0.5"]), fe(pr[lay]["R@0.5_pts"]),
                                       fe(pr[lay]["mIoU"], "%+.3f"), inter]) + "\n")
        for mach, blk in res["QN1"]["full_test_const_p4nat"]["machines"].items():
            for nm, pr in blk["paired"].items():
                a, b = nm.split(" - ")
                for lay in ("all", "EN_const", "PS_const"):
                    f.write("\t".join(["p4nat_full3720_" + mach, nm, lay, str(blk["n"][lay]), f3(blk["per_model"][a][lay]["R@0.5"]),
                                       f3(blk["per_model"][b][lay]["R@0.5"]), fe(pr[lay]["R@0.5_pts"]), fe(pr[lay]["mIoU"], "%+.3f"),
                                       fe(pr["interaction_EN_const_minus_PS_const_R@0.5_pts"]) if lay == "EN_const" else ""]) + "\n")
    with open(os.path.join(HERE, "qn2_halluc_stacked.tsv"), "w", encoding="utf-8") as f:
        f.write("source\tlayer\tmodel\tcondition\tn\tno_interval\t(explicit_refusal)\tconf_wrong@0.3\tcorrect@0.3\tconf_wrong@0.5\tcorrect@0.5\n")
        blocks = [("p5n_qwen(tm/node1)", res["QN2"]["qwen_p5n"]), ("p5n_v1_node4", res["QN2"]["node4_p5n_v1_instrument"])] + \
                 [("p5n_" + pid, b) for pid, b in res["QN2"]["other_backbones"].items()]
        for src, blk in blocks:
            for lay, lb in blk["layers"].items():
                for m, rr in lb["rates"].items():
                    for op, cc in rr.items():
                        if op == "E_minus_O_pts":
                            continue
                        f.write("\t".join([src, lay, m, op, str(lb["n"])] + [f3(cc[c]) for c in
                                          ("no_interval", "explicit_refusal_wide", "conf_wrong@0.3", "correct@0.3", "conf_wrong@0.5", "correct@0.5")]) + "\n")
        f.write("\n# contrasts (pts): diff under E, diff under O, DiD = (a-b) x (E-O)\n")
        f.write("source\tlayer\tcomparison\taxis\tcategory\tdiff_E\tdiff_O\tDiD\n")
        for src, blk in blocks:
            for lay, lb in blk["layers"].items():
                for nm, r in lb["contrasts"].items():
                    for ax in ("mask", "swap"):
                        if ax not in r:
                            continue
                        for c in ("no_interval", "conf_wrong@0.3", "conf_wrong@0.5", "correct@0.5"):
                            x = r[ax][c]
                            f.write("\t".join([src, lay, nm, ax, c, fe(x["diff_under_E_pts"]), fe(x["diff_under_O_pts"]), fe(x["DiD_E_minus_O_pts"])]) + "\n")
    with open(os.path.join(HERE, "qn3_accmatched.tsv"), "w", encoding="utf-8") as f:
        f.write("machine\tlabel\tstep\tclean_R@0.5(q0-999)\tdR@0.5_vs_base(pts)\tdR@0.5_vs_ours_s0(pts)\tGamma_mask\tGamma_swap\tR@0.5_q0-199(p4nat)\tR@0.5_q0-199(14Ln)"
                "\tdR_vs_base_EN_both(pts)\tdR_vs_base_PS_any(pts)\tours_s0_minus_this_EN_both(pts)\tours_s0_minus_this_PS_any(pts)\tinteraction(pts)\n")
        g = lambda s: "NA" if not s else "%+.2f [%+.2f, %+.2f]" % (s["est"], s["ci95"][0], s["ci95"][1])
        for r in res["QN3"]["rows"]:
            lb = r.get("layers_dR@0.5_vs_base_pts") or {}; lo = r.get("layers_dR@0.5_ours_s0_minus_this_pts") or {}
            f.write("\t".join([r["machine"], r["label"], str(r["step"]), "%.3f" % r["clean_R@0.5_q0_999"] if "clean_R@0.5_q0_999" in r else "NA",
                               fe(r.get("dR@0.5_vs_base_pts")), fe(r.get("dR@0.5_vs_ours_s0_pts")), g(r.get("Gamma_mask")), g(r.get("Gamma_swap")),
                               "%.3f" % r["clean_R@0.5_q0_199"] if "clean_R@0.5_q0_199" in r else "NA",
                               str((r.get("14Ln_clean_R50_q0_199") or {}).get("arm")),
                               fe(lb.get("EN_both")), fe(lb.get("PS_any")), fe(lo.get("EN_both")), fe(lo.get("PS_any")),
                               fe(r.get("interaction_ours_s0_minus_this_ENboth_minus_PSany_pts"))]) + "\n")


if __name__ == "__main__":
    main()
