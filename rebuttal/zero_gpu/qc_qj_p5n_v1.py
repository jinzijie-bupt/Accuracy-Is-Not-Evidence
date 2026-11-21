#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qc_qj_p5n_v1.py (revision, zero-GPU, 2026-11-06)

Reads 12p5n behavioural products (1,000 Charades-STA test questions, qid 0..999; per question one greedy
generation under none / mask_evid / mask_rand / swap_evid / swap_rand / black) and answers:

  Q-C(1)  answered-subset accuracy: on questions where BOTH base and arm output a parseable interval under the
          evidence edit, is the arm still less accurate than the base (relative to clean / placebo)?
  Q-C(2)  refusal / unparsed rate per condition, mask vs swap (swap has no black frames), E vs O vs clean.
          Plus a swap version of the paper's Table 20 left block (lost answers -> no interval / wrong interval).
  Q-J     stratify questions by prior-solvability and report the behavioural audit effect per stratum.
          Prior strata (defined WITHOUT any trained arm):
            const  : the constant (0,10) s window has IoU>=0.5 with the annotation (no model)
            blind  : at least one public/base model (6 base products on tm) hits IoU>=0.5 on the ALL-BLACK video
            ownblk : the pair's own base hits on the all-black video (reported only if n>=30)

Only SAME-MACHINE pairs are used (paired by qid; pairing keys n_estar/nframes/swap_donor/mask_rand_idx/
swap_rand_idx asserted identical).  Video-cluster bootstrap, B=2000, seed 0, two-sided p = 2*min(P<=0,P>=0).
Every input file is recorded with sha16.  Outputs: qc_qj_p5n_v1.json + three .tsv tables.
"""
import os, sys, json, hashlib, importlib.util, datetime
import numpy as np
LOCAL_ROOT = os.environ.get("LOCAL_ROOT", ".")  # analysis directory holding the pulled result files (results/, rebuttal/)

HERE = os.path.dirname(os.path.abspath(__file__))
R = (LOCAL_ROOT + "/results/")
B = 2000
SEED = 0
OPS = ("none", "mask_evid", "mask_rand", "swap_evid", "swap_rand", "black")
PAIR_KEYS = ("n_estar", "nframes", "swap_donor", "mask_rand_idx", "swap_rand_idx")

# abstention taxonomy (strict = verbatim 'There are none.'; wide = any refusal wording) from the registered reader
_SELP = (LOCAL_ROOT + "/analysis/read_p4nat_selective_v1.py")


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


SEL = _imp("sel_reader", _SELP)

# ------------------------------------------------------------------ products (identified by meta argv + sha16)
P = {
    "node5_qwenbase": (R + "pull/node5/native_eval_20260920/p5n6_qwenbase_timer1_default_node5_n1000_o0.json", "d064992d6c9d5f22"),
    "node5_natbase_a05": (R + "pull/node5/native_eval_20260920/p5n6_natbase_a05_g1_s0_timer1_default_node5_n1000_o0.json", "9547525ade59ea56"),
    "node5_natsft": (R + "pull/node5/native_eval_20260920/p5n6_natsft_b0_s0_timer1_default_node5_n1000_o0.json", "0ea58b16fb8a6310"),
    "node5_natmdpo": (R + "pull/node5/native_eval_20260920/p5n6_natmdpo_a05_s0_timer1_default_node5_n1000_o0.json", "80d7f3bfba180c66"),
    "node1_qwenbase": (R + "pull/node1/review_r3_20260924/p5n6_qwenbase_timer1_default_node1_n1000_o0.json", "d064992d6c9d5f22"),
    "node1_iougrpo": (R + "pull/node1/review_r3_20260924/p5n6_x0s0iou700_timer1_default_node1_n1000_o0.json", "8c39be04837129c7"),
    "node1_abl_single": (R + "pull/node1/review_r3_20260924/p5n6_abl_single_timer1_default_node1_n1000_o0.json", "72f0993a4e4d649c"),
    "node5_tzbase_tf": (R + "pull/node5/native_eval_20260920/p5n6_tzbase_timer1_default_node5_n1000_o0.json", "f0301961481c3001"),
    "node5_nattz_a05_tf": (R + "pull/node5/native_eval_20260920/p5n6_nattz_a05_g1_s0_timer1_default_node5_n1000_o0.json", "6a7101d391630a40"),
    "node5_tzbase_vf": (R + "pull/node5/native_eval_20260920/p5n6_tzbase_timer1_default_vf_node5_n1000_o0.json", "6df04f91b5d473ca"),
    "node5_tr1base": (R + "pull/node5/native_eval_20260920/p5n6_tr1base_timer1_default_node5_n1000_o0.json", "4d15445ef892718a"),
    "node5_nattr1_a025": (R + "pull/node5/native_eval_20260920/p5n6_nattr1_a025_g1_s0_timer1_default_node5_n1000_o0.json", "e1039d379cdfd231"),
    "node5_vcbase_tf": (R + "pull/node5/native_eval_20260920/p5n6_vcbase_vchat_node5_n1000_o0.json", "6bba22c9666eae98"),
    "node5_natvc_a05_tf": (R + "pull/node5/native_eval_20260920/p5n6_natvc_a05_g1_s0_vchat_node5_n1000_o0.json", "8673cfa3425f7dfe"),
    "node5_qwenbase_vchat": (R + "pull/node5/native_eval_20260920/p5n6_qwenbase_vchat_node5_n1000_o0.json", "9121075997d75a90"),
    "node2_tzbase": (R + "pull/node2/native_eval_20260918/p5n6_tzbase_timer1_default_node2_n1000_o0.json", "886a68f75b2e4caf"),
    "node2_nattz_a05": (R + "pull/node2/native_eval_20260918/p5n6_nattz_a05_g1_s0_timer1_default_node2_n1000_o0.json", "894ffa588102c465"),
    "node2_tzbase_tf": (R + "pull/node2/native_eval_20260918/p5n6_tzbase_timer1_default_textfirst_node2_n1000_o0.json", "2ecb02fa6bd9f335"),
    "node2_nattz_a05_trtf": (R + "pull/node2/native_eval_20260918/p5n6_nattz_a05_g1_tf_s0_timer1_default_textfirst_node2_n1000_o0.json", "24ee31824e739ec0"),
    "node4_vcbase": (R + "pull_20260924/p5n6_vcbase_vchat_node4_n1000_o0.json", "73bce56c05775449"),
    "node4_natvc_a05": (R + "pull_20260924/p5n6_natvc_a05_g1_s0_vchat_node4_n1000_o0.json", "f17981243c0d0c4e"),
}

# (pair id, machine, base key, arm key, arm kind, paper Table 19 row)
PAIRS = [
    ("Qwen7B_a05_node5", "tm", "node5_qwenbase", "node5_natbase_a05", "placebo-controlled a=0.5", "T19 row 1"),
    ("Qwen7B_SFT_node5", "tm", "node5_qwenbase", "node5_natsft", "SFT (accuracy)", "T19 row 4"),
    ("Qwen7B_mDPO_node5", "tm", "node5_qwenbase", "node5_natmdpo", "mDPO-style (accuracy)", "T19 row 3"),
    ("Qwen7B_IoUGRPO_node1", "node1", "node1_qwenbase", "node1_iougrpo", "IoU-GRPO (accuracy)", "T19 row 2"),
    ("Qwen7B_ablsingle_node1", "node1", "node1_qwenbase", "node1_abl_single", "ablation abl_single a=0.5", "not in T19"),
    ("TZ_a05_node5_evaltf", "tm", "node5_tzbase_tf", "node5_nattz_a05_tf", "placebo-controlled a=0.5 (trained vf, eval tf)", "T19 row 5"),
    ("TZ_a05_node2_vf", "node2", "node2_tzbase", "node2_nattz_a05", "placebo-controlled a=0.5 (vf)", "T19 row 6"),
    ("TZ_a05trtf_node2_tf", "node2", "node2_tzbase_tf", "node2_nattz_a05_trtf", "placebo-controlled a=0.5 (trained tf, eval tf)", "not in T19"),
    ("TR1_a025_node5", "tm", "node5_tr1base", "node5_nattr1_a025", "placebo-controlled a=0.25", "T19 row 7"),
    ("VC_a05_node5_evaltf", "tm", "node5_vcbase_tf", "node5_natvc_a05_tf", "placebo-controlled a=0.5 (trained vf, eval tf)", "T19 row 8"),
    ("VC_a05_node4_vf", "node4", "node4_vcbase", "node4_natvc_a05", "placebo-controlled a=0.5 (vf)", "T19 row 9"),
]
# base-only products used to DEFINE the blind stratum (no trained arm involved)
BLIND_REF = ["node5_qwenbase", "node5_qwenbase_vchat", "node5_tzbase_tf", "node5_tzbase_vf", "node5_tr1base", "node5_vcbase_tf"]


def load(key):
    path, want = P[key]
    got = sha16(path)
    if got != want:
        raise SystemExit("sha mismatch %s: %s != %s" % (key, got, want))
    rows = json.load(open(path, encoding="utf-8"))
    d = {int(r["qid"]): r for r in rows}
    if sorted(d) != list(range(1000)):
        raise SystemExit("qid set != 0..999 for %s" % key)
    meta = path + ".meta.json"
    m = json.load(open(meta, encoding="utf-8")) if os.path.exists(meta) else None
    info = {"key": key, "file": path, "sha16": got,
            "meta_sha16": sha16(meta) if m else None,
            "meta_model": (m or {}).get("args", {}).get("model"), "meta_adapter": (m or {}).get("args", {}).get("adapter"),
            "meta_prompt_style": (m or {}).get("args", {}).get("prompt_style"),
            "meta_text_first": (m or {}).get("args", {}).get("text_first"), "meta_out": (m or {}).get("args", {}).get("out"),
            "instrument": (m or {}).get("instrument")}
    return d, info


# ------------------------------------------------------------------ bootstrap machinery
def groups_of(vids):
    ix = {}
    for i, v in enumerate(vids):
        ix.setdefault(v, []).append(i)
    return [np.array(g, int) for g in ix.values()]


def draws_idx(groups, B=B, seed=SEED):
    rng = np.random.default_rng(seed); G = len(groups)
    return [np.concatenate([groups[k] for k in rng.integers(0, G, G)]) for _ in range(B)]


def summ(point, d):
    d = np.asarray([x for x in d if x is not None and np.isfinite(x)], float)
    if point is None or not np.isfinite(point):
        return {"est": None, "ci95": [None, None], "p": None}
    if not d.size:
        return {"est": float(point), "ci95": [None, None], "p": None}
    return {"est": float(point), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def stat(fn, idx_all, D):
    """fn(index array) -> float or None ; returns point + bootstrap summary"""
    pt = fn(idx_all)
    return summ(pt, [fn(ix) for ix in D])


def mean_or_none(x):
    return float(x.mean()) if x.size else None


# ------------------------------------------------------------------ per-row vectors
def vec(rows, qs, op, field):
    return np.array([float(rows[q]["%s_%s" % (op, field)]) for q in qs])


def refusal_flags(rows, qs, op):
    strict, wide = [], []
    for q in qs:
        r = rows[q]
        if r["%s_parsed" % op]:
            strict.append(0); wide.append(0); continue
        bw, bs = SEL.classify_abstain(r.get("%s_raw" % op) or "")
        strict.append(int(bs == "literal_there_are_none"))
        wide.append(int(bw in ("literal_there_are_none", "nl_refusal")))
    return np.array(strict, float), np.array(wide, float)


def const_window_hit(rows, qs, a=0.0, b=10.0):
    out = []
    for q in qs:
        gs, ge = float(rows[q]["gsA"]), float(rows[q]["geA"])
        inter = max(0.0, min(b, ge) - max(a, gs)); uni = max(b, ge) - min(a, gs)
        out.append(int((inter / uni if uni > 0 else 0.0) >= 0.5))
    return np.array(out, float)


# ------------------------------------------------------------------ analyses
def assert_pairing(Bq, Aq, qs):
    bad = []
    for q in qs:
        for k in PAIR_KEYS:
            if Bq[q].get(k, "<absent>") != Aq[q].get(k, "<absent>"):
                bad.append((q, k)); break
    return bad


def qc2_refusal(base, arm, qs, vids, D):
    """unparsed / strict / wide per condition, per model, + E-O contrasts and DiD (arm - base)."""
    out = {"per_condition": {}, "contrasts": {}}
    V = {}
    for tag, rows in (("base", base), ("arm", arm)):
        for op in OPS:
            par = vec(rows, qs, op, "parsed")
            s, w = refusal_flags(rows, qs, op)
            V[(tag, op, "unparsed")] = 1.0 - par
            V[(tag, op, "strict")] = s
            V[(tag, op, "wide")] = w
    for op in OPS:
        out["per_condition"][op] = {}
        for tag in ("base", "arm"):
            out["per_condition"][op][tag] = {k: float(V[(tag, op, k)].mean()) for k in ("unparsed", "strict", "wide")}
        out["per_condition"][op]["delta_unparsed_arm_minus_base"] = stat(
            lambda ix, op=op: V[("arm", op, "unparsed")][ix].mean() - V[("base", op, "unparsed")][ix].mean(), np.arange(len(qs)), D)
    for ax in ("mask", "swap"):
        e, o = "%s_evid" % ax, "%s_rand" % ax
        for kind in ("unparsed", "strict"):
            f_b = lambda ix, e=e, o=o, kind=kind: V[("base", e, kind)][ix].mean() - V[("base", o, kind)][ix].mean()
            f_a = lambda ix, e=e, o=o, kind=kind: V[("arm", e, kind)][ix].mean() - V[("arm", o, kind)][ix].mean()
            f_c = lambda ix, e=e, kind=kind: V[("arm", e, kind)][ix].mean() - V[("arm", "none", kind)][ix].mean()
            out["contrasts"]["%s_%s" % (ax, kind)] = {
                "base_E_minus_O": stat(f_b, np.arange(len(qs)), D),
                "arm_E_minus_O": stat(f_a, np.arange(len(qs)), D),
                "arm_E_minus_clean": stat(f_c, np.arange(len(qs)), D),
                "DiD_arm_minus_base": stat(lambda ix, f_a=f_a, f_b=f_b: f_a(ix) - f_b(ix), np.arange(len(qs)), D)}
    # swap-specific refusal as a fraction of mask-specific refusal (arm-minus-base DiD in each)
    def ratio(ix):
        dm = (V[("arm", "mask_evid", "unparsed")][ix].mean() - V[("arm", "mask_rand", "unparsed")][ix].mean()) - \
             (V[("base", "mask_evid", "unparsed")][ix].mean() - V[("base", "mask_rand", "unparsed")][ix].mean())
        ds = (V[("arm", "swap_evid", "unparsed")][ix].mean() - V[("arm", "swap_rand", "unparsed")][ix].mean()) - \
             (V[("base", "swap_evid", "unparsed")][ix].mean() - V[("base", "swap_rand", "unparsed")][ix].mean())
        return ds / dm if abs(dm) > 1e-9 else None
    out["contrasts"]["swap_DiD_over_mask_DiD_unparsed"] = stat(ratio, np.arange(len(qs)), D)
    # edit-induced no-interval: restricted to questions BOTH models answer (parse) on the clean video
    cl = (V[("base", "none", "unparsed")] == 0) & (V[("arm", "none", "unparsed")] == 0)
    sub = np.where(cl)[0]
    blk = {"n": int(cl.sum()), "definition": "questions parsed on the clean video by both base and arm; rate of no interval under each edit"}
    if sub.size >= 30:
        Gs = groups_of([vids[i] for i in sub]); Ds = draws_idx(Gs)
        W = {k: v[sub] for k, v in V.items() if k[2] == "unparsed"}
        ia = np.arange(sub.size)
        blk["rates"] = {t: {op: float(W[(t, op, "unparsed")].mean()) for op in OPS} for t in ("base", "arm")}
        for ax in ("mask", "swap"):
            e, o = "%s_evid" % ax, "%s_rand" % ax
            fb = lambda ix, e=e, o=o: W[("base", e, "unparsed")][ix].mean() - W[("base", o, "unparsed")][ix].mean()
            fa = lambda ix, e=e, o=o: W[("arm", e, "unparsed")][ix].mean() - W[("arm", o, "unparsed")][ix].mean()
            blk[ax] = {"base_E_minus_O": stat(fb, ia, Ds), "arm_E_minus_O": stat(fa, ia, Ds),
                       "DiD_arm_minus_base": stat(lambda ix, fa=fa, fb=fb: fa(ix) - fb(ix), ia, Ds)}
    out["on_clean_answered_both"] = blk
    return out


def lost_answers(base, arm, qs, vids, D):
    """Paper Table 20 left block, for mask (reproduction) and swap (new).  Per model, on its OWN set of questions
    correct under the placebo edit: share with no interval under the evidence edit, and among intervals returned,
    share wrong.  Entry = arm - base."""
    out = {}
    for ax in ("mask", "swap"):
        e, o = "%s_evid" % ax, "%s_rand" % ax
        M = {}
        for tag, rows in (("base", base), ("arm", arm)):
            M[tag] = (vec(rows, qs, o, "hit"), vec(rows, qs, e, "parsed"), vec(rows, qs, e, "hit"))

        def share_noint(ix, tag):
            oh, ep, eh = (x[ix] for x in M[tag]); s = oh == 1
            return float((ep[s] == 0).mean()) if s.sum() else None

        def share_wrong(ix, tag):
            oh, ep, eh = (x[ix] for x in M[tag]); s = (oh == 1) & (ep == 1)
            return float((eh[s] == 0).mean()) if s.sum() else None

        def dif(fn):
            def g(ix):
                a, b = fn(ix, "arm"), fn(ix, "base")
                return None if a is None or b is None else a - b
            return g
        ia = np.arange(len(qs))
        out[ax] = {"n_correct_under_placebo": {t: int(M[t][0].sum()) for t in M},
                   "no_interval_share": {t: share_noint(ia, t) for t in M},
                   "wrong_among_returned_share": {t: share_wrong(ia, t) for t in M},
                   "delta_no_interval_pts": stat(lambda ix: (lambda v: None if v is None else 100 * v)(dif(share_noint)(ix)), ia, D),
                   "delta_wrong_interval_pts": stat(lambda ix: (lambda v: None if v is None else 100 * v)(dif(share_wrong)(ix)), ia, D)}
    return out


def qc1_answered(base, arm, qs, vids, D_full):
    """answered-subset accuracy under the evidence edit."""
    out = {}
    for ax in ("mask", "swap"):
        e, o = "%s_evid" % ax, "%s_rand" % ax
        bp_e, ap_e = vec(base, qs, e, "parsed"), vec(arm, qs, e, "parsed")
        bp_o, ap_o = vec(base, qs, o, "parsed"), vec(arm, qs, o, "parsed")
        bp_n, ap_n = vec(base, qs, "none", "parsed"), vec(arm, qs, "none", "parsed")
        res = {"coverage_E": {"base": float(bp_e.mean()), "arm": float(ap_e.mean())}}
        # (a) J_E: both parse under E
        for sname, mask in (("J_E", (bp_e == 1) & (ap_e == 1)),
                            ("J_all", (bp_e == 1) & (ap_e == 1) & (bp_o == 1) & (ap_o == 1) & (bp_n == 1) & (ap_n == 1))):
            sub = [q for q, m in zip(qs, mask) if m]
            blk = {"n": len(sub)}
            if len(sub) < 10:
                blk["note"] = "subset too small (<10): not estimated (absent, not 0)"; res[sname] = blk; continue
            sv = [base[q]["video"] for q in sub]; G = groups_of(sv); D = draws_idx(G); ia = np.arange(len(sub))
            blk["n_videos"] = len(G)
            X = {}
            for tag, rows in (("base", base), ("arm", arm)):
                for op in ("none", e, o):
                    X[(tag, op, "hit")] = vec(rows, sub, op, "hit")
                    X[(tag, op, "iou")] = vec(rows, sub, op, "iou")
            for tag in ("base", "arm"):
                blk[tag] = {"R@0.5_clean": float(X[(tag, "none", "hit")].mean()), "R@0.5_E": float(X[(tag, e, "hit")].mean()),
                            "R@0.5_O": float(X[(tag, o, "hit")].mean()), "mIoU_E": float(X[(tag, e, "iou")].mean()),
                            "mIoU_clean": float(X[(tag, "none", "iou")].mean())}
            blk["delta_R@0.5_E_arm_minus_base"] = stat(lambda ix: X[("arm", e, "hit")][ix].mean() - X[("base", e, "hit")][ix].mean(), ia, D)
            blk["delta_mIoU_E_arm_minus_base"] = stat(lambda ix: X[("arm", e, "iou")][ix].mean() - X[("base", e, "iou")][ix].mean(), ia, D)
            dropb = lambda ix: X[("base", "none", "hit")][ix].mean() - X[("base", e, "hit")][ix].mean()
            dropa = lambda ix: X[("arm", "none", "hit")][ix].mean() - X[("arm", e, "hit")][ix].mean()
            blk["drop_clean_minus_E"] = {"base": stat(dropb, ia, D), "arm": stat(dropa, ia, D),
                                         "delta_arm_minus_base": stat(lambda ix: dropa(ix) - dropb(ix), ia, D)}
            cb = lambda ix: X[("base", o, "hit")][ix].mean() - X[("base", e, "hit")][ix].mean()
            ca = lambda ix: X[("arm", o, "hit")][ix].mean() - X[("arm", e, "hit")][ix].mean()
            blk["contrast_O_minus_E"] = {"base": stat(cb, ia, D), "arm": stat(ca, ia, D),
                                         "DiD_arm_minus_base": stat(lambda ix: ca(ix) - cb(ix), ia, D)}

            def share(ix, tag):
                ro = X[(tag, o, "hit")][ix].mean()
                return None if ro <= 0 else 1 - X[(tag, e, "hit")][ix].mean() / ro
            blk["evidence_share_1_minus_aE_over_aO"] = {
                "base": stat(lambda ix: share(ix, "base"), ia, D), "arm": stat(lambda ix: share(ix, "arm"), ia, D),
                "delta_pts": stat(lambda ix: (None if share(ix, "arm") is None or share(ix, "base") is None
                                              else 100 * (share(ix, "arm") - share(ix, "base"))), ia, D)}
            res[sname] = blk
        # (b) each model on its OWN answered set (selective accuracy; unpaired sets)
        own = {}
        for tag, rows, pe, po in (("base", base, bp_e, bp_o), ("arm", arm, ap_e, ap_o)):
            he, ho = vec(rows, qs, e, "hit"), vec(rows, qs, o, "hit")
            own[tag] = {"R@0.5_E_given_parsed_E": mean_or_none(he[pe == 1]), "n_parsed_E": int(pe.sum()),
                        "R@0.5_O_given_parsed_O": mean_or_none(ho[po == 1]), "n_parsed_O": int(po.sum())}
        ia = np.arange(len(qs))
        HB = (vec(base, qs, e, "hit"), vec(base, qs, o, "hit")); HA = (vec(arm, qs, e, "hit"), vec(arm, qs, o, "hit"))

        def gap(ix, H, pe, po):
            a, b = pe[ix] == 1, po[ix] == 1
            return (H[1][ix][b].mean() - H[0][ix][a].mean()) if a.sum() and b.sum() else None
        gb = lambda ix: gap(ix, HB, bp_e, bp_o); ga = lambda ix: gap(ix, HA, ap_e, ap_o)
        own["selective_gap_O_minus_E"] = {"base": stat(gb, ia, D_full), "arm": stat(ga, ia, D_full),
                                          "delta_arm_minus_base": stat(lambda ix: (None if ga(ix) is None or gb(ix) is None else ga(ix) - gb(ix)), ia, D_full)}
        res["own_answered_set"] = own
        out[ax] = res
    return out


def qj_strata(base, arm, qs, strata, D_by_stratum):
    """behavioural audit effect per prior stratum: contrast O-E per model, DiD, evidence share, interaction."""
    out = {}
    for sname, flag in strata.items():
        blk = {}
        for lev, lab in ((1, "prior_solvable"), (0, "not_prior_solvable")):
            sub = [q for q, f in zip(qs, flag) if f == lev]
            if len(sub) < 30:
                blk[lab] = {"n": len(sub), "note": "stratum < 30 questions: not estimated (absent, not 0)"}; continue
            sv = [base[q]["video"] for q in sub]; G = groups_of(sv); D = draws_idx(G); ia = np.arange(len(sub))
            b2 = {"n": len(sub), "n_videos": len(G)}
            for ax in ("mask", "swap"):
                e, o = "%s_evid" % ax, "%s_rand" % ax
                X = {(t, op): vec(rows, sub, op, "hit") for t, rows in (("base", base), ("arm", arm)) for op in ("none", e, o)}
                cb = lambda ix: X[("base", o)][ix].mean() - X[("base", e)][ix].mean()
                ca = lambda ix: X[("arm", o)][ix].mean() - X[("arm", e)][ix].mean()

                def share(ix, t):
                    ro = X[(t, o)][ix].mean()
                    return None if ro <= 0 else 1 - X[(t, e)][ix].mean() / ro
                b2[ax] = {"R@0.5": {t: {"clean": float(X[(t, "none")].mean()), "E": float(X[(t, e)].mean()), "O": float(X[(t, o)].mean())} for t in ("base", "arm")},
                          "contrast_base": stat(cb, ia, D), "contrast_arm": stat(ca, ia, D),
                          "DiD_pts": stat(lambda ix: 100 * (ca(ix) - cb(ix)), ia, D),
                          "delta_share_pts": stat(lambda ix: (None if share(ix, "arm") is None or share(ix, "base") is None
                                                              else 100 * (share(ix, "arm") - share(ix, "base"))), ia, D)}
            blk[lab] = b2
        # interaction (prior - not prior) on the full set, one joint video bootstrap
        if all("mask" in blk.get(l, {}) for l in ("prior_solvable", "not_prior_solvable")):
            D = D_by_stratum
            ia = np.arange(len(qs)); fl = np.asarray(flag)
            inter = {}
            for ax in ("mask", "swap"):
                e, o = "%s_evid" % ax, "%s_rand" % ax
                X = {(t, op): vec(rows, qs, op, "hit") for t, rows in (("base", base), ("arm", arm)) for op in (e, o)}

                def did(ix, lev):
                    ix = ix[fl[ix] == lev]
                    if not ix.size:
                        return None
                    return (X[("arm", o)][ix].mean() - X[("arm", e)][ix].mean()) - (X[("base", o)][ix].mean() - X[("base", e)][ix].mean())
                inter[ax] = stat(lambda ix: (None if did(ix, 1) is None or did(ix, 0) is None else 100 * (did(ix, 1) - did(ix, 0))), ia, D)
            blk["interaction_DiD_prior_minus_notprior_pts"] = inter
        out[sname] = blk
    return out


def main():
    data, info = {}, {}
    for k in P:
        data[k], info[k] = load(k)
    qs = list(range(1000))
    # ---- prior strata (no trained arm involved)
    ref = data["node5_qwenbase"]
    const = const_window_hit(ref, qs)
    blind_hits = np.vstack([vec(data[k], qs, "black", "hit") for k in BLIND_REF])
    blind = (blind_hits.sum(0) > 0).astype(float)
    strata_def = {"const": {"definition": "constant (0,10)s window IoU>=0.5 with annotation (no model)", "n_prior": int(const.sum())},
                  "blind": {"definition": "hit (IoU>=0.5) on the ALL-BLACK video by >=1 of %s" % BLIND_REF, "n_prior": int(blind.sum()),
                            "per_ref_black_hits": {k: int(h.sum()) for k, h in zip(BLIND_REF, blind_hits)}},
                  "overlap_const_and_blind": int(((const == 1) & (blind == 1)).sum())}
    # sanity: annotation fields identical across every product
    for k in P:
        for q in qs:
            a, b = data[k][q], ref[q]
            if (a["video"], a["gsA"], a["geA"]) != (b["video"], b["gsA"], b["geA"]):
                raise SystemExit("annotation mismatch %s qid %d" % (k, q))
    res = {"script": os.path.basename(__file__), "script_sha16": sha16(os.path.abspath(__file__)),
           "imports": {"read_p4nat_selective_v1.py": sha16(_SELP)},
           "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "params": {"B": B, "seed": SEED, "cluster": "video", "p": "2*min(P(d<=0),P(d>=0))", "thr": 0.5},
           "inputs": info, "strata_definition": strata_def, "pairs": {}}
    for pid, mach, bk, ak, kind, t19 in PAIRS:
        base, arm = data[bk], data[ak]
        bad = assert_pairing(base, arm, qs)
        if bad:
            raise SystemExit("pairing assertion failed for %s: %s" % (pid, bad[:5]))
        usable = [q for q in qs if not base[q].get("degenerate") and not arm[q].get("degenerate")
                  and all("%s_hit" % o in base[q] and "%s_hit" % o in arm[q] for o in OPS)]
        vids = [base[q]["video"] for q in usable]
        G = groups_of(vids); D = draws_idx(G)
        pr = {"machine": mach, "base": bk, "arm": ak, "arm_kind": kind, "paper_table19": t19,
              "n": len(usable), "n_videos": len(G), "pairing_keys_asserted": list(PAIR_KEYS),
              "clean_R@0.5": {"base": float(vec(base, usable, "none", "hit").mean()), "arm": float(vec(arm, usable, "none", "hit").mean())},
              "R@0.5_by_op": {t: {op: float(vec(rows, usable, op, "hit").mean()) for op in OPS} for t, rows in (("base", base), ("arm", arm))}}
        pr["QC1_answered_subset"] = qc1_answered(base, arm, usable, vids, D)
        pr["QC2_refusal"] = qc2_refusal(base, arm, usable, vids, D)
        pr["QC2_lost_answers_T20style"] = lost_answers(base, arm, usable, vids, D)
        uix = {q: i for i, q in enumerate(qs)}
        strata = {"const": [int(const[uix[q]]) for q in usable], "blind": [int(blind[uix[q]]) for q in usable]}
        ob = [int(base[q]["black_hit"]) for q in usable]
        strata["ownblk"] = ob
        pr["QJ_prior_strata"] = qj_strata(base, arm, usable, strata, D)
        pr["QJ_prior_strata"]["ownblk_n_prior"] = int(sum(ob))
        res["pairs"][pid] = pr
        print("done", pid, flush=True)
    out = os.path.join(HERE, "qc_qj_p5n_v1.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    write_tsv(res)
    print("wrote", out)


def f3(x):
    return "NA" if x is None else "%.3f" % x


def fe(s, scale=1.0, fmt="%+.1f"):
    if not s or s.get("est") is None:
        return "NA"
    lo, hi = s["ci95"]
    if lo is None:
        return (fmt % (scale * s["est"]))
    return (fmt + " [" + fmt + ", " + fmt + "]") % (scale * s["est"], scale * lo, scale * hi)


def write_tsv(res):
    # QC1
    with open(os.path.join(HERE, "qc1_answered_subset.tsv"), "w", encoding="utf-8") as f:
        f.write("pair\tmachine\taxis\tcov_E_base\tcov_E_arm\tn_JE\tR_E_base\tR_E_arm\tdR_E(pts)\tdmIoU_E\tdrop_clean-E_base(pts)\tdrop_clean-E_arm(pts)\tddrop(pts)\tn_Jall\tcontrast_base(pts)\tcontrast_arm(pts)\tDiD(pts)\n")
        for pid, pr in res["pairs"].items():
            for ax in ("mask", "swap"):
                r = pr["QC1_answered_subset"][ax]; je = r["J_E"]; ja = r["J_all"]
                if "base" not in je:
                    f.write("%s\t%s\t%s\t%.3f\t%.3f\t%d\tNA\n" % (pid, pr["machine"], ax, r["coverage_E"]["base"], r["coverage_E"]["arm"], je["n"])); continue
                f.write("\t".join([pid, pr["machine"], ax, f3(r["coverage_E"]["base"]), f3(r["coverage_E"]["arm"]), str(je["n"]),
                                   f3(je["base"]["R@0.5_E"]), f3(je["arm"]["R@0.5_E"]), fe(je["delta_R@0.5_E_arm_minus_base"], 100),
                                   fe(je["delta_mIoU_E_arm_minus_base"], 1, "%+.3f"),
                                   fe(je["drop_clean_minus_E"]["base"], 100), fe(je["drop_clean_minus_E"]["arm"], 100),
                                   fe(je["drop_clean_minus_E"]["delta_arm_minus_base"], 100), str(ja["n"]),
                                   fe(ja.get("contrast_O_minus_E", {}).get("base"), 100) if "base" in ja else "NA",
                                   fe(ja.get("contrast_O_minus_E", {}).get("arm"), 100) if "base" in ja else "NA",
                                   fe(ja.get("contrast_O_minus_E", {}).get("DiD_arm_minus_base"), 100) if "base" in ja else "NA"]) + "\n")
    # QC2
    with open(os.path.join(HERE, "qc2_refusal_by_condition.tsv"), "w", encoding="utf-8") as f:
        f.write("pair\tmachine\tmodel\t" + "\t".join("U_%s" % o for o in OPS) + "\t" + "\t".join("S_%s" % o for o in OPS) +
                "\tmask_DiD_U(pts)\tswap_DiD_U(pts)\tswap/mask\tT20_mask_noint(pts)\tT20_mask_wrong(pts)\tT20_swap_noint(pts)\tT20_swap_wrong(pts)\n")
        for pid, pr in res["pairs"].items():
            q = pr["QC2_refusal"]; L = pr["QC2_lost_answers_T20style"]
            for t in ("base", "arm"):
                cells = [pid, pr["machine"], t] + [f3(q["per_condition"][o][t]["unparsed"]) for o in OPS] + \
                        [f3(q["per_condition"][o][t]["strict"]) for o in OPS]
                if t == "arm":
                    cells += [fe(q["contrasts"]["mask_unparsed"]["DiD_arm_minus_base"], 100), fe(q["contrasts"]["swap_unparsed"]["DiD_arm_minus_base"], 100),
                              fe(q["contrasts"]["swap_DiD_over_mask_DiD_unparsed"], 1, "%.2f"),
                              fe(L["mask"]["delta_no_interval_pts"]), fe(L["mask"]["delta_wrong_interval_pts"]),
                              fe(L["swap"]["delta_no_interval_pts"]), fe(L["swap"]["delta_wrong_interval_pts"])]
                f.write("\t".join(cells) + "\n")
    with open(os.path.join(HERE, "qc2b_induced_refusal.tsv"), "w", encoding="utf-8") as f:
        f.write("pair\tmachine\tn_clean_answered_both\tmodel\t" + "\t".join("U_%s" % o for o in OPS) +
                "\tmask_E-O(pts)\tswap_E-O(pts)\tmask_DiD(pts)\tswap_DiD(pts)\n")
        for pid, pr in res["pairs"].items():
            b = pr["QC2_refusal"]["on_clean_answered_both"]
            if "rates" not in b:
                f.write("%s\t%s\t%d\tNA\n" % (pid, pr["machine"], b["n"])); continue
            for t in ("base", "arm"):
                cells = [pid, pr["machine"], str(b["n"]), t] + [f3(b["rates"][t][o]) for o in OPS] + \
                        [fe(b["mask"]["%s_E_minus_O" % t], 100), fe(b["swap"]["%s_E_minus_O" % t], 100)]
                if t == "arm":
                    cells += [fe(b["mask"]["DiD_arm_minus_base"], 100), fe(b["swap"]["DiD_arm_minus_base"], 100)]
                f.write("\t".join(cells) + "\n")
    # QJ
    with open(os.path.join(HERE, "qj_prior_strata_behav.tsv"), "w", encoding="utf-8") as f:
        f.write("pair\tmachine\tstratum_def\tlevel\tn\taxis\tR_clean_base\tR_clean_arm\tcontrast_base(pts)\tcontrast_arm(pts)\tDiD(pts)\tdShare(pts)\tinteraction_DiD(pts)\n")
        for pid, pr in res["pairs"].items():
            for sd in ("const", "blind", "ownblk"):
                blk = pr["QJ_prior_strata"][sd]
                for lab in ("prior_solvable", "not_prior_solvable"):
                    b = blk.get(lab, {})
                    for ax in ("mask", "swap"):
                        if ax not in b:
                            f.write("%s\t%s\t%s\t%s\t%s\t%s\tNA\n" % (pid, pr["machine"], sd, lab, b.get("n"), ax)); continue
                        a = b[ax]
                        f.write("\t".join([pid, pr["machine"], sd, lab, str(b["n"]), ax, f3(a["R@0.5"]["base"]["clean"]), f3(a["R@0.5"]["arm"]["clean"]),
                                           fe(a["contrast_base"], 100), fe(a["contrast_arm"], 100), fe(a["DiD_pts"]), fe(a["delta_share_pts"]),
                                           fe(blk.get("interaction_DiD_prior_minus_notprior_pts", {}).get(ax)) if lab == "prior_solvable" else ""]) + "\n")


if __name__ == "__main__":
    main()
