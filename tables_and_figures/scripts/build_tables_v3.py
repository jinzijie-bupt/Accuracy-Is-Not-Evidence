#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_tables_v3.py -- main-text tables under canonical_v3 (2026-09-23):
  T1_audit.tex    (tab:audit)    Six post-trained versions of Qwen2.5-VL-7B-Instruct (re-laid-out)
  T2_objectives.tex (tab:objectives) objective ablation (unchanged layout)
  T3_families.tex (tab:families) the placebo-controlled objective on six base models

Sources: the v2 registry through canonical_v3, plus, for the public-checkpoint rows of T1 only, the analysis
products named by the author: analysis/a1_public_rl_vs_qwen.json (TimeZero in A, Time-R1 in B) and
analysis/a6_vc_vs_qwen.json (VideoChat-R1 Gamma; a pair whose base and arm both ran on node5 is preferred).
VideoChat-R1 Delta: if results/native_eval_20260924/p4nat_qwenbase_vchat_tf5meta_node5_full.json exists, the
registered reader read_p4nat_selective_v1 is run in pair mode (ref = that file, arm = the node3 VideoChat-R1
run), its product is written next to the other registered pair products and read back; otherwise \\pending.
"""
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..", "..")).replace("\\", "/")
import hashlib, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from canonical_v3 import Canon, M, base_of, key_str, Q7, Q3, TZ, TR, VC, TL, PEND, CAPTION_NOTE  # noqa: E402
from table_emph_v3 import emph  # noqa: E402  (bold: the entries discussed in the text; located by row label and column)

ROOT = os.path.abspath(os.path.join(HERE, ".."))
T = os.path.join(ROOT, "tables")
LOGD = os.path.join(T, "_v3_logs")
ANA = os.path.join(_SUPP, "readers", "derived")
RES = _SUPP + "/results"
READER = _SUPP + "/readers/read_p4nat_selective_v1.py"
SEL = os.path.join(RES, "judgment_20260921", "selective")
QWEN_VCHAT_FULL = os.path.join(RES, "native_eval_20260924", "p4nat_qwenbase_vchat_tf5meta_node5_full.json")
VC_BASE_FULL = os.path.join(RES, "native_eval_20260918_node3", "p4nat_vcbase_vchat_tf5meta_node3_full.json")
CS = "charades-sta"
# round 7 : three Charades-STA R@0.5 changes had no pair_acc record in the registry, so delta3 fell back to
# the difference of marginals (dagger).  Base and model were run on the same 3,720 questions, videos and frames, so the
# pair products of the registered reader (read_pair: B=4000, seed 0, video clusters) replace them; the point estimates
# are unchanged (checked below), only the interval is added.  Products: round7/work/P1/p1_pair_products.py.
R7_PAIRS = os.path.join(RES, "judgment_20260926_r7", "pairs")
PAIR_ADD = {
    "Qwen2.5-VL-7B|iou-grpo|train=v|eval=v":
        "pair_p4nat_base_timer1_default_tf5meta_node1_full__p4nat_x0s0iou700_timer1_default_tf5meta_node1_full.json",
    "Qwen2.5-VL-7B|single-margin|a=0.5|b=0.1|s=0|train=v|eval=v":
        "pair_p4nat_base_timer1_default_tf5meta_node1_full__p4nat_natbase_a05_abl_single_s0_timer1_default_tf5meta_node1_full.json",
    "Time-R1-7B|grouped|a=0.5|b=0.1|s=0|train=v|eval=v":
        "pair_p4nat_tr1base_timer1_default_full_node3__p4nat_nattr1_a05_g1_s0_timer1_default_tf5meta_node3_full.json",
}


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


# ---------------------------------------------------------------- formatting (as v1/v2)
def fr(x):
    if x is None:
        return PEND
    if x >= 0.9995:
        return "1.00"
    return f"{x:.3f}"


def fpp(x, ci=None, dagger=False):
    if x is None:
        return PEND
    s = f"{100*x:+.1f}"
    if ci:
        s += f" [{100*ci[0]:.1f}, {100*ci[1]:.1f}]"
    if dagger:
        s += "$^\\dagger$"
    return s


def fg(x, ci=None):
    if x is None:
        return PEND
    s = "0.00" if abs(x) < 0.005 else f"{x:+.2f}"
    if ci:
        s += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    return s


def f2(x):
    return PEND if x is None else f"{0.0 if abs(x) < 0.005 else x:.2f}"


def gv(r, k):
    return None if r is None else r.get(k)


# ---------------------------------------------------------------- caption-note bookkeeping
class Track:
    """per table: environments of displayed base values and of paired Deltas, per (family, dataset)."""
    def __init__(self):
        self.base, self.delta = {}, {}

    def base_shown(self, fam, ds, env):
        if env:
            self.base.setdefault((fam, ds), set()).add(env)

    def delta_paired(self, fam, ds, env):
        if env:
            self.delta.setdefault((fam, ds), set()).add(env)

    def needs_note(self):
        return any(self.delta[k] - self.base.get(k, set()) for k in self.delta if k in self.base)


def base_cell(C, trk, kind, bk, ds, field, column):
    r = C.pick(kind, bk, ds, column=column)
    if r is not None:
        trk.base_shown(bk[0], ds, r["_env"]) if kind == "acc" else None
    return gv(r, field), r


def delta_cell(C, k, ds=CS, trk=None, srcs=None):
    v, ci, dag, env = C.delta3(k, ds)
    if dag and ds == CS and key_str(k) in PAIR_ADD:
        p = os.path.join(R7_PAIRS, PAIR_ADD[key_str(k)])
        dd = load_json(p)
        if abs(dd["point"]["delta_marginal"] - v) > 1e-9:
            raise SystemExit(f"PAIR_ADD {key_str(k)}: paired {dd['point']['delta_marginal']} != difference of marginals {v}")
        v, ci, dag = dd["point"]["delta_marginal"], dd["delta_marginal_boot"]["ci95"], False
        if srcs is not None:
            srcs.append(f"{PAIR_ADD[key_str(k)]} {sha16(p)}")
    if trk is not None and v is not None:
        trk.delta_paired(k[0], ds, env)
    return fpp(v, ci, dagger=dag)


def add_note(caption, trk):
    return caption


# ---------------------------------------------------------------- analysis products for public rows
def load_json(p):
    return json.load(open(p, encoding="utf-8"))


def a1_rows():
    p = os.path.join(ANA, "a1_public_rl_vs_qwen.json")
    d = load_json(p)
    g = {(x["label"], x["env"]): x for x in d["gamma_pairs"]}
    f = {(x["label"], x["env"], x["order"]): x for x in d["full_set_pairs"]}
    return p, g, f


def a6_pair():
    p = os.path.join(ANA, "a6_vc_vs_qwen.json")
    if not os.path.exists(p):
        return None, None
    d = load_json(p)
    node5 = [x for x in d["pairs"] if x.get("node") == "node5" or ("_node5_" in x.get("base", "") and "_node5_" in x.get("arm", ""))]
    return p, (node5[0] if node5 else d["pairs"][0])


def vc_delta_product():
    """pair product of the registered reader for VideoChat-R1 (vchat) vs Qwen base (vchat), both tf5.12."""
    if not os.path.exists(QWEN_VCHAT_FULL):
        return None
    out = os.path.join(SEL, "pair_p4nat_qwenbase_vchat_tf5meta_node5_full__p4nat_vcbase_vchat_tf5meta_node3_full.json")
    if not os.path.exists(out):
        subprocess.run([sys.executable, READER, "--label", os.path.basename(out)[:-5], "--pair", QWEN_VCHAT_FULL,
                        VC_BASE_FULL, "--B", "4000", "--seed", "0", "--note",
                        "v3 T1: VideoChat-R1 public checkpoint vs Qwen2.5-VL-7B base, vchat template, env B (node5/node3)",
                        "--out", out], check=True)
    return out


# ---------------------------------------------------------------- behavioural share (review round)
P5N = os.path.join(RES, "judgment_20260921", "p5n_v2")        # trained arms against their own base (also feed S6)
P5Q = os.path.join(RES, "judgment_20260921", "p5n_vsqwen")    # public checkpoints against the Qwen base (T1 only)
BEHAV = {   # row -> registered read_p5n_v2 product (base and arm run on one machine with identical frames and donors)
    #: the six-operator behavioural battery (S6, Fig. 5c) exists only for the key-converted runs of SFT and
    # a=0.5, so their share stays there (the node4 runs hold the mask operator only: judgment_20260925_native/)
    "sft": (P5N, "read_p5n_v2_node5_sft.json"), "grpo": (P5N, "read_p5n_v2_node1_grpo.json"), "mdpo": (P5N, "read_p5n_v2_node5_mdpo.json"),
    "tz": (P5Q, "read_p5n_v2_node5_qwen_vs_tzbasepub_vf.json"),   # video-first TimeZero run (tzbasepub.json paired a text-first run)
    "tr1": (P5Q, "read_p5n_v2_node5_qwen_vs_tr1basepub.json"),
    "vc": (P5Q, "read_p5n_v2_node5_qwenvchat_vs_vcbasepub.json"), "pc": (P5N, "read_p5n_v2_node5_qwen.json"),
    "base_level": (P5N, "read_p5n_v2_node5_qwen.json")}   # base row level only (the base run shared by the node5 rows)


def behav_cell(C, key, srcs, op="mask"):
    """change in the share of placebo-condition correct answers lost when the evidence frames are edited, trained minus base (pp)."""
    p = os.path.join(*BEHAV[key])
    if not os.path.exists(p):
        C.log_pending(f"behav share {key}", "behav", None, CS, None, "read_p5n_v2 output not on disk: " + BEHAV[key][1])
        return PEND, None
    d = load_json(p)
    x = d["all_questions"]["paired"][op]["normalised_contrast_diff"]
    srcs.append(f"{BEHAV[key][1]} {sha16(p)}")
    base_share = d["all_questions"]["paired"][op]["base_contrast"]["est"] / d["all_questions"]["base"]["conditions"][f"{op}_rand"]["R@0.5"]["est"]
    return fpp(x["est"], x["ci95"]), base_share


# ---------------------------------------------------------------- stacked intervals (main-text tables, round 2)
# The main-text tables print a 95% interval under its point estimate in a smaller grey face.  The macro is defined in
# the table file itself (\providecommand), so main.tex does not need it.
EST_DEF = (r"\providecommand{\est}[3]{\begin{tabular}[t]{@{}c@{}}#1\\[-1.5pt]"
           r"{\fontsize{6}{7}\selectfont\color{black!55}[#2,\,#3]}\end{tabular}}")
_CELL_CI = re.compile(r"^(\s*)(\\textcolor\{black!45\}\{)?(.*?)\s\[\s*([^\[\],]+?)\s*,\s*([^\[\],]+?)\s*\](\})?(\s*)$", re.S)


def stack_ci(text):
    """after emph(): every data cell "est [lo, hi]" (bold or grey as emph left it) becomes \\est{est}{lo}{hi}."""
    out = []
    for ln in text.split("\n"):
        s = ln.strip()
        if s.startswith("%") or "&" not in s or not s.endswith("\\\\"):
            out.append(ln)
            continue
        body, end = re.match(r"^(.*?)(\s*\\\\\s*)$", ln).groups()
        pieces = re.split(r"(?<!\\)&", body)
        for k, c in enumerate(pieces):
            m = _CELL_CI.match(c)
            if not m:
                continue
            lead, grey, est, lo, hi, close, trail = m.groups()
            if bool(grey) != bool(close):
                continue
            e = "\\textcolor{black!45}{" + est + "}" if grey else est
            pieces[k] = f"{lead}\\est{{{e}}}{{{lo}}}{{{hi}}}{trail}"
        out.append("&".join(pieces) + end)
    return "\n".join(out)


R3 = os.path.join(RES, "judgment_20260924_r3")   # moved-event readings (read_r3), base and model run on one machine
FOLLOW = {"sft": "read_r3_node3_sft_tf5.json", "grpo": "read_r3_node1_grpo.json",   # grpo: node1, trained and read there (2026-09-25)
          "mdpo": "read_r3_node5_mdpo.json",   # mdpo: base and model on node5, the machine of its behavioural run (2026-09-25)
          "tz": "read_r3_node3_tzpub.json", "tr1": "read_r3_node3_tr1pub.json",
          "vc": "read_r3_node3_vcpub.json", "pc": "read_r3_node3_pc_a05_tf5.json"}


def follows_cell(key, srcs):
    """change in how often the prediction follows the event after its frames are moved, model minus base (pp)."""
    if key not in FOLLOW:
        return "\\NA"
    p = os.path.join(R3, FOLLOW[key])
    if not os.path.exists(p):
        return PEND
    v = load_json(p)["paired"]["Gamma_reloc_follow_iou"]
    srcs.append(f"{FOLLOW[key]} {sha16(p)}")
    return fpp(v["est"], v["ci95"])


# ---------------------------------------------------------------- T1
def build_T1(C, out):
    """Summary table: what each way of post-training does to Qwen2.5-VL-7B-Instruct (review round 2, 2026-09-24).
    The base row gives levels; every other row gives changes paired against the base run of the same evaluation, so
    no absolute R@0.5 of a trained model is printed (its base run can differ from the one shown in the base row)."""
    srcs = []
    cap = ("\\caption{\\textbf{Training for accuracy adds correct answers but not evidence dependence; the placebo-controlled "
           "objective adds both.} All models start from Qwen2.5-VL-7B-Instruct. The base row gives levels. Every other entry is "
           "a change from the base on the same questions under the same template, except $\\dO$, which is each model's own placebo drop. "
           "Accuracy changes are in points, and brackets give 95\\% intervals.}")
    rows = []

    def trained(label, k, bkey):
        C.ctx("T1", f"{label} / {key_str(k)}")
        d = delta_cell(C, k, CS, srcs=srcs)
        gam = C.pick("gamma", k, CS, column="Gamma", instrument="14Ln")
        sh, _ = behav_cell(C, bkey, srcs)
        rows.append([label, d, fg(gv(gam, "gamma_mask"), gv(gam, "gamma_mask_ci")), fg(gv(gam, "gamma_swap")),
                     f2(gv(gam, "dO_mask_arm")), sh, follows_cell(bkey, srcs)])

    qb = M(Q7, "base")
    pck = M(Q7, "grouped", 0.5, 0.1, 0)
    C.ctx("T1", "Qwen base")
    r50, _ = base_cell(C, Track(), "acc", qb, CS, "R50", "R50")
    _, base_share = behav_cell(C, "base_level", [])
    # 09-25: the base dO comes from the same pair as the alpha=0.5 row (canonical env, training stack), as in T3/S7
    gpc = C.pick("gamma", pck, CS, column="base dO (base of the placebo-controlled pair)", instrument="14Ln")
    rows.append(["Qwen2.5-VL-7B (base)", fr(r50), "\\NA", "\\NA", f2(gv(gpc, "dO_mask_base")),
                 f"{100*base_share:.0f}\\%" if base_share else PEND, "\\NA"])
    rows.append("\\addlinespace[2pt]\\multicolumn{7}{@{}l}{\\emph{Trained for accuracy}} \\\\")
    trained("\\quad SFT", M(Q7, "sft", None, 0.0, 0), "sft")
    trained("\\quad IoU-GRPO", M(Q7, "iou-grpo"), "grpo")
    trained("\\quad mDPO-style", M(Q7, "mdpo", 0.5, None, 0), "mdpo")
    p1, g1, f1 = a1_rows()
    srcs.append(f"{os.path.basename(p1)} {sha16(p1)}")
    for label, fam, lab_a1, env, bkey in (("\\quad TimeZero (public)", TZ, "TimeZero vs Qwen", "A", "tz"),
                                          ("\\quad Time-R1 (public)", TR, "Time-R1 vs Qwen", "B", "tr1")):
        C.ctx("T1", label)
        fs = f1.get((lab_a1, env, "video-first"))
        gp = g1.get((lab_a1, env))
        sh, _ = behav_cell(C, bkey, srcs)
        rows.append([label, fpp(fs["delta_R50"], fs["delta_R50_ci95"]) if fs else PEND,
                     fg(gp["mask"]["gamma"], gp["mask"]["gamma_ci95"]) if gp else PEND, fg(gp["swap"]["gamma"]) if gp else PEND,
                     f2(gp["mask"]["dO_arm"]) if gp else PEND, sh, follows_cell(bkey, srcs)])
    C.ctx("T1", "VideoChat-R1 (public)")
    p6, g6 = a6_pair()
    if p6:
        srcs.append(f"{os.path.basename(p6)} {sha16(p6)} ({os.path.basename(g6['base'])} vs {os.path.basename(g6['arm'])})")
    vcp = vc_delta_product()
    if vcp:
        dd = load_json(vcp)
        vcd = fpp(dd["point"]["delta_marginal"], dd["delta_marginal_boot"]["ci95"])
        srcs.append(f"{os.path.basename(vcp)} {sha16(vcp)}")
    else:
        vcd = PEND
        C.log_pending("Delta vs Qwen (vchat)", "pair_acc", M(VC, "base"), CS, "B", "Qwen base vchat full run not on disk")
    sh, _ = behav_cell(C, "vc", srcs)
    rows.append(["\\quad VideoChat-R1 (public)", vcd,
                 fg(g6["mask"]["gamma"], g6["mask"]["gamma_ci95"]) if p6 else PEND, fg(g6["swap"]["gamma"]) if p6 else PEND,
                 f2(g6["mask"]["dO_arm"]) if p6 else PEND, sh, follows_cell("vc", srcs)])
    rows.append("\\addlinespace[2pt]\\multicolumn{7}{@{}l}{\\emph{Placebo-controlled objective (ours)}} \\\\")
    trained("\\quad $\\alpha{=}0.5$", pck, "pc")
    note = ("{\\scriptsize Changes in points; brackets: 95\\% intervals.\\par}")
    L = ["% T1 -- generated by scripts/build_tables_v3.py (canonical_v3); do not edit by hand",
         EST_DEF,
         "\\begin{table}[tb]", "\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.12}",
         cap, "\\label{tab:audit}",
         "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
         " & Accuracy & \\multicolumn{3}{c}{\\shortstack{Evidence dependence\\\\(nats per token)}} & "
         "\\multicolumn{2}{c@{}}{\\shortstack{Regenerated answers\\\\(points)}} \\\\",
         "\\cmidrule(lr){2-2}\\cmidrule(lr){3-5}\\cmidrule(l){6-7}",
         "Model & \\Rfive{} & \\shortstack{$\\Gamma$\\\\mask} & \\shortstack{$\\Gamma$\\\\swap} & $\\dO$ & \\shortstack{evidence\\\\share} & \\shortstack{follows\\\\moved event} \\\\",
         "\\midrule"]
    for r in rows:
        L.append(r if isinstance(r, str) else " & ".join(r) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", "% analysis sources: " + "; ".join(srcs)]
    txt = stack_ci(emph("\n".join(L) + "\n", os.path.basename(out)[:-4]))
    open(out, "w", encoding="utf-8", newline="\n").write(txt)


def build_T1_v3_old(C, out):
    trk = Track()
    srcs = []
    cap = ("\\caption{\\textbf{Post-training for accuracy raises \\Rfive{} by 10 to 20 points and leaves evidence dependence where it was.} "
           "Each row is a post-trained version of Qwen2.5-VL-7B-Instruct, audited against that base "
           "run under the same prompt template: Time-R1's for every row except VideoChat-R1, which uses its own. \\Rfive{} on the full 3{,}720-question Charades-STA test split, unparsed outputs as misses; "
           "$\\Delta$\\Rfive{}: paired change against the base with a 95\\% video-cluster interval ($^\\dagger$: difference of marginals); "
           "$\\Gm$, $\\Gs$: paired evidence dependence in nats per token on the $n_\\cap$ questions both localise.}")
    L = []
    rows = []

    def trained(label, data, k, black=True):
        C.ctx("T1", f"{label} / {key_str(k)}")
        acc = C.pick("acc", k, CS, column="R50")
        d = delta_cell(C, k, CS, trk)
        bl = C.pick("black", k, CS, column="black ans.") if black else None
        gam = C.pick("gamma", k, CS, column="Gamma", instrument="14Ln")
        rows.append([label, data, fr(gv(acc, "R50")), d,
                     fg(gv(gam, "gamma_mask"), gv(gam, "gamma_mask_ci")), fg(gv(gam, "gamma_swap")),
                     PEND if gam is None else str(gam["n_H_int"])])

    # base row
    qb = M(Q7, "base")
    C.ctx("T1", "Qwen base")
    r50, _ = base_cell(C, trk, "acc", qb, CS, "R50", "R50")
    bl, _ = base_cell(C, trk, "black", qb, CS, "black_answer_rate", "black ans.")
    rows.append(["Qwen2.5-VL-7B-Instruct (base)", "--", fr(r50), "\\NA", "0 (def.)", "\\NA", "\\NA"])
    trained("\\quad pure SFT ($\\beta{\\to}0$)", "ours", M(Q7, "sft", None, 0.0, 0))
    trained("\\quad IoU-reward GRPO", "ours", M(Q7, "iou-grpo"))
    # public checkpoints (model = that family's base configuration; Delta/Gamma against the Qwen base, same env + template)
    p1, g1, f1 = a1_rows()
    srcs.append(f"{os.path.basename(p1)} {sha16(p1)}")
    for label, fam, lab_a1, env, black in (("\\quad TimeZero (public)", TZ, "TimeZero vs Qwen", "A", True),
                                           ("\\quad Time-R1 (public)", TR, "Time-R1 vs Qwen", "B", True)):
        C.ctx("T1", label)
        k = M(fam, "base")
        acc = C.pick("acc", k, CS, column="R50 (global value of this configuration)")
        blr = C.pick("black", k, CS, column="black ans.")
        fs = f1.get((lab_a1, env, "video-first"))
        gp = g1.get((lab_a1, env))
        if fs:
            trk.delta_paired(Q7, CS, env)
        rows.append([label, "public", fr(gv(acc, "R50")),
                     fpp(fs["delta_R50"], fs["delta_R50_ci95"]) if fs else PEND,
                     fg(gp["mask"]["gamma"], gp["mask"]["gamma_ci95"]) if gp else PEND,
                     fg(gp["swap"]["gamma"]) if gp else PEND, str(gp["n_H_int"]) if gp else PEND])
    C.ctx("T1", "VideoChat-R1 (public)")
    acc = C.pick("acc", M(VC, "base"), CS, column="R50")
    p6, g6 = a6_pair()
    if p6:
        srcs.append(f"{os.path.basename(p6)} {sha16(p6)} ({os.path.basename(g6['base'])} vs {os.path.basename(g6['arm'])})")
    vcp = vc_delta_product()
    if vcp:
        dd = load_json(vcp)
        vcd = fpp(dd["point"]["delta_marginal"], dd["delta_marginal_boot"]["ci95"])
        trk.delta_paired(Q7, CS, "B")
        srcs.append(f"{os.path.basename(vcp)} {sha16(vcp)}")
    else:
        vcd = PEND
        C.log_pending("Delta vs Qwen (vchat)", "pair_acc", M(VC, "base"), CS, "B",
                      "Qwen base vchat full run not on disk (" + os.path.basename(QWEN_VCHAT_FULL) + ")")
    if not p6:
        C.log_pending("Gamma vs Qwen (vchat)", "gamma", M(VC, "base"), CS, "B", "a6 output not on disk")
    rows.append(["\\quad VideoChat-R1 (public)", "public", fr(gv(acc, "R50")), vcd,
                 fg(g6["mask"]["gamma"], g6["mask"]["gamma_ci95"]) if p6 else PEND,
                 fg(g6["swap"]["gamma"]) if p6 else PEND, str(g6["n_H_int"]) if p6 else PEND])
    #: the placebo-controlled and mDPO-style rows live in Table 2 / Figure 1 only (one number, one place)
    L = ["% T1 -- generated by scripts/build_tables_v3.py (canonical_v3); do not edit by hand",
         "\\begin{table}[tb]", "\\centering\\scriptsize\\setlength{\\tabcolsep}{2.4pt}\\renewcommand{\\arraystretch}{1.03}",
         add_note(cap, trk), "\\label{tab:audit}",
         "\\begin{tabular}{@{}llccccc@{}}", "\\toprule",
         "Model & Trained by & \\Rfive & $\\Delta$\\Rfive{} (pp, 95\\% CI) & $\\Gm$ [CI] & $\\Gs$ & $n_\\cap$ \\\\",
         "\\midrule"]
    for r in rows:
        L.append(" & ".join(r) + " \\\\")
    L += ["\\midrule", "Constant window $(0,10)$~s, no model & -- & 0.317 & \\NA & \\NA & \\NA & \\NA \\\\",
          "\\bottomrule", "\\end{tabular}", "\\end{table}", "% analysis sources: " + "; ".join(srcs)]
    open(out, "w", encoding="utf-8", newline="\n").write(emph("\n".join(L) + "\n", os.path.basename(out)[:-4]))


# ---------------------------------------------------------------- T2 (round 2: four ablation rows, d_O column)
def build_T2(C, out):
    """Objective ablation next to Figure 3 (inside a minipage).  The full objective is the last row of Table 1 and is
    not repeated; d_O is the trained model's own placebo drop under masking (the gamma record's dO_mask_arm)."""
    L = ["% T2 objective ablation -- generated by scripts/build_tables_v3.py (canonical_v3); do not edit by hand",
         EST_DEF,
         "\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.12}",
         "\\captionof{table}{\\textbf{With the evidence margin alone, $\\Gamma$ is a third of what the full objective reaches.} "
         "Variants of the full objective (last row of Table~\\ref{tab:audit}) are trained from Qwen2.5-VL-7B at $\\alpha{=}0.5$; "
         "\\emph{uniform} variants drop the Group-DRO weights. $\\dO$ and bold follow Table~\\ref{tab:audit}.}",
         "\\label{tab:objectives}",
         "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
         " & points & \\multicolumn{3}{c@{}}{nats per token} \\\\",
         "\\cmidrule(lr){2-2}\\cmidrule(l){3-5}",
         "Objective & $\\Delta$\\Rfive{} & \\shortstack{$\\Gamma$\\\\mask} & \\shortstack{$\\Gamma$\\\\swap} & $\\dO$ \\\\", "\\midrule"]
    for name, k in [("uniform weights", M(Q7, "grouped-nodro", 0.5, 0.1, 0)),
                    ("no placebo bound", M(Q7, "grouped-nohinge", 0.5, 0.1, 0)),
                    ("no bound, uniform", M(Q7, "grouped-uniform-nohinge", 0.5, 0.1, 0)),
                    ("evidence margin only", M(Q7, "single-margin", 0.5, 0.1, 0))]:
        C.ctx("T2", f"{name} / {key_str(k)}")
        d = delta_cell(C, k, CS)
        gam = C.pick("gamma", k, CS, column="Gamma", instrument="14Ln")
        L.append(f"{name} & {d} & {fg(gv(gam, 'gamma_mask'), gv(gam, 'gamma_mask_ci'))} & {fg(gv(gam, 'gamma_swap'))} & "
                 f"{f2(gv(gam, 'dO_mask_arm'))} \\\\")
    L.append("\\bottomrule\n\\end{tabular}")
    txt = stack_ci(emph("\n".join(L) + "\n", os.path.basename(out)[:-4]))
    open(out, "w", encoding="utf-8", newline="\n").write(txt)


# ---------------------------------------------------------------- T3 families
def build_T3(C, out):
    trk = Track()
    rows = [("Qwen2.5-VL-7B", M(Q7, "grouped", 0.5, 0.1, 0)),
            ("TimeZero", M(TZ, "grouped", 0.5, 0.1, 0)), ("Time-R1", M(TR, "grouped", 0.5, 0.1, 0)),
            ("VideoChat-R1", M(VC, "grouped", 0.5, 0.1, 0)), ("TimeLens-8B", M(TL, "grouped", 0.5, 0.1, 0))]
    body = []
    for name, k in rows:
        C.ctx("T3", f"{name} / {key_str(k)}")
        bk = base_of(k)
        r50, _ = base_cell(C, trk, "acc", bk, CS, "R50", "base R50")
        d = delta_cell(C, k, CS, trk)
        gam = C.pick("gamma", k, CS, column="Gamma", instrument="14Ln")
        do = PEND if gam is None else f"{f2(gam['dO_mask_arm'])} / {f2(gam['dO_mask_base'])}"
        body.append(" & ".join([name, fr(r50), d, fg(gv(gam, "gamma_mask"), gv(gam, "gamma_mask_ci")), fg(gv(gam, "gamma_swap")), do]) + " \\\\")
    cap = ("\\caption{\\textbf{The placebo-controlled objective raises evidence dependence in every base model; the change in "
           "accuracy varies with the base.} Each base model is paired with its adapter, trained at $\\alpha{=}0.5$ on the same "
           "2{,}400 Charades-STA pairs and run under the base's own template. Changes are in points, $\\Gamma$ and $\\dO$ are in "
           "nats per token, and $\\dO$ is given for the adapter and for its base; brackets give 95\\% intervals.}")
    L = ["% T3 -- generated by scripts/build_tables_v3.py (canonical_v3); do not edit by hand",
         "\\begin{table}[tb]", "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.05}",
         cap, "\\label{tab:families}", "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
         "Base model & base \\Rfive & $\\Delta$\\Rfive{} & \\shortstack{$\\Gamma$\\\\mask} & \\shortstack{$\\Gamma$\\\\swap} & $\\dO$ trained / base \\\\",
         "\\midrule"] + body + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(out, "w", encoding="utf-8", newline="\n").write(emph("\n".join(L) + "\n", os.path.basename(out)[:-4]))


def main():
    C = Canon()
    build_T1(C, os.path.join(T, "T1_audit.tex"))
    build_T2(C, os.path.join(T, "T2_objectives.tex"))
    build_T3(C, os.path.join(T, "T3_families.tex"))
    os.makedirs(LOGD, exist_ok=True)
    json.dump({"sources": C.sources, "pending": C.pending}, open(os.path.join(LOGD, "build_tables_v3.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    json.dump(C.sources, open(os.path.join(T, "TABLE_SOURCES_v3.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"T1/T2/T3 written; {len(C.sources)} sourced; {len(C.pending)} pending logged")


if __name__ == "__main__":
    main()
