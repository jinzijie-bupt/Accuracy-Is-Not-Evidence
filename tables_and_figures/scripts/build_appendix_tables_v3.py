#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_appendix_tables_v3.py -- appendix S1-S9 under canonical_v3 (2026-09-23; copy of the v2 builder).
v3: environments per (configuration, kind, dataset) from canonical_v3.env_for; stack by pairing completeness;
S1/S8 captions get the same-environment Delta sentence when a displayed base value and a paired Delta come
from different environments; S9 base "as pred." R@0.5 is the registered selective reading.

v2 docstring:

Differences from build_appendix_tables_v1.py:
  * no environment appears in any selector; every configuration is read in its canonical environment
    (canonical_v2), one row per configuration;
  * v1's cross-run merging is gone: S1 no longer fills a row's empty cells from a replicate run in
    another environment, and S3/S6 no longer keep "the most complete / larger reading" replicate
    (dedup_best); a configuration whose canonical stack lacks the product prints \\pending;
  * S5: the TL-label full-split column and the reproducibility column (ident.) are printed only when the
    TL run sits in the configuration's canonical stack (v1 compared an node2/node4 STA run with a node5 TL run);
  * trained message order and evaluated message order are separate parts of the configuration key.
Outputs tables/S1..S9 (.tex), tables/APPENDIX_SOURCES_v2.json, tables/_v2_logs/build_appendix_tables_v2.json.
"""
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..", "..")).replace("\\", "/")
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from canonical_v3 import Canon, M, base_of, key_str, rec_name, Q7, Q3, TZ, TR, VC, TL, PEND  # noqa: E402
from canonical_v2 import ENV_OF_MACHINE  # noqa: E402
import build_tables_v3 as _bt3  # noqa: E402
from build_tables_v3 import fr, fpp, fg, gv, Track, add_note  # noqa: E402
TRK = Track()


def delta_cell(C, k, ds="charades-sta"):
    return _bt3.delta_cell(C, k, ds, TRK)


def dans_cell(C, k, ds="charades-sta"):
    """round 4: trained rows of S1's ans. column give the paired change in R@0.5 on the questions the trained model
    answers (the pair record C.delta3 uses: delta_conditional_paired); without a pair record, the difference of the two
    runs' R@0.5 on answered questions inside the arm's environment, marked with a dagger (as delta3's else branch)."""
    env = C.env_for(k, "pair_acc", ds)
    out = [r for r in C.R if r.get("kind") == "pair_acc" and r["_key"] == k and r["_env"] == env
           and r["_ref_env"] == env and r.get("dataset") == ds and r.get("_ref_key") == base_of(k)]
    if out and out[0].get("delta_conditional_paired") is not None:
        return fpp(out[0]["delta_conditional_paired"])
    ea = C.env_for(k, "acc", ds)
    a = C.pick("acc", k, ds, column="Dans:arm", pending=False)
    b = C.pick("acc", base_of(k), ds, column="Dans:base (same env)", env=ea, pending=False)
    if a and b and a.get("R50_answered") is not None and b.get("R50_answered") is not None:
        return fpp(a["R50_answered"] - b["R50_answered"], dagger=True)
    C.log_pending("ans.", "acc", k, ds, ea, "no paired or same-env conditional reading")
    return PEND


def note_caption(L):
    for i, l in enumerate(L):
        if l.startswith("\\caption{"):
            L[i] = add_note(l, TRK)
    return L


ROOT = os.path.abspath(os.path.join(HERE, ".."))
T = os.path.join(ROOT, "tables")
LOGD = os.path.join(T, "_v3_logs")
REF_SEEDS = (None, 0)


def f2(x):
    return PEND if x is None else f"{x:.2f}"


def fint(x):
    return PEND if x is None else str(int(x))


def write(lines, out):
    from table_emph_v3 import emph   # bold: the entries discussed in the text (located by row label and column)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(emph("\n".join(lines) + "\n", os.path.basename(out)[:-4]))
    print("wrote", os.path.relpath(out, ROOT))


FAM = {Q7: "Qwen2.5-VL-7B", Q3: "Qwen2.5-VL-3B", TZ: "TimeZero", TR: "Time-R1", VC: "VideoChat-R1", TL: "TimeLens-8B"}
FAM_ORDER = list(FAM)
OBJ = {
    "base": "base",
    "sft": "SFT",                     # round 2 (2026-09-24): one printed name per model across all tables
    "iou-grpo": "IoU-GRPO",
    "mdpo": "mDPO-style",
    "grouped-nodro": "grouped, no group wts",
    "grouped-nohinge": "grouped, no hinge",
    "grouped-uniform-nohinge": "grouped, no hinge, no group wts",
    "single-margin": "evidence margin only",
    "grouped-4800": "grouped $\\alpha{=}0.5$, 4,800 pairs",
    "grouped-full-budget": "grouped $\\alpha{=}0.5$, 8,434 pairs, trained on 16 frames",
    "grouped-k16": "grouped $\\alpha{=}0.5$, 2,400 pairs, trained on 16 frames",
    "two-stage": "two-stage (SFT then grouped)",
}


def obj_label(k):
    fam, o, a, b, s, tr, ev = k
    if o == "grouped":
        s_ = "grouped $\\alpha{=}%s$" % f"{a:g}"
        if b is not None and abs(b - 0.1) > 1e-9:
            s_ += ", $\\beta{=}%g$" % b
        return s_
    if o == "two-stage":     # two Qwen two-stage adapters exist (alpha 0.5 / beta 0.03); v1 S7 showed one of them only
        s_ = OBJ[o]
        if a is not None:
            s_ += ", $\\alpha{=}%g$" % a
        if b is not None and abs(b - 0.1) > 1e-9:
            s_ += ", $\\beta{=}%g$" % b
        return s_
    return OBJ.get(o, o)


def row_label(k, with_family=True):
    s = (FAM.get(k[0], k[0]) + ", ") if with_family else ""
    s += obj_label(k)
    tr, ev = k[5], k[6]
    if tr == "t" and ev == "v":
        s += ", trained text-first, evaluated video-first"
    elif tr == "v" and ev == "t" and k[1] != "base":
        s += ", trained video-first, evaluated text-first"
    elif tr == "t" or ev == "t":
        s += ", text-first"
    return s


def sort_key(k, ds=None, dsorder=None):
    return ((dsorder.index(ds) if dsorder else 0), FAM_ORDER.index(k[0]), k[1] != "base", k[1],
            str(k[2]), str(k[3]), k[5], k[6])


def head(title, ncol):
    return "\\multicolumn{%d}{@{}l}{\\textit{%s}} \\\\" % (ncol, title)


CS = "charades-sta"
G = dict(beta=0.1, seed=0)


def g(fam, a, b=0.1, **kw):
    return M(fam, "grouped", a, b, 0, **kw)


# ---------------------------------------------------------------- S1
def s1_spec():
    return [
        ("HEAD", "Qwen2.5-VL-7B"),
        M(Q7, "base"), g(Q7, 1.0), g(Q7, 0.5), g(Q7, 0.5, 0.03), g(Q7, 0.25), g(Q7, 1.0, 0.03),
        M(Q7, "sft", None, 0.0, 0), M(Q7, "iou-grpo"), M(Q7, "mdpo", 0.5, None, 0),
        ("HEAD", "Qwen2.5-VL-7B: objective ablation at $\\alpha{=}0.5$"),
        M(Q7, "grouped-nodro", 0.5, 0.1, 0), M(Q7, "grouped-nohinge", 0.5, 0.1, 0),
        M(Q7, "grouped-uniform-nohinge", 0.5, 0.1, 0), M(Q7, "single-margin", 0.5, 0.1, 0),
        ("HEAD", "Qwen2.5-VL-7B: training budget"),
        M(Q7, "grouped-4800", 0.5, 0.1, 0), M(Q7, "grouped-k16", 0.5, 0.1, 0), M(Q7, "grouped-full-budget", 0.5, 0.1, 0),
        ("HEAD", "TimeZero"),
        M(TZ, "base"), M(TZ, "base", eval="t"), g(TZ, 0.5), g(TZ, 0.25), g(TZ, 0.5, train="t", eval="t"),
        g(TZ, 1.0), g(TZ, 0.5, 0.03),
        ("HEAD", "Time-R1"),
        M(TR, "base"), g(TR, 0.25), g(TR, 0.5), g(TR, 0.1), g(TR, 1.0),
        ("HEAD", "VideoChat-R1"),
        M(VC, "base"), g(VC, 0.5), g(VC, 0.25), g(VC, 1.0),
        ("HEAD", "TimeLens-8B"),
        M(TL, "base"), g(TL, 0.25), g(TL, 0.5), g(TL, 1.0),
    ]


SEL_DIR = _SUPP + "/results/judgment_20260921/selective"
SEL_READER = _SUPP + "/readers/read_p4nat_selective_v1.py"
TZ_TF_node3 = _SUPP + "/results/native_eval_20260918_node3/p4nat_tzbase_timer1_default_textfirst_tf5meta_node3_full.json"


def tz_textfirst_base_acc(C):
    """TimeZero base, text-first: the run Table S13 pairs against the video-first run (C7, round 2).  S1 prints the
    registered selective reading of that same run, so the two tables show one value."""
    lab = os.path.basename(TZ_TF_node3)[:-5]
    out = os.path.join(SEL_DIR, f"read_selective_{lab}.json")
    if not os.path.exists(out):
        import subprocess
        subprocess.run([sys.executable, SEL_READER, "--label", lab, "--files", TZ_TF_node3, "--expect-n", "3720",
                        "--B", "2000", "--note", "v3 S1: TimeZero base text-first, the run of Table S13", "--out", out], check=True)
    d = json.load(open(out, encoding="utf-8"))
    import hashlib
    C.sources.append({"table": "S1", "row": "TimeZero base, text-first", "column": "acc", "kind": "acc",
                      "src_file": os.path.basename(out), "sha16": hashlib.sha256(open(out, "rb").read()).hexdigest()[:16],
                      "run": os.path.basename(TZ_TF_node3)})
    return {"R50": d["marginal"]["R@0.5"], "parse_rate": d["marginal"]["parse_rate"], "coverage": d["coverage"],
            "R50_answered": d["conditional"]["R@0.5"]}


def build_S1(C, out):
    """round 2 (2026-09-24): base rows give R@0.5, trained rows only its paired change (their base run can differ from
    the base row's run, so an absolute R@0.5 next to it would not subtract to the change); parse is dropped when it
    equals coverage on every row."""
    global TRK
    TRK = Track()
    rows = []
    for e in s1_spec():
        if isinstance(e, tuple) and e[0] == "HEAD":
            rows.append(e[1])
            continue
        k = e
        lab = row_label(k, with_family=False)
        C.ctx("S1", f"{FAM[k[0]]} / {lab} / env {C.env(k)}")
        acc = tz_textfirst_base_acc(C) if k == M(TZ, "base", eval="t") else C.pick("acc", k, CS, column="acc")
        if k[1] == "base":
            r, gm, gs, nn = fr(gv(acc, "R50")), "\\NA", "\\NA", "\\NA"
        else:
            r = delta_cell(C, k, CS)
            gam = C.pick("gamma", k, CS, column="gamma", instrument="14Ln")
            gm, gs, nn = fg(gv(gam, "gamma_mask")), fg(gv(gam, "gamma_swap")), fint(gv(gam, "n_H_int"))
        if acc is not None:
            for fld, col in (("coverage", "cov."), ("R50_answered", "ans.")):
                if acc.get(fld) is None:
                    C.log_pending(col, "acc", k, CS, C.env(k), "this acc reader has no refusal/coverage fields")
        ans = fr(gv(acc, "R50_answered")) if k[1] == "base" else dans_cell(C, k, CS)
        rows.append([lab, k[6], r, fr(gv(acc, "parse_rate")), fr(gv(acc, "coverage")), ans, gm, gs, nn])
    rows.append(None)
    rows.append(["constant window $(0,10)$\\,s, no model", "\\NA", "0.317", "1.00", "1.00", "0.317", "\\NA", "\\NA", "\\NA"])
    same = all(r[3] == r[4] for r in rows if isinstance(r, list))
    keep = [0, 1, 2, 4, 5, 6, 7, 8] if same else list(range(9))
    NC = len(keep)
    print(f"   S1: parse {'equals coverage on every row: column dropped' if same else 'differs from coverage: kept'}")
    hdr = ["Model / objective", "ord.", "\\Rfive{} / $\\Delta$", "parse", "cov.", "ans. / $\\Delta$", "$\\Gm$", "$\\Gs$", "$n_\\cap$"]
    L = ["% S1 -- generated by scripts/build_appendix_tables_v3.py (canonical_v3); do not edit by hand",
         "\\begin{table}[tbp]",
         "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.05}",
         "\\caption{\\textbf{The full audit matrix on Charades-STA compares every trained configuration with the base it was "
         "trained from.} Each block of rows belongs to one base model. A base row gives its \\Rfive{} and, in the column "
         "\\emph{ans.}, its \\Rfive{} on the questions it answers; a trained row gives the paired changes of these two values "
         "from its base in points, the second on the questions the trained model answers. A $^\\dagger$ marks a change that is "
         "not paired by question but is the difference of the two models' values. The column \\emph{ord.} gives the message "
         "order, video first (v) or query text first (t); \\emph{cov.} is the coverage, $1-$strict abstention; $\\Gm$ and $\\Gs$ "
         "are in nats per token on the $n_\\cap$ questions both models localise. The last row is a constant $(0,10)$\\,s window, "
         "which uses no model.}",
         "\\label{tab:s1}",
         "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{5.0cm}" + "c" * (NC - 1) + "@{}}",
         "\\toprule",
         " & ".join(hdr[i] for i in keep) + " \\\\",
         "\\midrule"]
    for r in rows:
        if r is None:
            L.append("\\midrule")
        elif isinstance(r, str):
            L.append("\\addlinespace[1pt]" + head(r, NC))
        else:
            L.append(" & ".join(r[i] for i in keep) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(note_caption(L), out)


# ---------------------------------------------------------------- S2
def build_S2(C, out):
    spec = [
        ("Qwen2.5-VL-7B", [("$\\alpha{=}0.25$", g(Q7, 0.25)), ("$\\alpha{=}0.5$", g(Q7, 0.5)), ("$\\alpha{=}1$", g(Q7, 1.0)),
                           ("$\\alpha{=}0.5$, $\\beta{=}0.03$", g(Q7, 0.5, 0.03)),
                           ("SFT", M(Q7, "sft", None, 0.0, 0)),
                           ("$\\alpha{=}0.5$, 4,800 pairs", M(Q7, "grouped-4800", 0.5, 0.1, 0))]),
        ("TimeZero", [("$\\alpha{=}0.25$", g(TZ, 0.25)), ("$\\alpha{=}0.5$", g(TZ, 0.5)), ("$\\alpha{=}1$", g(TZ, 1.0)),
                      ("$\\alpha{=}0.5$, $\\beta{=}0.03$", g(TZ, 0.5, 0.03))]),
        ("Time-R1", [("$\\alpha{=}0.1$", g(TR, 0.1)), ("$\\alpha{=}0.25$", g(TR, 0.25)), ("$\\alpha{=}0.5$", g(TR, 0.5)),
                     ("$\\alpha{=}1$", g(TR, 1.0))]),
        ("VideoChat-R1", [("$\\alpha{=}0.25$", g(VC, 0.25)), ("$\\alpha{=}0.5$", g(VC, 0.5)), ("$\\alpha{=}1$", g(VC, 1.0))]),
        ("TimeLens-8B", [("$\\alpha{=}0.25$", g(TL, 0.25)), ("$\\alpha{=}0.5$", g(TL, 0.5)), ("$\\alpha{=}1$", g(TL, 1.0))]),
    ]
    L = ["% S2 -- generated by scripts/build_appendix_tables_v3.py (canonical_v3)",
         "\\begin{table}[t]",
         "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.05}",
         "\\caption{\\textbf{Within every base, evidence dependence falls monotonically as the clean-supervision weight rises.} "
         "Each row compares one trained model with its base. $\\Delta$\\Rfive{} is the change in points, and $^\\dagger$ marks a "
         "difference of the two accuracies that is not paired by question; $\\Gm$ and $\\Gs$ are in nats per token on the "
         "$n_\\cap$ questions both models localise, and the sign fraction is the fraction of them on which $\\psi$ rises. Rows "
         "other than SFT use $\\beta{=}0.1$ unless they name another value, and brackets give 95\\% intervals.}",
         "\\label{tab:s2}",
         "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{3.6cm}ccccc@{}}",
         "\\toprule",
         "Setting & $\\Delta$\\Rfive{} & $\\Gm$ & $\\Gs$ & $n_\\cap$ & sign frac. \\\\",
         "\\midrule"]
    for fam, rows in spec:
        L.append("\\addlinespace[1pt]" + head(fam, 6))
        for lab, k in rows:
            C.ctx("S2", f"{fam} / {lab} / env {C.env(k)}")
            delta = delta_cell(C, k, CS)
            gam = C.pick("gamma", k, CS, column="gamma", instrument="14Ln")
            L.append(" & ".join([lab, delta, fg(gv(gam, "gamma_mask"), gv(gam, "gamma_mask_ci")),
                                 fg(gv(gam, "gamma_swap"), gv(gam, "gamma_swap_ci")),
                                 fint(gv(gam, "n_H_int")), fr(gv(gam, "pos_frac_mask"))]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(L, out)


# ---------------------------------------------------------------- exhaustive helper
def exhaustive(C, kind, table, dsorder=None, **extra):
    rows = [x for x in C.rows_for(kind, **extra) if x[0][4] in REF_SEEDS]
    rows.sort(key=lambda x: sort_key(x[0], x[1], dsorder))
    for k, ds, rec, envs in rows:
        if rec is None:
            C.ctx(table, row_label(k) + f" / env {C.env(k)} / {ds}")
            C.log_pending("whole row", kind, k, ds, C.env(k), "canonical stack has no such output" if not envs else
                          f"this output exists only on {','.join(envs)} stack", {kk: vv for kk, vv in extra.items() if kk != "dataset"})
    return rows


# ---------------------------------------------------------------- S3
def build_S3(C, out):
    L = ["% S3 -- generated by scripts/build_appendix_tables_v3.py (canonical_v3)",
         "\\begin{table}[t]",
         "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.05}",
         "\\caption{\\textbf{On an all-black video, SFT and the public checkpoints keep answering.} The placebo-controlled "
         "Qwen2.5-VL-7B models almost never answer, and those trained from TimeZero and Time-R1 answer less often than their "
         "bases. Every frame is set to zero; duration and video token count are kept. The column \\emph{data} names the question "
         "set: Charades-STA (STA), QVHighlights (QVH) or Charades-STA with TimeLens labels (C-TL). \\emph{ans.} is the fraction "
         "of questions still given an interval, and \\emph{black \\Rfive} is the \\Rfive{} over all questions. \\emph{abst.} is the "
         "strict (s) or wide (w) abstention rate: the strict criterion counts only the verbatim output \\texttt{There are "
         "none.}, and the wide criterion counts any refusal wording (Appendix~\\ref{app:protocols}). Text-first rows put the "
         "query before the video.}",
         "\\label{tab:s3}",
         "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{5.8cm}ccccc@{}}",
         "\\toprule",
         "Model / objective & data & ans. & black \\Rfive & abst.\\ (s) & abst.\\ (w) \\\\",
         "\\midrule"]
    DS3 = {"charades-sta": ("STA", "Charades-STA"), "qvhighlights": ("QVH", "QVHighlights"),
           "charades-timelens": ("C-TL", "Charades, TimeLens re-annotation")}
    seen = None
    for k, ds, r, envs in exhaustive(C, "black", "S3", dsorder=list(DS3)):
        if r is None:   # author ruling: a configuration with no black run in its canonical environment is not listed
            C.pending[:] = [p for p in C.pending if not (p["table"] == "S3" and p["key"] == key_str(k) and p["dataset"] == ds)]
            continue
        if ds != seen:
            L.append("\\addlinespace[1pt]" + head(DS3[ds][1], 6))
            seen = ds
        C.ctx("S3", row_label(k) + f" / env {C.env(k)} / {ds}")
        if r is not None:
            C.note(r, "black row", "canonical")
        L.append(" & ".join([row_label(k), DS3[ds][0], fr(gv(r, "black_answer_rate")), fr(gv(r, "black_R50")),
                             fr(gv(r, "black_hard_refusal_strict")), fr(gv(r, "black_any_refusal_wide"))]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(L, out)


# ---------------------------------------------------------------- S4
def build_S4(C, out):
    L = ["% S4 -- generated by scripts/build_appendix_tables_v3.py (canonical_v3)",
         "\\begin{table}[t]",
         "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.05}",
         "\\caption{\\textbf{Given the query of another video, the placebo-controlled Qwen2.5-VL-7B and Time-R1 models decline "
         "more often than their bases, and no TimeZero configuration declines.} Each question is asked once with its own "
         "annotation sentence as the query and once with the sentence of another video, which gives $n$ pairs. \\emph{abst.} is "
         "the strict abstention rate, the share of outputs that are verbatim \\texttt{There are none.}, and \\Rfive{} is measured "
         "on the same pairs. Each $\\Delta$ is the foreign-query value minus the own-query value, in points, with its 95\\% "
         "interval. Text-first rows put the query before the video.}",
         "\\label{tab:s4}",
         "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{4.8cm}cccccc@{}}",
         "\\toprule",
         "Model / objective & $n$ & abst.\\ own & abst.\\ foreign & $\\Delta$ abst. & \\Rfive{} own & $\\Delta$\\Rfive{} \\\\",
         "\\midrule"]
    for k, ds, r, envs in exhaustive(C, "tneg", "S4"):
        C.ctx("S4", row_label(k) + f" / env {C.env(k)}")
        if r is not None:
            C.note(r, "tneg row", "canonical")
        L.append(" & ".join([row_label(k), fint(gv(r, "n_pairs")), fr(gv(r, "clean_abstain_strict")),
                             fr(gv(r, "foreign_abstain_strict")),
                             fpp(gv(r, "delta_abstain_strict"), gv(r, "delta_abstain_strict_ci")),
                             fr(gv(r, "clean_R50")), fpp(gv(r, "delta_R50"), gv(r, "delta_R50_ci"))]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(L, out)


# ---------------------------------------------------------------- S5
def _tl_env(r):
    ins = r.get("inputs") or {}
    files = []
    for v in (ins.values() if isinstance(ins, dict) else [ins]):
        for i in (v if isinstance(v, list) else [v]):
            files.append(i.get("file") if isinstance(i, dict) else str(i))
    tl = [f for f in files if f and "chtl" in f]
    if len(tl) != 1:
        raise SystemExit(f"S5: cannot identify the TL run of {rec_name(r)}: {tl}")
    m = re.search(r"_([a-z0-9]+)\.json$", tl[0])
    return ENV_OF_MACHINE[m.group(1)], tl[0]


# Qwen3-VL-8B-Instruct (base of TimeLens-8B) is not a family of the registry; its one labelswap reading is read here
Q3VL_LABELSWAP = (_SUPP + "/results/judgment_20260921/labelswap/read_tl_labelswap_Q3VL8B_base.json", "952a24c90add913d")


def _q3vl_labelswap_row():
    import hashlib
    p, want = Q3VL_LABELSWAP
    got = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    if got != want:
        raise SystemExit(f"S5: {p} sha16 {got} != {want}")
    d = json.load(open(p, encoding="utf-8"))
    if d["match"]["n_common"] != 870 or not d["machines"]["same_env"]:
        raise SystemExit("S5: Qwen3-VL-8B labelswap reading is not the 870-question same-environment matrix")
    m = {c: v["R@0.5"] for c, v in d["common"]["matrix"].items()}
    return " & ".join(["Qwen3-VL-8B, base of TimeLens-8B", fr(m["sta_run|STA_labels"]), fr(m["sta_run|TL_labels"]), "", ""]) + " \\\\"


def build_S5(C, out):
    L = ["% S5 -- generated by scripts/build_appendix_tables_v3.py (canonical_v3)",
         "\\begin{table}[t]",
         "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.05}",
         "\\caption{\\textbf{A second annotation of the same videos reorders the configurations by accuracy.} Measured against "
         "it, every placebo-controlled model keeps a positive $\\Gamma$ and no accuracy-oriented model rises beyond the audit's "
         "resolution. The upper block scores one run's predictions on the 870 questions present in both label sets under each "
         "set of labels, where STA denotes the original Charades-STA labels and TL the TimeLens labels. The lower block gives "
         "$\\Gm$ and $\\Gs$ in nats per token, with 95\\% intervals, measured with the re-annotated intervals as the fixed answer, "
         "together with the number $n_\\cap$ of questions both models localise and the fraction of them on which $\\psi$ rises.}",
         "\\label{tab:s5}",
         "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{5.2cm}cccc@{}}",
         "\\toprule",
         "Model / objective & \\Rfive{}, STA labels & \\Rfive{}, TL labels & & \\\\",
         "\\midrule"]
    by = {}
    for r in C.R:
        if r.get("kind") == "labelswap" and r.get("_key") and r["_key"][4] in REF_SEEDS:
            by.setdefault(r["_key"], []).append(r)
    for k in sorted(by, key=lambda k: sort_key(k)):
        env = C.env_for(k, "labelswap", "charades-sta")
        # 09-25: the Qwen2.5-VL-7B trained rows are read on their training machine (C); the base row
        # of this block must come from the same machine so the three rows compare like with like
        if k[0] == Q7 and k[1] == "base" and any(r["_env"] == "C" for r in by[k]):
            env = "C"
        tl_env = C.env_for(k, "acc", "charades-timelens")
        C.ctx("S5", "labels / " + row_label(k) + f" / env {env}")
        recs = by[k]
        sta = [r for r in recs if r["_env"] == env]
        if len(sta) > 1:
            raise SystemExit(f"S5: {key_str(k)} has {len(sta)} labelswap records in {env}")
        # only the frozen-prediction block is printed : the full-split columns came from independent runs
        # and added a reproducibility column that the text does not need
        s = sta[0] if sta else None
        if s is None:
            C.pending[:] = [p for p in C.pending if not (p["table"] == "S5" and p["key"] == key_str(k))]
            continue
        C.note(s, "labelswap 870 columns", "canonical")
        m = s.get("matrix_870") or {}
        # 09-25: the TimeLens-8B row is the public checkpoint; its own base Qwen3-VL-8B is printed just above it
        lab = "TimeLens-8B (public checkpoint)" if (k[0] == TL and k[1] == "base") else row_label(k)
        if k[0] == TL and k[1] == "base":
            L.append(_q3vl_labelswap_row())
        L.append(" & ".join([lab, fr(m.get("sta_run|STA_labels")), fr(m.get("sta_run|TL_labels")), "", ""]) + " \\\\")
    L.append("\\midrule")
    L.append(head("Evidence dependence measured against the re-annotated intervals", 5))
    L.append("Model / objective & $\\Gm$ & $\\Gs$ & $n_\\cap$ & sign frac. \\\\")
    L.append("\\midrule")
    for k, ds, r, envs in exhaustive(C, "gamma", "S5", dataset="charades-timelens", instrument="14Ln"):
        C.ctx("S5", "gamma / " + row_label(k) + f" / env {C.env(k)}")
        if r is not None:
            C.note(r, "gamma (TL labels)", "canonical")
        # 09-25: the one gamma record keyed (TimeLens-8B, base) is TimeLens-8B itself against its base Qwen3-VL-8B
        lab = "TimeLens-8B against Qwen3-VL-8B" if (k[0] == TL and k[1] == "base") else row_label(k)
        L.append(" & ".join([lab, fg(gv(r, "gamma_mask"), gv(r, "gamma_mask_ci")),
                             fg(gv(r, "gamma_swap"), gv(r, "gamma_swap_ci")),
                             fint(gv(r, "n_H_int")), fr(gv(r, "pos_frac_mask"))]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(L, out)


# ---------------------------------------------------------------- S6
OPS6 = ["none", "mask_evid", "mask_rand", "swap_evid", "swap_rand", "black"]
OPS6_H = ["clean", "mask $E$", "mask $O$", "swap $E$", "swap $O$", "black"]


def _cond(d, op):
    if d is None:
        return None
    c = d.get(op)
    return None if c is None else c["R@0.5"]["est"]


# moved-event readings of the three placebo-controlled models of Table 4's lower block other than Qwen (whose value is in
# Table 1): (family, objective, alpha, trained order, evaluated order) -> read_r3 product.  The rows are the ones
# whose behavioural pair Table 4 uses (behav_split.json: the node5 p5n6 runs).
R3_DIR = _SUPP + "/results/judgment_20260924_r3"
S6_FOLLOW = {(TZ, "grouped", 0.5, "v", "t"): "read_r3_node3_pc_tz_a05.json",
             (TR, "grouped", 0.25, "v", "v"): "read_r3_node3_pc_tr1_a025.json",
             (VC, "grouped", 0.5, "v", "t"): "read_r3_node3_pc_vc_a05.json"}


def s6_follows(C, k):
    f = S6_FOLLOW.get((k[0], k[1], k[2], k[5], k[6]))
    if f is None:
        return "\\NA"
    p = os.path.join(R3_DIR, f)
    v = json.load(open(p, encoding="utf-8"))["paired"]["Gamma_reloc_follow_iou"]
    import hashlib
    C.sources.append({"table": "S6", "row": row_label(k), "column": "follows", "src_file": f,
                      "sha16": hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]})
    return fpp(v["est"], v["ci95"])


def build_S6(C, out):
    L = ["% S6 -- generated by scripts/build_appendix_tables_v3.py (canonical_v3)",
         "\\begin{table}[t]",
         "\\centering\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\\renewcommand{\\arraystretch}{1.05}",
         "\\caption{\\textbf{In regenerated answers, every placebo-controlled model loses a larger share of its correct answers "
         "to evidence edits than its base does.} Under the mask, SFT and IoU-GRPO lose a smaller share. Each block compares a "
         "trained model with its base on the same 1{,}000 questions, edited frames and placebo draws. The first six columns "
         "give \\Rfive{} on the clean video, with the evidence frames ($E$) or as many frames outside the interval ($O$) masked "
         "or swapped, and on all-black versions of these questions. The $\\Delta$ columns give trained-minus-base changes in "
         "points in two readouts (Appendix~\\ref{app:regen}). The evidence share is $1-a_E/a_O$, with $a_E$ and $a_O$ the "
         "\\Rfive{} under the evidence edit and under the placebo edit, and is given under swap and under mask. \\emph{Follows} "
         "is the fraction of the audit questions both models localise on which the prediction, regenerated after the event's "
         "frames are exchanged with another block, has IoU $\\ge0.5$ with the moved interval; Table~\\ref{tab:audit} gives it for "
         "the Qwen2.5-VL-7B models. Table~\\ref{tab:s3} gives answering and refusal on black videos over the full test split. "
         "Brackets give 95\\% intervals.}",
         "\\label{tab:s6}",
         "\\begin{tabular}{@{}l" + "c" * 6 + "ccc@{}}",   # round 2: each block's name is a heading row, no role column
         "\\toprule",
         " & " + " & ".join(OPS6_H) + " & $\\Delta$share swap & $\\Delta$share mask & $\\Delta$follows \\\\",
         "\\midrule"]
    # every row states its message order; the video-first configuration of each audited block is always
    # listed (\pending when its canonical stack has no video-first run -- a text-first run never stands in)
    REQUIRED = [g(Q7, 0.5), M(Q7, "sft", None, 0.0, 0), g(TR, 0.25), g(TZ, 0.5), g(VC, 0.5)]
    rows = exhaustive(C, "behav", "S6")
    have = {x[0] for x in rows}
    for k in REQUIRED:
        if k not in have:
            C.ctx("S6", row_label(k) + f" / env {C.env(k)}")
            envs = sorted({r["_env"] for r in C.R if r.get("kind") == "behav" and r["_key"] == k})
            C.log_pending("whole row", "behav", k, "charades-sta", C.env(k),
                          "canonical (video-first) config has no same-order behavioral audit output; same model only has text-first eval outputs, not substituted")
            rows.append((k, "charades-sta", None, envs))
    def complete(x):
        return x is not None and all(_cond(gv(x, side), o) is not None for side in ("base_conditions", "arm_conditions") for o in OPS6)
    for i, (k, ds, r, envs) in enumerate(rows):   # a block missing operators is replaced by the key's six-operator block
        if not complete(r):
            alts = sorted((x for x in C.R if x.get("kind") == "behav" and x.get("_key") == k and complete(x)), key=rec_name)
            if alts:
                rows[i] = (k, ds, alts[0], envs)
                C.pending[:] = [p for p in C.pending if not (p["table"] == "S6" and p["key"] == key_str(k))]
    rows.sort(key=lambda x: sort_key(x[0]))
    for k, ds, r, envs in rows:
        C.ctx("S6", row_label(k) + f" / env {C.env(k)}")
        if r is not None:
            C.note(r, "behav block (order basis " + str(r.get("_order_basis")) + ")", "canonical")
        lab6 = row_label(k) + (" (video-first)" if k[6] == "v" else (" (text-first)" if "text-first" not in row_label(k) else ""))
        base = [fr(_cond(gv(r, "base_conditions"), o)) for o in OPS6]
        arm = [fr(_cond(gv(r, "arm_conditions"), o)) for o in OPS6]
        L.append(head(lab6, 10))
        L.append(" & ".join(["\\quad base"] + base + ["\\NA", "\\NA", "\\NA"]) + " \\\\")
        L.append(" & ".join(["\\quad trained"] + arm +
                            [fpp(gv(r, "nshare_swap"), gv(r, "nshare_swap_ci")), fpp(gv(r, "nshare_mask"), gv(r, "nshare_mask_ci")),
                             s6_follows(C, k)]) + " \\\\")
        L.append("\\addlinespace[1pt]")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(L, out)


# ---------------------------------------------------------------- S7
def build_S7(C, out):
    L = ["% S7 -- generated by scripts/build_appendix_tables_v3.py (canonical_v3)",
         "\\begin{table}[tbp]",
         "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.05}",
         "\\caption{\\textbf{Per edit operator, the placebo-controlled objective moves the evidence drop and keeps the placebo "
         "drop within $0.2$ nats per token of the base model's; without the placebo bound the placebo drop rises.} For the mask "
         "and the swap operator, the columns give the evidence drop $\\dE$ and the placebo drop $\\dO$ of the trained model (tr) "
         "and of its base on the same questions, in nats per token, followed by $\\Gamma$; under mask they also give the sign "
         "fraction, the fraction of questions on which $\\psi$ rises. Text-first rows put the query before the video.}",
         "\\label{tab:s7}",
         "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{4.1cm}cccccc@{\\hspace{8pt}}ccccc@{}}",
         "\\toprule",
         " & \\multicolumn{6}{c}{mask} & \\multicolumn{5}{c}{swap} \\\\",
         "\\cmidrule(lr){2-7}\\cmidrule(l){8-12}",
         "Model / objective & $\\dE^{\\text{tr}}$ & $\\dO^{\\text{tr}}$ & $\\dE^{\\text{base}}$ & $\\dO^{\\text{base}}$ & $\\Gamma$ & sign frac. "
         "& $\\dE^{\\text{tr}}$ & $\\dO^{\\text{tr}}$ & $\\dE^{\\text{base}}$ & $\\dO^{\\text{base}}$ & $\\Gamma$ \\\\",
         "\\midrule"]
    prev_fam = None
    for k, ds, r, envs in exhaustive(C, "gamma", "S7", dataset=CS, instrument="14Ln"):
        C.ctx("S7", row_label(k) + f" / env {C.env(k)}")
        if k[0] != prev_fam:   # one heading per base; rows then drop the family prefix
            L.append("\\addlinespace[1pt]" + head(FAM.get(k[0], k[0]), 12))
            prev_fam = k[0]
        if r is not None:
            C.note(r, "gamma components", "canonical")
        for op in ("mask", "swap"):   # the four components must reproduce the reader's Gamma
            c = [gv(r, f"{a}_{op}_{b}") for a, b in (("dE", "arm"), ("dO", "arm"), ("dE", "base"), ("dO", "base"))]
            if r is not None and None not in c and gv(r, f"gamma_{op}") is not None:
                if abs((c[0] - c[1]) - (c[2] - c[3]) - gv(r, f"gamma_{op}")) > 1e-6:
                    raise SystemExit(f"S7: {key_str(k)} {op} components do not reproduce Gamma")
        # 09-25: the one gamma record keyed (TimeLens-8B, base) is TimeLens-8B itself against its base Qwen3-VL-8B
        lab = "public checkpoint against Qwen3-VL-8B" if (k[0] == TL and k[1] == "base") else row_label(k, with_family=False)
        L.append(" & ".join([lab, f2(gv(r, "dE_mask_arm")), f2(gv(r, "dO_mask_arm")),
                             f2(gv(r, "dE_mask_base")), f2(gv(r, "dO_mask_base")),
                             fg(gv(r, "gamma_mask")), fr(gv(r, "pos_frac_mask")),
                             f2(gv(r, "dE_swap_arm")), f2(gv(r, "dO_swap_arm")),
                             f2(gv(r, "dE_swap_base")), f2(gv(r, "dO_swap_base")),
                             fg(gv(r, "gamma_swap"))]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(L, out)


# ---------------------------------------------------------------- S8
def build_S8(C, out):
    global TRK
    TRK = Track()
    QV = "qvhighlights"
    rows = [("HEAD", "Qwen2.5-VL-7B"),
            M(Q7, "base"), M(Q7, "base", eval="t"), g(Q7, 0.5), g(Q7, 1.0), g(Q7, 0.5, 0.03), M(Q7, "sft", None, 0.0, 0),
            ("HEAD", "TimeZero"),
            M(TZ, "base"), M(TZ, "base", eval="t"), g(TZ, 0.25), g(TZ, 0.5)]
    L = ["% S8 -- generated by scripts/build_appendix_tables_v3.py (canonical_v3)",
         "\\begin{table}[t]",
         "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.05}",
         "\\caption{\\textbf{On QVHighlights, every placebo-controlled model keeps a positive $\\Gm$ and SFT stays at zero.} All "
         "models are trained on Charades-STA and evaluated on QVHighlights with the TimeLens re-annotation (1{,}541 questions). "
         "$\\Delta$\\Rfive{} is the paired change from the model's base in points, and $\\Gm$ and $\\Gs$ are in nats per token on "
         "the $n_\\cap$ questions both models localise. The column \\emph{ord.} gives the message order, video first (v) or query "
         "text first (t); \\emph{parse} is the share of outputs that parse to an interval, and \\emph{abst.} is the strict "
         "abstention rate, the share of outputs that are verbatim \\texttt{There are none.}}",
         "\\label{tab:s8}",
         "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{4.4cm}cccccccc@{}}",
         "\\toprule",
         "Model / objective & ord. & \\Rfive & parse & abst. & $\\Delta$\\Rfive{} & $\\Gm$ & $\\Gs$ & $n_\\cap$ \\\\",
         "\\midrule"]
    for e in rows:
        if isinstance(e, tuple) and e[0] == "HEAD":
            L.append("\\addlinespace[1pt]" + head(e[1], 9))
            continue
        k = e
        lab = row_label(k, with_family=False)
        C.ctx("S8", f"{FAM[k[0]]} / {lab} / env {C.env(k)}")
        acc = C.pick("acc", k, QV, column="acc")
        if k[1] == "base":
            # a base has no paired Gamma against itself: \NA (v1 printed \pending here)
            if acc is not None:
                TRK.base_shown(k[0], QV, acc["_env"])
            delta, gcells = "\\NA", ["\\NA", "\\NA", "\\NA"]
        else:
            delta = delta_cell(C, k, QV)
            gam = C.pick("gamma", k, QV, column="gamma", instrument="14Ln")
            gcells = [fg(gv(gam, "gamma_mask"), gv(gam, "gamma_mask_ci")), fg(gv(gam, "gamma_swap")), fint(gv(gam, "n_H_int"))]
        L.append(" & ".join([lab, k[6], fr(gv(acc, "R50")), fr(gv(acc, "parse_rate")),
                             fr(gv(acc, "abstain_strict_rate")), delta] + gcells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(note_caption(L), out)


# ---------------------------------------------------------------- S9
ORACLE_LAB = {"tl8b_a05_sta": "TimeLens-8B, grouped $\\alpha{=}0.5$", "tl8b_base_sta": "TimeLens-8B, base"}
ORACLE_MODE = {"gt": "length of the annotated interval", "const": "constant 8\\,s"}


S9_DIFF = []


def build_S9(C, out):
    L = ["% S9 -- generated by scripts/build_appendix_tables_v3.py (canonical_v3)",
         "\\begin{table}[t]",
         "\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.05}",
         "\\caption{\\textbf{Keeping each prediction's centre and replacing only its length reproduces the largest accuracy gain "
         "in the audit matrix.} The predictions come unchanged from the Charades-STA run, and only each interval's length is "
         "replaced, either by a constant 8\\,s or by the length of the annotated interval. The columns give \\Rfive{} and mIoU, "
         "the mean IoU with the annotation, for the predictions as made (\\emph{as pred.}) and after the substitution "
         "(\\emph{subst.}).}",
         "\\label{tab:s9}",
         "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{4.4cm}>{\\raggedright\\arraybackslash}p{3.3cm}cccc@{}}",
         "\\toprule",
         "Configuration & length substituted & \\multicolumn{2}{c}{\\Rfive{}} & \\multicolumn{2}{c@{}}{mIoU} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}",
         " & & as pred. & subst. & as pred. & subst. \\\\",
         "\\midrule"]
    recs = sorted([r for r in C.R if r.get("kind") == "oracle"], key=lambda r: (r["label"], r["mode"]))
    for r in recs:
        C.ctx("S9", r["label"] + "/" + r["mode"])
        ok = r.get("_key") is not None and r.get("_env") == C.env_for(r["_key"], "acc", "charades-sta")
        if ok:
            C.note(r, "oracle row", "canonical")
        else:
            C.log_pending("whole row", "oracle", r.get("_key"), "charades-sta", C.env(r["_key"]) if r.get("_key") else None,
                          f"oracle readout run is on {r.get('_env')} stack, not canonical")
        v = (lambda x: x) if ok else (lambda x: None)
        orig = r["original_R50"]
        acc = C.pick("acc", r["_key"], "charades-sta", column="as pred. R@0.5 (registered selective reading)") if r.get("_key") else None
        if acc is not None and abs(acc["R50"] - orig) > 1e-12:
            S9_DIFF.append((r["label"], r["mode"], orig, acc["R50"]))
            orig = acc["R50"]      # one value per (configuration, acc): the registered reader's
        L.append(" & ".join([ORACLE_LAB.get(r["label"], r["label"]), ORACLE_MODE[r["mode"]],
                             fr(v(orig)), fr(v(r["oracle_R50"])),
                             fr(v(r["original_mIoU"])), fr(v(r["oracle_mIoU"]))]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(L, out)


# ---------------------------------------------------------------- S17 (round 4): edit operators never used in training
UNSEEN_DIR = _SUPP + "/results/judgment_20260921/gamma"
UNSEEN = [(g(Q7, 0.5), "gamma_14Ln7_natbase_a05_g1_s0_tf5_timer1_default_node2__vs__14Ln7_base_timer1_default_node2.json"),
          (g(Q7, 0.5, 0.03), "gamma_14Ln7_natbase_a05_b003_g1_s0_timer1_default_node2__vs__14Ln7_base_timer1_default_node2.json"),
          (g(Q7, 1.0), "gamma_14Ln7_natbase_a10_g1_s0_timer1_default_node2__vs__14Ln7_base_timer1_default_node2.json"),
          (g(TZ, 0.25), "gamma_14Ln7_nattz_a025_g1_s0_timer1_default_node2__vs__14Ln7_tzbase_timer1_default_node2.json"),
          (g(TZ, 0.5), "gamma_14Ln7_nattz_a05_g1_s0_timer1_default_node2__vs__14Ln7_tzbase_timer1_default_node2.json")]
OPS17 = (("noise", "noise"), ("pscram", "patch scrambling"), ("gblur", "Gaussian blur"))


def build_S17(C, out):
    import hashlib
    L = ["% S17 -- generated by scripts/build_appendix_tables_v3.py (canonical_v3)",
         "\\begin{table}[t]",
         "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.05}",
         "\\caption{\\textbf{Under edit operators never used in training, every placebo-controlled model keeps a positive "
         "$\\Gamma$, large under patch scrambling and small under noise and Gaussian blur.} Each model is audited against its "
         "base on the same 200 questions. For each operator, the table gives $\\Gamma$ with its 95\\% interval on the $n_\\cap$ "
         "questions both models localise, and the trained model's placebo drop $\\dO$, both in nats per token.}",
         "\\label{tab:unseen}",
         "\\begin{tabular}{@{}lc" + "cc" * len(OPS17) + "@{}}",
         "\\toprule",
         " & & " + " & ".join("\\multicolumn{2}{c}{%s}" % h for _, h in OPS17) + " \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(l){7-8}",
         "Model / objective & $n_\\cap$" + " & $\\Gamma$ & $\\dO$" * len(OPS17) + " \\\\",
         "\\midrule"]
    prev = None
    for k, f in UNSEEN:
        if k[0] != prev:
            L.append("\\addlinespace[1pt]" + head(FAM[k[0]], 2 + 2 * len(OPS17)))
            prev = k[0]
        p = os.path.join(UNSEEN_DIR, f)
        d = json.load(open(p, encoding="utf-8"))
        C.sources.append({"table": "S17", "row": row_label(k), "column": "noise/pscram/gblur", "src_file": f,
                          "sha16": hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]})
        cells = [row_label(k, with_family=False), fint(d["n_H_int"])]
        for op, _ in OPS17:
            v = d["per_op"][op]
            cells += [fg(v["gamma"], v["gamma_ci95"]), f2(v["arm_on_cohort"]["dO"])]
        L.append(" & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(L, out)


def main():
    C = Canon()
    build_S1(C, os.path.join(T, "S1_audit_matrix.tex"))
    build_S2(C, os.path.join(T, "S2_alpha_beta_grid.tex"))
    build_S3(C, os.path.join(T, "S3_black_screen.tex"))
    build_S4(C, os.path.join(T, "S4_foreign_query.tex"))
    build_S5(C, os.path.join(T, "S5_reannotation.tex"))
    build_S6(C, os.path.join(T, "S6_behavioural.tex"))
    build_S7(C, os.path.join(T, "S7_operator_components.tex"))
    build_S8(C, os.path.join(T, "S8_out_of_distribution.tex"))
    build_S9(C, os.path.join(T, "S9_window_oracle.tex"))
    build_S17(C, os.path.join(T, "S17_unseen_operators.tex"))
    os.makedirs(LOGD, exist_ok=True)
    with open(os.path.join(T, "APPENDIX_SOURCES_v3.json"), "w", encoding="utf-8") as f:
        json.dump(C.sources, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(LOGD, "build_appendix_tables_v3.json"), "w", encoding="utf-8") as f:
        json.dump({"sources": C.sources, "pending": C.pending, "s9_as_pred_replaced": S9_DIFF}, f, ensure_ascii=False, indent=1, default=str)
    print(f"{len(C.sources)} sourced cells; {len(C.pending)} pending logged")


if __name__ == "__main__":
    main()
