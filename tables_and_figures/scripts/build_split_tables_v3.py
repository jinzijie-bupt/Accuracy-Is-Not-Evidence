#!/usr/bin/env python3
"""build_split_tables_v3.py (2026-09-24): what changes when a model depends more on the evidence (main-text table).

T4_split.tex: per pair (trained model against its base, mask),
  - behavioural split of the evidence share (analysis/behav_split.py): among answers correct with the placebo frames
    masked, the change in the share that returns no interval, and in the share that is a wrong interval, once the
    evidence frames are masked instead;
  - where in the answer string Gamma sits (analysis/read_r3_tokclass.py): answer onset, timestamps, tags.
Base and trained model of every pair were run on one machine with identical frames and draws.

Round 2 (2026-09-24): the moved-event column went to Table 1 (and, for the three placebo-controlled models of the
other bases, to Table S6); the token parts carry no bold/grey rule; the VideoChat-R1 public checkpoint gets its
"wrong interval" change from the same registered behav_split computation, run on the pair Table 1 uses for its
evidence share (Qwen base and VideoChat-R1, both under the VideoChat template).  That product is written once to
results/judgment_20260924_r3/behav_split_vcpub.json and read back.
"""
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..", "..")).replace("\\", "/")
import hashlib, importlib.util, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "..", "tables")
J = _SUPP + "/results/judgment_20260924_r3"
P5N_DIR = _SUPP + "/results/native_p5n_20260920"
BEHAV_SPLIT = _SUPP + "/readers/behav_split.py"
VCPUB = os.path.join(J, "behav_split_vcpub.json")
VCPUB_PAIR = ("VideoChat-R1 public", "p5n6_qwenbase_vchat_node5_n1000_o0.json", "p5n6_vcbase_vchat_node5_n1000_o0.json")
# the two per-question files must be the inputs of the product Table 1 reads (read_p5n_v2_node5_qwenvchat_vs_vcbasepub.json)
VCPUB_SHA = {"p5n6_qwenbase_vchat_node5_n1000_o0.json": "9121075997d75a90", "p5n6_vcbase_vchat_node5_n1000_o0.json": "6bba22c9666eae98"}
PEND = "\\pending"


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def ld(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fpp(v, mark=""):
    return PEND if v is None else f"{100*v['est']:+.1f}{mark} [{100*v['ci95'][0]:.1f}, {100*v['ci95'][1]:.1f}]"


def fe(v):
    if v is None:
        return PEND
    return "0.00" if abs(v["est"]) < 0.005 else f"{v['est']:+.2f}"


def vcpub_product():
    """behav_split on the VideoChat-R1 public pair, with the registered reader's own pair() (file not modified)."""
    if os.path.exists(VCPUB):
        return ld(VCPUB)
    lab, b, t = VCPUB_PAIR
    fb, ft = os.path.join(P5N_DIR, b), os.path.join(P5N_DIR, t)
    if not (os.path.exists(fb) and os.path.exists(ft)):
        return None
    for f in (fb, ft):
        if sha16(f) != VCPUB_SHA[os.path.basename(f)]:
            raise SystemExit(f"T4: {os.path.basename(f)} is not the input of the Table 1 evidence-share product")
    spec = importlib.util.spec_from_file_location("behav_split", BEHAV_SPLIT)
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    B, A = bs.load(fb), bs.load(ft)
    row = {"label": lab, "base": {"file": b, "sha16": sha16(fb)}, "arm": {"file": t, "sha16": sha16(ft)}}
    for op in ("mask", "swap"):
        row[op] = bs.pair(B, A, op)
    res = {"reader": "behav_split", "reader_sha16": sha16(BEHAV_SPLIT), "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "B": 2000, "seed": 0, "called_by": "scripts/build_split_tables_v3.py (pair() of the reader, same B and seed)",
           "pairs": [row]}
    json.dump(res, open(VCPUB + ".tmp", "w", encoding="utf-8"), indent=1)
    os.replace(VCPUB + ".tmp", VCPUB)
    return res


# row label, behav_split label (None: no behavioural pair), tokclass label
ROWS = [("SFT", "SFT", "node3_sft_tf5"),
        ("TimeZero (public)", "TimeZero public", "node3_tzpub"),
        ("Time-R1 (public)", "Time-R1 public", "node3_tr1pub"),
        ("VideoChat-R1 (public)", "VideoChat-R1 public", "node3_vcpub"),
        None,
        ("Qwen2.5-VL-7B, $\\alpha{=}0.5$", "pc Qwen a=0.5", "node3_pc_a05_tf5"),
        ("TimeZero, $\\alpha{=}0.5$$^\\ddagger$", "pc TimeZero a=0.5", "node3_pc_tz_a05"),
        ("Time-R1, $\\alpha{=}0.25$", "pc Time-R1 a=0.25", "node3_pc_tr1_a025"),
        ("VideoChat-R1, $\\alpha{=}0.5$$^\\ddagger$", "pc VideoChat-R1 a=0.5", "node3_pc_vc_a05")]


def main():
    bs_p = os.path.join(J, "behav_split.json")
    bs = {r["label"]: r for r in ld(bs_p)["pairs"]}
    srcs = [f"behav_split.json {sha16(bs_p)}"]
    vp = vcpub_product()
    if vp is not None:
        bs.update({r["label"]: r for r in vp["pairs"]})
        srcs.append(f"behav_split_vcpub.json {sha16(VCPUB)}")
    body = ["\\multicolumn{6}{@{}l}{\\emph{Trained for accuracy, against the Qwen2.5-VL-7B base}} \\\\"]
    for row in ROWS:
        if row is None:
            body += ["\\addlinespace[2pt]", "\\multicolumn{6}{@{}l}{\\emph{Placebo-controlled, against the base it was trained from}} \\\\"]
            continue
        lab, bl, rl = row
        b = bs.get(bl, {}).get("mask", {}).get("diff") if bl else None
        tp = os.path.join(J, f"tokclass_{rl}.json"); tc = ld(tp)
        if tc is not None:
            srcs.append(f"{os.path.basename(tp)} {sha16(tp)}")
        P = (tc or {}).get("paired", {})
        vc = "_vc" in rl      # VideoChat-R1's template has no tags and no way to decline: those cells do not exist
        # a model that never returns "no interval" under the mask has minus the base's rate as its change (footnote *)
        star = "$^\\ast$" if (b and not vc and abs(b["no_answer"]["est"] + bs[bl]["mask"]["base"]["no_answer"]) < 1e-12) else ""
        body.append(" & ".join(["\\quad " + lab,
                                "n/a" if vc else (fpp(b["no_answer"], star) if b else PEND),
                                fpp(b["wrong_ans"]) if b else PEND,
                                "n/a" if vc else fe(P.get("Gamma_onset")), fe(P.get("Gamma_ts")),
                                "n/a" if vc else fe(P.get("Gamma_rest"))]) + " \\\\")
    cap = ("\\caption{\\textbf{Evidence dependence shows in whether a model answers and where it places the interval.} Each entry "
           "is a trained model minus its base, with the evidence frames masked. The left block starts from the regenerated "
           "answers that are correct under the placebo mask (1{,}000 questions) and gives the change in the share that returns no "
           "interval and, among the intervals returned, in the share that is wrong. The right block splits $\\Gamma$ over the "
           "tokens of the scored answer, using a separate token-level audit of 200 questions; the \\emph{onset} is the first token "
           "after the opening tag.}")
    note = ("{\\scriptsize\\raggedright $^\\ast$Both models return an interval for every question under the mask, so both changes "
            "equal minus the base's rate. $^\\ddagger$The left block is evaluated with the text before the video, as released, and "
            "the right block with the video first. The entries marked n/a belong to VideoChat-R1, whose template fixes its tags "
            "and the bracket after the opening tag, leaving only the timestamps free, and has no way to decline. Brackets give "
            "95\\% intervals over videos.\\par}")
    L = ["% T4 -- generated by scripts/build_split_tables_v3.py; do not edit by hand",
         "\\begin{table}[tb]", "\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.1}",
         cap, "\\label{tab:split}",
         "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{Lost answers become (points)} & \\multicolumn{3}{c@{}}{Where $\\Gamma$ sits (nats per token)} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(l){4-6}",
         "Model & no interval & wrong interval & onset & timestamps & tags \\\\",
         "\\midrule"] + body + ["\\bottomrule", "\\end{tabular}", "\\par\\smallskip", note, "\\end{table}", "% sources: " + "; ".join(srcs)]
    out = os.path.join(T, "T4_split.tex")
    sys.path.insert(0, HERE)
    from table_emph_v3 import emph   # main-text rule: bold a change whose interval excludes zero; shade our rows
    open(out + ".tmp", "w", encoding="utf-8", newline="\n").write(emph("\n".join(L) + "\n", "T4_split")); os.replace(out + ".tmp", out)
    print(out, "pending:", sum(x.count(PEND) for x in L))


if __name__ == "__main__":
    main()
