# v3 copy of build_cd_table_v2.py: identical logic, environments from canonical_v3 (per configuration, kind, dataset).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_cd_table_v2.py -- (2026-09-23; canonical_v2 copy of v1; output goes to the rebuild copy).
v2: a cell prints only when its p4cd_ run sits in the configuration's canonical environment
(canonical_v2.ext_pick); v1 took the first glob hit on any machine.
v1: build_cd_table_v1.py -- S14: every configuration on the Charades-CD debiased re-split
(Yuan et al. 2021, test-iid / test-ood). Source: results/cd_20260923/p4cd_<arm>_<split>_<mach>.json
(r6cd evaluator adfb321c for the 7B families, r7cd af06fc5c for TimeLens-8B, chain manifests in the
ledger). Scoring identical to read_cd_v1: R@0.5 with unparsed rows counted as misses. A missing
run prints \\pending. The constant-window floor row is computed from the annotation files alone.
"""
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..", "..")).replace("\\", "/")
import json, glob, os, hashlib
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from canonical_v3 import Canon, M, key_str, Q7, Q3, TZ, TR, VC, TL, PEND as _PEND  # canonical_v2

D = _SUPP + r"/results/cd_20260923"
ANN = {"iid": _SUPP + "/data_ids/charades_cd/charades_cd_iid_items.jsonl",
       "ood": _SUPP + "/data_ids/charades_cd/charades_cd_ood_items.jsonl"}
OUT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "tables", "S14_charades_cd.tex")
SRC = _os.path.join(_os.path.dirname(OUT), "S14_SOURCES.json")
assert "_iclr2027" not in OUT.replace(chr(92), "/")
CAN = Canon()
PEND = "\\pending"

ROWS = [
    ("Qwen2.5-VL-7B", [("base", "qwenbase"), ("SFT", "qwensft"), ("IoU-GRPO", "grpo"),
                       ("grouped $\\alpha{=}0.5$", "qwena05")]),
    ("TimeZero", [("base (text-first)", "tzsttf_NONE"), ("base", "tzbase"), ("base, text-first", "tzbasetf"),
                  ("SFT, text-first", "tzsfttf"), ("grouped $\\alpha{=}0.5$", "tza05")]),
    ("Time-R1", [("base", "tr1base"), ("grouped $\\alpha{=}0.5$", "tr1a05"), ("grouped $\\alpha{=}0.1$", "tr1a01")]),
    ("VideoChat-R1", [("base", "vcbase"), ("base, text-first", "vcbasetf"),
                      ("SFT, text-first", "vcsfttf"), ("grouped $\\alpha{=}1$", "vca10")]),
    ("TimeLens-8B", [("base", "tlbase"), ("SFT", "tlsft"), ("grouped $\\alpha{=}0.5$", "tlsft_a05")]),
]
TAG_FIX = {"tzsttf_NONE": None, "tlsft_a05": "tla05"}


def iou(p, g):
    if not p: return 0.0
    s1, e1 = float(p[0]), float(p[1]); s2, e2 = g
    inter = max(0.0, min(e1, e2) - max(s1, s2)); union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def score(path):
    rows = json.load(open(path, encoding="utf-8"))
    n = len(rows)
    r5 = sum(iou(r.get("pred"), (float(r["gsA"]), float(r["geA"]))) >= 0.5 for r in rows)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    return r5 / n, n, sha


def cell(tag, split, log=False):
    ks = [x[2] for x in CAN.ext if x[0] == "cd" and x[5]["tag"] == tag]
    if not ks:
        return None
    if log:
        CAN.ctx("S14", "%s / %s" % (tag, key_str(ks[0])))
        hit = CAN.ext_pick("cd", ks[0], dataset="charades-cd-" + split, column=split, tag=tag)
    else:
        hits = [x for x in CAN.ext if x[0] == "cd" and x[5]["tag"] == tag and x[5]["split"] == split and x[3] == CAN.env_for(ks[0], "cd", "charades-cd-" + split)]
        hit = hits[0] if len(hits) == 1 else None
    if hit is None:
        return None
    return score(hit[4]) + (os.path.basename(hit[4]),)


DID = _os.path.join(_os.path.dirname(OUT), "S14_did.json")


def _qhits(tag, split):
    """per-question hit (R@0.5, unparsed as miss) and video of the run cell() resolves, keyed by qid."""
    ks = [x[2] for x in CAN.ext if x[0] == "cd" and x[5]["tag"] == tag]
    hits = [x for x in CAN.ext if x[0] == "cd" and x[5]["tag"] == tag and x[5]["split"] == split
            and ks and x[3] == CAN.env_for(ks[0], "cd", "charades-cd-" + split)]
    if len(hits) != 1:
        return None
    rows = json.load(open(hits[0][4], encoding="utf-8"))
    return {r["qid"]: (r["video"], float(iou(r.get("pred"), (float(r["gsA"]), float(r["geA"]))) >= 0.5)) for r in rows}


def did(tag_a, tag_b, B=2000, seed=0):
    """round 4: 100*[(ood_a - ood_b) - (iid_a - iid_b)]; questions paired by qid within each split, videos resampled
    independently in each split (B draws, fixed seed). Returns dict(point, ci95, n_iid, n_ood, videos_iid, videos_ood)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    per = {}
    for s in ("iid", "ood"):
        a, b = _qhits(tag_a, s), _qhits(tag_b, s)
        if a is None or b is None:
            raise SystemExit(f"S14 did: {tag_a}/{tag_b} {s} run missing")
        q = sorted(set(a) & set(b))
        vids = sorted({a[i][0] for i in q})
        ix = {v: j for j, v in enumerate(vids)}
        dsum, cnt = np.zeros(len(vids)), np.zeros(len(vids))
        for i in q:
            dsum[ix[a[i][0]]] += a[i][1] - b[i][1]
            cnt[ix[a[i][0]]] += 1
        per[s] = (dsum, cnt, len(q), len(vids))
    pt = per["ood"][0].sum() / per["ood"][2] - per["iid"][0].sum() / per["iid"][2]
    bs = []
    for _ in range(B):
        m = []
        for s in ("iid", "ood"):
            j = rng.integers(0, len(per[s][0]), len(per[s][0]))
            m.append(per[s][0][j].sum() / per[s][1][j].sum())
        bs.append(m[1] - m[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"point": 100 * pt, "ci95": [100 * lo, 100 * hi], "n_iid": per["iid"][2], "n_ood": per["ood"][2],
            "videos_iid": per["iid"][3], "videos_ood": per["ood"][3], "B": B, "seed": seed}


def th(n):
    return f"{n:,}".replace(",", "{,}")


def floor(split):
    rows = [json.loads(l) for l in open(ANN[split], encoding="utf-8") if l.strip()]
    r5 = sum(iou((0.0, 10.0), (float(r["gsA"]), float(r["geA"]))) >= 0.5 for r in rows)
    return r5 / len(rows)


def main():
    src = {}
    any_pending = any(cell(TAG_FIX.get(t, t), s) is None
                      for _, arms in ROWS for _, t in arms if TAG_FIX.get(t, t) is not None
                      for s in ("iid", "ood"))
    pend_note = " \\pending{}: run not yet landed." if any_pending else ""
    D1, D2 = did("qwensft", "grpo"), did("qwensft", "qwena05")
    json.dump({"what": "SFT minus other objective, change from test-iid to test-ood, points (Qwen2.5-VL-7B)",
               "sft_minus_iou_grpo": D1, "sft_minus_placebo_controlled_a05": D2}, open(DID, "w", encoding="utf-8"), indent=1)
    L = ["% generated by scripts/build_cd_table_v3.py (canonical_v3) -- do not edit by hand",
         "\\begin{table}[t]", "\\centering\\papertablestyle",
         "\\caption{\\textbf{Across the shift to the Charades-CD test-ood split, SFT loses more accuracy than the other objectives on Qwen2.5-VL-7B.} "
         + "The table uses the test-iid (%s questions, %s videos) and test-ood (%s questions, %s videos) splits of Charades-CD \\citep{yuan2021closer}. "
         % (th(D2["n_iid"]), th(D2["videos_iid"]), th(D2["n_ood"]), th(D2["videos_ood"]))
         + "The re-split shifts the moment distribution, so the constant ten-second window falls from iid to ood; "
         + "the column \\emph{ood$-$iid} is each model's change across that shift, in points. "
         + "On Qwen2.5-VL-7B, SFT's change is lower than IoU-GRPO's by $%.1f$ $[%.1f, %.1f]$ points and lower than the placebo-controlled model's by $%.1f$ $[%.1f, %.1f]$; "
         % (-D1["point"], -D1["ci95"][1], -D1["ci95"][0], -D2["point"], -D2["ci95"][1], -D2["ci95"][0])
         + "questions are paired within each split, and the 95\\% intervals resample videos. Text-first rows put the query before the video." + pend_note + "}",
         "\\label{tab:s14}",
         "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
         "Model / objective & \\Rfive{} iid & \\Rfive{} ood & ood$-$iid \\\\", "\\midrule"]
    fi, fo = floor("iid"), floor("ood")
    L.append(f"constant $(0,10)$\\,s window (no model) & {fi:.3f} & {fo:.3f} & {100*(fo-fi):+.1f} \\\\")
    L.append("\\midrule")
    for fam, arms in ROWS:
        L.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{fam}}}}} \\\\")
        for label, tag in arms:
            tag = TAG_FIX.get(tag, tag)
            if tag is None:
                continue
            ci, co = cell(tag, "iid", log=True), cell(tag, "ood", log=True)
            si = f"{ci[0]:.3f}" if ci else PEND
            so = f"{co[0]:.3f}" if co else PEND
            gap = f"{100*(co[0]-ci[0]):+.1f}" if (ci and co) else PEND
            L.append(f"\\quad {label} & {si} & {so} & {gap} \\\\")
            src[f"{tag}"] = {s: (c[3] if c else None) for s, c in (("iid", ci), ("ood", co))}
        L.append("\\addlinespace[2pt]")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    from table_emph_v3 import emph   # bold: the entries discussed in the text (located by row label and column)
    open(OUT, "w", encoding="utf-8", newline="\n").write(emph("\n".join(L) + "\n", "S14_charades_cd"))
    json.dump(src, open(SRC, "w", encoding="utf-8"), indent=1)
    os.makedirs(os.path.join(os.path.dirname(OUT), "_v3_logs"), exist_ok=True)
    json.dump({"sources": CAN.sources, "pending": CAN.pending}, open(os.path.join(os.path.dirname(OUT), "_v3_logs", "build_cd_table_v3.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    landed = sum(1 for v in src.values() for x in v.values() if x)
    print(f"wrote {OUT}; landed cells {landed}/{2*len(src)}")


if __name__ == "__main__":
    main()
