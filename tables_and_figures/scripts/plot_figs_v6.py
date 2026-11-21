#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_figs_v6.py -- round-2 unified redraw of every data figure (2026-09-24).

Data: every number comes from the canonical_v3 readers through plot_figs_v5 (its figure functions are called with
their save() replaced by an in-memory capture, so the manifests they build are exactly those behind the previous
figures; CIs the manifests do not carry are re-read with the same V5 helpers).  One deliberate data-source change:
Figure 5c now follows the S6 table rule (a behavioural block missing operators is replaced by the key's six-operator
block, build_appendix_tables_v3.build_S6), so each Figure 5c value equals the corresponding S6 cell; the script
asserts that equality against tables/S6_behavioural.tex.

Style: scripts/paper_style.py.  Every figure is drawn at its placed width (see PLACED below), so the printed type
size equals the size set here (minimum 7 pt; the script asserts it).

Outputs overwrite the paths the paper includes:
  fig1_map       figures/data_redesign_v3/   (Figure 1, right; 0.56 linewidth)
  fig2_findings  figures/data_redesign_v4/   (Figure 2; linewidth)
  fig5_mechanism figures/data_redesign_v4/   (Figure 3; 0.46 linewidth, beside Table 2)
  fig6_dose      figures/data_redesign_v4/   (Figure 4; linewidth)
  fig5_readouts  figures/data_redesign_v3/   (Figure 5; linewidth)
  fig9_ood       figures/data_redesign_v3/   (Figure 9; linewidth)
  fig8_cases     figures/data/               (Figure 10; 0.84 linewidth)
"""
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..", "..")).replace("\\", "/")
import copy, hashlib, json, os, re, sys, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import blended_transform_factory

import plot_figs_v5 as V5                     # canonical data access (imports canonical_v3)
CAPTURE = {}
V5.save = lambda fig, name, man: CAPTURE.update({name: (fig, copy.deepcopy(man))})

import paper_style as PS
from paper_style import C, MARK, SHORT, FS_TICK, FS_LABEL, FS_NOTE, FS_SMALL

FIG = os.path.join(ROOT, "figures")
OUTDIR = {"fig1_map": os.path.join(FIG, "data_redesign_v3"), "fig2_findings": os.path.join(FIG, "data_redesign_v4"),
          "fig5_mechanism": os.path.join(FIG, "data_redesign_v4"), "fig6_dose": os.path.join(FIG, "data_redesign_v4"),
          "fig5_readouts": os.path.join(FIG, "data_redesign_v3"), "fig9_ood": os.path.join(FIG, "data_redesign_v3"),
          "fig8_cases": os.path.join(FIG, "data"), "fig10_dynamics": os.path.join(FIG, "data_redesign_v4")}
PLACED = {"fig1_map": 0.56, "fig2_findings": 1.0, "fig5_mechanism": 0.46, "fig6_dose": 1.0, "fig5_readouts": 1.0,
          "fig9_ood": 1.0, "fig8_cases": 0.84, "fig10_dynamics": 1.0}
RESO_GAMMA = 0.23          # audit resolution under mask (larger interval end of the two seed pairs; swap 0.31)
QA = {}


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def capture(name, fn):
    fn()
    fig, man = CAPTURE[name]
    plt.close(fig)
    return man


def parse_key(s):
    f = s.split("|")
    d = dict(fam=f[0], obj=f[1] if len(f) > 1 else None)
    for t in f[2:]:
        if "=" in t:
            a, b = t.split("=", 1)
            d[a] = b
    for kk in ("a", "b"):
        if kk in d:
            d[kk] = float(d[kk])
    return d


def fnum(v):
    return f"{v:g}"


# ------------------------------------------------------------------ QA helpers
def text_qa(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    seen, items = set(), []
    for t in fig.findobj(Text):
        if id(t) in seen or not t.get_visible() or not t.get_text().strip():
            continue
        seen.add(id(t))
        bb = t.get_window_extent(r)
        if bb.width <= 0:
            continue
        items.append((t, bb))
    small = [(t.get_text(), t.get_fontsize()) for t, _ in items if t.get_fontsize() < 7.0 - 1e-9]
    assert not small, (name, "text below 7 pt", small)
    over = []
    for i, (a, ab) in enumerate(items):
        for b, bb in items[i + 1:]:
            dx = min(ab.x1, bb.x1) - max(ab.x0, bb.x0)
            dy = min(ab.y1, bb.y1) - max(ab.y0, bb.y0)
            if dx > 0.5 and dy > 0.5:
                over.append((a.get_text(), b.get_text()))
    W, H = fig.bbox.width, fig.bbox.height
    outside = [t.get_text() for t, bb in items if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5]
    words = " ".join(t.get_text() for t, _ in items).lower()
    assert "support" not in words, (name, "the word 'support' appears in the figure")
    QA[name] = dict(size_in=list(fig.get_size_inches()), placed_linewidth=PLACED[name],
                    min_font_pt=min(t.get_fontsize() for t, _ in items), text_overlaps=over, outside_canvas=outside)
    if over or outside:
        print("  [layout]", name, "overlaps:", over, "outside:", outside)


def export(fig, name, manifest):
    text_qa(fig, name)
    stem = os.path.join(OUTDIR[name], name)
    PS.save(fig, stem)
    manifest = dict(generator="scripts/plot_figs_v6.py", style="scripts/paper_style.py",
                    size_inches=list(fig.get_size_inches()), placed_width_linewidth=PLACED[name], **manifest)
    mname = stem + (".sources.json" if name == "fig8_cases" else "_manifest.json")
    json.dump(manifest, open(mname, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    plt.close(fig)
    print("wrote", os.path.relpath(stem, ROOT), fig.get_size_inches() if False else "")


def mk_ours(ax, x, y, fam, s=22, fill="full", z=4, color=None, edge="white"):
    col = color or C["ours"]
    ax.plot([x], [y], ls="", marker=MARK[fam], ms=np.sqrt(s), mfc=col if fill != "none" else "white",
            mec=edge if fill == "full" else col, mew=0.5 if fill == "full" else 0.8, fillstyle=fill if fill in ("left", "full") else "full",
            markerfacecoloralt="white", zorder=z)


def leader(ax, xy, xytext, color=C["note"]):
    ax.plot([xy[0], xytext[0]], [xy[1], xytext[1]], color=color, lw=0.45, zorder=3)


# ================================================================== Figure 1, right: the audit map
def fig1_map():
    """round7/P5 redesign: every group is named in the figure (no shape key to decode from the caption);
    same points, same CIs, same physical size (3.08 x 2.36 in)."""
    name = "fig1_map"
    man = capture(name, V5.fig1_map)
    W, H = 3.08, 2.36
    fig = plt.figure(figsize=(W, H))
    ax = PS.axes_in(fig, 0.50, 0.38, 2.50, 1.90)
    ax.set_xlim(-11, 24)
    ax.set_ylim(-0.55, 2.8)
    ax.axhspan(-RESO_GAMMA, RESO_GAMMA, color=C["band"], lw=0, zorder=0)
    pts = []
    for p in man["points"]:
        s = p["point"]
        if "(public)" in s:
            fam, obj = "public", "public"
            short = s.split(" (public)")[0]
        else:
            d = parse_key(s)
            fam, obj, short = SHORT[d["fam"]], d["obj"], None
        pts.append(dict(p, fam=fam, obj=obj, short=short, beta=(parse_key(s).get("b") if "(public)" not in s else None),
                        alpha=(parse_key(s).get("a") if "(public)" not in s else None)))
    # CIs first
    for p in pts:
        ours = p["obj"] == "grouped"
        col = C["ours_ci"] if ours else C["acc_ci"]
        if p.get("delta_ci"):
            PS.ci_h(ax, p["delta_ci"][0], p["delta_ci"][1], p["gamma"], col, lw=0.6, zorder=1)
        if p.get("gamma_ci"):
            PS.ci_v(ax, p["delta_pp"], p["gamma_ci"][0], p["gamma_ci"][1], col, lw=0.6, zorder=1)
    for p in pts:
        x, y = p["delta_pp"], p["gamma"]
        if p["obj"] == "grouped":
            if p["beta"] == 0.03:
                ax.plot([x], [y], ls="", marker=MARK[p["fam"]], ms=4.6, mfc=C["ours_light"], mec=C["ours"], mew=0.7, zorder=4)
            else:
                ax.plot([x], [y], ls="", marker=MARK[p["fam"]], ms=4.6 if p["fam"] != "TimeLens-8B" else 5.2,
                        mfc=C["ours"], mec="white", mew=0.45, zorder=4)
        elif p["obj"] == "public":
            ax.plot([x], [y], ls="", marker=MARK["public"], ms=4.3, mfc="white", mec=C["acc"], mew=0.8, zorder=5)
        else:   # SFT / IoU-GRPO / mDPO-style, all from Qwen-7B
            ax.plot([x], [y], ls="", marker=MARK[p["fam"]], ms=4.6, mfc=C["acc"], mec="white", mew=0.45, zorder=5)
    # the origin is each model's own base
    ax.plot([0], [0], ls="", marker="o", ms=4.6, mfc="white", mec=C["base_edge"], mew=0.9, zorder=6)
    ax.text(1.5, -0.38, "own base", ha="left", va="center", fontsize=FS_SMALL, color=C["note"], zorder=7)
    leader(ax, (0.35, -0.07), (1.4, -0.31))
    ax.text(-10.6, 0.0, "within audit\nresolution", ha="left", va="center", fontsize=FS_SMALL, color=C["band_text"],
            linespacing=0.95)
    # ---- accuracy-oriented training: every point named
    by = {}
    for p in pts:
        if p["obj"] == "public":
            by[p["short"]] = p
        elif p["obj"] in ("sft", "iou-grpo", "mdpo"):
            by[{"sft": "SFT", "iou-grpo": "IoU-GRPO", "mdpo": "mDPO-style"}[p["obj"]]] = p
        elif p["obj"] == "grouped":
            by.setdefault(("g", p["fam"], p["beta"]), []).append(p)
    lab = {  # name: (text x, text y, ha)
        "mDPO-style": (7.6, 0.42, "right"), "Time-R1": (12.4, 0.42, "center"), "IoU-GRPO": (14.0, -0.42, "center"),
        "SFT": (8.7, -0.42, "center"), "VideoChat-R1": (20.3, 0.42, "center"), "TimeZero": (20.6, -0.42, "center"),
    }
    for nm, (tx, ty, ha) in lab.items():
        p = by[nm]
        ax.text(tx, ty, nm, ha=ha, va="center", fontsize=FS_SMALL, color=C["acc"], zorder=7)
        leader(ax, (p["delta_pp"], p["gamma"] + (0.07 if ty > 0 else -0.07)), (tx if ha == "center" else tx + 0.2, ty - (0.1 if ty > 0 else -0.1)))
    ax.text(23.6, 0.86, "accuracy-oriented", ha="right", va="center", fontsize=FS_NOTE, color=C["acc"], fontweight="bold")
    ax.text(23.6, 0.67, "◇ public checkpoint", ha="right", va="center", fontsize=FS_SMALL, color=C["acc"])
    # ---- placebo-controlled: each base named next to its points (shape = base, as elsewhere in the paper)
    ax.text(23.6, 2.72, "placebo-controlled (ours)", ha="right", va="top", fontsize=FS_NOTE, color=C["ours"], fontweight="bold")
    ax.text(23.6, 2.50, "labelled by base model", ha="right", va="top", fontsize=FS_SMALL, color=C["ours"])
    G = lambda fam, b=0.1: by[("g", fam, b)]
    top = lambda fam: max(G(fam), key=lambda q: q["gamma"])
    low = lambda fam: min(G(fam), key=lambda q: q["gamma"])
    t = top("Time-R1")
    ax.text(t["delta_pp"] + 1.0, t["gamma"] + 0.06, "Time-R1", ha="left", va="center", fontsize=FS_SMALL, color=C["ours"], zorder=7)
    t = low("Time-R1")
    ax.text(t["delta_pp"] - 0.1, t["gamma"] - 0.36, "Time-R1", ha="center", va="center", fontsize=FS_SMALL, color=C["ours"], zorder=7)
    t = top("TimeZero")
    ax.text(t["delta_pp"] - 0.9, t["gamma"] + 0.17, "TimeZero", ha="right", va="center", fontsize=FS_SMALL, color=C["ours"], zorder=7)
    vc = G("VideoChat-R1")
    t = max(vc, key=lambda q: q["gamma"])
    ax.text(t["delta_pp"] - 3.0, t["gamma"] + 0.25, "VideoChat-R1", ha="center", va="center", fontsize=FS_SMALL, color=C["ours"], zorder=7)
    q = sorted(G("Qwen-7B"), key=lambda q: q["alpha"])[1]
    ax.text(q["delta_pp"] + 1.9, q["gamma"] + 0.04, "Qwen2.5-VL-7B", ha="left", va="center", fontsize=FS_SMALL, color=C["ours"], zorder=7)
    t = G("TimeLens-8B")[0]
    ax.text(t["delta_pp"], t["gamma"] + 0.30, "TimeLens-8B", ha="center", va="center", fontsize=FS_SMALL, color=C["ours"], zorder=7)
    for p in by[("g", "Qwen-7B", 0.03)] + by[("g", "TimeZero", 0.03)]:
        if p["gamma"] < 1.0:
            left = True      # both labels to the left of their point (the right side holds the accuracy-oriented header)
            ax.text(p["delta_pp"] + (-0.9 if left else 0.9), p["gamma"] - 0.02, "β=0.03", ha="right" if left else "left",
                    va="center", fontsize=FS_SMALL, color=C["ours"], zorder=7)
    ax.set_xticks([-10, 0, 10, 20])
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel("change in accuracy, R@0.5 (points)")
    ax.set_ylabel("change in evidence dependence,\nΓ (mask, nats per token)", linespacing=1.0)
    PS.clean_axis(ax)
    export(fig, name, dict(points=man["points"], dropped=man["dropped"],
                           encoding="green filled = placebo-controlled (light fill: beta=0.03); charcoal filled = SFT / IoU-GRPO / mDPO-style; "
                                    "charcoal open diamond = public checkpoint vs Qwen base; open grey circle at origin = own base; shape = base family; "
                                    "every base family named in the figure (round7/P5)"))


# ================================================================== Figure 2: findings
def findings_data():
    man = capture("fig2_findings", V5.fig2_findings)
    # panels by record shape (: the SFT label-swap run joined panel b, so fixed slices no longer hold)
    blind = [p for p in man["points"] if "answer_rate" in p]
    paired = [p for p in man["points"] if "sta" in p and "tl_src" not in p]
    rest = [p for p in man["points"] if "tl_src" in p]
    assert len(blind) + len(paired) + len(rest) == len(man["points"]), "Fig 2 manifest has an unknown point shape"
    rows = []
    for k, ds, tl, envs in V5.C.rows_for("gamma", dataset=V5.CTL, instrument="14Ln"):
        if tl is None or k[1] not in ("sft", "grouped"):   # same row filter as V5.draw_relabel
            continue
        sta = V5.gam(k, V5.CS)
        if sta is not None and sta.get("_env") != tl.get("_env"):
            same = [r for r in V5.C.R if r.get("kind") == "gamma" and r.get("_key") == k and r.get("dataset") == V5.CS
                    and r.get("_env") == tl.get("_env") and "14Ln" in str(r.get("instrument", "14Ln"))]
            sta = same[0] if len(same) == 1 else None
        sft = k[1] == "sft"
        rows.append(dict(key=V5.key_str(k), fam=SHORT[k[0]], sft=sft, alpha=k[2],
                         label=f"{SHORT[k[0]]}, " + ("SFT" if sft else f"α={fnum(k[2])}"),
                         sta=None if sta is None else sta["gamma_mask"], sta_ci=None if sta is None else sta["gamma_mask_ci"],
                         tl=tl["gamma_mask"], tl_ci=tl["gamma_mask_ci"],
                         sta_src=V5.src(sta), tl_src=V5.src(tl)))
    rows.sort(key=lambda r: (r["sft"], -r["tl"]))
    # the same labels, in the same order, as the manifest rows of the previous figure
    assert sorted(r["label"] for r in rows) == sorted(p["point"] for p in rest), "Fig 2c row set drifted"
    return man, blind, paired, rows


def fig2_findings():
    name = "fig2_findings"
    man, blind, paired, contr = findings_data()
    W, H = 5.5, 2.12
    fig = plt.figure(figsize=(W, H))
    top = 1.66
    # ---------------- a: blind answers
    a = PS.axes_in(fig, 0.93, 0.36, 0.74, top - 0.36)
    labels = ["Qwen-7B base", "SFT", "IoU-GRPO", "mDPO-style", "TimeZero", "Time-R1", "ours, α=1", "ours, α=0.5"]
    groups = ["base", "acc", "acc", "acc", "public", "public", "ours", "ours"]
    col = {"base": C["base_edge"], "acc": C["acc"], "public": C["acc"], "ours": C["ours"]}
    n = len(blind)
    tr = blended_transform_factory(fig.transFigure, a.transData)
    for i, (r, lab, g) in enumerate(zip(blind, labels, groups)):
        y = n - 1 - i
        c = col[g]
        a.plot([0, r["answer_rate"]], [y, y], color=C["faint"], lw=0.75, zorder=1, solid_capstyle="butt")
        a.plot([r["answer_rate"]], [y], ls="", marker="o", ms=4.4, mfc="white", mec=c, mew=0.9, zorder=3, clip_on=False)
        a.plot([r["black_R50"]], [y], ls="", marker="o", ms=2.8, mfc=c, mec=c, mew=0.0, zorder=4, clip_on=False)
        a.text(0.05 / W, y, lab, transform=tr, ha="left", va="center", fontsize=FS_NOTE,
               color=C["ours"] if g == "ours" else C["ink"])
    cw = man["constant_window"]["value"]
    a.plot([cw, cw], [-0.6, n - 0.35], color=C["ref"], lw=0.6, ls=(0, (2, 2)), zorder=0)
    a.text(cw - 0.03, n - 0.2, "fixed 0–10 s window", ha="left", va="bottom", fontsize=FS_SMALL, color=C["note"])
    a.set_ylim(-0.6, n + 0.55)
    a.set_xlim(0, 1.02)
    a.set_xticks([0, 0.5, 1])
    a.set_xticklabels(["0", "0.5", "1"])
    a.set_xlabel("fraction of black videos")
    PS.clean_axis(a, left=False)
    a.legend(handles=[Line2D([], [], ls="", marker="o", ms=4.4, mfc="white", mec=C["ink"], mew=0.9, label="answered"),
                      Line2D([], [], ls="", marker="o", ms=2.8, mfc=C["ink"], mec=C["ink"], mew=0, label="correct")],
             loc="lower left", bbox_to_anchor=(0.05 / W, (H - 0.30) / H), bbox_transform=fig.transFigure, ncol=2,
             fontsize=FS_NOTE, handlelength=0.6, columnspacing=0.9)
    PS.panel_letter(fig, 0.0, H - 0.02, "a")
    # ---------------- b: label sensitivity
    b = PS.axes_in(fig, 2.07, 0.36, 1.58, top - 0.36)
    xl, xr = 0.0, 1.0
    b.set_xlim(-1.02, 1.95)
    b.set_ylim(0.28, 0.76)
    lines = []
    for r in paired:
        d = parse_key(r["point"])
        fam, obj = SHORT[d["fam"]], d["obj"]
        if fam == "TimeLens-8B":
            kind = "tl_base" if obj == "base" else "tl_ours"
        elif obj == "base":
            kind = "qwen_base" if fam == "Qwen-7B" else "public"
        elif obj == "sft":
            kind = "sft"
        else:
            kind = "ours7b"
        lines.append((kind, r))
    style = {"ours7b": dict(color=C["ours_ci"], lw=0.7, z=2, mk="o", ms=2.4, mfc=C["ours_ci"], mec=C["ours_ci"]),
             "sft": dict(color=C["acc"], lw=0.85, z=3, mk="o", ms=3.0, mfc=C["acc"], mec="white"),
             "public": dict(color=C["acc"], lw=0.85, z=3, mk="D", ms=3.0, mfc="white", mec=C["acc"]),
             "qwen_base": dict(color=C["base_edge"], lw=0.85, z=3, mk="o", ms=3.4, mfc="white", mec=C["base_edge"]),
             "tl_base": dict(color=C["acc"], lw=1.4, z=4, mk="P", ms=4.2, mfc=C["acc"], mec="white"),
             "tl_ours": dict(color=C["ours"], lw=1.4, z=4, mk="P", ms=4.2, mfc=C["ours"], mec="white")}
    for kind, r in lines:
        s = style[kind]
        b.plot([xl, xr], [r["sta"], r["tl"]], color=s["color"], lw=s["lw"], zorder=s["z"], solid_capstyle="round")
        b.plot([xl, xr], [r["sta"], r["tl"]], ls="", marker=s["mk"], ms=s["ms"], mfc=s["mfc"], mec=s["mec"], mew=0.7 if s["mfc"] == "white" else 0.4,
               zorder=s["z"] + 0.5)
    get = lambda kind: [r for k, r in lines if k == kind]
    pub = get("public")
    ymid = float(np.mean([r["sta"] for r in pub]))
    b.text(-0.09, max(r["sta"] for r in pub) + 0.012, "public RL\ncheckpoints", ha="right", va="bottom", fontsize=FS_SMALL, color=C["acc"], linespacing=0.95)
    qb = get("qwen_base")[0]
    b.text(-0.09, qb["sta"] + 0.004, "Qwen-7B\nbase", ha="right", va="bottom", fontsize=FS_SMALL, color=C["note"], linespacing=0.95)
    tb, to = get("tl_base")[0], get("tl_ours")[0]
    b.text(xr + 0.09, tb["tl"], "TimeLens-8B", ha="left", va="center", fontsize=FS_SMALL, color=C["acc"])
    b.text(xr + 0.09, to["tl"] + 0.012, "+ our\nadapter", ha="left", va="bottom", fontsize=FS_SMALL, color=C["ours"], linespacing=0.95)
    for r in get("sft"):
        b.text(xr + 0.09, r["tl"] - 0.004, "SFT", ha="left", va="top", fontsize=FS_SMALL, color=C["acc"])
    b.set_xticks([xl, xr])
    b.set_xticklabels(["STA\nlabels", "TimeLens\nlabels"])
    b.tick_params(axis="x", length=0, pad=2)
    b.set_yticks([0.3, 0.5, 0.7])
    b.set_ylabel("R@0.5", labelpad=2)
    PS.clean_axis(b)
    b.spines["bottom"].set_visible(False)
    b.spines["left"].set_bounds(0.3, 0.7)
    PS.panel_letter(fig, 1.83, H - 0.02, "b")
    # ---------------- c: Gamma under both label sets
    c = PS.axes_in(fig, 4.72, 0.36, 0.74, top - 0.36)
    c.axvspan(-RESO_GAMMA, RESO_GAMMA, color=C["band"], lw=0, zorder=0)
    c.axvline(0, color=C["ref"], lw=0.5, zorder=0)
    trc = blended_transform_factory(fig.transFigure, c.transData)
    nrow = len(contr)
    for i, r in enumerate(contr):
        y = nrow - 1 - i - (0.35 if r["sft"] else 0.0)
        c.text(3.80 / W, y, r["label"], transform=trc, ha="left", va="center", fontsize=FS_NOTE,
               color=C["acc"] if r["sft"] else C["ours"])
        for key, off, fill in (("sta", 0.17, True), ("tl", -0.17, False)):
            if r[key] is None:
                continue
            yy = y + off
            if r[key + "_ci"]:
                PS.ci_h(c, r[key + "_ci"][0], r[key + "_ci"][1], yy, C["acc_ci"] if fill else C["acc_ci"], lw=0.75, zorder=2)
            c.plot([r[key]], [yy], ls="", marker="o", ms=3.4, mfc=C["ink"] if fill else "white", mec=C["ink"], mew=0.75, zorder=3)
    c.set_ylim(-0.95, nrow - 0.4)
    c.set_xlim(-0.45, 2.15)
    c.set_xticks([0, 1, 2])
    c.set_xlabel("Γ (mask), nats per token", ha="right")
    c.xaxis.set_label_coords(1.0, -0.12)
    PS.clean_axis(c, left=False)
    c.legend(handles=[Line2D([], [], ls="", marker="o", ms=3.4, mfc=C["ink"], mec=C["ink"], mew=0.75, label="STA labels"),
                      Line2D([], [], ls="", marker="o", ms=3.4, mfc="white", mec=C["ink"], mew=0.75, label="TimeLens labels")],
             loc="lower left", bbox_to_anchor=(3.80 / W, (H - 0.30) / H), bbox_transform=fig.transFigure, ncol=2,
             fontsize=FS_NOTE, handlelength=0.6, columnspacing=0.9)
    PS.panel_letter(fig, 3.62, H - 0.02, "c")
    export(fig, name, dict(points=[p for p in man["points"] if "tl_src" not in p], contrast_rows=contr, dropped=man["dropped"],
                           constant_window=man["constant_window"], summary=man.get("summary")))


# ================================================================== Figure 3: mechanism (beside Table 2)
MECH_ROWS = [("base", "base model (from the full objective's paired run)"), ("SFT", "pure SFT"), ("mDPO-style", "mDPO-style"),
             ("evidence margin only", "evidence margin only"), ("no placebo bound", "no placebo bound"),
             ("no bound, uniform", "no bound, uniform weights"), ("uniform weights", "uniform weights"),
             ("full objective", "all terms")]
MECH_GAP_AFTER = {"mDPO-style", "evidence margin only", "no bound, uniform"}


def fig5_mechanism():
    name = "fig5_mechanism"
    man = capture(name, V5.fig5_mechanism)
    src = {p["point"]: p for p in man["points"]}
    W, H = 2.53, 2.02
    fig = plt.figure(figsize=(W, H))
    ax = PS.axes_in(fig, 1.02, 0.36, 1.44, 1.36)
    ys, y = [], 0.0
    for lab, key in MECH_ROWS:
        ys.append(y)
        y -= 1.0 + (0.45 if lab in MECH_GAP_AFTER else 0.0)
    base_dO = src[MECH_ROWS[0][1]]["dO"]
    ax.axvline(base_dO, color=C["placebo"], lw=0.6, ls=(0, (2.5, 2)), alpha=0.8, zorder=0)
    ax.text(base_dO + 0.06, ys[0] + 0.72, "base $d_O$", ha="left", va="bottom", fontsize=FS_SMALL, color=C["placebo"])
    data = []
    for yy, (lab, key) in zip(ys, MECH_ROWS):
        p = src[key]
        ax.plot([p["dO"], p["dE"]], [yy, yy], color=C["faint"], lw=0.8, zorder=1)
        ax.plot([p["dE"]], [yy], ls="", marker="o", ms=4.0, mfc=C["evidence"], mec="white", mew=0.4, zorder=3)
        ax.plot([p["dO"]], [yy], ls="", marker="s", ms=3.7, mfc=C["placebo"], mec="white", mew=0.4, zorder=4)
        data.append(dict(row=lab, source_point=key, dE=p["dE"], dO=p["dO"]))
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in MECH_ROWS])
    for t in ax.get_yticklabels():
        t.set_fontsize(FS_NOTE)
        if t.get_text() == "full objective":
            t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
    ax.set_xlim(-0.3, 3.75)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("likelihood drop (nats per token)")
    PS.clean_axis(ax)
    ax.spines["left"].set_visible(False)
    fig.legend(handles=[Line2D([], [], ls="", marker="o", ms=4.0, mfc=C["evidence"], mec="white", mew=0.4, label="evidence edit, $d_E$"),
                        Line2D([], [], ls="", marker="s", ms=3.7, mfc=C["placebo"], mec="white", mew=0.4, label="placebo edit, $d_O$")],
               loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, fontsize=FS_NOTE, handlelength=0.6, columnspacing=1.0)
    export(fig, name, dict(points=man["points"], rows_top_to_bottom=data, base_dO_reference=base_dO, dropped=man["dropped"]))


# ================================================================== Figure 4: dose
def fig6_dose():
    name = "fig6_dose"
    man = capture(name, V5.fig6_dose)
    keys = {V5.key_str(k): k for k in V5.GROUPED + V5.BETA003 + [V5.SFT]}
    rows = []
    for p in man["points"]:
        k = keys[p["point"]]
        g = V5.gam(k)
        assert g["gamma_mask"] == p["gamma"]
        rows.append(dict(p, fam=SHORT[k[0]], obj=k[1], alpha=k[2], beta=k[3], gamma_ci=g["gamma_mask_ci"],
                         panel=p.get("panel", "alpha")))
    # 09-25: Qwen2.5-VL-3B and TimeLens-8B get alpha columns of their own; every registered setting is drawn
    have = {r["point"] for r in rows}
    for r in rows:
        r["single"] = False
    for fam in (V5.TL,):   # 09-25 ruling: no Qwen2.5-VL-3B in the paper
        for a in (0.25, 0.5, 1.0):
            k = V5.g_(fam, a)
            g = V5.gam(k)
            if V5.key_str(k) in have or g is None:
                continue
            dv, dci, dag, denv = V5.C.delta3(k, V5.CS)
            p = dict(point=V5.key_str(k), gamma=g["gamma_mask"], gamma_src=V5.src(g), delta=dv, delta_ci=dci, dagger=dag, delta_env=denv)
            man["points"].append(p)
            rows.append(dict(p, fam=SHORT[fam], obj="grouped", alpha=a, beta=0.1, gamma_ci=g["gamma_mask_ci"], panel="alpha"))
    W, H = 5.5, 1.80
    fig = plt.figure(figsize=(W, H))
    fams = ["Qwen-7B", "TimeZero", "Time-R1", "VideoChat-R1", "TimeLens-8B"]
    NA = len(fams)
    left, gap, bw = 0.60, 0.06, 0.62
    colw = (W - left - 0.06 - 0.24 - bw - gap * (NA - 1)) / NA
    xs = [left + i * (colw + gap) for i in range(NA)]
    xs.append(xs[-1] + colw + 0.24)            # beta
    widths = [colw] * NA + [bw]
    titles = fams + ["Qwen-7B"]
    y_lo, h = 0.36, 0.52
    y_up = y_lo + h + 0.08
    out = []
    for i, (x0, w, title) in enumerate(zip(xs, widths, titles)):
        up, lo = PS.axes_in(fig, x0, y_up, w, h), PS.axes_in(fig, x0, y_lo, w, h)
        up.axhspan(-RESO_GAMMA, RESO_GAMMA, color=C["band"], lw=0, zorder=0)
        for ax in (up, lo):
            ax.axhline(0, color=C["ref"], lw=0.5, zorder=0)
        if i < NA:
            recs = sorted([r for r in rows if r["panel"] == "alpha" and r["fam"] == fams[i] and not r.get("single")], key=lambda r: r["alpha"])
            xsv = [r["alpha"] for r in recs]
            for ax, met in ((up, "gamma"), (lo, "delta")):
                vals = [r[met] if met == "gamma" else 100 * r[met] for r in recs]
                for r, xv in zip(recs, xsv):
                    ci = r["gamma_ci"] if met == "gamma" else ([100 * c for c in r["delta_ci"]] if r["delta_ci"] else None)
                    if ci:
                        PS.ci_v(ax, xv, ci[0], ci[1], C["ours_ci"], lw=0.8, zorder=1)
                ax.plot(xsv, vals, color=C["ours"], lw=1.0, zorder=2)
                ax.plot(xsv, vals, ls="", marker=MARK[fams[i]], ms=3.6, mfc=C["ours"], mec="white", mew=0.4, zorder=3)
            out += [dict(column=title, point=r["point"], alpha=r["alpha"], gamma=r["gamma"], gamma_ci=r["gamma_ci"],
                         delta_pp=None if r["delta"] is None else 100 * r["delta"]) for r in recs]
            for ax in (up, lo):
                ax.set_xscale("log")
                ax.set_xlim(0.085, 1.4)
                ax.set_xticks(xsv)                       # only the settings that were trained
                ax.set_xticklabels([fnum(v) for v in xsv])              # 0.25 0.5 1 (leading zeros kept)
                ax.minorticks_off()
        else:
            recs = sorted([r for r in rows if r["panel"] == "beta"], key=lambda r: r["beta"])
            xp = [0.01 if r["beta"] == 0 else r["beta"] for r in recs]
            for ax, met in ((up, "gamma"), (lo, "delta")):
                vals = [r[met] if met == "gamma" else 100 * r[met] for r in recs]
                ax.plot(xp[1:], vals[1:], color=C["ours"], lw=1.0, zorder=2)
                for r, xv, v in zip(recs, xp, vals):
                    ci = r["gamma_ci"] if met == "gamma" else ([100 * c for c in r["delta_ci"]] if r["delta_ci"] else None)
                    sft = r["beta"] == 0
                    if ci:
                        PS.ci_v(ax, xv, ci[0], ci[1], C["acc_ci"] if sft else C["ours_ci"], lw=0.8, zorder=1)
                    ax.plot([xv], [v], ls="", marker="o", ms=3.6, mfc=C["acc"] if sft else C["ours"], mec="white", mew=0.4, zorder=3)
            out += [dict(column="beta", point=r["point"], beta=r["beta"], gamma=r["gamma"], gamma_ci=r["gamma_ci"],
                         delta_pp=None if r["delta"] is None else 100 * r["delta"]) for r in recs]
            for ax in (up, lo):
                ax.set_xscale("log")
                ax.set_xlim(0.0065, 0.15)
                ax.set_xticks([0.01, 0.03, 0.1])
                ax.set_xticklabels(["SFT", "0.03", "0.1"])
                ax.minorticks_off()
        up.set_ylim(-0.3, 2.9)
        up.set_yticks([0, 1, 2])
        lo.set_ylim(-11, 27)
        lo.set_yticks([-10, 0, 10, 20])
        PS.clean_axis(up, bottom=False)
        PS.clean_axis(lo)
        if i not in (0, NA):
            for ax in (up, lo):
                ax.tick_params(axis="y", left=False, labelleft=False)
                ax.spines["left"].set_visible(False)
        if i == NA:
            for ax in (up, lo):
                ax.tick_params(axis="y", labelleft=False)
        fig.text((x0 + w / 2) / W, (y_up + h + 0.05) / H, title, ha="center", va="bottom", fontsize=FS_NOTE)
    fig.axes[0].set_ylabel("Γ (mask)\nnats per token", fontsize=FS_LABEL, linespacing=1.0)
    fig.axes[1].set_ylabel("ΔR@0.5\n(points)", fontsize=FS_LABEL, linespacing=1.0)
    fig.axes[0].yaxis.set_label_coords(-0.30, 0.5)
    fig.axes[1].yaxis.set_label_coords(-0.30, 0.5)
    fig.text((xs[0] + xs[NA - 1] + widths[NA - 1]) / 2 / W, 0.03 / H, "α, weight of clean supervision", ha="center", va="bottom", fontsize=FS_LABEL)
    fig.text((xs[NA] + widths[NA] / 2) / W, 0.03 / H, "β", ha="center", va="bottom", fontsize=FS_LABEL)
    PS.panel_letter(fig, 0.0, H - 0.01, "a")
    PS.panel_letter(fig, xs[NA] - 0.20, H - 0.01, "b")
    export(fig, name, dict(points=man["points"], drawn=out, dropped=man["dropped"]))


# ================================================================== Figure 5: readouts
def s6_table_values():
    """Delta share (mask) of each trained row of tables/S6_behavioural.tex, keyed by the row label.
    The column is located by its header, so the parser survives added columns / changed number formatting."""
    lines = open(os.path.join(ROOT, "tables", "S6_behavioural.tex"), encoding="utf-8").read().splitlines()
    out, cur, col = {}, None, None
    for ln in lines:
        if "share mask" in ln and " & " in ln:
            col = [i for i, cell in enumerate(ln.rstrip("\\ ").split(" & ")) if "share mask" in cell][0]
        elif ln.startswith("\\multicolumn{") and "\\textit{" in ln:     # 2026-09-24 layout: one group header per model
            cur = re.search(r"\\textit\{(.*)\}\}\s*\\\\\s*$", ln).group(1).strip()
        elif "& base &" in ln:
            cur = ln.split(" & ")[0].strip()
        elif ln.strip().startswith("\\quad trained") and cur and col is not None:
            cell = ln.rstrip("\\ ").split(" & ")[col]
            m = re.findall(r"(\$-\$|-|−|\+)?(\d+\.\d)", cell)
            v = [(-1 if s in ("$-$", "-", "−") else 1) * float(x) for s, x in m]
            out[cur] = dict(est=v[0], lo=v[1], hi=v[2], cell=cell)
    return out


def s6_match(labels, fam, sft, alpha):
    """the S6 row of one Figure 5c configuration (video-first rows only)"""
    disp = {"Qwen2.5-VL-7B": "Qwen2.5-VL-7B", "TimeZero-7B": "TimeZero", "Time-R1-7B": "Time-R1", "VideoChat-R1-7B": "VideoChat-R1"}[fam]
    hit = [l for l in labels if l.startswith(disp + ",") and "(video-first)" in l and "text-first" not in l
           and (("SFT" in l) if sft else ("{=}" + fnum(alpha) + "$" in l or "{=}" + fnum(alpha) + " " in l))]
    assert len(hit) == 1, (fam, sft, alpha, hit)
    return hit[0]


def behav_rows_s6_rule():
    def cond(d, op):
        if d is None:
            return None
        c = d.get(op)
        return None if c is None else c["R@0.5"]["est"]
    OPS6 = ["none", "mask_evid", "mask_rand", "swap_evid", "swap_rand", "black"]

    def complete(x):
        return x is not None and all(cond(x.get(side), o) is not None for side in ("base_conditions", "arm_conditions") for o in OPS6)
    rows, dropped = [], []
    for k, ds, r, envs in V5.C.rows_for("behav"):
        if k[6] != "v":
            dropped.append(dict(point=V5.key_str(k), reason="evaluated text-first")); continue
        if k[1] not in ("sft", "grouped") or (k[1] == "grouped" and k[3] != 0.1):
            dropped.append(dict(point=V5.key_str(k), reason="not a placebo-controlled model at beta=0.1 or SFT")); continue
        rule = "canonical"
        if not complete(r):   # build_appendix_tables_v3.build_S6: replaced by the key's six-operator block
            alts = sorted((x for x in V5.C.R if x.get("kind") == "behav" and x.get("_key") == k and complete(x)), key=V5.rec_name if hasattr(V5, "rec_name") else (lambda x: x.get("name") or x.get("label") or ""))
            if alts:
                r, rule = alts[0], "S6 six-operator block"
        if r is None or r.get("nshare_mask") is None:
            dropped.append(dict(point=V5.key_str(k), reason=f"no video-first behavioural run (runs in {envs})")); continue
        rows.append((k, r, rule))
    return rows, dropped


def fig5_readouts():
    """round7/P5 redesign: no shared shape key; each panel names its models and states its reading rule in place.
    Same points and values as before (panel c still asserted cell by cell against S6), same size (5.5 x 2.42 in)."""
    name = "fig5_readouts"
    man = capture(name, V5.fig5_readouts)
    cross = [p for p in man["points"] if "d_psi_own" in p]
    abst = [p for p in man["points"] if "clean" in p]
    W, H = 5.5, 2.42
    fig = plt.figure(figsize=(W, H))
    y0, hh = 0.50, 1.74
    ytop = y0 + hh
    # ---------------- a: crossed queries
    a = PS.axes_in(fig, 0.58, y0, 1.36, hh)
    xl, yl = (-0.7, 2.9), (-2.05, 1.55)
    a.plot([-2.1, 1.6], [-2.1, 1.6], color=C["ref"], lw=0.5, ls=(0, (3, 2)), zorder=1)
    a.axhline(0, color=C["ref"], lw=0.5, zorder=1)
    pos = {}
    for p in cross:
        d = parse_key(p["point"])
        fam = SHORT[d["fam"]]
        tf = d.get("eval") == "t"
        colr = {"sft": C["acc"], "single-margin": C["em"]}.get(d["obj"], C["ours"])
        a.plot([p["d_psi_own"]], [p["d_psi_cross"]], ls="", marker=MARK[fam], ms=4.3, mfc=colr,
               mec=colr if tf else "white", mew=0.8 if tf else 0.45, fillstyle="left" if tf else "full",
               markerfacecoloralt="white", zorder=5 if d["obj"] in ("sft", "single-margin") else 4)
        pos[(d["obj"], fam, tf, d.get("a"))] = (p["d_psi_own"], p["d_psi_cross"])
    a.set_xlim(*xl)
    a.set_ylim(*yl)
    # reading rule, written into the panel
    fig.canvas.draw()
    p0, p1 = a.transData.transform((0, 0)), a.transData.transform((1, 1))
    ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    a.text(0.62, 0.86, "reacts only to where\nthe edit lands", rotation=ang, rotation_mode="anchor", ha="center", va="bottom",
           fontsize=FS_SMALL, color=C["note"], linespacing=0.95, zorder=6)
    a.text(2.88, -0.10, "below 0:\nreads the query", ha="right", va="top", fontsize=FS_SMALL, color=C["note"], linespacing=0.95)
    sx, sy = pos[("sft", "Qwen-7B", False, None)]
    a.text(-0.62, -0.30, "SFT", ha="left", va="center", fontsize=FS_SMALL, color=C["acc"])
    leader(a, (sx - 0.03, sy - 0.08), (-0.42, -0.22))
    ex, ey = pos[("single-margin", "Qwen-7B", False, 0.5)]
    a.text(-0.62, -1.05, "evidence\nmargin only", ha="left", va="center", fontsize=FS_SMALL, color=C["em"], linespacing=0.95)
    leader(a, (ex - 0.07, ey), (0.42, -1.02))
    tx, ty = pos[("grouped", "TimeZero", True, 0.5)]
    a.text(tx, ty - 0.17, "TimeZero,\ntext-first", ha="center", va="top", fontsize=FS_SMALL, color=C["ours"], linespacing=0.95)
    a.text(2.12, 0.33, "VideoChat-R1", ha="center", va="center", fontsize=FS_SMALL, color=C["ours"])
    a.text(2.88, -1.62, "placebo-controlled:\nQwen-7B, TimeZero, Time-R1", ha="right", va="top", fontsize=FS_SMALL,
           color=C["ours"], linespacing=0.95)
    a.set_xticks([0, 1, 2])
    a.set_yticks([-1, 0, 1])
    a.set_xlabel("change in partner contrast,\nevent's own query", linespacing=1.0)
    a.set_ylabel("change in partner contrast,\npartner event's query", linespacing=1.0)
    PS.clean_axis(a)
    PS.panel_letter(fig, 0.02, H - 0.01, "a")
    # ---------------- b: matched vs mismatched abstention, one row per model
    brows_spec = [("Qwen-7B", "base"), ("Qwen-7B", "sft"), ("Qwen-7B", ("grouped", 0.5, 0.1)), ("Qwen-7B", ("grouped", 0.5, 0.03)),
                  ("Time-R1", "base"), ("Time-R1", ("grouped", 0.5, 0.1)), ("Time-R1", ("grouped", 1.0, 0.1)),
                  ("TimeZero", "base"), ("TimeZero", "sft"), ("TimeZero", ("grouped", 1.0, 0.1)), ("TimeZero", ("grouped", 0.5, 0.03))]
    lookup = {}
    for p in abst:
        d = parse_key(p["point"])
        fam = SHORT[d["fam"]]
        if d["obj"] in ("base", "sft"):
            lookup[(fam, d["obj"])] = (p, d)
        else:
            lookup[(fam, (d["obj"], d["a"], d["b"]))] = (p, d)
    assert len(lookup) == len(abst) == len(brows_spec), (len(lookup), len(abst))
    bh = 1.50
    b = PS.axes_in(fig, 3.00, y0, 0.86, bh)
    trb = blended_transform_factory(fig.transFigure, b.transData)
    ydraw, yy, prev = [], 0.0, None
    brow_out = []
    for fam, spec in brows_spec:
        if prev is not None and fam != prev:
            yy -= 0.45
        prev = fam
        p, d = lookup[(fam, spec)]
        if spec == "base":
            lab = f"{fam} public" if fam in ("Time-R1", "TimeZero") else f"{fam} base"
            colr, mk, mfc = (C["acc"], "D", "white") if fam in ("Time-R1", "TimeZero") else (C["base_edge"], "o", "white")
        elif spec == "sft":
            lab = f"{fam} SFT" + (" (text-first)" if d.get("train") == "t" else "")
            colr, mk, mfc = C["acc"], MARK[fam], C["acc"]
        else:
            lab = f"{fam} " + (f"β={fnum(spec[2])}" if spec[2] != 0.1 else f"α={fnum(spec[1])}")
            colr, mk, mfc = C["ours"], MARK[fam], C["ours"]
        m_, f_ = p["clean"], p["foreign"]
        b.plot([m_, f_], [yy, yy], color=C["faint"] if colr != C["ours"] else C["ours_ci"], lw=1.1, zorder=1, solid_capstyle="butt")
        b.plot([m_], [yy], ls="", marker="|", ms=5.0, mec=colr, mew=1.1, zorder=3)
        b.plot([f_], [yy], ls="", marker=mk, ms=3.9, mfc=mfc, mec=colr if mfc == "white" else "white", mew=0.8 if mfc == "white" else 0.4, zorder=4)
        b.text(2.95 / W, yy, lab, transform=trb, ha="right", va="center", fontsize=FS_SMALL,
               color=C["ours"] if colr == C["ours"] else C["ink"])
        brow_out.append(dict(row=lab, point=p["point"], matched=m_, mismatched=f_))
        ydraw.append(yy)
        yy -= 1.0
    b.set_ylim(yy + 0.4, 0.75)
    b.set_xlim(-4, 92)
    b.set_xticks([0, 40, 80])
    b.set_xlabel("declines to answer (%)")
    PS.clean_axis(b, left=False)
    b.legend(handles=[Line2D([], [], ls="", marker="|", ms=5.0, mec=C["ink"], mew=1.1, label="query about this video (matched)"),
                      Line2D([], [], ls="", marker="o", ms=3.9, mfc=C["ink"], mec="white", mew=0.4, label="query about another video (mismatched)")],
             loc="upper left", bbox_to_anchor=(2.20 / W, (H - 0.04) / H), bbox_transform=fig.transFigure, ncol=1,
             fontsize=FS_SMALL, handlelength=0.6, handletextpad=0.4, labelspacing=0.25)
    PS.panel_letter(fig, 2.04, H - 0.01, "b")
    # ---------------- c: evidence share, S6 rule
    rows, dropped_c = behav_rows_s6_rule()
    s6 = s6_table_values()
    crow = []
    for k, r, rule in rows:
        lab_s6 = s6_match(list(s6), k[0], k[1] == "sft", k[2])
        est, ci = 100 * r["nshare_mask"], [100 * v for v in r["nshare_mask_ci"]]
        t = s6[lab_s6]
        assert (round(est, 1), round(ci[0], 1), round(ci[1], 1)) == (t["est"], t["lo"], t["hi"]), (lab_s6, est, ci, t)
        crow.append(dict(key=V5.key_str(k), fam=SHORT[k[0]], sft=k[1] == "sft", alpha=k[2], est=est, ci=ci, rule=rule,
                         src=dict(file=os.path.basename((r.get("src") or {}).get("file") or ""), sha16=(r.get("src") or {}).get("sha16"), env=r.get("_env")),
                         s6_row=lab_s6, s6_cell=t["cell"]))
    assert len(crow) == 5, [c["key"] for c in crow]
    crow.sort(key=lambda c: -c["est"])
    ch = 1.50
    c = PS.axes_in(fig, 4.60, y0, 0.84, ch)
    trc = blended_transform_factory(fig.transFigure, c.transData)
    c.axvline(0, color=C["ref"], lw=0.5, zorder=0)
    n = len(crow)
    for i, rr in enumerate(crow):
        y = n - 1 - i
        colr = C["acc"] if rr["sft"] else C["ours"]
        PS.ci_h(c, rr["ci"][0], rr["ci"][1], y, C["acc_ci"] if rr["sft"] else C["ours_ci"], lw=0.9, zorder=2)
        c.plot([rr["est"]], [y], ls="", marker=MARK[rr["fam"]], ms=4.3, mfc=colr, mec="white", mew=0.45, zorder=3)
        lab = f"{rr['fam']}, " + ("SFT" if rr["sft"] else f"α={fnum(rr['alpha'])}")
        c.text(4.55 / W, y, lab, transform=trc, ha="right", va="center", fontsize=FS_SMALL, color=colr)
    c.set_ylim(-0.6, n - 0.4)
    c.set_xlim(-22, 36)
    c.set_xticks([-20, 0, 20])
    c.set_xlabel("change in evidence\nshare (points)", linespacing=1.0)
    PS.clean_axis(c, left=False)
    # reading rule above the panel
    fig.text(5.44 / W, (y0 + ch + 0.08) / H, "right of 0: needs the event's frames\nmore than its base does", ha="right", va="bottom",
             fontsize=FS_SMALL, color=C["note"], linespacing=0.95)
    PS.panel_letter(fig, 3.93, H - 0.01, "c")
    export(fig, name, dict(points_a=cross, points_b=brow_out, points_c=crow, dropped=man["dropped"], dropped_c=dropped_c,
                           note_c="panel c follows the S6 rule (six-operator block); each value asserted equal to tables/S6_behavioural.tex"))
    return crow


# ================================================================== Figure 9: OOD
def fig9_ood():
    name = "fig9_ood"
    man = capture(name, V5.fig9_ood)
    keys = {V5.key_str(k): k for k in V5.GROUPED + V5.BETA003 + [V5.SFT]}
    W, H = 5.5, 2.15
    fig = plt.figure(figsize=(W, H))
    a = PS.axes_in(fig, 0.56, 0.42, 2.55, 1.58)
    RESO = 4.9
    a.axvspan(-RESO, RESO, color=C["band"], lw=0, zorder=0)
    a.axhline(0, color=C["ref"], lw=0.5, zorder=1)
    a.text(0, -0.31, "accuracy resolution", ha="center", va="bottom", fontsize=FS_SMALL, color=C["band_text"])
    pa = []
    for p in [q for q in man["points"] if q["panel"] == "a"]:
        k = keys[p["point"]]
        g = V5.gam(k, V5.QV)
        assert g["gamma_mask"] == p["gamma"]
        fam = SHORT[k[0]]
        ours = k[1] == "grouped"
        colr = C["ours"] if ours else C["acc"]
        cic = C["ours_ci"] if ours else C["acc_ci"]
        if p["delta_ci"]:
            PS.ci_h(a, p["delta_ci"][0], p["delta_ci"][1], p["gamma"], cic, lw=0.75, zorder=2)
        PS.ci_v(a, p["delta"], g["gamma_mask_ci"][0], g["gamma_mask_ci"][1], cic, lw=0.75, zorder=2)
        a.plot([p["delta"]], [p["gamma"]], ls="", marker=MARK[fam], ms=4.6, mfc=colr, mec="white", mew=0.45, zorder=4)
        lab = "SFT" if k[1] == "sft" else f"{fam}, " + (f"β={fnum(k[3])}" if k[3] == 0.03 else f"α={fnum(k[2])}")
        pa.append(dict(p, gamma_ci=g["gamma_mask_ci"], label=lab, x=p["delta"], y=p["gamma"], colr=colr))
    place = {"TimeZero, α=0.25": (-9.3, 2.02, "left", (-5.2, 1.95)), "TimeZero, α=0.5": (-9.3, 0.62, "left", (-6.0, 0.69)),
             "Qwen-7B, α=1": (1.2, 0.62, "left", (1.1, 0.64)), "Qwen-7B, β=0.03": (0.4, 1.90, "left", (0.3, 1.86)),
             "Qwen-7B, α=0.5": (1.9, 1.02, "left", (1.8, 1.06)), "SFT": (6.8, 0.30, "center", (6.8, 0.23))}
    for q in pa:
        tx, ty, ha, end = place[q["label"]]
        a.text(tx, ty, q["label"], ha=ha, va="center", fontsize=FS_SMALL, color=q["colr"], zorder=6)
        leader(a, (q["x"], q["y"]), end)
    a.set_xlim(-9.5, 10.5)
    a.set_ylim(-0.35, 2.12)
    a.set_xticks([-5, 0, 5, 10])
    a.set_yticks([0, 0.5, 1, 1.5, 2])
    a.set_xlabel("accuracy change on QVHighlights, ΔR@0.5 (points)")
    a.set_ylabel("Γ (mask), nats per token")
    PS.clean_axis(a)
    PS.panel_letter(fig, 0.02, H - 0.02, "a")
    # ---------------- b: black-video answering, base (open) vs trained (filled), no printed values
    b = PS.axes_in(fig, 3.78, 0.42, 1.62, 1.58)
    pb = [q for q in man["points"] if q["panel"] == "b"]
    xs, xt, grp = [], [], []
    x = 0.0
    for q in pb:
        fam, ds = q["point"].rsplit(" ", 1)
        fam = SHORT[fam]
        if grp and grp[-1][0] != fam:
            x += 0.6
        b.plot([x, x], [q["base"], q["trained"]], color=C["faint"], lw=1.0, zorder=1)
        b.plot([x], [q["base"]], ls="", marker=MARK[fam], ms=4.6, mfc="white", mec=C["base_edge"], mew=0.9, zorder=3)
        b.plot([x], [q["trained"]], ls="", marker=MARK[fam], ms=4.6, mfc=C["ours"], mec="white", mew=0.45, zorder=4)
        xs.append(x); xt.append("STA" if ds == "STA" else "QVH"); grp.append((fam, x))
        x += 1.0
    b.set_xticks(xs)
    b.set_xticklabels(xt)
    b.tick_params(axis="x", length=0, pad=2)
    for fam in dict.fromkeys(g[0] for g in grp):
        xx = [g[1] for g in grp if g[0] == fam]
        b.text(float(np.mean(xx)), -0.20, fam, transform=b.get_xaxis_transform(), ha="center", va="top", fontsize=FS_NOTE)
    b.set_xlim(-0.5, x - 0.5)
    b.set_ylim(-0.04, 1.05)
    b.set_yticks([0, 0.5, 1])
    b.set_yticklabels(["0", "0.5", "1"])
    b.set_ylabel("black-video answer rate")
    PS.clean_axis(b)
    b.spines["bottom"].set_visible(False)
    from matplotlib.legend_handler import HandlerTuple
    fams_b = list(dict.fromkeys(g[0] for g in grp))   # one marker per base drawn in the panel (legend matches the plot)
    h_base = tuple(Line2D([], [], ls="", marker=MARK[f], ms=4.6, mfc="white", mec=C["base_edge"], mew=0.9) for f in fams_b)
    h_ours = tuple(Line2D([], [], ls="", marker=MARK[f], ms=4.6, mfc=C["ours"], mec="white", mew=0.45) for f in fams_b)
    b.legend(handles=[h_base, h_ours], labels=["base", "placebo-controlled, α=0.5"],
             handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
             loc="upper right", bbox_to_anchor=(5.45 / W, 0.995), bbox_transform=fig.transFigure, ncol=1, fontsize=FS_SMALL,
             handlelength=0.6 * len(fams_b) + 0.3)
    PS.panel_letter(fig, 3.22, H - 0.02, "b")
    export(fig, name, dict(points=man["points"], a_with_gamma_ci=[{k: v for k, v in q.items() if k != "colr"} for q in pa],
                           dropped=man["dropped"], note="panel b prints no values (the STA ones are in Figure 2a and Section 3)"))


# ================================================================== Figure 10: two cases
def fig8_cases():
    import cv2
    from matplotlib.patches import Rectangle
    import plot_fig8_cases_v3 as SRC
    name = "fig8_cases"
    srcman = json.load(open(SRC.MAN, encoding="utf-8"))
    cases = {c["qid"]: c for c in srcman["cases"]}
    W, H = 4.62, 3.02
    fig = plt.figure(figsize=(W, H))

    def txt(x, y, s, **kw):
        return fig.text(x / W, y / H, s, **kw)

    def frame160(cap, t):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        assert ok
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        h, w = fr.shape[:2]
        if w / h > 4 / 3:
            nw = int(h * 4 / 3); x0 = (w - nw) // 2; fr = fr[:, x0:x0 + nw]
        else:
            nh = int(w * 3 / 4); y0 = (h - nh) // 2; fr = fr[y0:y0 + nh]
        return cv2.resize(fr, (160, 120), interpolation=cv2.INTER_AREA)
    record = dict(sources=[dict(file=os.path.basename(SRC.MAN), sha16=sha16(SRC.MAN))], cases=[])
    cw, gapc = 2.18, 0.20
    for col, qid in enumerate(SRC.CASE_ORDER):
        c = cases[qid]
        x0 = 0.02 + col * (cw + gapc)
        path = os.path.join(ROOT, "assets", "source_videos", c["video"])
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == c["source_video"]["sha256"]
        _, ev, times, end = SRC.case_strip(c)
        cap = cv2.VideoCapture(path)
        frames = [frame160(cap, t) for t in times]
        cap.release()
        query = c["query"].rstrip(".")
        txt(x0, H - 0.02, "ab"[col], fontsize=PS.FS_PANEL, fontweight="bold", va="top")
        txt(x0 + 0.17, H - 0.03, "“" + "\n".join(textwrap.wrap(query, 34)) + "”", fontsize=FS_NOTE, va="top", style="italic", linespacing=1.0)
        tw = (cw - 0.03 * 4) / 5
        th = tw * 0.75
        ytop = H - 0.40
        for i, (im, e, t) in enumerate(zip(frames, ev, times)):
            r, j = divmod(i, 5)
            xx = x0 + j * (tw + 0.03)
            yy = ytop - th - r * (th + 0.17)
            ia = fig.add_axes([xx / W, yy / H, tw / W, th / H])
            ia.imshow(im, interpolation="bilinear")
            ia.set_axis_off()
            ia.add_patch(Rectangle((-0.5, -0.5), 160, 120, fill=False, ec=C["evidence"] if e else "#D0D0D0", lw=1.1 if e else 0.4, clip_on=False))
            txt(xx + tw / 2, yy - 0.015, f"{t:.1f} s", ha="center", va="top", fontsize=FS_SMALL, color=C["note"])
        # intervals
        ybase = ytop - 2 * th - 0.17 - 0.24
        tl = fig.add_axes([(x0 + 0.62) / W, (ybase - 0.40) / H, (cw - 0.64) / W, 0.40 / H])
        for yv, (lab, key, colr) in zip([2, 1, 0], [("annotation", "gt", C["evidence"]), ("base", "base", C["base_edge"]),
                                                    ("trained", "grouped", C["ours"])]):
            iv = c["gt_seconds"] if key == "gt" else c["records"][key]["prediction"]
            tl.plot([0, end], [yv, yv], color="#E6E6E6", lw=0.6)
            tl.plot(iv, [yv, yv], color=colr, lw=2.4, solid_capstyle="butt")
            record["cases"].append(dict(qid=qid, kind="interval", key=key, seconds=iv))
        tl.set_xlim(0, end)
        tl.set_ylim(-0.5, 2.5)
        tl.set_yticks([2, 1, 0])
        tl.set_yticklabels(["annotation", "base", "trained"])
        tl.set_xticks([0, end])
        tl.set_xticklabels(["0 s", f"{end:.1f} s"])
        tl.get_xticklabels()[0].set_ha("left")
        tl.get_xticklabels()[-1].set_ha("right")
        tl.tick_params(axis="both", length=0, pad=2, labelsize=FS_SMALL)
        for sp in tl.spines.values():
            sp.set_visible(False)
        for t_ in tl.get_yticklabels():
            t_.set_color({"annotation": C["evidence"], "base": C["note"], "trained": C["ours"]}[t_.get_text()])
        # likelihood drop block
        yb = ybase - 0.40 - 0.30
        txt(x0, yb, "likelihood drop under swap", fontsize=FS_NOTE, fontweight="bold", va="baseline")
        txt(x0 + cw, yb, "nats per token", fontsize=FS_SMALL, color=C["note"], ha="right", va="baseline")
        cellx = [x0 + 0.62, x0 + 0.62 + (cw - 0.62) / 2 + 0.04]
        cellw = (cw - 0.62) / 2 - 0.06
        for xx, lab in zip(cellx, ["evidence frames", "placebo frames"]):
            txt(xx + cellw / 2, yb - 0.17, lab, ha="center", va="baseline", fontsize=FS_SMALL,
                color=C["evidence"] if lab.startswith("evidence") else C["placebo"])
        for row, key in enumerate(["base", "grouped"]):
            yy = yb - 0.36 - row * 0.21
            txt(x0, yy, {"base": "base", "grouped": "trained"}[key], va="baseline", fontsize=FS_NOTE,
                color=C["note"] if key == "base" else C["ours"])
            rec = c["records"][key]
            for xx, field, colr in zip(cellx, ["dE_swap", "dO_swap"], [C["evidence"], C["placebo"]]):
                v = rec[field]
                s = f"{v:.2f}" if v >= 0 else f"−{abs(v):.2f}"
                txt(xx + cellw, yy, s, ha="right", va="baseline", fontsize=FS_NOTE)
                bar = fig.add_axes([xx / W, (yy - 0.035) / H, (cellw - 0.30) / W, 0.09 / H])
                bar.set_xlim(-0.1, 4.2)
                bar.set_ylim(-1, 1)
                bar.set_axis_off()
                bar.plot([0, 4], [0, 0], color="#EBEBEB", lw=2.2, solid_capstyle="butt")
                bar.plot([0, max(v, 0)], [0, 0], color=colr, lw=2.2, solid_capstyle="butt")
                bar.plot([0, 0], [-0.8, 0.8], color=C["ref"], lw=0.4)
                record["cases"].append(dict(qid=qid, kind="likelihood_drop_swap", arm=key, field=field, value=v,
                                            source=rec["source"]["path"], sha256=rec["source"]["sha256"]))
    fig.add_artist(Line2D([(0.02 + cw + gapc / 2) / W] * 2, [0.03, 0.97], transform=fig.transFigure, color="#DDDDDD", lw=0.5))
    export(fig, name, record)


# ================================================================== Figure 10 (appendix): training dynamics
import glob                                                  # noqa: E402
DYN = _SUPP + "/results/judgment_20260925_native/dyn"


def fig10_dynamics():
    """Gamma (mask) and the trained model's own placebo drop at checkpoints every 50 steps (8 pairs per step), each
    checkpoint audited against the base on the machine it was trained on (read_14Ln_v1 products in DYN)."""
    name = "fig10_dynamics"
    runs = [("pc_a05_s1", "node2", C["ours"], C["ours_ci"]), ("sft_s1", "node4", C["acc"], C["acc_ci"]),
            ("sft_s2", "node4", C["acc"], C["acc_ci"])]
    W, H = 5.5, 1.75
    fig = plt.figure(figsize=(W, H))
    axg = PS.axes_in(fig, 0.62, 0.40, 2.05, 1.22)
    axo = PS.axes_in(fig, 3.36, 0.40, 2.05, 1.22)
    axg.axhspan(-RESO_GAMMA, RESO_GAMMA, color=C["band"], lw=0, zorder=0)
    for ax in (axg, axo):
        ax.axhline(0, color=C["ref"], lw=0.5, zorder=0)
    drawn, base_dO = [], []
    for tag, mach, col, cicol in runs:
        pts = []
        for st in (50, 100, 150, 200, 250, 300):
            hits = glob.glob(os.path.join(DYN, f"gamma_14Ln_dyn_{tag}_step{st}_{mach}_nat_o0n200R3*.json"))
            assert len(hits) == 1, (tag, st, hits)
            d = json.load(open(hits[0], encoding="utf-8"))
            m = d["per_op"]["mask"]
            pts.append((8 * st, m["gamma"], m["gamma_ci95"], m["arm_on_cohort"]["dO"]))
            base_dO.append(m["base_on_cohort"]["dO"])
            drawn.append(dict(run=tag, step=st, pairs=8 * st, gamma=m["gamma"], gamma_ci=m["gamma_ci95"],
                              dO_arm=m["arm_on_cohort"]["dO"], dO_base=m["base_on_cohort"]["dO"],
                              src=os.path.basename(hits[0]), sha16=sha16(hits[0])))
        xs = [p[0] for p in pts]
        for p in pts:
            PS.ci_v(axg, p[0], p[2][0], p[2][1], cicol, lw=0.8, zorder=1)
        axg.plot(xs, [p[1] for p in pts], color=col, lw=1.0, marker="o", ms=3.2, mec="white", mew=0.4, zorder=3)
        axo.plot(xs, [p[3] for p in pts], color=col, lw=1.0, marker="o", ms=3.2, mec="white", mew=0.4, zorder=3)
    bo = float(np.mean(base_dO))
    axo.axhline(bo, color=C["note"], lw=0.7, ls=(0, (3, 2)), zorder=1)
    axo.text(2440, bo - 0.012, "base", fontsize=FS_NOTE, color=C["note"], ha="right", va="top")
    axg.text(420, 1.62, "placebo-controlled, α = 0.5", fontsize=FS_NOTE, color=C["ours"], ha="left", va="bottom")
    axg.text(2440, -0.22, "SFT, two seeds", fontsize=FS_NOTE, color=C["acc"], ha="right", va="top")
    axg.set_ylim(-0.45, 2.05)
    axg.set_yticks([0, 1, 2])
    axo.set_ylim(-0.12, 0.12)
    axo.set_yticks([-0.1, 0, 0.1])
    axo.set_yticklabels(["−0.1", "0", "0.1"])
    for ax in (axg, axo):
        ax.set_xlim(250, 2550)
        ax.set_xticks([400, 800, 1200, 1600, 2000, 2400])
        ax.set_xticklabels(["400", "800", "1,200", "1,600", "2,000", "2,400"])
        ax.set_xlabel("training pairs seen", fontsize=FS_LABEL)
        PS.clean_axis(ax)
    axg.set_ylabel("Γ (mask)\nnats per token", fontsize=FS_LABEL, linespacing=1.0)
    axo.set_ylabel("trained model's\nplacebo drop $d_O$", fontsize=FS_LABEL, linespacing=1.0)
    PS.panel_letter(fig, 0.0, H - 0.01, "a")
    PS.panel_letter(fig, 2.78, H - 0.01, "b")
    export(fig, name, dict(points=drawn, base_dO_mean=bo))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    PS.apply()
    F = dict(fig1_map=fig1_map, fig2_findings=fig2_findings, fig5_mechanism=fig5_mechanism, fig6_dose=fig6_dose,
             fig5_readouts=fig5_readouts, fig9_ood=fig9_ood, fig8_cases=fig8_cases, fig10_dynamics=fig10_dynamics)
    todo = sys.argv[1:] or list(F)
    for k in todo:
        PS.apply()
        F[k]()
    qa_path = os.path.join(ROOT, "figures", "figures_qa.json")
    old = json.load(open(qa_path, encoding="utf-8")) if os.path.exists(qa_path) else {}
    old.update(QA)
    json.dump(old, open(qa_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: {kk: v for kk, v in QA[k].items() if kk != "size_in"} for k in QA}, ensure_ascii=False, indent=1))
