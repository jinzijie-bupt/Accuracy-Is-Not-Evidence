#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_figs_v3.py -- data figures, third pass.

Changes over v2, following the K-Dense checklist, SciencePlots and Rougier:
  * SciencePlots "science" base style: serif type matching the paper body, boxed axes,
    inward ticks; overrides for size and the Okabe-Ito palette;
  * one visual form per message, varied across the paper: scatter map (fig2, fig9),
    slopegraph + dumbbell (fig3), annotated phase plane (fig5), dose curves with CI
    bands (fig6), diagonal scatter + forest (fig7);
  * shaded confidence bands instead of bare error bars wherever a line exists;
  * direct labels before legends; no numbers printed on the canvas; no twin axes.
Data access inherited from plot_figs_v1 (registry, selectors, paired deltas).
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_figs_v1 as V1
from plot_figs_v1 import sel, one, paired_delta, save, FAM_MARK, W

import scienceplots  # noqa: F401
plt.style.use(["science", "no-latex"])

# ---- palette (Okabe-Ito); objective colours are global, per-figure encodings are local ----
P = {"grouped": "#009E73", "sft": "#E69F00", "iou-grpo": "#CC79A7", "ablation": "#D55E00",
     "base": "#6E6E6E", "sta": "#0072B2", "tl": "#D55E00", "ink": "#1A1A1A",
     "band": "#ECECEC", "grid": "#E4E4E4", "note": "#777777"}
OBJ = {"grouped": P["grouped"], "sft": P["sft"], "iou-grpo": P["iou-grpo"], "single-margin": P["ablation"],
       "grouped-nohinge": P["ablation"], "grouped-uniform-nohinge": P["ablation"], "grouped-nodro": P["ablation"], "base": P["base"]}
SHORT = {"Qwen2.5-VL-7B": "Qwen-7B", "Qwen2.5-VL-3B": "Qwen-3B", "TimeZero-7B": "TimeZero", "Time-R1-7B": "Time-R1",
         "VideoChat-R1-7B": "VideoChat-R1", "TimeLens-8B": "TimeLens-8B"}
FAMC = {"Qwen2.5-VL-7B": "#0072B2", "TimeZero-7B": "#009E73", "Time-R1-7B": "#CC79A7", "VideoChat-R1-7B": "#D55E00"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "pdf.fonttype": 42, "savefig.dpi": 300,
    "legend.frameon": False, "legend.handlelength": 1.2, "legend.handletextpad": 0.5, "legend.labelspacing": 0.3,
    "figure.constrained_layout.use": True, "figure.constrained_layout.w_pad": 0.04, "figure.constrained_layout.h_pad": 0.04,
})
NOTE = dict(fontsize=6.3, color=P["note"])
LAB = dict(fontsize=6.5, color=P["ink"])


def panel(ax, letter, title):
    ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                fontsize=8.5, fontweight="bold", va="bottom", ha="left", annotation_clip=False)
    ax.annotate(title, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(11, 4), textcoords="offset points",
                fontsize=7.8, va="bottom", ha="left", annotation_clip=False)


def open_axis(ax, keep="x"):
    """forest / dumbbell panels: no box, no y ticks"""
    for s in ("top", "right", "left" if keep == "x" else "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y" if keep == "x" else "x", length=0, which="both")
    ax.tick_params(which="minor", length=0)


def ci_err(v, ci):
    return [[v - ci[0]], [ci[1] - v]] if ci else None


# =============================================================== Fig 2: map
def fig2_map():
    pts, man = [], []
    for g in sel("gamma", instrument="14Ln", dataset="charades-sta", text_first=False):
        if g.get("seed") not in (None, 0) or g["objective"] in (None, "base"):
            continue
        d, ci, src = paired_delta(g["family"], g["objective"], g.get("alpha"), g.get("beta"), g["env"], seed=g.get("seed"))
        if d is None:
            continue
        pts.append(dict(fam=g["family"], obj=g["objective"], alpha=g.get("alpha"), beta=g.get("beta"), env=g["env"],
                        x=100 * d, xci=[100 * c for c in ci] if ci else None, y=g["gamma_mask"], yci=g["gamma_mask_ci"]))
        man.append(dict(point=f"{g['family']} {g['objective']} a={g.get('alpha')} b={g.get('beta')}", gamma_src=g["src"], acc_src=src))
    fig, ax = plt.subplots(figsize=(W, 3.2))
    ax.axhspan(-0.11, 0.11, color=P["band"], zorder=0, lw=0)
    ax.axhline(0, color=P["base"], lw=0.6, zorder=1)
    ax.axvline(0, color=P["base"], lw=0.6, zorder=1, ls=(0, (3, 2)))
    # region of grouped arms (light hull)
    try:
        from scipy.spatial import ConvexHull
        G = np.array([(p["x"], p["y"]) for p in pts if p["obj"] == "grouped"])
        if len(G) >= 3:
            h = ConvexHull(G)
            ax.fill(G[h.vertices, 0], G[h.vertices, 1], color=P["grouped"], alpha=0.07, lw=0, zorder=0)
    except Exception:
        pass
    # trajectories in alpha (same family, environment, beta=0.1)
    ends = []
    for fam in FAM_MARK:
        for env in ("A", "B", "C"):
            tr = sorted([p for p in pts if p["fam"] == fam and p["env"] == env and p["obj"] == "grouped" and p["beta"] == 0.1], key=lambda p: p["alpha"])
            if len(tr) >= 2:
                ax.plot([p["x"] for p in tr], [p["y"] for p in tr], color=P["grouped"], lw=0.9, alpha=0.5, zorder=2)
                for a1, b1 in zip(tr[:-1], tr[1:]):
                    ax.add_patch(FancyArrowPatch((a1["x"], a1["y"]), (b1["x"], b1["y"]), arrowstyle="-|>", mutation_scale=7,
                                                 color=P["grouped"], lw=0.0, alpha=0.85, shrinkA=6, shrinkB=6, zorder=2))
                ends.append((fam, env, tr))
    for p in pts:
        col = OBJ.get(p["obj"], P["ablation"])
        ax.errorbar(p["x"], p["y"], yerr=ci_err(p["y"], p["yci"]), xerr=ci_err(p["x"], p["xci"]),
                    fmt="none", ecolor=col, elinewidth=0.55, capsize=0, alpha=0.35, zorder=3)
        ax.scatter(p["x"], p["y"], marker=FAM_MARK[p["fam"]], s=30, facecolor=col, edgecolor="white", linewidth=0.6, zorder=4)
    off = {"Qwen2.5-VL-7B": ("last", 5, -8), "Time-R1-7B": ("first", -6, 2), "TimeZero-7B": ("first", -6, 3), "VideoChat-R1-7B": ("first", -6, 4)}
    for fam, env, tr in ends:
        which, dx, dy = off.get(fam, ("last", 4, 4))
        e = tr[-1] if which == "last" else tr[0]
        ax.annotate(SHORT[fam], (e["x"], e["y"]), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", va="center", zorder=6, **LAB)
    special = {("Qwen2.5-VL-7B", "iou-grpo"): ("IoU-reward GRPO", (-6, 0), "right"),
               ("Qwen2.5-VL-7B", "sft"): ("pure SFT", (5, -2), "left"),
               ("Qwen2.5-VL-3B", "grouped"): ("Qwen-3B", (0, 9), "center"),
               ("TimeLens-8B", "grouped"): ("TimeLens-8B", (0, -10), "center"),
               ("Qwen2.5-VL-7B", "single-margin"): ("evidence margin only", (-6, 0), "right"),
               ("Qwen2.5-VL-7B", "grouped-nohinge"): ("no hinge", (-5, 0), "right"),
               ("Qwen2.5-VL-7B", "grouped-uniform-nohinge"): ("no hinge, no group wts", (5, 2), "left"),
               ("Qwen2.5-VL-7B", "grouped-nodro"): ("no group wts", (5, 2), "left")}
    for p in pts:
        key = (p["fam"], p["obj"])
        if key in special:
            t, (dx, dy), ha = special[key]
            ax.annotate(t, (p["x"], p["y"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", zorder=6, **LAB)
        if p["obj"] == "grouped" and p["beta"] == 0.03:
            ax.annotate("β=0.03", (p["x"], p["y"]), xytext=(5, -7), textcoords="offset points", ha="left", va="center", fontsize=6.0, color="#555555", zorder=6)
    ax.set_xlabel("Paired change in R@0.5 against own base (pp)")
    ax.set_ylabel(r"$\Gamma_{\mathrm{mask}}$ against own base (nats / token)")
    ax.set_xlim(-10.5, 22.5); ax.set_ylim(-0.38, 3.0)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.text(-10.0, 0.0, "audit\nresolution", va="center", ha="left", **NOTE)
    ax.text(21.8, 2.9, "evidence-dependent at any accuracy", ha="right", va="top", **NOTE)
    ax.text(21.8, -0.31, "accurate without the evidence", ha="right", va="bottom", **NOTE)
    h_obj = [Line2D([], [], marker="o", ls="", color=c, markeredgecolor="white", markersize=5.2, label=l) for l, c in
             [("grouped objective", P["grouped"]), ("pure SFT", P["sft"]), ("IoU-reward GRPO", P["iou-grpo"]), ("objective ablation", P["ablation"])]]
    h_fam = [Line2D([], [], marker=m, ls="", color="white", markeredgecolor=P["ink"], markeredgewidth=0.7, markersize=5.2, label=SHORT[f]) for f, m in FAM_MARK.items()]
    l1 = ax.legend(handles=h_obj, loc="upper right", bbox_to_anchor=(1.0, 0.90), title="fill: objective", title_fontsize=6.6, alignment="left")
    ax.add_artist(l1)
    ax.legend(handles=h_fam, loc="lower right", bbox_to_anchor=(1.0, 0.13), ncol=2, title="shape: family", title_fontsize=6.6, alignment="left")
    save(fig, "fig2_map", man)


# =============================================================== Fig 3: relabel
def fig3_relabel():
    man = []
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.75), gridspec_kw=dict(width_ratios=[0.9, 1.1]))
    # ---- (a) slopegraph: same predictions, two label sets ----
    ls = [r for r in sel("labelswap") if r["STA_R50_full"] is not None and r["TL_R50_full"] is not None]
    x0, x1 = 0.0, 1.0
    band = [r["TL_R50_full"] for r in ls if r["family"] != "TimeLens-8B"]
    for r in ls:
        y0, y1 = r["STA_R50_full"], r["TL_R50_full"]
        tl8 = r["family"] == "TimeLens-8B"
        if tl8:
            col = "#4D4D4D" if r["objective"] == "base" else P["iou-grpo"]
            a.plot([x0, x1], [y0, y1], color=col, lw=1.5, zorder=3, solid_capstyle="round")
            a.scatter([x0, x1], [y0, y1], s=14, color=col, zorder=4)
            lab = "TimeLens-8B base" if r["objective"] == "base" else "TimeLens-8B trained"
            a.annotate(lab, (x1, y1), xytext=(6, 3 if r["objective"] == "base" else 5), textcoords="offset points",
                       ha="left", va="center", fontsize=6.5, color=col)
        else:
            a.plot([x0, x1], [y0, y1], color="#B9B9B9", lw=0.7, zorder=2)
            a.scatter([x0, x1], [y0, y1], s=5, color="#B9B9B9", zorder=2)
        man.append(dict(panel="a", point=r["label"], src=r["src"]))
    if band:
        a.annotate("nine 7B\nconfigurations", (x1, float(np.median(band))), xytext=(6, 0), textcoords="offset points",
                   ha="left", va="center", **NOTE)
    a.set_xlim(-0.25, 1.62); a.set_ylim(0.28, 0.72)
    a.tick_params(right=False, top=False, which="both")
    a.set_xticks([x0, x1]); a.set_xticklabels(["Charades-STA\nlabels", "TimeLens\nlabels"])
    a.set_ylabel("R@0.5, identical predictions")
    a.tick_params(axis="x", length=0, which="both")
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False); a.spines["bottom"].set_visible(False)
    a.tick_params(which="minor", length=0)
    panel(a, "a", "accuracy: the ranking flips")
    # ---- (b) dumbbell: Gamma under each label set ----
    rows = []
    for g in sel("gamma", instrument="14Ln", dataset="charades-timelens"):
        fam, obj, al, be = g["family"], g["objective"], g.get("alpha"), g.get("beta")
        sta = [s for s in sel("gamma", instrument="14Ln", dataset="charades-sta", family=fam, objective=obj, alpha=al, beta=be, text_first=False)
               if s.get("seed") in (None, 0) and (s["env"] == g["env"] or fam == "TimeLens-8B")]
        sta = sta[0] if sta else None
        lab = f"{SHORT[fam]} " + ("pure SFT" if obj == "sft" else f"grouped, α={al:g}")
        rows.append((obj == "sft", lab, sta, g))
        man.append(dict(panel="b", point=lab, tl_src=g["src"], sta_src=sta["src"] if sta else None))
    rows.sort(key=lambda t: (t[0], -(t[3]["gamma_mask"])))
    y = np.arange(len(rows))[::-1]
    for yi, (is_sft, lab, sta, tl) in zip(y, rows):
        if sta:
            b.plot([sta["gamma_mask"], tl["gamma_mask"]], [yi, yi], color="#C9C9C9", lw=1.6, zorder=2, solid_capstyle="round")
            b.plot([sta["gamma_mask_ci"][0], sta["gamma_mask_ci"][1]], [yi + 0.13, yi + 0.13], color=P["sta"], lw=0.7, alpha=0.65, zorder=3)
            b.scatter(sta["gamma_mask"], yi, s=17, color=P["sta"], zorder=4)
        b.plot([tl["gamma_mask_ci"][0], tl["gamma_mask_ci"][1]], [yi - 0.13, yi - 0.13], color=P["tl"], lw=0.7, alpha=0.65, zorder=3)
        b.scatter(tl["gamma_mask"], yi, s=17, color=P["tl"], zorder=4)
    n_sft = sum(1 for r in rows if r[0])
    if 0 < n_sft < len(rows):
        b.axhline(n_sft - 0.5, color=P["grid"], lw=0.7)
    b.axvline(0, color=P["base"], lw=0.6)
    b.set_yticks(y); b.set_yticklabels([r[1] for r in rows])
    open_axis(b)
    b.tick_params(top=False, right=False, which="both")
    b.set_xlabel(r"$\Gamma_{\mathrm{mask}}$ against own base (nats / token)")
    b.set_xlim(-0.45, 2.15)
    b.legend(handles=[Line2D([], [], marker="o", ls="", color=P["sta"], markersize=4, label="under STA labels"),
                      Line2D([], [], marker="o", ls="", color=P["tl"], markersize=4, label="under TimeLens labels")],
             loc="lower right", bbox_to_anchor=(1.0, -0.02))
    panel(b, "b", "evidence dependence: the sign holds")
    save(fig, "fig3_relabel", man)


# =============================================================== Fig 5: mechanism
def fig5_mechanism():
    man = []
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.6))
    items = [("full objective", dict(objective="grouped", alpha=0.5, beta=0.1, env="A", seed=0), P["grouped"], "o", (6, -6, "left")),
             ("no group weights", dict(objective="grouped-nodro", env="B"), P["ablation"], "s", (-7, 6, "right")),
             ("no hinge", dict(objective="grouped-nohinge", env="B"), P["ablation"], "^", (-6, -2, "right")),
             ("no hinge, no group weights", dict(objective="grouped-uniform-nohinge", env="B"), P["ablation"], "v", (-7, 5, "right")),
             ("evidence margin only", dict(objective="single-margin", env="B"), P["ablation"], "D", (0, -9, "center")),
             ("pure SFT", dict(objective="sft"), P["sft"], "P", (6, -1, "left"))]
    seq = []
    base_pt = None
    for lab, kw, col, mk, (dx, dy, ha) in items:
        g = one("gamma", instrument="14Ln", dataset="charades-sta", family="Qwen2.5-VL-7B", text_first=False, **kw)
        if g is None:
            print("  fig5: missing", lab); continue
        x, yv = g["dO_mask_arm"], g["dE_mask_arm"]
        ax.scatter(x, yv, marker=mk, s=44 if lab == "full objective" else 38, facecolor=col, edgecolor="white",
                   linewidth=0.6, zorder=5 if lab == "full objective" else 4)
        ax.annotate(lab, (x, yv), xytext=(dx, dy), textcoords="offset points", fontsize=6.4, ha=ha, va="center", zorder=6)
        man.append(dict(point=lab, src=g["src"]))
        if lab != "pure SFT":
            seq.append((x, yv))
        if kw.get("objective") == "grouped" and base_pt is None:
            base_pt = (g["dO_mask_base"], g["dE_mask_base"])
    # the ablation path: removing terms walks the model into the shaded region
    for a1, b1 in zip(seq[:-1], seq[1:]):
        ax.add_patch(FancyArrowPatch(a1, b1, arrowstyle="-|>", mutation_scale=6, color="#9A9A9A",
                                     lw=0.7, ls=(0, (2.5, 2)), shrinkA=7, shrinkB=7, zorder=2))
    if base_pt:
        ax.scatter(*base_pt, marker="o", s=38, facecolor=P["base"], edgecolor="white", linewidth=0.6, zorder=4)
        ax.annotate("base model", base_pt, xytext=(6, -1), textcoords="offset points", fontsize=6.4, ha="left", va="center")
    ax.set_xscale("symlog", linthresh=0.05, linscale=0.6); ax.set_yscale("symlog", linthresh=0.05, linscale=0.6)
    ax.set_xticks([-0.1, 0, 0.1, 0.5, 1, 2]); ax.set_xticklabels(["−0.1", "0", "0.1", "0.5", "1", "2"])
    ax.set_yticks([0, 0.1, 0.5, 1, 2, 4]); ax.set_yticklabels(["0", "0.1", "0.5", "1", "2", "4"]); ax.minorticks_off()
    ax.set_xlim(-0.15, 3.2); ax.set_ylim(-0.02, 6.2)
    ax.axvspan(0.11, 3.2, color=P["band"], zorder=0, lw=0)
    ax.text(2.9, 0.017, "reacts to edits anywhere", ha="right", va="center", **NOTE)
    ax.text(-0.135, 1.05, "uses evidence,\nignores the placebo", ha="left", va="top", **NOTE)
    ax.set_xlabel(r"placebo drop $d_O$ (nats / token, symlog)"); ax.set_ylabel(r"evidence drop $d_E$ (nats / token, symlog)")
    save(fig, "fig5_mechanism", man)


# =============================================================== Fig 6: dose
def fig6_dose():
    man = []
    fig, axs = plt.subplots(2, 2, figsize=(W, 3.4), sharex="col", gridspec_kw=dict(width_ratios=[1.45, 1.0]))
    for ax in axs.flat:
        ax.axhline(0, color=P["base"], lw=0.6, zorder=1)
    series = {}
    for g in sel("gamma", instrument="14Ln", dataset="charades-sta", objective="grouped", beta=0.1, text_first=False):
        if g.get("seed") not in (None, 0):
            continue
        d, ci, src = paired_delta(g["family"], "grouped", g["alpha"], 0.1, g["env"], seed=g.get("seed"))
        series.setdefault((g["family"], g["env"]), []).append((g["alpha"], g["gamma_mask"], g["gamma_mask_ci"], d, ci, g["src"], src))
    lab_off = {"Qwen2.5-VL-7B": ("last", 5, 4), "Time-R1-7B": ("last", 5, -4), "TimeZero-7B": ("first", -5, 4), "VideoChat-R1-7B": ("first", -5, -4)}
    for (fam, env), pts in sorted(series.items()):
        pts.sort()
        if len(pts) < 2:
            continue
        col = FAMC.get(fam, P["ink"]); mk = FAM_MARK[fam]
        xs = [p[0] for p in pts]
        axs[0, 0].fill_between(xs, [p[2][0] for p in pts], [p[2][1] for p in pts], color=col, alpha=0.14, lw=0, zorder=1)
        axs[0, 0].plot(xs, [p[1] for p in pts], marker=mk, ms=3.6, color=col, lw=1.1, markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        acc = [(p[0], 100 * p[3], [100 * p[4][0], 100 * p[4][1]] if p[4] else None) for p in pts if p[3] is not None]
        if len(acc) >= 2:
            xs2 = [t[0] for t in acc]
            has_ci = [t for t in acc if t[2]]
            if len(has_ci) >= 2:
                axs[1, 0].fill_between([t[0] for t in has_ci], [t[2][0] for t in has_ci], [t[2][1] for t in has_ci], color=col, alpha=0.14, lw=0, zorder=1)
            axs[1, 0].plot(xs2, [t[1] for t in acc], marker=mk, ms=3.6, color=col, lw=1.1, markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        which, dx, dy = lab_off[fam]
        e = pts[-1] if which == "last" else pts[0]
        ha = "left" if dx > 0 else "right"
        axs[0, 0].annotate(SHORT[fam], (e[0], e[1]), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, va="center", ha=ha, color=col)
        if e[3] is not None:
            axs[1, 0].annotate(SHORT[fam], (e[0], 100 * e[3]), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, va="center", ha=ha, color=col)
        for p in pts:
            man.append(dict(panel="alpha", family=fam, env=env, alpha=p[0], gamma_src=p[5], acc_src=p[6]))
    axs[0, 0].set_ylabel(r"$\Gamma_{\mathrm{mask}}$ (nats / token)"); axs[1, 0].set_ylabel(r"$\Delta$R@0.5 vs. own base (pp)")
    axs[1, 0].set_xlabel(r"$\alpha$, weight of clean supervision")
    axs[0, 0].set_xscale("log"); axs[0, 0].set_xlim(0.085, 1.35)
    axs[1, 0].set_xticks([0.1, 0.25, 0.5, 1.0]); axs[1, 0].set_xticklabels(["0.1", "0.25", "0.5", "1.0"]); axs[1, 0].minorticks_off(); axs[0, 0].minorticks_off()
    panel(axs[0, 0], "a", "the clean-supervision dial, four families")
    # ---- beta panels (Qwen, alpha .5) ----
    bpts = []
    for be in (0.03, 0.1):
        g = one("gamma", instrument="14Ln", dataset="charades-sta", family="Qwen2.5-VL-7B", objective="grouped", alpha=0.5, beta=be, env="A", text_first=False)
        if g:
            d, ci, src = paired_delta("Qwen2.5-VL-7B", "grouped", 0.5, be, "A")
            bpts.append((be, g, d, ci, src))
    gs = one("gamma", instrument="14Ln", dataset="charades-sta", family="Qwen2.5-VL-7B", objective="sft", env="A")
    if gs:
        d, ci, src = paired_delta("Qwen2.5-VL-7B", "sft", None, 0.0, "A")
        bpts.append((0.0, gs, d, ci, src))
    bpts.sort(key=lambda t: t[0])
    xpos = [0.01 if b[0] == 0 else b[0] for b in bpts]
    if len(bpts) >= 2:
        axs[0, 1].fill_between(xpos, [b[1]["gamma_mask_ci"][0] for b in bpts], [b[1]["gamma_mask_ci"][1] for b in bpts], color=P["grouped"], alpha=0.14, lw=0, zorder=1)
        axs[0, 1].plot(xpos, [b[1]["gamma_mask"] for b in bpts], color=P["grouped"], lw=1.1, zorder=2)
        acc = [(xp, 100 * b[2], [100 * b[3][0], 100 * b[3][1]] if b[3] else None) for xp, b in zip(xpos, bpts) if b[2] is not None]
        if len(acc) >= 2:
            has_ci = [t for t in acc if t[2]]
            if len(has_ci) >= 2:
                axs[1, 1].fill_between([t[0] for t in has_ci], [t[2][0] for t in has_ci], [t[2][1] for t in has_ci], color=P["ink"], alpha=0.10, lw=0, zorder=1)
            axs[1, 1].plot([t[0] for t in acc], [t[1] for t in acc], color=P["ink"], lw=1.1, alpha=0.85, zorder=2)
    for (be, g, d, ci, src), xp in zip(bpts, xpos):
        fc = P["sft"] if be == 0 else P["grouped"]
        axs[0, 1].scatter(xp, g["gamma_mask"], s=16, color=fc, zorder=3, edgecolor="white", linewidth=0.5)
        if d is not None:
            axs[1, 1].scatter(xp, 100 * d, s=16, color=P["ink"] if be else P["sft"], zorder=3, edgecolor="white", linewidth=0.5)
        man.append(dict(panel="beta", beta=be, gamma_src=g["src"], acc_src=src))
    axs[0, 1].set_xscale("log"); axs[0, 1].set_xlim(0.0065, 0.15)
    axs[1, 1].set_xticks([0.01, 0.03, 0.1]); axs[1, 1].set_xticklabels(["0 (SFT)", "0.03", "0.1"]); axs[1, 1].minorticks_off(); axs[0, 1].minorticks_off()
    axs[1, 1].set_xlabel(r"$\beta$, weight of the evidence margins")
    axs[0, 1].text(0.012, 2.6, "Qwen-7B, α = 0.5", fontsize=6.4, color="#555555")
    if not gs:
        axs[0, 1].annotate("β→0 is pure SFT\n(audit pending)", (0.01, 0.0), xytext=(4, 8), textcoords="offset points", fontsize=6.0, color=P["note"])
    axs[0, 0].set_ylim(-0.3, 2.9); axs[0, 1].set_ylim(-0.3, 2.9)
    axs[1, 0].set_ylim(-11, 9); axs[1, 1].set_ylim(-11, 9)
    axs[1, 1].tick_params(labelleft=False); axs[0, 1].tick_params(labelleft=False)
    panel(axs[0, 1], "b", "the evidence-margin dial")
    save(fig, "fig6_dose", man)


# =============================================================== Fig 7: signature
def fig7_signature():
    man = []
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.75), gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    # ---- (a) clean vs foreign abstention, diagonal scatter ----
    lim = 92
    a.fill_between([0, lim], [0, lim], [lim, lim], color=P["grouped"], alpha=0.05, lw=0, zorder=0)
    a.plot([0, lim], [0, lim], color=P["base"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    a.text(58, 51, "answers either way", rotation=38, ha="center", va="top", **NOTE)
    a.text(48, 79, "abstains only when the\nevent is absent", ha="center", va="top", **NOTE)
    recs = sorted(sel("tneg"), key=lambda t: (t["family"], str(t["objective"])))
    dose = {}
    for t in recs:
        fam, obj = t["family"], t["objective"]
        x = 100 * t["clean_abstain_strict"]; yv = 100 * t["foreign_abstain_strict"]
        col = P["base"] if obj == "base" else (P["sft"] if obj == "sft" else P["grouped"])
        a.scatter(x, yv, marker=FAM_MARK[fam], s=30, facecolor=col, edgecolor="white", linewidth=0.6, zorder=4)
        man.append(dict(panel="a", point=f"{fam} {obj} a={t.get('alpha')} b={t.get('beta')} tf={t.get('text_first')}", src=t["src"]))
        if obj == "grouped":
            dose.setdefault(fam, []).append((t.get("beta") or 0, t.get("alpha") or 0, x, yv))
        lab = None
        if obj == "grouped" and t.get("beta") == 0.03 and yv > 50:
            lab, dx, dy, ha = "Qwen, β=0.03", 2, 9, "center"
        elif obj == "grouped" and fam == "Time-R1-7B" and t.get("alpha") == 1.0:
            lab, dx, dy, ha = "Time-R1, α=1", 6, 0, "left"
        elif obj == "grouped" and fam == "Time-R1-7B" and t.get("alpha") == 0.5:
            lab, dx, dy, ha = "Time-R1, α=0.5", 6, 0, "left"
        elif obj == "grouped" and fam == "Qwen2.5-VL-7B" and t.get("beta") == 0.1:
            lab, dx, dy, ha = "Qwen, β=0.1", 6, 0, "left"
        if lab:
            a.annotate(lab, (x, yv), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=6.4, zorder=6)
    # dose connectors within a family (light)
    for fam, ptss in dose.items():
        if len(ptss) >= 2:
            ptss.sort()
            a.plot([p[2] for p in ptss], [p[3] for p in ptss], color=P["grouped"], lw=0.7, alpha=0.45, zorder=2)
    a.annotate("pure SFT and every\nTimeZero configuration", (2.5, 1.0), xytext=(38, 6), textcoords="offset points",
               ha="left", va="bottom", fontsize=6.4, color="#555555",
               arrowprops=dict(arrowstyle="-", lw=0.55, color="#999999", shrinkB=2), zorder=6)
    qb = one("tneg", family="Qwen2.5-VL-7B", objective="base")
    if qb is not None:
        a.annotate("Qwen base", (100 * qb["clean_abstain_strict"], 100 * qb["foreign_abstain_strict"]),
                   xytext=(6, 6), textcoords="offset points", ha="left", va="center", fontsize=6.4, color="#555555", zorder=6)
    a.set_xlim(-3, lim); a.set_ylim(-3, lim)
    a.set_xticks([0, 20, 40, 60, 80]); a.set_yticks([0, 20, 40, 60, 80])
    a.set_xlabel("strict abstention, clean queries (%)"); a.set_ylabel("strict abstention, foreign queries (%)")
    panel(a, "a", "abstention tracks the evidence")
    # ---- (b) DiD forest ----
    brows = []
    for r in sel("behav"):
        if r.get("did_swap") is None:
            continue
        lab = SHORT[r["family"]] + (" SFT" if r["objective"] == "sft" else f" α={r['alpha']:g}")
        brows.append((lab, r))
        man.append(dict(panel="b", point=lab, src=r["src"]))
    brows.sort(key=lambda t: -t[1]["did_swap"])
    yb = np.arange(len(brows))[::-1]
    for yi, (lab, r) in zip(yb, brows):
        d, ci = 100 * r["did_swap"], r["did_swap_ci"]
        col = P["sft"] if r["objective"] == "sft" else P["grouped"]
        if ci:
            b.plot([100 * ci[0], 100 * ci[1]], [yi, yi], color=col, lw=1.0, zorder=3)
        b.scatter(d, yi, marker=FAM_MARK[r["family"]], s=26, facecolor=col, edgecolor="white", linewidth=0.6, zorder=4)
    b.axvline(0, color=P["base"], lw=0.6)
    b.set_yticks(yb); b.set_yticklabels([t[0] for t in brows])
    open_axis(b)
    b.set_xlabel("behavioural evidence contrast,\narm − base (pp, paired DiD)")
    b.set_xlim(-3, 10)
    panel(b, "b", "predictions need the evidence more")
    save(fig, "fig7_signature", man)


# =============================================================== Fig 9: OOD map
def fig9_ood():
    man = []
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.6))
    RES = 4.9
    ax.axvspan(-RES, RES, color=P["band"], lw=0, zorder=0)
    ax.axhline(0, color=P["base"], lw=0.6, zorder=1)
    ax.axvline(0, color=P["base"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    gs = {(g["family"], g["objective"], g.get("alpha"), bool(g.get("text_first"))): g
          for g in sel("gamma", dataset="qvhighlights")}
    pts = []
    for p in sel("pair_acc", dataset="qvhighlights"):
        arm, ref = p["arm"], p["ref"]
        if ref.get("objective") != "base" or ref.get("family") != arm.get("family"):
            continue
        if arm.get("seed") not in (None, 0) or bool(arm.get("text_first")) != bool(ref.get("text_first")):
            continue
        if "_g1_tf_" in str(p["src"]):   # arm trained text-first: a protocol variant, not the T1 arm
            continue
        key = (arm["family"], arm["objective"], arm.get("alpha"), bool(arm.get("text_first")))
        g = gs.get(key)
        if g is None or (arm.get("beta") not in (None, 0.0, 0.1)):
            continue
        pts.append((p, g))
        man.append(dict(point=f"{arm['family']} {arm['objective']} a={arm.get('alpha')} tf={arm.get('text_first')}",
                        pair_src=p["src"], gamma_src=g["src"]))
    lab_of = {("Qwen2.5-VL-7B", "grouped", 0.5): ("grouped α=0.5", 6, 2, "left"),
              ("Qwen2.5-VL-7B", "grouped", 1.0): ("grouped α=1", 6, 2, "left"),
              ("Qwen2.5-VL-7B", "sft", None): ("pure SFT", 0, 9, "center"),
              ("TimeZero-7B", "grouped", 0.5): ("TimeZero grouped α=0.5", 6, -4, "left")}
    for p, g in pts:
        arm = p["arm"]
        col = OBJ.get(arm["objective"], P["ablation"])
        x = 100 * p["delta_R50"]; xci = [100 * c for c in p["delta_R50_ci"]]
        yv = g["gamma_mask"]; yci = g["gamma_mask_ci"]
        ax.errorbar(x, yv, xerr=ci_err(x, xci), yerr=ci_err(yv, yci), fmt="none", ecolor=col, elinewidth=0.55, alpha=0.4, capsize=0, zorder=3)
        ax.scatter(x, yv, marker=FAM_MARK[arm["family"]], s=32, facecolor=col, edgecolor="white", linewidth=0.6, zorder=4)
        key = (arm["family"], arm["objective"], arm.get("alpha"))
        if key in lab_of:
            t, dx, dy, ha = lab_of[key]
            ax.annotate(t, (x, yv), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=6.5, zorder=6)
    ax.set_xlim(-9.5, 10.5); ax.set_ylim(-0.35, 1.75)
    ax.text(0, 1.68, "OOD accuracy resolution", ha="center", va="top", **NOTE)
    ax.set_xlabel(r"$\Delta$R@0.5 on QVHighlights, arm $-$ base (pp)")
    ax.set_ylabel(r"$\Gamma_{\mathrm{mask}}$ on QVHighlights (nats / token)")
    save(fig, "fig9_ood", man)


if __name__ == "__main__":
    which = sys.argv[1:] or ["fig2", "fig3", "fig5", "fig6", "fig7", "fig9"]
    F = {"fig2": fig2_map, "fig3": fig3_relabel, "fig5": fig5_mechanism, "fig6": fig6_dose, "fig7": fig7_signature, "fig9": fig9_ood}
    for k in which:
        F[k]()
