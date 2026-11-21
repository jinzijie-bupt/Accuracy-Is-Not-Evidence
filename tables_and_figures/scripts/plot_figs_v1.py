#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_figs_v1.py -- data figures for the paper from tables/paper_numbers_v1.json.

fig2_map        accuracy x evidence-dependence map (paired dR@0.5 vs Gamma_mask, alpha trajectories)
fig3_relabel    (a) STA vs TL R@0.5 of the same predictions  (b) forest: Gamma under STA vs TL labels
fig5_mechanism  dO vs dE of the objective ablations (symlog)
fig6_dose       2x2: Gamma / dR@0.5 against alpha and beta
fig7_signature  (a) foreign-query abstention delta forest  (b) behavioural DiD forest

Rules: one message per figure; no numbers printed on the canvas (axes/CI only); no inset; no twin axis.
Every figure writes a data_manifest.json next to it listing the source files + sha16 of each point.
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "figures", "data")
os.makedirs(OUT, exist_ok=True)

# ---- style ----
C = {"evidence": "#C8553D", "outside": "#3B6EA8", "grouped": "#2A9D8F", "base": "#7F7F7F",
     "sft": "#E9C46A", "iou": "#8E6BB5", "ablation": "#F4A261", "band": "#E6E6E6", "ink": "#222222"}
FAM_MARK = {"Qwen2.5-VL-7B": "o", "Qwen2.5-VL-3B": "D", "TimeZero-7B": "s", "Time-R1-7B": "^", "VideoChat-R1-7B": "v", "TimeLens-8B": "P"}
FAM_SHORT = {"Qwen2.5-VL-7B": "Qwen2.5-VL-7B", "Qwen2.5-VL-3B": "Qwen2.5-VL-3B", "TimeZero-7B": "TimeZero", "Time-R1-7B": "Time-R1", "VideoChat-R1-7B": "VideoChat-R1", "TimeLens-8B": "TimeLens-8B"}
OBJ_COLOR = {"grouped": C["grouped"], "sft": C["sft"], "iou-grpo": C["iou"], "single-margin": C["ablation"],
             "grouped-nohinge": C["ablation"], "grouped-uniform-nohinge": C["ablation"], "grouped-nodro": C["ablation"], "base": C["base"]}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "savefig.dpi": 300, "legend.frameon": False,
})
W = 5.5  # inches, ICLR text width

R = json.load(open(os.path.join(ROOT, "tables", "paper_numbers_v1.json"), encoding="utf-8"))["records"]


def sel(kind, **kw):
    out = []
    for r in R:
        if r.get("kind") != kind:
            continue
        ok = True
        for k, v in kw.items():
            if k == "name_not":
                ok &= v not in (r.get("name") or r.get("label") or "")
            elif callable(v):
                ok &= bool(v(r.get(k)))
            else:
                ok &= r.get(k) == v
            if not ok:
                break
        if ok:
            out.append(r)
    return out


def one(kind, **kw):
    o = sel(kind, **kw)
    if len(o) > 1:
        raise SystemExit(f"{kind} {kw}: {len(o)} matches: {[x.get('name') or x.get('label') for x in o]}")
    return o[0] if o else None


def paired_delta(fam, obj, alpha, beta, env, text_first=False, seed=None):
    """paired dR@0.5 (pair reader) else single-run difference vs same-env base; returns (delta, ci or None, src)"""
    kw = dict(family=fam, objective=obj, alpha=alpha, beta=beta, env=env)
    pairs = [p for p in sel("pair_acc", dataset="charades-sta") if all((p["arm"].get(k) == v) for k, v in kw.items())
             and p["arm"].get("text_first") == text_first and p["ref"].get("objective") == "base"
             and (seed is None or p["arm"].get("seed") == seed)]
    if len(pairs) == 1:
        p = pairs[0]
        return p["delta_R50"], p["delta_R50_ci"], [p["src"]]
    if len(pairs) > 1:
        raise SystemExit(f"multiple pairs for {kw}")
    arm = sel("acc", dataset="charades-sta", text_first=text_first, name_not="official", **kw)
    arm = [a for a in arm if seed is None or a.get("seed") == seed]
    base = sel("acc", dataset="charades-sta", family=fam, objective="base", env=env, text_first=False, name_not="official")
    base = [b for b in base if "r1_full" not in (b.get("name") or "")]
    if len(arm) == 1 and len(base) == 1:
        return arm[0]["R50"] - base[0]["R50"], None, [arm[0]["src"], base[0]["src"]]
    return None, None, []


def save(fig, name, manifest):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.02)
    with open(os.path.join(OUT, name + "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    plt.close(fig)
    print("wrote", name, f"({len(manifest)} points)")


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
        man.append(dict(point=f"{g['family']} {g['objective']} a={g.get('alpha')} b={g.get('beta')} env={g['env']}", gamma_src=g["src"], acc_src=src))
    fig, ax = plt.subplots(figsize=(W, 3.3))
    ax.axhspan(-0.11, 0.11, color=C["band"], zorder=0, lw=0)
    ax.axhline(0, color=C["base"], lw=0.6, zorder=1)
    ax.axvline(0, color=C["base"], lw=0.6, zorder=1)
    # alpha trajectories: same family, env, objective grouped, beta .1
    for fam in FAM_MARK:
        for env in ("A", "B", "C"):
            tr = sorted([p for p in pts if p["fam"] == fam and p["env"] == env and p["obj"] == "grouped" and p["beta"] == 0.1], key=lambda p: p["alpha"])
            if len(tr) >= 2:
                for a, b in zip(tr[:-1], tr[1:]):
                    ax.annotate("", xy=(b["x"], b["y"]), xytext=(a["x"], a["y"]),
                                arrowprops=dict(arrowstyle="-|>", color=C["grouped"], lw=0.8, alpha=0.7, shrinkA=3, shrinkB=3), zorder=2)
    for p in pts:
        col = OBJ_COLOR.get(p["obj"], C["ablation"])
        ax.errorbar(p["x"], p["y"], yerr=[[p["y"] - p["yci"][0]], [p["yci"][1] - p["y"]]] if p["yci"] else None,
                    xerr=[[p["x"] - p["xci"][0]], [p["xci"][1] - p["x"]]] if p["xci"] else None,
                    fmt="none", ecolor=col, elinewidth=0.6, capsize=0, alpha=0.6, zorder=3)
        ax.scatter(p["x"], p["y"], marker=FAM_MARK[p["fam"]], s=34, facecolor=col, edgecolor=C["ink"], linewidth=0.5, zorder=4)
    ax.set_xlabel("Paired change in R@0.5 vs. own base (pp)")
    ax.set_ylabel(r"$\Gamma_{\mathrm{mask}}$ (nats / token)")
    h1 = [Line2D([], [], marker=m, ls="", color="white", markeredgecolor=C["ink"], markersize=5.5, label=FAM_SHORT[f]) for f, m in FAM_MARK.items()]
    h2 = [Line2D([], [], marker="o", ls="", color=c, markeredgecolor=C["ink"], markersize=5.5, label=l) for l, c in
          [("grouped objective", C["grouped"]), ("pure SFT", C["sft"]), ("IoU-reward GRPO", C["iou"]), ("objective ablation", C["ablation"])]]
    leg1 = ax.legend(handles=h1, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="family", title_fontsize=7)
    ax.add_artist(leg1)
    ax.legend(handles=h2, loc="upper left", bbox_to_anchor=(1.01, 0.45), title="objective", title_fontsize=7)
    save(fig, "fig2_map", man)


# =============================================================== Fig 3: relabel
def fig3_relabel():
    man = []
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw=dict(width_ratios=[1.0, 1.15], wspace=1.05))
    # (a) labelswap scatter
    ls = sel("labelswap")
    lo, hi = 0.28, 0.72
    a.plot([lo, hi], [lo, hi], color=C["base"], lw=0.6, ls="--", zorder=1)
    tl_pts = {}
    for r in ls:
        x, y = r["STA_R50_full"], r["TL_R50_full"]
        if x is None or y is None:
            continue
        col = OBJ_COLOR.get(r["objective"], C["ablation"])
        a.scatter(x, y, marker=FAM_MARK.get(r["family"], "o"), s=34, facecolor=col, edgecolor=C["ink"], linewidth=0.5, zorder=3)
        man.append(dict(panel="a", point=r["label"], src=r["src"]))
        if r["family"] == "TimeLens-8B":
            tl_pts[r["objective"]] = (x, y)
    if "base" in tl_pts and "grouped" in tl_pts:
        a.annotate("", xy=tl_pts["grouped"], xytext=tl_pts["base"], arrowprops=dict(arrowstyle="-|>", color=C["evidence"], lw=0.9, shrinkA=4, shrinkB=4), zorder=2)
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
    a.set_xlabel("R@0.5, Charades-STA labels"); a.set_ylabel("R@0.5, TimeLens labels (same predictions)")
    a.set_title("(a) one prediction set, two annotations", loc="left")
    # (b) forest: Gamma under STA labels vs TL labels for the arms that have both
    rows = []
    for g in sel("gamma", instrument="14Ln", dataset="charades-timelens"):
        fam, obj, al, be = g["family"], g["objective"], g.get("alpha"), g.get("beta")
        sta = [s for s in sel("gamma", instrument="14Ln", dataset="charades-sta", family=fam, objective=obj, alpha=al, beta=be, text_first=False)
               if s.get("seed") in (None, 0) and (s["env"] == g["env"] or fam == "TimeLens-8B")]
        sta = sta[0] if len(sta) >= 1 else None
        lab = f"{FAM_SHORT[fam].replace('Qwen2.5-VL-7B', 'Qwen-7B')} " + ("SFT" if obj == "sft" else f"grouped α={al:g}")
        rows.append((lab, sta, g))
        man.append(dict(panel="b", point=lab, tl_src=g["src"], sta_src=sta["src"] if sta else None))
    rows.sort(key=lambda t: (t[2]["objective"] == "sft", t[0]))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, sta, tl) in zip(y, rows):
        if sta:
            b.errorbar(sta["gamma_mask"], yi + 0.17, xerr=[[sta["gamma_mask"] - sta["gamma_mask_ci"][0]], [sta["gamma_mask_ci"][1] - sta["gamma_mask"]]],
                       fmt="o", ms=3.8, color=C["base"], ecolor=C["base"], elinewidth=0.8, capsize=1.5, zorder=3)
        b.errorbar(tl["gamma_mask"], yi - 0.17, xerr=[[tl["gamma_mask"] - tl["gamma_mask_ci"][0]], [tl["gamma_mask_ci"][1] - tl["gamma_mask"]]],
                   fmt="o", ms=3.8, color=C["evidence"], ecolor=C["evidence"], elinewidth=0.8, capsize=1.5, zorder=3)
    b.axvline(0, color=C["base"], lw=0.6)
    b.set_yticks(y); b.set_yticklabels([r[0] for r in rows])
    b.set_xlabel(r"$\Gamma_{\mathrm{mask}}$ vs. own base (nats / token)")
    b.set_title("(b) same arms, evidence axis under each label set", loc="left")
    b.legend(handles=[Line2D([], [], marker="o", ls="", color=C["base"], label="Charades-STA labels"),
                      Line2D([], [], marker="o", ls="", color=C["evidence"], label="TimeLens labels")], loc="lower right")
    save(fig, "fig3_relabel", man)


# =============================================================== Fig 5: mechanism
def fig5_mechanism():
    man = []
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.6))
    items = [("full objective", dict(objective="grouped", alpha=0.5, beta=0.1, env="A", seed=0), C["grouped"], "o"),
             ("no group weights", dict(objective="grouped-nodro", env="B"), C["ablation"], "s"),
             ("no hinge", dict(objective="grouped-nohinge", env="B"), C["ablation"], "^"),
             ("no hinge, no group weights", dict(objective="grouped-uniform-nohinge", env="B"), C["ablation"], "v"),
             ("evidence margin only", dict(objective="single-margin", env="B"), C["ablation"], "D"),
             ("pure SFT", dict(objective="sft"), C["sft"], "P")]
    base_pt = None
    for lab, kw, col, mk in items:
        g = one("gamma", instrument="14Ln", dataset="charades-sta", family="Qwen2.5-VL-7B", text_first=False, **kw)
        if g is None:
            print("  fig5: missing", lab); continue
        ax.scatter(g["dO_mask_arm"], g["dE_mask_arm"], marker=mk, s=40, facecolor=col, edgecolor=C["ink"], linewidth=0.5, zorder=3, label=lab)
        man.append(dict(point=lab, src=g["src"]))
        if kw.get("objective") == "grouped" and base_pt is None:
            base_pt = (g["dO_mask_base"], g["dE_mask_base"])
    if base_pt:
        ax.scatter(*base_pt, marker="o", s=40, facecolor=C["base"], edgecolor=C["ink"], linewidth=0.5, zorder=3, label="base model")
    ax.set_xscale("symlog", linthresh=0.05); ax.set_yscale("symlog", linthresh=0.05)
    ax.set_xticks([-0.1, 0, 0.1, 0.5, 1, 2]); ax.set_xticklabels(["-0.1", "0", "0.1", "0.5", "1", "2"])
    ax.set_yticks([0, 0.1, 0.5, 1, 2, 4]); ax.set_yticklabels(["0", "0.1", "0.5", "1", "2", "4"]); ax.minorticks_off()
    ax.set_xlabel(r"placebo drop $d_O$ (nats / token)"); ax.set_ylabel(r"evidence drop $d_E$ (nats / token)")
    ax.axvline(0.11, color=C["band"], lw=0.8, zorder=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, "fig5_mechanism", man)


# =============================================================== Fig 6: dose
def fig6_dose():
    man = []
    fig, axs = plt.subplots(2, 2, figsize=(W, 3.6), sharex="col", gridspec_kw=dict(hspace=0.15, wspace=0.35))
    # alpha panels: for each (family, env) with >=2 grouped beta=.1 points
    series = {}
    for g in sel("gamma", instrument="14Ln", dataset="charades-sta", objective="grouped", beta=0.1, text_first=False):
        if g.get("seed") not in (None, 0):
            continue
        d, ci, src = paired_delta(g["family"], "grouped", g["alpha"], 0.1, g["env"], seed=g.get("seed"))
        series.setdefault((g["family"], g["env"]), []).append((g["alpha"], g["gamma_mask"], g["gamma_mask_ci"], d, ci, g["src"], src))
    for (fam, env), pts in sorted(series.items()):
        pts.sort()
        if len(pts) < 2:
            continue
        col = C["grouped"]; mk = FAM_MARK[fam]
        xs = [p[0] for p in pts]
        axs[0, 0].errorbar(xs, [p[1] for p in pts], yerr=[[p[1] - p[2][0] for p in pts], [p[2][1] - p[1] for p in pts]], marker=mk, ms=4, color=col, ecolor=col, elinewidth=0.7, capsize=1.5, label=f"{FAM_SHORT[fam]}")
        ys = [100 * p[3] if p[3] is not None else np.nan for p in pts]
        yerr = [[100 * (p[3] - p[4][0]) if p[4] else 0 for p in pts], [100 * (p[4][1] - p[3]) if p[4] else 0 for p in pts]]
        axs[1, 0].errorbar(xs, ys, yerr=yerr, marker=mk, ms=4, color=C["ink"], ecolor=C["ink"], elinewidth=0.7, capsize=1.5, alpha=0.8)
        for p in pts:
            man.append(dict(panel="alpha", family=fam, env=env, alpha=p[0], gamma_src=p[5], acc_src=p[6]))
    axs[0, 0].set_ylabel(r"$\Gamma_{\mathrm{mask}}$ (nats / token)"); axs[1, 0].set_ylabel(r"$\Delta$R@0.5 vs. base (pp)")
    axs[1, 0].set_xlabel(r"$\alpha$ (weight of clean supervision)"); axs[1, 0].axhline(0, color=C["base"], lw=0.6)
    axs[0, 0].set_xscale("log"); axs[0, 0].set_xticks([0.1, 0.25, 0.5, 1.0]); axs[0, 0].set_xticklabels(["0.1", "0.25", "0.5", "1.0"])
    axs[0, 0].minorticks_off(); axs[1, 0].minorticks_off()
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4)
    # beta panels: Qwen env A alpha .5 : beta .03, .1 ; SFT (beta->0) where same env
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
    # (no cross-environment fill: the beta->0 point appears only when the same-environment SFT audit has landed)
    bpts.sort(key=lambda t: t[0])
    xb = [0.0 if b[0] == 0 else b[0] for b in bpts]
    xpos = [0.01 if x == 0 else x for x in xb]
    for (be, g, d, ci, src), xp in zip(bpts, xpos):
        fc = C["sft"] if be == 0 else C["grouped"]
        axs[0, 1].errorbar(xp, g["gamma_mask"], yerr=[[g["gamma_mask"] - g["gamma_mask_ci"][0]], [g["gamma_mask_ci"][1] - g["gamma_mask"]]], marker="o", ms=4, color=fc, ecolor=fc, elinewidth=0.7, capsize=1.5)
        if d is not None:
            axs[1, 1].errorbar(xp, 100 * d, yerr=[[100 * (d - ci[0])], [100 * (ci[1] - d)]] if ci else None, marker="o", ms=4, color=C["ink"], ecolor=C["ink"], elinewidth=0.7, capsize=1.5)
        man.append(dict(panel="beta", beta=be, gamma_src=g["src"], acc_src=src))
    axs[0, 1].plot([p for p in xpos], [b[1]["gamma_mask"] for b in bpts], color=C["grouped"], lw=0.8, zorder=1)
    axs[0, 1].set_xscale("log"); axs[0, 1].set_xticks([0.01, 0.03, 0.1]); axs[0, 1].set_xticklabels([r"0 (SFT)", "0.03", "0.1"]); axs[0, 1].set_xlim(0.007, 0.14)
    axs[0, 1].minorticks_off(); axs[1, 1].minorticks_off()
    axs[1, 1].set_xlabel(r"$\beta$ (weight of the evidence margins), Qwen2.5-VL-7B, $\alpha{=}0.5$"); axs[1, 1].axhline(0, color=C["base"], lw=0.6)
    for ax in axs.flat:
        ax.axhline(0, color=C["base"], lw=0.5, zorder=0)
    save(fig, "fig6_dose", man)


# =============================================================== Fig 7: signature
def fig7_signature():
    man = []
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=1.15))
    rows = []
    for t in sel("tneg"):
        fam = t["family"]; obj = t["objective"]
        lab = FAM_SHORT[fam].replace("Qwen2.5-VL-7B", "Qwen-7B") + " " + ("base" if obj == "base" else ("SFT" if obj == "sft" else f"α={t['alpha']:g}" + (f", β={t['beta']:g}" if t['beta'] not in (None, 0.1) else "")))
        if t.get("text_first"):
            lab += " (text-first)"
        rows.append((fam, obj, t["alpha"] or 0, t["beta"] or 0, lab, t))
        man.append(dict(panel="a", point=lab, src=t["src"]))
    order = {"Qwen2.5-VL-7B": 0, "Time-R1-7B": 1, "TimeZero-7B": 2}
    rows.sort(key=lambda r: (order.get(r[0], 9), r[1] != "base", r[2], -r[3]))
    y = np.arange(len(rows))[::-1]
    for yi, (fam, obj, al, be, lab, t) in zip(y, rows):
        d = t["delta_abstain_strict"]; ci = t["delta_abstain_strict_ci"]
        col = C["base"] if obj == "base" else (C["sft"] if obj == "sft" else C["grouped"])
        a.errorbar(100 * d, yi, xerr=[[100 * (d - ci[0])], [100 * (ci[1] - d)]] if ci else None, fmt=FAM_MARK[fam], ms=4, color=col, ecolor=col, elinewidth=0.8, capsize=1.5)
    a.axvline(0, color=C["base"], lw=0.6)
    a.set_yticks(y); a.set_yticklabels([r[4] for r in rows])
    a.set_xlabel("Change in strict abstention, foreign minus clean query (pp)")
    a.set_title("(a) abstention when the event is absent", loc="left")
    # (b) DiD forest
    brows = []
    for r in sel("behav"):
        if r.get("did_swap") is None:
            continue
        lab = FAM_SHORT[r["family"]].replace("Qwen2.5-VL-7B", "Qwen-7B") + (" SFT" if r["objective"] == "sft" else f" α={r['alpha']:g}")
        brows.append((lab, r))
        man.append(dict(panel="b", point=lab, src=r["src"]))
    brows.sort(key=lambda t: -t[1]["did_swap"])
    yb = np.arange(len(brows))[::-1]
    for yi, (lab, r) in zip(yb, brows):
        d, ci = r["did_swap"], r["did_swap_ci"]
        col = C["sft"] if r["objective"] == "sft" else C["grouped"]
        b.errorbar(100 * d, yi, xerr=[[100 * (d - ci[0])], [100 * (ci[1] - d)]] if ci else None, fmt=FAM_MARK[r["family"]], ms=4, color=col, ecolor=col, elinewidth=0.8, capsize=1.5)
    b.axvline(0, color=C["base"], lw=0.6)
    b.set_yticks(yb); b.set_yticklabels([t[0] for t in brows])
    b.set_xlabel("Behavioural evidence contrast,\narm minus base (pp, paired DiD)")
    b.set_title("(b) generation loses more to evidence swaps", loc="left")
    save(fig, "fig7_signature", man)


if __name__ == "__main__":
    fig2_map(); fig3_relabel(); fig5_mechanism(); fig6_dose(); fig7_signature()
