#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_figs_v5.py -- data figures under canonical_v3 (2026-09-23): one model, one point.

Every number is read through canonical_v3 (per configuration, kind and dataset one environment; paired
quantities close inside one specific environment).  Public-checkpoint points of fig2 come from the analysis
products named by the author (analysis/a1_public_rl_vs_qwen.json: TimeZero row of env A, Time-R1 row of env B;
analysis/a6_vc_vs_qwen.json).  Style, palette, FAM_MARK and save() are those of plot_figs_v3/v4.
Rules (author, 2026-09-23): no in-figure titles; no explanatory text annotations; leading zeros kept; only
Charades-STA, evaluated video-first, unless a figure says otherwise.  Each figure's manifest lists every point
drawn (with source file + sha16 + environment) and every point dropped (with the reason).
"""
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..", "..")).replace("\\", "/")
import hashlib, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_figs_v3 as V3                                   # noqa: E402  (style + palette)
from plot_figs_v1 import save, FAM_MARK, W                  # noqa: E402
from plot_figs_v3 import P, OBJ, SHORT, FAMC, open_axis, ci_err   # noqa: E402
from canonical_v3 import Canon, M, base_of, key_str, Q7, Q3, TZ, TR, VC, TL   # noqa: E402

ANA = os.path.join(_SUPP, "readers", "derived")
CS, QV, CTL = "charades-sta", "qvhighlights", "charades-timelens"
C = Canon()
CONSTWIN = 0.317


def panel(ax, letter):
    ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                fontsize=8.5, fontweight="bold", va="bottom", ha="left", annotation_clip=False)


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def src(r):
    return None if r is None else dict(file=(r.get("src") or {}).get("file"), sha16=(r.get("src") or {}).get("sha16"), env=r.get("_env"))


def gam(k, ds=CS):
    return C.pick("gamma", k, ds, column="fig", instrument="14Ln", pending=False)


def kpick(kind, k, ds=CS):
    return C.pick(kind, k, ds, column="fig", pending=False)


def g_(fam, a, b=0.1, seed=0, **kw):
    return M(fam, "grouped", a, b, seed, **kw)


SFT, GRPO, MDPO = M(Q7, "sft", None, 0.0, 0), M(Q7, "iou-grpo"), M(Q7, "mdpo", 0.5, None, 0)
GROUPED = [g_(Q7, 0.25), g_(Q7, 0.5), g_(Q7, 1.0), g_(TZ, 0.25), g_(TZ, 0.5), g_(TZ, 1.0),
           g_(TR, 0.1), g_(TR, 0.25), g_(TR, 0.5), g_(TR, 1.0), g_(VC, 0.25), g_(VC, 0.5), g_(VC, 1.0), g_(TL, 0.5)]
BETA003 = [g_(Q7, 0.5, 0.03), g_(Q7, 1.0, 0.03), g_(TZ, 0.5, 0.03)]


def fmt(v, nd=2):
    return f"{v:.{nd}f}"


# ============================================== fig1: black screens (one base, its trained models, public)
def fig1_black():
    man = dict(points=[], dropped=[])
    fig, ax = plt.subplots(figsize=(2.6, 2.3))
    draw_black(ax, man)
    save(fig, "fig1_black", man)
    return man


def draw_black(ax, man, compact=False):
    rows = [("Qwen2.5-VL-7B base", M(Q7, "base"), "base"), ("+ pure SFT", SFT, "ours"), ("+ IoU-reward GRPO", GRPO, "ours"),
            ("+ mDPO-style", MDPO, "ours"),
            ("TimeZero", M(TZ, "base"), "public"), ("Time-R1", M(TR, "base"), "public"),
            ("+ placebo-controlled, α=1.0", g_(Q7, 1.0), "ours2"), ("+ placebo-controlled, α=0.5", g_(Q7, 0.5), "ours2")]
    ys, labs = [], []
    y = 0.0
    prev = None
    for lab, k, grp in rows:
        if prev is not None and grp != prev:
            y += 0.45
            ax.axhline(y - 0.72, color="#CFCFCF", lw=0.5, zorder=1)
        prev = grp
        r = kpick("black", k)
        if r is None:
            man["dropped"].append(dict(point=lab, key=key_str(k), reason="no black run in the canonical environment"))
            continue
        v = r["black_answer_rate"]
        ax.barh(y, v, height=0.62, color="#B9B2A7", edgecolor="none", zorder=2)
        ax.scatter(r["black_R50"], y, s=14, color=P["ink"], zorder=4, edgecolor="white", linewidth=0.5)
        man["points"].append(dict(point=lab, key=key_str(k), answer_rate=v, black_R50=r["black_R50"], src=src(r)))
        ys.append(y); labs.append(lab)
        y += 1.0
    ax.axvline(CONSTWIN, color=P["ink"], lw=0.8, ls=(0, (3, 2)), zorder=3)
    ax.annotate("constant window", (CONSTWIN, 1.0), xycoords=("data", "axes fraction"), xytext=(2, 1),
                textcoords="offset points", ha="left", va="bottom", fontsize=5.8, color=P["ink"], annotation_clip=False)
    # group tags, small type, right margin
    grp_y = {}
    for (lab, k, grp), yy in zip([r for r in rows if r[0] in labs], ys):
        grp_y.setdefault(grp, []).append(yy)
    tag = {"ours": "trained from this base by us", "public": "public, same base", "ours2": "trained from this base by us"}
    for grp, yy in grp_y.items():
        if grp in tag:
            ax.text(1.0, min(yy) - 0.52, tag[grp], transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=5.0, color=P["note"], style="italic", clip_on=False)
    ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=6.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "", "0.5", "", "1.0"])
    open_axis(ax)
    ax.tick_params(top=False, right=False, which="both")
    ax.set_xlabel("fraction of questions, all-black video", fontsize=7)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#B9B2A7", label="answer rate"),
                       Line2D([], [], marker="o", ls="", color=P["ink"], markersize=3.5, label="R@0.5")],
              loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=5.8, frameon=False, handlelength=1.0, labelspacing=0.25)
    man["constant_window"] = dict(value=CONSTWIN, src="tables/T1_audit.tex reference row")
    man["template"] = "timer1_default for every row"


# ============================================== fig2: the audit map (single panel)
def fig1_map():
    """Figure 1(b): the audit map at half width, beside the concept panel."""
    return fig2_map(name="fig1_map", figsize=(3.05, 2.4), compact=True)


# label offsets (dx, dy, ha) at half width, where the default placements collide
_COMPACT_LAB = {"pure SFT": (-4, -4, "right"), "IoU-reward GRPO": (0, -9, "center"), "mDPO-style": (-4, 6, "right"),
                "TimeZero": (0, -9, "center"), "VideoChat-R1": (0, 8, "center"), "Time-R1": (-3, 8, "center")}


def fig2_map(name="fig2_map", figsize=(W, 2.8), compact=False):
    man = dict(points=[], dropped=[])
    fig, a = plt.subplots(figsize=figsize)
    a.axhspan(-0.10, 0.10, color=P["band"], zorder=0, lw=0)
    pts = []
    for k in GROUPED + BETA003 + [SFT, GRPO, MDPO]:
        gm = gam(k)
        dv, dci, dag, denv = C.delta3(k, CS)
        if gm is None or dv is None:
            man["dropped"].append(dict(point=key_str(k), reason=("no Gamma" if gm is None else "no Delta R@0.5 in the environment of its accuracy run")))
            continue
        p = dict(k=k, fam=k[0], obj=k[1], alpha=k[2], beta=k[3], x=100 * dv, xci=[100 * c for c in dci] if dci else None,
                 y=gm["gamma_mask"], yci=gm["gamma_mask_ci"])
        pts.append(p)
        man["points"].append(dict(point=key_str(k), delta_pp=p["x"], delta_ci=p["xci"], dagger=dag, delta_env=denv,
                                  gamma=p["y"], gamma_ci=p["yci"], gamma_src=src(gm)))
    for fam in FAM_MARK:
        tr = sorted([p for p in pts if p["fam"] == fam and p["obj"] == "grouped" and p["beta"] == 0.1], key=lambda p: p["alpha"])
        if len(tr) >= 2:
            a.plot([p["x"] for p in tr], [p["y"] for p in tr], color=P["grouped"], lw=0.9, alpha=0.5, zorder=2)
            for p1, p2 in zip(tr[:-1], tr[1:]):
                a.add_patch(FancyArrowPatch((p1["x"], p1["y"]), (p2["x"], p2["y"]), arrowstyle="-|>", mutation_scale=6,
                                            color=P["grouped"], lw=0.0, alpha=0.85, shrinkA=5, shrinkB=5, zorder=2))
            e = tr[-1]
            a.annotate(SHORT[fam], (e["x"], e["y"]), xytext=(4, -5), textcoords="offset points", ha="left", va="center",
                       fontsize=6.2, color=P["ink"], zorder=6)
    for p in pts:
        col = OBJ.get(p["obj"], P["ablation"]) if p["obj"] != "mdpo" else "#8C8C8C"
        a.errorbar(p["x"], p["y"], yerr=ci_err(p["y"], p["yci"]), xerr=ci_err(p["x"], p["xci"]), fmt="none",
                   ecolor=col, elinewidth=0.5, capsize=0, alpha=0.35, zorder=3)
        a.scatter(p["x"], p["y"], marker=FAM_MARK[p["fam"]], s=26, facecolor=col, edgecolor="white", linewidth=0.5, zorder=4)
        lab = None
        if p["obj"] == "sft":
            lab = ("pure SFT", 4, -2, "left")
        elif p["obj"] == "iou-grpo":
            lab = ("IoU-reward GRPO", 4, 7, "left")
        elif p["obj"] == "mdpo":
            lab = ("mDPO-style", 4, 2, "left")
        elif p["obj"] == "grouped" and p["beta"] == 0.03:
            lab = ("β=0.03", 4, -6, "left")
        elif p["fam"] in (Q3, TL):
            lab = (SHORT[p["fam"]], 0, 8, "center")
        if lab and compact:
            lab = None if lab[0] == "β=0.03" else ((lab[0],) + _COMPACT_LAB[lab[0]] if lab[0] in _COMPACT_LAB else lab)
        if lab:
            a.annotate(lab[0], (p["x"], p["y"]), xytext=(lab[1], lab[2]), textcoords="offset points", ha=lab[3], va="center",
                       fontsize=5.8 if lab[0] == "β=0.03" else 6.2, color="#555555" if lab[0] == "β=0.03" else P["ink"], zorder=6)
    # public checkpoints against the Qwen base (same environment, same template): hollow diamonds
    a1 = json.load(open(os.path.join(ANA, "a1_public_rl_vs_qwen.json"), encoding="utf-8"))
    pub = [("TimeZero", "TimeZero vs Qwen", "A"), ("Time-R1", "Time-R1 vs Qwen", "B")]
    for pname, lab_a1, env in pub:
        gp = [x for x in a1["gamma_pairs"] if x["label"] == lab_a1 and x["env"] == env]
        fs = [x for x in a1["full_set_pairs"] if x["label"] == lab_a1 and x["env"] == env and x["order"] == "video-first"]
        if not gp or not fs:
            man["dropped"].append(dict(point=f"{pname} (public) vs Qwen", reason="a1 row missing")); continue
        x, y = 100 * fs[0]["delta_R50"], gp[0]["mask"]["gamma"]
        xci, yci = [100 * c for c in fs[0]["delta_R50_ci95"]], gp[0]["mask"]["gamma_ci95"]
        a.errorbar(x, y, xerr=ci_err(x, xci), yerr=ci_err(y, yci), fmt="none", ecolor=P["ink"], elinewidth=0.5, alpha=0.35, zorder=3)
        a.scatter(x, y, marker="D", s=24, facecolor="white", edgecolor=P["ink"], linewidth=0.8, zorder=5)
        dx, dy, ha = _COMPACT_LAB[pname] if compact else (0, 7, "center")
        a.annotate(pname, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=6.2, zorder=6)
        man["points"].append(dict(point=f"{pname} (public) vs Qwen base", env=env, delta_pp=x, delta_ci=xci, gamma=y, gamma_ci=yci,
                                  src=dict(file="a1_public_rl_vs_qwen.json", sha16=sha16(os.path.join(ANA, "a1_public_rl_vs_qwen.json")))))
    a6p = os.path.join(ANA, "a6_vc_vs_qwen.json")
    vc_pair = os.path.join(_SUPP + "/results/judgment_20260921/selective",
                           "pair_p4nat_qwenbase_vchat_tf5meta_node5_full__p4nat_vcbase_vchat_tf5meta_node3_full.json")
    if os.path.exists(a6p) and os.path.exists(vc_pair):
        a6 = json.load(open(a6p, encoding="utf-8"))
        node5 = [x for x in a6["pairs"] if x.get("node") == "node5" or ("_node5_" in x.get("base", "") and "_node5_" in x.get("arm", ""))]
        gp = node5[0] if node5 else a6["pairs"][0]
        pr = json.load(open(vc_pair, encoding="utf-8"))
        x, xci = 100 * pr["point"]["delta_marginal"], [100 * c for c in pr["delta_marginal_boot"]["ci95"]]
        y, yci = gp["mask"]["gamma"], gp["mask"]["gamma_ci95"]
        a.errorbar(x, y, xerr=ci_err(x, xci), yerr=ci_err(y, yci), fmt="none", ecolor=P["ink"], elinewidth=0.5, alpha=0.35, zorder=3)
        a.scatter(x, y, marker="D", s=24, facecolor="white", edgecolor=P["ink"], linewidth=0.8, zorder=5)
        dx, dy, ha = _COMPACT_LAB["VideoChat-R1"] if compact else (0, 7, "center")
        a.annotate("VideoChat-R1", (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=6.2, zorder=6)
        man["points"].append(dict(point="VideoChat-R1 (public) vs Qwen base", env="B", delta_pp=x, gamma=y,
                                  src=[dict(file=os.path.basename(a6p), sha16=sha16(a6p)), dict(file=os.path.basename(vc_pair), sha16=sha16(vc_pair))]))
    else:
        man["dropped"].append(dict(point="VideoChat-R1 (public) vs Qwen base",
                                   reason="Delta R@0.5 needs the Qwen base vchat full run (not yet landed); Gamma alone is not drawn"))
    a.set_xlabel("accuracy change, ΔR@0.5 vs. base (pp)")
    a.set_ylabel("evidence dependence change,\n" + r"$\Gamma_{\mathrm{mask}}$ vs. base (nats / token)")
    a.set_xlim(-10.5, 23.5); a.set_ylim(-0.38, 3.0)
    a.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    a.tick_params(which="both", top=False, right=False)
    save(fig, name, man)
    return man


# ============================================== fig3: relabel (canonical labelswap)
def fig3_relabel():
    man = dict(points=[], dropped=[])
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.52), gridspec_kw=dict(width_ratios=[0.9, 1.1]))
    draw_relabel(a, b, man)
    save(fig, "fig3_relabel", man)
    return man


def fig2_findings():
    """Section 3 figure: (a) blind answering, (b) frozen predictions under two label sets, (c) Gamma under both."""
    man = dict(points=[], dropped=[])
    fig = plt.figure(figsize=(W, 2.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.7, 1.0], wspace=0.95)
    a0, a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    draw_black(a0, man)
    panel(a0, "a")
    draw_relabel(a1, a2, man, letters=("b", "c"), compact=True)
    save(fig, "fig2_findings", man)
    return man


def draw_relabel(a, b, man, letters=("a", "b"), compact=False):
    # 09-25: the same rows and environments as the upper block of Table S5 (build_appendix_tables_v3.build_S5)
    by = {}
    for r in C.R:
        if r.get("kind") == "labelswap" and r.get("_key") and r["_key"][4] in (None, 0):
            by.setdefault(r["_key"], []).append(r)
    ls = []
    for k in by:
        env = C.env_for(k, "labelswap", "charades-sta")
        if k[0] == Q7 and k[1] == "base" and any(x["_env"] == "C" for x in by[k]):
            env = "C"
        rr = [x for x in by[k] if x["_env"] == env]
        r = rr[0] if rr else None
        if r is None or (r.get("matrix_870") or {}).get("sta_run|TL_labels") is None:
            man["dropped"].append(dict(point=key_str(k), reason=f"no labelswap record in {env}"))
            continue
        ls.append((k, r))
    x0, x1 = 0.0, 1.0
    for k, r in ls:
        y0, y1 = r["matrix_870"]["sta_run|STA_labels"], r["matrix_870"]["sta_run|TL_labels"]
        if k[0] == TL:
            col = "#4D4D4D" if k[1] == "base" else P["iou-grpo"]
            a.plot([x0, x1], [y0, y1], color=col, lw=1.5, zorder=3, solid_capstyle="round")
            a.scatter([x0, x1], [y0, y1], s=14, color=col, zorder=4)
            if not compact:   # compact: the caption names the two coloured lines
                a.annotate("TimeLens-8B base" if k[1] == "base" else "TimeLens-8B trained", (x1, y1),
                           xytext=(6, 3 if k[1] == "base" else 5), textcoords="offset points", ha="left", va="center", fontsize=6.5, color=col)
                for xx, yy in ((x0, y0), (x1, y1)):
                    a.annotate(f"{yy:.2f}", (xx, yy), xytext=(0, -8), textcoords="offset points", ha="center", va="top", fontsize=6.0, color=col)
        else:
            a.plot([x0, x1], [y0, y1], color="#B9B9B9", lw=0.7, zorder=2)
            a.scatter([x0, x1], [y0, y1], s=5, color="#B9B9B9", zorder=2)
        man["points"].append(dict(point=key_str(k), sta=y0, tl=y1, src=src(r)))
    band0 = [r["matrix_870"]["sta_run|STA_labels"] for k, r in ls if k[0] != TL]
    band1 = [r["matrix_870"]["sta_run|TL_labels"] for k, r in ls if k[0] != TL]
    for xx, vals, dx in ((x0, band0, -0.10), (x1, band1, 0.10)):
        lo, hi = min(vals), max(vals)
        a.plot([xx + dx, xx + dx], [lo, hi], color=P["note"], lw=0.8, zorder=2, solid_capstyle="butt")
        for yy in (lo, hi):
            a.plot([xx + dx - 0.02, xx + dx + 0.02], [yy, yy], color=P["note"], lw=0.8, zorder=2)
        a.annotate(f"{100 * (hi - lo):.0f} pp", (xx + dx, (lo + hi) / 2), xytext=(-3 if dx < 0 else 3, 0), textcoords="offset points",
                   ha="right" if dx < 0 else "left", va="center", fontsize=5.8, color=P["note"], rotation=90)
    man["summary"] = dict(n_7B=len(band0), span_STA_pp=100 * (max(band0) - min(band0)), span_TL_pp=100 * (max(band1) - min(band1)),
                          TL8B={("base" if k[1] == "base" else "trained"): [r["matrix_870"]["sta_run|STA_labels"], r["matrix_870"]["sta_run|TL_labels"]]
                                for k, r in ls if k[0] == TL})
    a.set_xlim(*((-0.3, 1.3) if compact else (-0.25, 1.62))); a.set_ylim(0.28, 0.76)
    a.set_xticks([x0, x1]); a.set_xticklabels(["STA\nlabels", "TimeLens\nlabels"] if compact else ["Charades-STA\nlabels", "TimeLens\nlabels"])
    a.set_ylabel("R@0.5, identical predictions")
    a.tick_params(axis="x", length=0, which="both"); a.tick_params(right=False, top=False, which="both")
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    panel(a, letters[0])
    rows = []
    for k, ds, tl, envs in C.rows_for("gamma", dataset=CTL, instrument="14Ln"):
        if tl is None:
            man["dropped"].append(dict(point=key_str(k) + " (TL labels)", reason=f"TL-label Gamma only in {envs}")); continue
        if k[1] not in ("sft", "grouped"):   # 09-25 r6: TimeLens-8B vs Qwen3-VL-8B (TL labels) is a base-vs-base row of Table S5, not a trained row
            man["dropped"].append(dict(point=key_str(k) + " (TL labels)", reason="not one of our trained models (Table S5 lower block)")); continue
        sta = gam(k, CS)
        if sta is not None and sta.get("_env") != tl.get("_env"):
            # both ends of a line must come from one environment: use the STA-label audit on the TL row's stack
            same = [r for r in C.R if r.get("kind") == "gamma" and r.get("_key") == k and r.get("dataset") == CS
                    and r.get("_env") == tl.get("_env") and "14Ln" in str(r.get("instrument", "14Ln"))]
            if len(same) == 1:
                sta = same[0]
            else:
                man["dropped"].append(dict(point=key_str(k) + " (STA end)", reason=f"no STA-label Gamma in env {tl.get('_env')}"))
                sta = None
        if compact:
            lab = f"{SHORT[k[0]]}, " + ("SFT" if k[1] == "sft" else f"α={k[2]:g}")
        else:
            lab = f"{SHORT[k[0]]} " + ("pure SFT" if k[1] == "sft" else f"placebo-controlled, α={k[2]:g}")
        rows.append((k[1] == "sft", lab, sta, tl))
        man["points"].append(dict(point=lab, tl_src=src(tl), sta_src=src(sta)))
    rows.sort(key=lambda t: (t[0], -(t[3]["gamma_mask"])))
    y = np.arange(len(rows))[::-1].astype(float) - 0.7 * np.array([r[0] for r in rows], dtype=float)
    for yi, (is_sft, lab, sta, tl) in zip(y, rows):
        if sta:
            b.plot([sta["gamma_mask"], tl["gamma_mask"]], [yi, yi], color="#C9C9C9", lw=1.6, zorder=2, solid_capstyle="round")
            b.plot(sta["gamma_mask_ci"], [yi + 0.13] * 2, color=P["sta"], lw=0.7, alpha=0.65, zorder=3)
            b.scatter(sta["gamma_mask"], yi, s=17, color=P["sta"], zorder=4)
        b.plot(tl["gamma_mask_ci"], [yi - 0.13] * 2, color=P["tl"], lw=0.7, alpha=0.65, zorder=3)
        b.scatter(tl["gamma_mask"], yi, s=17, color=P["tl"], zorder=4)
    b.axvline(0, color=P["base"], lw=0.6)
    b.set_yticks(y); b.set_yticklabels([r[1] for r in rows])
    open_axis(b); b.tick_params(top=False, right=False, which="both")
    b.set_xlabel(r"$\Gamma_{\mathrm{mask}}$ against own base (nats / token)")
    b.set_xlim(-0.45, 2.15)
    b.legend(handles=[Line2D([], [], marker="o", ls="", color=P["sta"], markersize=4, label="STA labels"),
                      Line2D([], [], marker="o", ls="", color=P["tl"], markersize=4, label="TimeLens labels")],
             loc="lower right", bbox_to_anchor=(1.0, -0.02))
    panel(b, letters[1])


# ============================================== fig5: phase plane only
def fig5_mechanism():
    man = dict(points=[], dropped=[])
    fig, a = plt.subplots(figsize=(2.7, 2.5))
    items = [("mDPO-style", MDPO, "#8C8C8C", "X", (6, 3, "left")),
             ("all terms", g_(Q7, 0.5), P["grouped"], "o", (7, -8, "left")),
             ("uniform weights", M(Q7, "grouped-nodro", 0.5, 0.1, 0), P["ablation"], "s", (-6, 7, "right")),
             ("no placebo bound", M(Q7, "grouped-nohinge", 0.5, 0.1, 0), P["ablation"], "^", (-6, 5, "right")),
             ("no bound, uniform weights", M(Q7, "grouped-uniform-nohinge", 0.5, 0.1, 0), P["ablation"], "v", (-4, 8, "right")),
             ("evidence margin only", M(Q7, "single-margin", 0.5, 0.1, 0), P["ablation"], "D", (0, -10, "center")),
             ("pure SFT", SFT, P["sft"], "P", (6, -1, "left"))]
    seq, base_pt = [], None
    for lab, k, col, mk, (dx, dy, ha) in items:
        g = gam(k)
        if g is None:
            man["dropped"].append(dict(point=lab, reason="no Gamma")); continue
        x, yv = g["dO_mask_arm"], g["dE_mask_arm"]
        a.scatter(x, yv, marker=mk, s=40 if lab == "all terms" else 34, facecolor=col, edgecolor="white", linewidth=0.6, zorder=5)
        a.annotate(lab, (x, yv), xytext=(dx, dy), textcoords="offset points", fontsize=5.8, ha=ha, va="center", zorder=6)
        man["points"].append(dict(point=lab, key=key_str(k), dO=x, dE=yv, src=src(g)))
        if lab not in ("pure SFT", "mDPO-style"):
            seq.append((x, yv))
        if lab == "all terms":
            base_pt = (g["dO_mask_base"], g["dE_mask_base"])
    if base_pt:
        a.scatter(*base_pt, marker="o", s=34, facecolor=P["base"], edgecolor="white", linewidth=0.6, zorder=4)
        a.annotate("base model", base_pt, xytext=(0, -9), textcoords="offset points", fontsize=5.8, ha="center", va="center")
        man["points"].append(dict(point="base model (from the full objective's paired run)", dO=base_pt[0], dE=base_pt[1]))
    a.set_xscale("symlog", linthresh=0.05, linscale=0.6); a.set_yscale("symlog", linthresh=0.05, linscale=0.6)
    a.set_xticks([-0.1, 0, 0.1, 0.5, 1, 2]); a.set_xticklabels(["−0.1", "0", "0.1", "0.5", "1.0", "2.0"])
    a.set_yticks([0, 0.1, 0.5, 1, 2, 4]); a.set_yticklabels(["0", "0.1", "0.5", "1.0", "2.0", "4.0"]); a.minorticks_off()
    a.set_xlim(-0.15, 3.2); a.set_ylim(-0.02, 6.2)
    a.axvspan(0.11, 3.2, color=P["band"], zorder=0, lw=0)
    a.set_xlabel(r"placebo drop $d_O$ (nats / token)"); a.set_ylabel(r"evidence drop $d_E$ (nats / token)")
    a.tick_params(which="both", top=False, right=False)
    save(fig, "fig5_mechanism", man)
    return man


# ============================================== fig6: dose
def fig6_dose():
    man = dict(points=[], dropped=[])
    fig, axs = plt.subplots(2, 2, figsize=(W, 3.15), sharex="col", gridspec_kw=dict(width_ratios=[1.45, 1.0]))
    for ax in axs.flat:
        ax.axhline(0, color=P["base"], lw=0.6, zorder=1)
        ax.tick_params(which="both", top=False, right=False)
    for ax in axs[0]:
        ax.axhspan(-0.10, 0.10, color=P["band"], zorder=0, lw=0)
    for fam in (Q7, TZ, TR, VC):
        pts = []
        for k in [x for x in GROUPED if x[0] == fam]:
            g = gam(k)
            dv, dci, dag, denv = C.delta3(k, CS)
            if g is None:
                man["dropped"].append(dict(point=key_str(k), reason="no Gamma")); continue
            pts.append((k[2], g, dv, dci))
            man["points"].append(dict(point=key_str(k), gamma=g["gamma_mask"], gamma_src=src(g), delta=dv, delta_ci=dci, dagger=dag, delta_env=denv))
        pts.sort(key=lambda t: t[0])
        if len(pts) < 2:
            continue
        col, mk = FAMC.get(fam, P["ink"]), FAM_MARK[fam]
        xs = [p[0] for p in pts]
        axs[0, 0].fill_between(xs, [p[1]["gamma_mask_ci"][0] for p in pts], [p[1]["gamma_mask_ci"][1] for p in pts], color=col, alpha=0.14, lw=0, zorder=1)
        axs[0, 0].plot(xs, [p[1]["gamma_mask"] for p in pts], marker=mk, ms=3.6, color=col, lw=1.1, markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        acc = [(p[0], 100 * p[2], [100 * c for c in p[3]] if p[3] else None) for p in pts if p[2] is not None]
        if len(acc) >= 2:
            ci_pts = [t for t in acc if t[2]]
            if len(ci_pts) >= 2:
                axs[1, 0].fill_between([t[0] for t in ci_pts], [t[2][0] for t in ci_pts], [t[2][1] for t in ci_pts], color=col, alpha=0.14, lw=0, zorder=1)
            axs[1, 0].plot([t[0] for t in acc], [t[1] for t in acc], marker=mk, ms=3.6, color=col, lw=1.1, markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        which, dx, dy = {Q7: ("last", 5, 5), TR: ("last", 5, -5), TZ: ("first", -5, 4), VC: ("first", -5, -5)}[fam]
        e = pts[-1] if which == "last" else pts[0]
        axs[0, 0].annotate(SHORT[fam], (e[0], e[1]["gamma_mask"]), xytext=(dx, dy), textcoords="offset points", fontsize=6.3,
                           va="center", ha="left" if dx > 0 else "right", color=col)
    # bases trained at one setting only (alpha=0.5): single points, so every trained base sits in this figure
    for fam, xj, dxy in ((TL, 0.55, (5, 0)),):   # small horizontal offset off the alpha=0.5 column
        k = g_(fam, 0.5)
        g = gam(k)
        dv, dci, dag, denv = C.delta3(k, CS)
        if g is None:
            man["dropped"].append(dict(point=key_str(k), reason="no Gamma")); continue
        axs[0, 0].errorbar(xj, g["gamma_mask"], yerr=ci_err(g["gamma_mask"], g["gamma_mask_ci"]), fmt="none", ecolor=P["base"],
                           elinewidth=0.6, capsize=0, alpha=0.6, zorder=2)
        axs[0, 0].scatter(xj, g["gamma_mask"], marker=FAM_MARK[fam], s=16, facecolor=P["base"], edgecolor="white", linewidth=0.5, zorder=5)
        axs[0, 0].annotate(SHORT[fam], (xj, g["gamma_mask"]), xytext=(dxy[0], -7), textcoords="offset points", fontsize=5.6,
                           va="center", ha="left" if dxy[0] > 0 else "right", color=P["base"])
        if dv is not None:
            axs[1, 0].scatter(xj, 100 * dv, marker=FAM_MARK[fam], s=16, facecolor=P["base"], edgecolor="white", linewidth=0.5, zorder=4)
            axs[1, 0].annotate(SHORT[fam], (xj, 100 * dv), xytext=dxy, textcoords="offset points", fontsize=5.8, va="center",
                               ha="left" if dxy[0] > 0 else "right", color=P["base"])
        man["points"].append(dict(point=key_str(k), gamma=g["gamma_mask"], gamma_src=src(g), delta=dv, delta_ci=dci, dagger=dag, single=True))
    axs[0, 0].set_ylabel("evidence dependence\n" + r"$\Gamma_{\mathrm{mask}}$ (nats / token)", fontsize=7.5)
    axs[1, 0].set_ylabel("accuracy change\nvs. own base (pp)", fontsize=7.5)
    axs[1, 0].set_xlabel(r"$\alpha$, weight of clean supervision")
    axs[0, 0].set_xscale("log"); axs[0, 0].set_xlim(0.085, 1.45)
    axs[1, 0].set_xticks([0.1, 0.25, 0.5, 1.0]); axs[1, 0].set_xticklabels(["0.1", "0.25", "0.5", "1.0"]); axs[1, 0].minorticks_off(); axs[0, 0].minorticks_off()
    panel(axs[0, 0], "a")
    bpts = []
    for be, k in ((0.0, SFT), (0.03, g_(Q7, 0.5, 0.03)), (0.1, g_(Q7, 0.5))):
        g = gam(k)
        dv, dci, dag, denv = C.delta3(k, CS)
        if g is None:
            man["dropped"].append(dict(point=key_str(k), reason="no Gamma")); continue
        bpts.append((be, g, dv, dci))
        man["points"].append(dict(point=key_str(k), panel="beta", gamma=g["gamma_mask"], gamma_src=src(g), delta=dv, delta_ci=dci, dagger=dag, delta_env=denv))
    xpos = [0.01 if b_[0] == 0 else b_[0] for b_ in bpts]
    axs[0, 1].fill_between(xpos, [b_[1]["gamma_mask_ci"][0] for b_ in bpts], [b_[1]["gamma_mask_ci"][1] for b_ in bpts], color=P["grouped"], alpha=0.14, lw=0, zorder=1)
    axs[0, 1].plot(xpos, [b_[1]["gamma_mask"] for b_ in bpts], color=P["grouped"], lw=1.1, zorder=2)
    acc = [(xp, 100 * b_[2], [100 * c for c in b_[3]] if b_[3] else None) for xp, b_ in zip(xpos, bpts) if b_[2] is not None]
    ci_pts = [t for t in acc if t[2]]
    if len(ci_pts) >= 2:
        axs[1, 1].fill_between([t[0] for t in ci_pts], [t[2][0] for t in ci_pts], [t[2][1] for t in ci_pts], color=P["ink"], alpha=0.10, lw=0, zorder=1)
    axs[1, 1].plot([t[0] for t in acc], [t[1] for t in acc], color=P["ink"], lw=1.1, alpha=0.85, zorder=2)
    for (be, g, dv, dci), xp in zip(bpts, xpos):
        fc = P["sft"] if be == 0 else P["grouped"]
        axs[0, 1].scatter(xp, g["gamma_mask"], s=16, color=fc, zorder=3, edgecolor="white", linewidth=0.5)
        if dv is not None:
            axs[1, 1].scatter(xp, 100 * dv, s=16, color=P["sft"] if be == 0 else P["ink"], zorder=3, edgecolor="white", linewidth=0.5)
    axs[0, 1].set_xscale("log"); axs[0, 1].set_xlim(0.0065, 0.15)
    axs[1, 1].set_xticks([0.01, 0.03, 0.1]); axs[1, 1].set_xticklabels(["0 (SFT)", "0.03", "0.1"]); axs[1, 1].minorticks_off(); axs[0, 1].minorticks_off()
    axs[1, 1].set_xlabel(r"$\beta$, weight of the evidence margins")
    axs[0, 0].set_ylim(-0.3, 2.9); axs[0, 1].set_ylim(-0.3, 2.9)
    axs[1, 0].set_ylim(-11, 24); axs[1, 1].set_ylim(-11, 24)
    axs[1, 1].tick_params(labelleft=False); axs[0, 1].tick_params(labelleft=False)
    panel(axs[0, 1], "b")
    save(fig, "fig6_dose", man)
    return man


# ============================================== fig7: behaviour signature
def fig7_signature():
    man = dict(points=[], dropped=[])
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.75), gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    draw_abst(a, man); panel(a, "a")
    draw_behav(b, man); panel(b, "b")
    save(fig, "fig7_signature", man)
    return man


def fig5_readouts():
    """Section 5 figure: (a) crossed queries, (b) mismatched queries, (c) regenerated predictions."""
    man = dict(points=[], dropped=[])
    fig = plt.figure(figsize=(W, 2.0), layout="none")   # manual placement: constrained layout squeezes the panels
    a = fig.add_axes([0.075, 0.19, 0.215, 0.74])
    b = fig.add_axes([0.395, 0.19, 0.215, 0.74])
    c = fig.add_axes([0.805, 0.19, 0.19, 0.74])
    draw_cross(a, man); panel(a, "a")
    draw_abst(b, man, compact=True); panel(b, "b")
    draw_behav(c, man, compact=True); panel(c, "c")
    save(fig, "fig5_readouts", man)
    return man


def draw_cross(ax, man):
    """Crossed-query design (qswap3, mask): x = change in the own-query contrast, y = the same contrast scored under the
    partner event's query, trained minus base.  y = x is a model that reacts to the edit's position only."""
    lim = (-1.8, 2.9)
    ax.plot(lim, lim, color=P["base"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0, color=P["base"], lw=0.5, zorder=1); ax.axvline(0, color=P["base"], lw=0.5, zorder=1)
    keys = []
    for x in C.ext:
        if x[0] == "qswap" and x[2] not in keys:
            keys.append(x[2])
    for k in sorted(keys, key=key_str):
        C.ctx("fig5a", key_str(k))
        hit = C.ext_pick("qswap", k, column="fig")
        if hit is None:
            man["dropped"].append(dict(panel="a", point=key_str(k), reason="no qswap3 product in the configuration's environment")); continue
        d = json.load(open(hit[4], encoding="utf-8"))
        s = d["mask"]["spec"]["vs_base"]
        xv, yv = s["d_psi_own"], s["d_psi_cross"]
        col = {"sft": P["sft"], "single-margin": P["ablation"]}.get(k[1], P["grouped"])
        tf = k[6] == "t"
        ax.scatter(xv, yv, marker=FAM_MARK[k[0]], s=22, facecolor="white" if tf else col, edgecolor=col if tf else "white",
                   linewidth=0.8 if tf else 0.5, zorder=4)
        man["points"].append(dict(panel="a", point=key_str(k), d_psi_own=xv, d_psi_cross=yv, gamma_spec=s["gamma_spec"],
                                  src=dict(file=os.path.basename(hit[4]), sha16=sha16(hit[4]))))
        lab = None
        if k[1] == "sft":
            lab = ("SFT", 5, -5, "left")
        elif k[0] == VC and not tf and k[2] == 0.5:
            lab = ("VideoChat-R1", 4, 6, "left")
        elif k[0] == TZ and tf:
            lab = ("TimeZero (t)", 5, 4, "left")
        if lab:
            ax.annotate(lab[0], (xv, yv), xytext=(lab[1], lab[2]), textcoords="offset points", ha=lab[3], va="center", fontsize=5.8, zorder=6)
    ax.set_xlim(-0.7, 2.9); ax.set_ylim(-1.8, 1.3)
    ax.set_xticks([0, 1, 2]); ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r"$\Delta\psi$, own query (nats / token)"); ax.set_ylabel(r"$\Delta\psi$, crossed query")
    ax.tick_params(which="both", top=False, right=False)


def draw_abst(a, man, compact=False):
    lim = 92
    a.plot([0, lim], [0, lim], color=P["base"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    dose = {}
    for k, ds, t, envs in C.rows_for("tneg"):
        if t is None:
            man["dropped"].append(dict(point=key_str(k), reason=f"tneg only in {envs}")); continue
        x, yv = 100 * t["clean_abstain_strict"], 100 * t["foreign_abstain_strict"]
        col = P["base"] if k[1] == "base" else (P["sft"] if k[1] == "sft" else P["grouped"])
        a.scatter(x, yv, marker=FAM_MARK[k[0]], s=30, facecolor=col, edgecolor="white", linewidth=0.6, zorder=4)
        man["points"].append(dict(panel="a", point=key_str(k), clean=x, foreign=yv, src=src(t)))
        if k[1] == "grouped":
            dose.setdefault(k[0], []).append((k[3] or 0, k[2] or 0, x, yv))
        lab = None
        if k[1] == "grouped" and k[3] == 0.03 and k[0] == Q7:
            lab = ("Qwen, β=0.03", 6, -9, "left") if compact else ("Qwen, β=0.03", -6, 4, "right")
        elif k[1] == "grouped" and k[0] == TR:
            lab = (f"Time-R1, α={k[2]:g}", 6, 0, "left")
        elif k[1] == "grouped" and k[0] == Q7:
            lab = ("Qwen, β=0.1", 6, 0, "left")
        elif k[1] == "base" and k[0] == Q7:
            lab = ("Qwen base", 6, -5, "left")
        if lab:
            a.annotate(lab[0], (x, yv), xytext=(lab[1], lab[2]), textcoords="offset points", ha=lab[3], va="center", fontsize=6.2, zorder=6)
    for fam, ptss in dose.items():
        if len(ptss) >= 2:
            ptss.sort()
            a.plot([p[2] for p in ptss], [p[3] for p in ptss], color=P["grouped"], lw=0.7, alpha=0.45, zorder=2)
    a.set_xlim(-3, 45 if compact else lim); a.set_ylim(-3, lim)
    a.set_xticks([0, 20, 40] if compact else [0, 20, 40, 60, 80]); a.set_yticks([0, 20, 40, 60, 80])
    a.tick_params(which="both", top=False, right=False)
    if compact:
        a.set_xlabel("abstention, matched queries (%)"); a.set_ylabel("abstention, mismatched queries (%)")
    else:
        a.set_xlabel("strict abstention, clean queries (%)"); a.set_ylabel("strict abstention, foreign queries (%)")


def draw_behav(b, man, compact=False):
    #: change in the evidence share (normalised contrast: of the answers correct with placebo frames masked,
    # the fraction lost with the evidence frames masked), trained minus base, mask, video-first runs only; the
    # placebo-controlled models and SFT as the reference (accuracy-oriented models are compared in Table 1)
    brows = []
    for k, ds, r, envs in C.rows_for("behav"):
        if k[6] != "v":
            man["dropped"].append(dict(panel="b", point=key_str(k), reason="evaluated text-first")); continue
        if k[1] not in ("sft", "grouped") or (k[1] == "grouped" and k[3] != 0.1):
            man["dropped"].append(dict(panel="b", point=key_str(k), reason="not a placebo-controlled model at beta=0.1 or SFT")); continue
        if r is None or r.get("nshare_mask") is None:
            man["dropped"].append(dict(panel="b", point=key_str(k), reason=f"no video-first behavioural run in the canonical environment (runs in {envs})")); continue
        lab = SHORT[k[0]] + (" SFT" if k[1] == "sft" else f" α={k[2]:g}") + ("" if compact else " (v)")
        brows.append((lab, k, r))
        man["points"].append(dict(panel="b", point=lab, nshare_mask=r["nshare_mask"], ci=r["nshare_mask_ci"], src=src(r)))
    brows.sort(key=lambda t: -t[2]["nshare_mask"])
    yb = np.arange(len(brows))[::-1]
    for yi, (lab, k, r) in zip(yb, brows):
        d, ci = 100 * r["nshare_mask"], r["nshare_mask_ci"]
        col = P["sft"] if k[1] == "sft" else P["grouped"]
        if ci:
            b.plot([100 * ci[0], 100 * ci[1]], [yi, yi], color=col, lw=1.0, zorder=3)
            for cv in ci:
                b.plot([100 * cv, 100 * cv], [yi - 0.10, yi + 0.10], color=col, lw=1.0, zorder=3)
        b.scatter(d, yi, marker=FAM_MARK[k[0]], s=26, facecolor=col, edgecolor="white", linewidth=0.6, zorder=4)
        b.annotate(f"+{d:.1f}" if d >= 0 else f"−{abs(d):.1f}", (d, yi), xytext=(0, 6), textcoords="offset points",
                   ha="center", va="bottom", fontsize=5.9, color=col, zorder=5)
    b.axvline(0, color=P["base"], lw=0.6)
    b.set_yticks(yb); b.set_yticklabels([t[0] for t in brows])
    open_axis(b); b.tick_params(which="both", top=False, right=False)
    b.set_xlabel("change in evidence share,\ntrained − base (pp)")
    b.set_xlim(-22, 36); b.set_xticks([-20, 0, 20])


# ============================================== fig9: OOD
def fig9_ood():
    man = dict(points=[], dropped=[])
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    RESO = 4.9
    a.axvspan(-RESO, RESO, color=P["band"], lw=0, zorder=0)
    a.axhline(0, color=P["base"], lw=0.6, zorder=1)
    for lab, k in (("placebo-controlled α=0.5", g_(Q7, 0.5)), ("placebo-controlled α=1.0", g_(Q7, 1.0)), ("β=0.03", g_(Q7, 0.5, 0.03)), ("pure SFT", SFT),
                   ("TimeZero α=0.25", g_(TZ, 0.25)), ("TimeZero α=0.5", g_(TZ, 0.5))):
        dv, dci, dag, denv = C.delta3(k, QV)
        g = gam(k, QV)
        if dv is None or g is None:
            man["dropped"].append(dict(panel="a", point=key_str(k), reason="no QVH Gamma for the video-first configuration" if g is None else "no QVH Delta"))
            continue
        col = OBJ.get(k[1], P["ablation"])
        x, xci, yv, yci = 100 * dv, [100 * c for c in dci] if dci else None, g["gamma_mask"], g["gamma_mask_ci"]
        a.errorbar(x, yv, xerr=ci_err(x, xci), yerr=ci_err(yv, yci), fmt="none", ecolor=col, elinewidth=0.55, alpha=0.4, capsize=0, zorder=3)
        a.scatter(x, yv, marker=FAM_MARK[k[0]], s=32, facecolor=col, edgecolor="white", linewidth=0.6, zorder=4)
        a.annotate(lab, (x, yv), xytext=(6, 2), textcoords="offset points", ha="left", va="center", fontsize=6.3, zorder=6)
        man["points"].append(dict(panel="a", point=key_str(k), delta=x, delta_ci=xci, delta_env=denv, gamma=yv, gamma_src=src(g)))
    a.set_xlim(-9.5, 10.5); a.set_ylim(-0.35, 1.75)
    a.set_xlabel(r"$\Delta$R@0.5 on QVHighlights, trained $-$ base (pp)")
    a.set_ylabel(r"$\Gamma_{\mathrm{mask}}$ on QVHighlights (nats / token)")
    a.tick_params(which="both", top=False, right=False)
    panel(a, "a")
    cells, xc = [], 0.0
    for fam in (TZ, Q7):
        for ds, dl in ((CS, "STA"), (QV, "QVH")):
            base, arm = kpick("black", M(fam, "base"), ds), kpick("black", g_(fam, 0.5), ds)
            if base is None or arm is None:
                man["dropped"].append(dict(panel="b", point=f"{fam} {dl}", reason="missing black run")); continue
            cells.append((xc, fam, dl, base, arm))
            man["points"].append(dict(panel="b", point=f"{fam} {dl}", base=base["black_answer_rate"], trained=arm["black_answer_rate"],
                                      base_src=src(base), arm_src=src(arm)))
            xc += 1.0
        xc += 0.55
    bw = 0.38
    for x, fam, dl, base, arm in cells:
        for dx, r, col in ((-bw / 2, base, "#B9B2A7"), (bw / 2, arm, P["grouped"])):
            v = r["black_answer_rate"]
            b.bar(x + dx, v, width=bw, color=col, zorder=3)
            b.text(x + dx, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=5.4, color="#555555", zorder=5)
    b.set_xticks([c[0] for c in cells]); b.set_xticklabels([c[2] for c in cells], fontsize=6.8)
    for fam in (TZ, Q7):
        xs_f = [c[0] for c in cells if c[1] == fam]
        if xs_f:
            b.text(float(np.mean(xs_f)), -0.16, SHORT[fam], ha="center", va="top", fontsize=6.8, color=P["ink"],
                   transform=b.get_xaxis_transform(), clip_on=False)
    b.set_ylim(0, 1.05); b.set_ylabel("black-screen answer rate")
    b.set_yticks([0, 0.5, 1.0]); b.set_yticklabels(["0", "0.5", "1.0"])
    open_axis(b, keep="y"); b.tick_params(axis="x", length=0, which="both"); b.tick_params(top=False, right=False, which="both")
    b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#B9B2A7", label="base"),
                      plt.Rectangle((0, 0), 1, 1, color=P["grouped"], label="placebo-controlled α=0.5")],
             loc="upper right", bbox_to_anchor=(1.02, 1.04), fontsize=6.2)
    panel(b, "b")
    save(fig, "fig9_ood", man)
    return man


if __name__ == "__main__":
    F = {"fig1": fig1_black, "fig1map": fig1_map, "fig2": fig2_map, "fig2f": fig2_findings, "fig3": fig3_relabel,
         "fig5": fig5_mechanism, "fig5r": fig5_readouts, "fig6": fig6_dose, "fig7": fig7_signature, "fig9": fig9_ood}
    summ = {}
    for k in (sys.argv[1:] or list(F)):
        m = F[k]()
        summ[k] = dict(n_points=len(m["points"]), dropped=m["dropped"], **({"summary": m["summary"]} if "summary" in m else {}))
    json.dump(summ, open(os.path.join(HERE, "..", "figures", "data", "FIGS_v5_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str)[:6000])
