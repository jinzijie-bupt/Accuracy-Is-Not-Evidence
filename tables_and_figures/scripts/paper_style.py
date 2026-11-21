#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper_style.py -- the one style module every data figure imports (round 2, 2026-09-24).

Spec: review/round2_20260924/figures_C_appendix_system.md §4 (+ figures_A_main.md §5).

Colour semantics, fixed across the paper:
  evidence #D55E00 / placebo #0072B2  -- ONLY for the edits themselves (evidence / placebo frames, d_E / d_O)
  ours     #009E73                    -- every model trained with the placebo-controlled objective
  acc      #3A3A3A                    -- accuracy-oriented training (SFT, IoU-GRPO, mDPO-style, public checkpoints)
  base     hollow, light grey edge    -- untrained base models
  em       #CC79A7                    -- the evidence-margin-only ablation (edit detector)
No other hue carries meaning.  Non-method binary contrasts (e.g. two label sets) use black filled vs. open.

Marker shape = base family, fixed:  Qwen-7B o, TimeZero s, Time-R1 ^, VideoChat-R1 v, Qwen-3B p, TimeLens-8B P;
public checkpoints are open diamonds (D).

Type sizes are for the FINAL printed size: every figure is drawn at its placed width (scale factor 1).
  ticks 7 pt, labels / legends / direct labels 7.5-8 pt, panel letters 9 pt bold; nothing below 7 pt.
Lines: axes 0.6 pt, data 1.0 pt, CI 0.75 pt without caps, references 0.5 pt.
Export: vector PDF with embedded TrueType (fonttype 42), SVG with live text, 300 dpi PNG preview.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

TEXTWIDTH = 5.5          # ICLR text width, inches

C = dict(
    evidence="#D55E00", evidence_ci="#EBAE80", evidence_bg="#FBE9DF",
    placebo="#0072B2", placebo_ci="#80B8D8", placebo_bg="#E3F0F7",
    ours="#009E73", ours_ci="#7FCEB9", ours_light="#A6DBCB",
    acc="#3A3A3A", acc_ci="#9A9A9A",
    base_fill="#FFFFFF", base_edge="#8A8A8A", base_ci="#C4C4C4",
    em="#CC79A7", em_ci="#E6BCD3",
    ref="#9A9A9A", band="#EEEEEE", band_text="#8A8A8A",
    ink="#1A1A1A", axis="#333333", note="#6A6A6A", faint="#D0D0D0",
)

MARK = {"Qwen-7B": "o", "TimeZero": "s", "Time-R1": "^", "VideoChat-R1": "v",
        "Qwen-3B": "p", "TimeLens-8B": "P", "public": "D"}

# canonical family ids (canonical_v2) -> short display names used inside figures
SHORT = {"Qwen2.5-VL-7B": "Qwen-7B", "Qwen2.5-VL-3B": "Qwen-3B", "TimeZero-7B": "TimeZero", "Time-R1-7B": "Time-R1",
         "VideoChat-R1-7B": "VideoChat-R1", "TimeLens-8B": "TimeLens-8B"}

FS_TICK = 7.0
FS_LABEL = 8.0
FS_NOTE = 7.5
FS_SMALL = 7.0
FS_PANEL = 9.0

RC = {
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": FS_LABEL, "axes.labelsize": FS_LABEL, "legend.fontsize": FS_NOTE,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "axes.titlesize": FS_LABEL,
    "axes.linewidth": 0.6, "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink"],
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.color": C["axis"], "ytick.color": C["axis"], "xtick.labelcolor": C["ink"], "ytick.labelcolor": C["ink"],
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "axes.labelpad": 3.0, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "text.color": C["ink"],
    "lines.linewidth": 1.0, "lines.markersize": 4.0, "lines.markeredgewidth": 0.5,
    "errorbar.capsize": 0,
    "legend.frameon": False, "legend.handlelength": 1.0, "legend.handletextpad": 0.4, "legend.borderpad": 0.0,
    "legend.borderaxespad": 0.0, "legend.columnspacing": 1.0, "legend.labelspacing": 0.3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 300, "savefig.bbox": None, "savefig.pad_inches": 0.0,
    "figure.constrained_layout.use": False, "figure.autolayout": False,
    "axes.grid": False,
}


def apply():
    mpl.rcParams.update(RC)


def panel_letter(fig, x_in, y_in, s):
    """9 pt bold lower-case letter; position in inches from the figure's lower-left corner (top-left anchored)."""
    w, h = fig.get_size_inches()
    return fig.text(x_in / w, y_in / h, s, fontsize=FS_PANEL, fontweight="bold", ha="left", va="top")


def axes_in(fig, x, y, w, h):
    """Axes placed in physical inches (lower-left x, y, width, height)."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


def clean_axis(ax, left=True, bottom=True):
    ax.minorticks_off()
    for side, sp in ax.spines.items():
        sp.set_visible((side == "left" and left) or (side == "bottom" and bottom))
        sp.set_linewidth(0.6)
        sp.set_color(C["axis"])
    ax.tick_params(which="both", top=False, right=False)
    if not left:
        ax.tick_params(axis="y", left=False, labelleft=False)
    if not bottom:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)


def ci_h(ax, lo, hi, y, color, lw=0.75, zorder=2, **kw):
    """horizontal CI segment, no caps"""
    return ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=zorder, **kw)[0]


def ci_v(ax, x, lo, hi, color, lw=0.75, zorder=2, **kw):
    return ax.plot([x, x], [lo, hi], color=color, lw=lw, solid_capstyle="butt", zorder=zorder, **kw)[0]


def save(fig, stem, formats=("pdf", "svg", "png")):
    for ext in formats:
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches=None, pad_inches=0, facecolor="white")
