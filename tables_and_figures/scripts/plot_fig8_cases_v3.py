"""Fig 8, v3: two real cases following the generated P4 case-layout skeleton
(<local>/P4): query line, film strip with evidence
frames outlined, three parallel interval bars (annotated / base / grouped), and a
quantified read-out block per model (evidence drop vs matched placebo drop, values
printed, shared scale across cases).

v3 vs v2: q191 (dark closet, ZDV60) replaced by q157 (bright bedroom, AK2KG,
identical base/arm prediction, dE 0.06 -> 3.81); read-out dumbbells replaced by
labelled bars with printed values; semantic colours: annotated interval & evidence
frames = vermillion, matched placebo = blue, grouped arm = green, base = gray.

Sources: assets/source_videos/REAL_CASES_MANIFEST_20260922.json (frames decoded from
byte-SHA-matched source videos; dE/dO and the predictions from the 14Ln records
named in the manifest).
"""
import json, os, sys, hashlib
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from plot_figs_v3 import P  # palette; rcParams applied at import
P.setdefault("evidence", "#D55E00")   # vermillion: evidence frames / annotated interval
P.setdefault("placebo", "#0072B2")    # blue: matched outside (placebo) replacement

MAN = os.path.join(ROOT, "assets", "source_videos", "REAL_CASES_MANIFEST_20260922.json")
VID_DIR = os.path.join(ROOT, "assets", "source_videos")
OUT = os.path.join(ROOT, "figures", "data")
N_THUMB = 10
TH, TW = 60, 80

CASE_ORDER = [157, 52]


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def read_frame(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, fr = cap.read()
    if not ok:
        return None
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    h, w = fr.shape[:2]
    if w / h > 4 / 3:
        nw = int(h * 4 / 3); x0 = (w - nw) // 2; fr = fr[:, x0:x0 + nw]
    else:
        nh = int(w * 3 / 4); y0 = (h - nh) // 2; fr = fr[y0:y0 + nh]
    return cv2.resize(fr, (TW, TH), interpolation=cv2.INTER_AREA)


def case_strip(c):
    times = c["native_times_seconds"]
    n = len(times)
    pick = np.unique(np.round(np.linspace(0, n - 1, N_THUMB)).astype(int))
    cap = cv2.VideoCapture(os.path.join(VID_DIR, c["video"]))
    frames, is_ev = [], []
    ev = set(c["evidence_indices"])
    for i in pick:
        fr = read_frame(cap, times[i])
        if fr is None:
            fr = np.full((TH, TW, 3), 200, np.uint8)
        frames.append(fr); is_ev.append(i in ev)
    cap.release()
    return frames, is_ev, [times[i] for i in pick], times[-1]


def main():
    man = json.load(open(MAN, encoding="utf-8"))
    by_qid = {c["qid"]: c for c in man["cases"]}
    cases = [by_qid[q] for q in CASE_ORDER]
    fig = plt.figure(figsize=(5.5, 2.74))
    gs = fig.add_gridspec(len(cases), 2, width_ratios=[2.9, 1.25], wspace=0.10, hspace=0.34,
                          left=0.012, right=0.968, top=0.955, bottom=0.135)
    manifest = {"figure": "fig8_cases", "sources": [{"file": os.path.basename(MAN), "sha16": sha16(MAN)}], "cases": []}
    BARS = [("annotated interval", "gt", P["evidence"], 2.8),
            ("base prediction", "base", P["base"], 1.6),
            ("trained prediction", "grouped", P["grouped"], 1.6)]
    XMAX = 4.35   # shared read-out scale across cases (max dE 3.81)
    for r, c in enumerate(cases):
        frames, is_ev, tpick, tend = case_strip(c)
        gap = 3
        mosaic = np.full((TH, len(frames) * TW + (len(frames) - 1) * gap, 3), 255, np.uint8)
        for k, fr in enumerate(frames):
            x = k * (TW + gap); mosaic[:, x:x + TW] = fr
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(mosaic, interpolation="nearest")
        for k, e in enumerate(is_ev):
            x = k * (TW + gap)
            ax.add_patch(Rectangle((x - 0.5, -0.5), TW, TH, fill=False, lw=1.7 if e else 0.6,
                                   ec=P["evidence"] if e else "#bbbbbb"))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        # ---- interval strip: annotated / base / grouped ----
        Wpx = mosaic.shape[1]
        y0, step = TH + 14, 22
        for j, (lab, key, col, lw) in enumerate(BARS):
            yj = y0 + j * step
            iv = c["gt_seconds"] if key == "gt" else c["records"][key]["prediction"]
            ax.plot([0, Wpx], [yj, yj], color="#e2e2e2", lw=0.7, clip_on=False, zorder=1)
            ax.plot([iv[0] / tend * Wpx, iv[1] / tend * Wpx], [yj, yj], color=col, lw=lw,
                    solid_capstyle="butt", clip_on=False, zorder=2)
            ax.text(max(iv[1] / tend * Wpx, 0.30 * Wpx) + 6, yj, lab, fontsize=6.2, va="center",
                    ha="left", color=col, clip_on=False)
        ax.text(0, y0 + 2 * step + 13, "0 s", fontsize=6.2, va="center", ha="left", color="#666666")
        ax.text(Wpx, y0 + 2 * step + 13, f"{tend:.0f} s", fontsize=6.2, va="center", ha="right", color="#666666")
        ax.set_ylim(y0 + 2 * step + 20, -TH * 0.55)
        ax.set_xlim(-0.5, Wpx - 0.5)
        ax.text(0, -TH * 0.16, f"“{c['query'].rstrip('.')}”", fontsize=7.5, va="bottom", ha="left", style="italic")
        ax.text(-0.012, 1.06, f"({'ab'[r]})", transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="bottom", ha="left")
        # ---- right: quantified read-out, evidence vs matched placebo, values printed ----
        bx = fig.add_subplot(gs[r, 1])
        rows = [("base", 1.0), ("grouped", 0.0)]
        bh = 0.30
        for key, y in rows:
            rec = c["records"][key]
            dE, dO = rec["dE_swap"], rec["dO_swap"]
            bx.barh(y + 0.19, dE, height=bh, color=P["evidence"], zorder=3)
            bx.barh(y - 0.19, dO, height=bh, color=P["placebo"], zorder=3)
            for v, yy in ((dE, y + 0.19), (dO, y - 0.19)):
                if v >= 0:
                    bx.text(v + 0.09, yy, f"{v:.2f}", fontsize=6.2, va="center", ha="left")
                else:
                    bx.text(0.09, yy, f"\u2212{abs(v):.2f}", fontsize=6.2, va="center", ha="left")
            manifest["cases"].append({"qid": c["qid"], "arm": key, "dE_swap": dE, "dO_swap": dO,
                                      "prediction": rec.get("prediction"), "gt_seconds": c["gt_seconds"],
                                      "source": rec["source"]["path"], "sha256": rec["source"]["sha256"]})
        if r == 0:
            bx.text(0.985, 0.97, "evidence frames replaced", fontsize=5.8, color=P["evidence"],
                    ha="right", va="top", transform=bx.transAxes)
            bx.text(0.985, 0.80, "matched outside replacement", fontsize=5.8, color=P["placebo"],
                    ha="right", va="top", transform=bx.transAxes)
        bx.axvline(0, color="#999999", lw=0.7, zorder=1)
        bx.set_yticks([y for _, y in rows]); bx.set_yticklabels([{"grouped": "trained"}.get(k, k) for k, _ in rows])
        bx.set_ylim(-0.65, 1.65)
        bx.set_xlim(-0.55, 4.85)
        bx.set_xticks([0, 2, 4])
        bx.tick_params(axis="y", which="both", length=0, pad=2)
        for sname in ("left", "top", "right"):
            bx.spines[sname].set_visible(False)
        bx.tick_params(which="both", top=False, right=False, left=False)
        if r == len(cases) - 1:
            bx.set_xlabel("support lost (nats / token)", fontsize=7)
            bx.xaxis.set_label_coords(0.55, -0.30)
        else:
            bx.set_xticklabels([])
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig8_cases.pdf"))
    fig.savefig(os.path.join(OUT, "fig8_cases.png"), dpi=200)
    json.dump(manifest, open(os.path.join(OUT, "fig8_cases.sources.json"), "w"), indent=1)
    print("wrote fig8_cases (v3, P4 layout)", len(cases), "cases")


if __name__ == "__main__":
    main()
