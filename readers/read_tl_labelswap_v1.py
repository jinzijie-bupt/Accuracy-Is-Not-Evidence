#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""read_tl_labelswap_v1.py — registered reader for the label-convention / window-length
diagnostic: Charades-STA products (12p4n rows, n=3720) vs charades-timelens products (12p6n rows, n=3363).

The two products are joined on (video stem, normalised query).  On the common subset the reader reports
  (1) whether the two runs emitted bit-identical predictions (identical fraction / mean & max L1 / parse
      mismatches).  This is only interpretable when both products come from the SAME environment class;
      the machine of each side and the `same_env` verdict are written into the output.
  (2) the 2x2 matrix {prediction source} x {label set}: parse, R@0.3/0.5/0.7, mIoU (+ video-cluster CI).
  (3) GT span lengths (both label sets) and predicted span lengths: mean / median / share of duration.
  (4) window oracles: keep the predicted CENTRE, force the span length to a constant L (default 4,6,8,10 s);
      and the GT-length oracle (keep the centre, use that question's GT length).
  (5) length-controlled re-scoring R@0.5* — identical estimator to the GT-length oracle.  *** DIAGNOSTIC ONLY:
      it consumes the ground-truth segment length, so it is NOT an accuracy number and must never be put in
      an accuracy table (label it "length-controlled re-scoring (diagnostic; uses ground-truth length)"). ***
  (6) centre error |dc|/dur on PARSED rows only, with parse rate reported as its own column (never folded in).
All point estimates carry video-cluster bootstrap percentile CIs (B draws, seed).
Unparsed rows count as IoU 0 and are never dropped; a row with a missing/None GT bound is a hard error,
never a silent n=0.  Reversed predictions (e<s) are sorted before use, exactly as read_p4nat_v1 scores them.

usage:
  read_tl_labelswap_v1.py --label NAME --sta F [F ...] --tl F [F ...] --out OUT
        [--sta-machine m] [--tl-machine m] [--norm ws|lower|raw|nopunct] [--oracle-lens 4,6,8,10]
        [--B 2000] [--seed 0] [--boundary clip|shift|none] [--expect-sta-n N] [--expect-tl-n N]
  read_tl_labelswap_v1.py --selftest
"""
import argparse, json, hashlib, os, re, sys, datetime, collections
import numpy as np

READER = "read_tl_labelswap_v1"

# environment classes: products from the same class are byte-comparable.  node1 == node3 == node5 was
# verified locally (p4nat_base_timer1_default_black_node1_full vs p4nat_qwenbase_timer1_default_black_node3_full:
# raw text 3720/3720 identical), node2 differs from that class on the same arm (1797/3720).
ENV_CLASS = {"node1": "B", "node3": "B", "node5": "B", "node5": "B", "node2": "A", "node4": "C"}


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def iou_1d(a, b):
    if not a or not b or len(a) != 2 or len(b) != 2:
        return 0.0
    s1, e1 = float(a[0]), float(a[1]); s2, e2 = float(b[0]), float(b[1])
    if e1 < s1: s1, e1 = e1, s1
    if e2 < s2: s2, e2 = e2, s2
    inter = max(0.0, min(e1, e2) - max(s1, s2)); uni = max(e1, e2) - min(s1, s2)
    return inter / uni if uni > 0 else 0.0


def sort_span(p):
    if not p or len(p) != 2 or p[0] is None or p[1] is None:
        return None
    s, e = float(p[0]), float(p[1])
    return (e, s) if e < s else (s, e)


def centre_len(p):
    sp = sort_span(p)
    return (None, None) if sp is None else ((sp[0] + sp[1]) / 2.0, sp[1] - sp[0])


BOUNDARY_MODES = ("clip", "shift", "none")


def recentre(p, L, dur=None, boundary="clip"):
    """Keep the predicted centre, force the span length to L.  None (unparsed) stays None -> scores 0.
    boundary handling for a window that falls off the video (this choice is worth ~1 pp, so it is explicit):
      'clip'  intersect with [0,dur]; the window may end up SHORTER than L  (reproduces the C5 numbers)
      'shift' slide the window back inside [0,dur], keeping length L        (reproduces the C14 numbers)
      'none'  no boundary handling at all (pure centre+length)
    """
    sp = sort_span(p)
    if sp is None or L is None:
        return None
    c = (sp[0] + sp[1]) / 2.0
    s, e = c - L / 2.0, c + L / 2.0
    if boundary == "none" or not dur:
        return (s, e)
    D = float(dur)
    if boundary == "clip":
        return (max(0.0, s), min(D, e))
    if boundary == "shift":
        if s < 0.0: s, e = 0.0, min(D, L)
        if e > D: e, s = D, max(0.0, D - L)
        return (s, e)
    raise ValueError("unknown boundary mode " + str(boundary))


def norm_q(s, mode):
    s = "" if s is None else str(s)
    if mode == "raw":
        return s
    if mode == "ws":                       # C5's convention -> 870 common questions
        return re.sub(r"\s+", " ", s.strip())
    if mode == "lower":                    # C14's convention -> 872 common questions
        return re.sub(r"\s+", " ", s.strip().lower())
    if mode == "nopunct":
        return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", s.strip().lower()))
    raise ValueError("unknown --norm " + mode)


def vid_stem(v):
    return os.path.splitext(str(v))[0]


def load_rows(files, tag):
    """concatenate shards by qid (duplicate qid or qid gap -> hard error); GT bounds must be present."""
    rows = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        rs = d["rows"] if isinstance(d, dict) and "rows" in d else d
        if not isinstance(rs, list) or not rs:
            raise ValueError("%s (%s): no rows" % (f, tag))
        for r in rs:
            for k in ("qid", "video", "caption", "gsA", "geA"):
                if k not in r or r[k] is None:
                    raise ValueError("%s (%s): row missing %s -> refusing to score (GT length absent is an "
                                     "error, not n=0): %s" % (f, tag, k, str(r)[:140]))
            q = int(r["qid"])
            if q in rows:
                raise ValueError("%s (%s): duplicate qid %d" % (f, tag, q))
            rows[q] = r
    qs = sorted(rows)
    if qs != list(range(qs[0], qs[0] + len(qs))):
        miss = sorted(set(range(qs[0], qs[-1] + 1)) - set(qs))
        raise ValueError("%s: non-contiguous qid coverage, %d missing, first %s" % (tag, len(miss), miss[:5]))
    return [rows[q] for q in qs]


def gt_of(r):
    gs, ge = r["gsA"], r["geA"]
    if gs is None or ge is None:
        raise ValueError("missing GT bound on qid %s" % r.get("qid"))
    gs, ge = float(gs), float(ge)
    return (ge, gs) if ge < gs else (gs, ge)


# ---------------------------------------------------------------- bootstrap ----
def cluster_groups(vids):
    idx = collections.OrderedDict()
    for i, v in enumerate(vids):
        idx.setdefault(v, []).append(i)
    return [np.array(v, dtype=int) for v in idx.values()]


def boot_many(values, vids, B, seed):
    """one shared video-cluster resampling for every named per-question value array."""
    groups = cluster_groups(vids); G = len(groups)
    keys = list(values); arrs = {k: np.asarray(values[k], dtype=float) for k in keys}
    out = {k: {"mean": (float(arrs[k].mean()) if arrs[k].size else None), "ci95": [None, None],
               "n": int(arrs[k].size)} for k in keys}
    if G == 0 or B <= 0:
        return out
    rng = np.random.default_rng(seed); acc = {k: np.empty(B) for k in keys}
    for b in range(B):
        pick = rng.integers(0, G, G); ii = np.concatenate([groups[p] for p in pick])
        for k in keys:
            acc[k][b] = arrs[k][ii].mean()
    for k in keys:
        out[k]["ci95"] = [float(np.percentile(acc[k], 2.5)), float(np.percentile(acc[k], 97.5))]
    out["_n_videos"] = G
    return out


def qstats(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if not a.size:
        return {"n": 0, "mean": None, "median": None}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a))}


# ---------------------------------------------------------------- scoring ----
def per_q(preds, gts, durs, lens, boundary):
    """per-question value arrays for one (prediction source, label set) cell."""
    n = len(preds)
    v = {"parsed": np.zeros(n), "iou": np.zeros(n), "hit03": np.zeros(n), "hit05": np.zeros(n),
         "hit07": np.zeros(n), "star_hit05": np.zeros(n), "star_iou": np.zeros(n)}
    for L in lens:
        v["oracle_L%g_hit05" % L] = np.zeros(n)
    predlen, predlen_ratio, gtlen, gtlen_ratio, cerr, cerr_s = [], [], [], [], [], []
    for i, (p, g, dur) in enumerate(zip(preds, gts, durs)):
        gl = g[1] - g[0]
        gtlen.append(gl)
        gtlen_ratio.append(gl / float(dur) if dur else None)
        sp = sort_span(p)
        v["parsed"][i] = 1.0 if sp is not None else 0.0
        io = iou_1d(sp, g); v["iou"][i] = io
        v["hit03"][i] = 1.0 if io >= 0.3 else 0.0
        v["hit05"][i] = 1.0 if io >= 0.5 else 0.0
        v["hit07"][i] = 1.0 if io >= 0.7 else 0.0
        st = recentre(sp, gl, dur, boundary)                  # R@0.5* == GT-length oracle
        sio = iou_1d(st, g); v["star_iou"][i] = sio
        v["star_hit05"][i] = 1.0 if sio >= 0.5 else 0.0
        for L in lens:
            oio = iou_1d(recentre(sp, float(L), dur, boundary), g)
            v["oracle_L%g_hit05" % L][i] = 1.0 if oio >= 0.5 else 0.0
        if sp is None:
            predlen.append(None); predlen_ratio.append(None); cerr.append(None); cerr_s.append(None)
        else:
            c, pl = centre_len(sp)
            predlen.append(pl); predlen_ratio.append(pl / float(dur) if dur else None)
            gc = (g[0] + g[1]) / 2.0
            cerr_s.append(abs(c - gc))
            cerr.append(abs(c - gc) / float(dur) if dur else None)
    aux = {"predlen": predlen, "predlen_ratio": predlen_ratio, "gtlen": gtlen,
           "gtlen_ratio": gtlen_ratio, "centre_err": cerr, "centre_err_s": cerr_s}
    return v, aux


def cell(preds, gts, durs, vids, lens, boundary, B, seed):
    v, aux = per_q(preds, gts, durs, lens, boundary)
    bt = boot_many(v, vids, B, seed)
    out = {"n": len(preds), "n_videos": bt.pop("_n_videos", None),
           "parse_rate": bt["parsed"]["mean"],
           "R@0.3": bt["hit03"]["mean"], "R@0.5": bt["hit05"]["mean"], "R@0.5_ci95": bt["hit05"]["ci95"],
           "R@0.7": bt["hit07"]["mean"], "mIoU": bt["iou"]["mean"], "mIoU_ci95": bt["iou"]["ci95"],
           "R@0.5_star_DIAGNOSTIC": bt["star_hit05"]["mean"], "R@0.5_star_ci95": bt["star_hit05"]["ci95"],
           "mIoU_star_DIAGNOSTIC": bt["star_iou"]["mean"],
           "oracle_const_L": {("L=%g" % L): bt["oracle_L%g_hit05" % L]["mean"] for L in lens},
           "oracle_const_L_ci95": {("L=%g" % L): bt["oracle_L%g_hit05" % L]["ci95"] for L in lens},
           "gt_len": qstats(aux["gtlen"]), "gt_len_ratio_of_dur": qstats(aux["gtlen_ratio"]),
           "pred_len_parsed_only": qstats(aux["predlen"]),
           "pred_len_ratio_of_dur_parsed_only": qstats(aux["predlen_ratio"])}
    # centre error: parsed rows only, its own cluster bootstrap; parse rate stays a separate column
    keep = [i for i, c in enumerate(aux["centre_err"]) if c is not None]
    if keep:
        cb = boot_many({"centre_err": [aux["centre_err"][i] for i in keep]}, [vids[i] for i in keep], B, seed)
        out["centre_err_parsed_only"] = {"n_parsed": len(keep), "mean": cb["centre_err"]["mean"],
                                         "ci95": cb["centre_err"]["ci95"],
                                         "median": qstats([aux["centre_err"][i] for i in keep])["median"],
                                         "abs_seconds": qstats([aux["centre_err_s"][i] for i in keep]),
                                         "parse_rate": out["parse_rate"]}
    else:
        out["centre_err_parsed_only"] = {"n_parsed": 0, "mean": None, "ci95": [None, None],
                                         "median": None, "parse_rate": out["parse_rate"],
                                         "note": "no parsed row (absence recorded, not silently 0)"}
    ratio = None
    if out["pred_len_parsed_only"]["median"] and out["gt_len"]["median"]:
        ratio = out["pred_len_parsed_only"]["median"] / out["gt_len"]["median"]
    out["pred_over_gt_len_median_ratio"] = ratio
    return out


def pred_identity(pa, pb):
    n = len(pa); ident = 0; l1 = []; mism = 0; both = 0
    for a, b in zip(pa, pb):
        sa, sb = sort_span(a), sort_span(b)
        if (sa is None) != (sb is None):
            mism += 1; continue
        if sa is None:
            ident += 1; continue
        both += 1
        d = abs(sa[0] - sb[0]) + abs(sa[1] - sb[1]); l1.append(d)
        if d == 0.0:
            ident += 1
    return {"n": n, "n_identical": ident, "frac_identical": (ident / n if n else None),
            "n_both_parsed": both, "n_parse_mismatch": mism,
            "mean_L1_both_parsed": (float(np.mean(l1)) if l1 else None),
            "max_L1_both_parsed": (float(np.max(l1)) if l1 else None)}


# ---------------------------------------------------------------- join ----
def join_pairs(sta, tl, mode):
    A, Bm = {}, {}
    for r in sta: A.setdefault((vid_stem(r["video"]), norm_q(r["caption"], mode)), []).append(r)
    for r in tl:  Bm.setdefault((vid_stem(r["video"]), norm_q(r["caption"], mode)), []).append(r)
    keys = sorted(set(A) & set(Bm))
    amb = [k for k in keys if len(A[k]) > 1 or len(Bm[k]) > 1]
    pairs = [(A[k][0], Bm[k][0]) for k in keys if k not in set(amb)]
    return pairs, {"n_keys": len(keys), "n_ambiguous_dropped": len(amb),
                   "ambiguous_examples": [list(k) for k in amb[:5]]}


def env_of(machine):
    return ENV_CLASS.get((machine or "").lower(), "unknown")


def guess_machine(files):
    for f in files:
        b = os.path.basename(f).lower()
        for m in ("node1", "node3", "node2", "node4", "node5"):
            if "_" + m in b or b.endswith(m + ".json"):
                return m
        if "_node5." in b or b.endswith("_node5.json"):
            return "node5"
    return "unknown"


def report(sta_rows, tl_rows, lens, boundary, B, seed, mode):
    out = {}
    durs_s = [r.get("dur") for r in sta_rows]; durs_t = [r.get("dur") for r in tl_rows]
    gts_s = [gt_of(r) for r in sta_rows]; gts_t = [gt_of(r) for r in tl_rows]
    vids_s = [vid_stem(r["video"]) for r in sta_rows]; vids_t = [vid_stem(r["video"]) for r in tl_rows]
    ps = [r.get("pred") for r in sta_rows]; pt = [r.get("pred") for r in tl_rows]
    out["full"] = {"sta_run_vs_STA_labels": cell(ps, gts_s, durs_s, vids_s, lens, boundary, B, seed),
                   "tl_run_vs_TL_labels": cell(pt, gts_t, durs_t, vids_t, lens, boundary, B, seed)}
    # ---- common subset
    pairs, jinfo = join_pairs(sta_rows, tl_rows, mode)
    alt = {m: join_pairs(sta_rows, tl_rows, m)[1]["n_keys"] for m in ("raw", "ws", "lower", "nopunct")}
    out["match"] = dict(jinfo); out["match"]["norm"] = mode
    out["match"]["n_common_by_norm"] = alt
    out["match"]["n_sta"] = len(sta_rows); out["match"]["n_tl"] = len(tl_rows)
    out["match"]["n_common"] = len(pairs)
    if not pairs:
        out["common"] = {"note": "no common question (absence recorded, not silently 0)"}
        return out
    rs = [a for a, _ in pairs]; rt = [b for _, b in pairs]
    vids = [vid_stem(a["video"]) for a in rs]
    out["match"]["n_videos_common"] = len(set(vids))
    gs = [gt_of(a) for a in rs]; gtl = [gt_of(b) for b in rt]
    ds = [a.get("dur") for a in rs]; dt = [b.get("dur") for b in rt]
    dd = [abs(float(x) - float(y)) for x, y in zip(ds, dt) if x and y]
    pS = [a.get("pred") for a in rs]; pT = [b.get("pred") for b in rt]
    out["common"] = {
        "pred_identity_sta_run_vs_tl_run": pred_identity(pS, pT),
        "duration_max_abs_diff_between_runs": (float(np.max(dd)) if dd else None),
        "matrix": {
            "sta_run|STA_labels": cell(pS, gs, ds, vids, lens, boundary, B, seed),
            "sta_run|TL_labels": cell(pS, gtl, ds, vids, lens, boundary, B, seed),
            "tl_run|STA_labels": cell(pT, gs, dt, vids, lens, boundary, B, seed),
            "tl_run|TL_labels": cell(pT, gtl, dt, vids, lens, boundary, B, seed)},
    }
    gg = [iou_1d(a, b) for a, b in zip(gs, gtl)]
    out["common"]["gt_vs_gt"] = {"mIoU": float(np.mean(gg)), "frac_iou_ge_0.5": float(np.mean(np.array(gg) >= 0.5)),
                                 "frac_iou_lt_0.3": float(np.mean(np.array(gg) < 0.3)),
                                 "median_abs_dstart": float(np.median([abs(a[0] - b[0]) for a, b in zip(gs, gtl)])),
                                 "median_abs_dend": float(np.median([abs(a[1] - b[1]) for a, b in zip(gs, gtl)])),
                                 "n_same_span": int(sum(1 for a, b in zip(gs, gtl) if a == b))}
    return out


def boundary_sensitivity(sta_rows, tl_rows, lens):
    """The boundary convention for oracle windows is worth ~1 pp, and the two verdict entries used different
    ones (C5 clip, C14 shift).  Report every mode side by side so neither number looks like a discrepancy."""
    out = {"note": "clip = intersect with [0,dur] (C5); shift = slide inside keeping length L (C14); "
                   "none = no boundary handling"}
    for tag, rows in (("sta_run_vs_STA_labels", sta_rows), ("tl_run_vs_TL_labels", tl_rows)):
        gts = [gt_of(r) for r in rows]; durs = [r.get("dur") for r in rows]; preds = [r.get("pred") for r in rows]
        blk = {}
        for mode in BOUNDARY_MODES:
            d = {"R@0.5_star_DIAGNOSTIC": float(np.mean([1.0 if iou_1d(recentre(p, g[1] - g[0], u, mode), g) >= 0.5
                                                         else 0.0 for p, g, u in zip(preds, gts, durs)]))}
            for L in lens:
                d["oracle_L%g_R@0.5" % L] = float(np.mean([1.0 if iou_1d(recentre(p, float(L), u, mode), g) >= 0.5
                                                           else 0.0 for p, g, u in zip(preds, gts, durs)]))
                d["oracle_L%g_mIoU" % L] = float(np.mean([iou_1d(recentre(p, float(L), u, mode), g)
                                                          for p, g, u in zip(preds, gts, durs)]))
            blk[mode] = d
        out[tag] = blk
    return out


# ---------------------------------------------------------------- selftest ----
def selftest():
    ok = True
    def chk(c, m):
        nonlocal ok; ok &= bool(c); print(("  PASS " if c else "  FAIL ") + m)
    L = [4.0, 8.0]
    mk = lambda q, v, cap, pred, gs, ge, dur=30.0: {"qid": q, "video": v, "caption": cap, "pred": pred,
                                                    "gsA": gs, "geA": ge, "dur": dur}

    # (1) unparsed rows count as IoU 0 and are NOT dropped
    rows = [mk(0, "a.mp4", "x", [0, 10], 0, 10), mk(1, "a.mp4", "y", [0, 5], 0, 10),
            mk(2, "b.mp4", "z", None, 0, 10), mk(3, "b.mp4", "w", [10, 0], 0, 10)]
    c = cell([r["pred"] for r in rows], [gt_of(r) for r in rows], [r["dur"] for r in rows],
             [vid_stem(r["video"]) for r in rows], L, "clip", 50, 0)
    chk(c["n"] == 4 and abs(c["mIoU"] - (1 + 0.5 + 0 + 1) / 4) < 1e-12 and abs(c["parse_rate"] - 0.75) < 1e-12,
        "unparsed row kept and scored 0 (n=4, mIoU exact, parse column separate)")

    # (2) R@0.5* == R@0.5 when the prediction already has the GT length (length control is a no-op)
    idr = [mk(i, "v%d" % (i % 3), "c%d" % i, [3 + i * 0.1, 8 + i * 0.1], 2.0, 7.0) for i in range(12)]
    c2 = cell([r["pred"] for r in idr], [gt_of(r) for r in idr], [r["dur"] for r in idr],
              [vid_stem(r["video"]) for r in idr], L, "clip", 30, 0)
    chk(abs(c2["R@0.5_star_DIAGNOSTIC"] - c2["R@0.5"]) < 1e-12 and abs(c2["mIoU_star_DIAGNOSTIC"] - c2["mIoU"]) < 1e-12,
        "identity case: pred length == GT length -> R@0.5* == R@0.5 exactly")

    # (3) pred := GT  ->  R@0.5* == 1 (and R@0.5 == 1)
    gtr = [mk(i, "v%d" % (i % 3), "c%d" % i, [2.0 + i, 7.0 + i], 2.0 + i, 7.0 + i) for i in range(9)]
    c3 = cell([r["pred"] for r in gtr], [gt_of(r) for r in gtr], [r["dur"] for r in gtr],
              [vid_stem(r["video"]) for r in gtr], L, "clip", 30, 0)
    chk(c3["R@0.5_star_DIAGNOSTIC"] == 1.0 and c3["R@0.5"] == 1.0, "pred := GT -> R@0.5* = 1")
    bad = [mk(0, "v0", "c", [0, 1], 0, 1)]; bad[0]["geA"] = None
    try:
        gt_of(bad[0]); chk(False, "missing GT length raises (never a silent n=0)")
    except ValueError:
        chk(True, "missing GT length raises (never a silent n=0)")
    import tempfile
    tmp = tempfile.mkdtemp(); fbad = os.path.join(tmp, "bad.json")
    json.dump([{"qid": 0, "video": "a.mp4", "caption": "x", "pred": [0, 1], "gsA": 0, "geA": None, "dur": 5}],
              open(fbad, "w"))
    try:
        load_rows([fbad], "sta"); chk(False, "loader rejects a row whose GT bound is None")
    except ValueError:
        chk(True, "loader rejects a row whose GT bound is None")

    # (4) destroying the label mapping collapses the 2x2 to one value (mechanism = two DIFFERENT label sets)
    base = [mk(i, "v%d" % (i % 4), "cap %d" % i, [1.0 + (i % 5), 6.0 + (i % 5)], 2.0, 9.0) for i in range(24)]
    tlab = [dict(r) for r in base]
    for i, r in enumerate(tlab):                         # TL labels: systematically shorter, like chtl
        r["gsA"] = 3.0; r["geA"] = 6.0
    r_real = report(base, tlab, L, "clip", 60, 0, "ws")
    M = r_real["common"]["matrix"]
    chk(r_real["match"]["n_common"] == 24 and M["sta_run|STA_labels"]["R@0.5"] != M["sta_run|TL_labels"]["R@0.5"],
        "positive control: same predictions, two label sets -> the two columns differ")
    same = [dict(r) for r in base]
    r_null = report(base, same, L, "clip", 60, 0, "ws")
    Mn = r_null["common"]["matrix"]
    vals = [Mn[k]["R@0.5"] for k in Mn] + [Mn[k]["mIoU"] for k in Mn]
    chk(len(set(vals[:4])) == 1 and len(set(vals[4:])) == 1,
        "negative control: label mapping removed (TL labels := STA labels) -> all four cells collapse to one value")
    chk(r_null["common"]["pred_identity_sta_run_vs_tl_run"]["frac_identical"] == 1.0
        and r_null["common"]["pred_identity_sta_run_vs_tl_run"]["mean_L1_both_parsed"] == 0.0,
        "bit-identity detector returns 1.000 / L1 0 on a self-pair")

    # (5) permuting the TL labels ACROSS questions must move the TL column (the scorer is question-paired)
    perm = [dict(r) for r in tlab]
    for i, r in enumerate(perm):
        src = base[(i + 7) % len(base)]
        r["gsA"], r["geA"] = src["gsA"] + (i % 3) * 2.0, src["geA"] + (i % 3) * 2.0
    r_perm = report(base, perm, L, "clip", 60, 0, "ws")
    chk(r_perm["common"]["matrix"]["sta_run|TL_labels"]["R@0.5"] != M["sta_run|TL_labels"]["R@0.5"],
        "shuffling the per-question label mapping changes the TL column (pairing is per question, not aggregate)")

    # (6) window oracle: forcing L == the constant GT length reproduces R@0.5*
    cst = [mk(i, "v%d" % (i % 3), "c%d" % i, [1.0 + i * 0.5, 9.0 + i * 0.5], 4.0, 9.0) for i in range(15)]
    c6 = cell([r["pred"] for r in cst], [gt_of(r) for r in cst], [r["dur"] for r in cst],
              [vid_stem(r["video"]) for r in cst], [5.0], "clip", 30, 0)
    chk(abs(c6["oracle_const_L"]["L=5"] - c6["R@0.5_star_DIAGNOSTIC"]) < 1e-12,
        "constant-L oracle with L == GT length equals the GT-length oracle / R@0.5*")

    # (6b) the three boundary conventions agree on an interior window and disagree on one that falls off
    inner = recentre([10.0, 14.0], 8.0, 30.0, "clip")
    chk(inner == recentre([10.0, 14.0], 8.0, 30.0, "shift") == recentre([10.0, 14.0], 8.0, 30.0, "none"),
        "boundary modes agree when the oracle window is interior")
    edge_c = recentre([1.0, 3.0], 8.0, 30.0, "clip"); edge_s = recentre([1.0, 3.0], 8.0, 30.0, "shift")
    chk(edge_c == (0.0, 6.0) and edge_s == (0.0, 8.0) and edge_c != edge_s,
        "boundary modes differ at the video edge: clip shortens the window, shift keeps length L")

    # (7) the join is exact on (video stem, normalised query); duplicates are reported, not silently merged
    a = [mk(0, "v1.mp4", "A  b", [0, 1], 0, 1), mk(1, "v2.mp4", "q", [0, 1], 0, 1)]
    b = [mk(0, "v1", "A b", [0, 1], 0, 1), mk(1, "v3", "q", [0, 1], 0, 1)]
    chk(join_pairs(a, b, "ws")[1]["n_keys"] == 1 and join_pairs(a, b, "raw")[1]["n_keys"] == 0,
        "join: extension stripped, whitespace normalised; raw mode is stricter")
    a2 = a + [mk(2, "v1.mp4", "A b", [0, 2], 0, 1)]
    chk(join_pairs(a2, b, "ws")[1]["n_ambiguous_dropped"] == 1,
        "duplicate (video, query) key dropped AND counted (absence is recorded)")
    print("[selftest]", "ALL PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label"); ap.add_argument("--sta", nargs="*", default=[]); ap.add_argument("--tl", nargs="*", default=[])
    ap.add_argument("--sta-machine", default=None); ap.add_argument("--tl-machine", default=None)
    ap.add_argument("--norm", default="ws", choices=["ws", "lower", "raw", "nopunct"])
    ap.add_argument("--oracle-lens", default="4,6,8,10")
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boundary", default="clip", choices=list(BOUNDARY_MODES),
                    help="how an oracle window that falls off the video is handled: clip (C5 convention, "
                         "default) / shift (C14 convention) / none")
    ap.add_argument("--expect-sta-n", type=int, default=None); ap.add_argument("--expect-tl-n", type=int, default=None)
    ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.sta or not a.tl:
        raise SystemExit("need --sta and --tl")
    lens = [float(x) for x in a.oracle_lens.split(",") if x.strip()]
    sta = load_rows(a.sta, "sta"); tl = load_rows(a.tl, "tl")
    if a.expect_sta_n is not None and len(sta) != a.expect_sta_n:
        raise SystemExit("expected sta n=%d, got %d" % (a.expect_sta_n, len(sta)))
    if a.expect_tl_n is not None and len(tl) != a.expect_tl_n:
        raise SystemExit("expected tl n=%d, got %d" % (a.expect_tl_n, len(tl)))
    sm = a.sta_machine or guess_machine(a.sta); node5 = a.tl_machine or guess_machine(a.tl)
    es, et = env_of(sm), env_of(node5)
    same_env = None if ("unknown" in (es, et)) else (es == et)
    out = {"reader": READER, "reader_sha16": sha16(os.path.abspath(__file__)), "label": a.label,
           "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "params": {"norm": a.norm, "oracle_lens": lens, "B": a.B, "seed": a.seed, "boundary": a.boundary,
                      "bootstrap_cluster": "video"},
           "inputs": {"sta": [{"file": os.path.basename(f), "sha16": sha16(f)} for f in a.sta],
                      "tl": [{"file": os.path.basename(f), "sha16": sha16(f)} for f in a.tl]},
           "machines": {"sta": sm, "tl": node5, "sta_env_class": es, "tl_env_class": et, "same_env": same_env,
                        "env_note": "env classes: node1/node3/node5=B (base products byte-identical), "
                                    "node2=A, node4=C; the bit-identity check is only interpretable when same_env "
                                    "is true"},
           "caveats": ["R@0.5* (== GT-length oracle) consumes the ground-truth segment length: DIAGNOSTIC, "
                       "never an accuracy number",
                       "centre error is computed on parsed rows only; the parse rate is reported beside it",
                       "unparsed rows count as IoU 0 in every R@t / mIoU / oracle column"]}
    out.update(report(sta, tl, lens, a.boundary, a.B, a.seed, a.norm))
    out["boundary_sensitivity"] = boundary_sensitivity(sta, tl, lens)
    M = out.get("common", {}).get("matrix")
    print("[%s] sta n=%d (%s) tl n=%d (%s) same_env=%s | common=%d (%s)" %
          (a.label, len(sta), sm, len(tl), node5, same_env, out["match"]["n_common"], a.norm))
    if M:
        pi = out["common"]["pred_identity_sta_run_vs_tl_run"]
        print("  pred identity on common: %d/%d identical, mean L1 %s, parse-mismatch %d" %
              (pi["n_identical"], pi["n"], ("%.4f" % pi["mean_L1_both_parsed"]) if pi["mean_L1_both_parsed"] is not None else "n/a", pi["n_parse_mismatch"]))
        for k in ("sta_run|STA_labels", "sta_run|TL_labels", "tl_run|STA_labels", "tl_run|TL_labels"):
            print("  %-22s R@0.5 %.4f  mIoU %.4f  parse %.4f" % (k, M[k]["R@0.5"], M[k]["mIoU"], M[k]["parse_rate"]))
    for k, v in out["full"].items():
        print("  FULL %-22s n=%d R@0.5 %.4f mIoU %.4f parse %.4f | predlen med %.2f gtlen med %.2f ratio %.2f | "
              "oracle %s | R@0.5* %.4f | dcentre %.4f" %
              (k, v["n"], v["R@0.5"], v["mIoU"], v["parse_rate"], v["pred_len_parsed_only"]["median"] or float("nan"),
               v["gt_len"]["median"], v["pred_over_gt_len_median_ratio"] or float("nan"),
               " ".join("%s:%.4f" % (kk, vv) for kk, vv in v["oracle_const_L"].items()),
               v["R@0.5_star_DIAGNOSTIC"], v["centre_err_parsed_only"]["mean"]))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
