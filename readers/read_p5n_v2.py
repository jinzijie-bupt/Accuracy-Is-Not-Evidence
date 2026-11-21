#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""read_p5n_v2.py — PAIRED reader for 12p5n_native_perturb_eval products.

read_p5n_v1.py stays untouched (it is imported here for load_rows / usable, so the question set is defined
identically).  v2 adds everything the design actually supports but v1 did not compute:

  --pair BASE ARM
   * hard pairing assertion first: for every shared qid the two products must agree on
     n_estar / nframes / swap_donor / mask_rand_idx / swap_rand_idx.  Any disagreement is an ERROR --
     without it a "paired" DiD would silently compare two different perturbations.
   * per-question paired DiD on the swap axis and the mask axis
     DiD = [R_arm(rand) - R_arm(evid)] - [R_base(rand) - R_base(evid)], video-cluster bootstrap, two-sided p.
     (v1's "do the two independent CIs overlap" test is systematically conservative on a matched design.)
   * capability-aligned estimator: on the questions BOTH products answer correctly clean, the survival rate
     of the evid condition; delta = base - arm (positive = the arm loses more when its evidence is removed).
   * normalised contrast 1 - R(evid)/R(rand), per product and as a paired arm-base difference.
   * floor: R(evid) - R(black) with CI, and the floor-corrected ratio [R(rand)-R(evid)]/[R(rand)-R(black)];
     a base already at its no-video prior has no headroom left for a difference-type contrast.
   * per condition: parse rate and the three-way split hit / parsed-but-wrong / refused.
   * every contrast in two versions: ALL questions, and the subset where ALL requested conditions parsed in
     BOTH products -- the gap between them is the refusal component.
   * mask - swap difference, both versions.

usage:
  read_p5n_v2.py --pair BASE.json ARM.json --label NAME [--ops none,mask_evid,mask_rand,swap_evid,swap_rand,black]
                 [--B 4000] [--seed 0] [--out OUT.json]
  read_p5n_v2.py --selftest
"""
import argparse, json, hashlib, os, sys, datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read_p5n_v1 as v1                                   # noqa: E402  (v1 is NOT modified)

READER = "read_p5n_v2"
PAIR_KEYS = ("n_estar", "nframes", "swap_donor", "mask_rand_idx", "swap_rand_idx")
AXES = {"swap": ("swap_evid", "swap_rand"), "mask": ("mask_evid", "mask_rand")}


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def two_sided_p(draws):
    d = np.asarray(draws, dtype=float)
    if not d.size:
        return None
    return float(min(1.0, 2.0 * min((d <= 0).mean(), (d >= 0).mean())))


def ci95(draws):
    d = np.asarray(draws, dtype=float)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d.size else [None, None]


def boot_means(arrays, vids, B, seed):
    """resample video clusters ONCE and return every array's mean per draw.
    Every estimator below is a function of condition means, so this keeps the pairing exact and is cheap."""
    idx = {}
    for i, v in enumerate(vids):
        idx.setdefault(v, []).append(i)
    groups = [np.array(g, dtype=int) for g in idx.values()]
    keys = list(arrays); A = {k: np.asarray(arrays[k], dtype=float) for k in keys}
    point = {k: (float(A[k].mean()) if A[k].size else None) for k in keys}
    if not groups or B <= 0 or not A[keys[0]].size:
        return point, {k: np.empty(0) for k in keys}, len(groups)
    rng = np.random.default_rng(seed); G = len(groups)
    draws = {k: np.empty(B) for k in keys}
    for b in range(B):
        pick = rng.integers(0, G, G); ii = np.concatenate([groups[p] for p in pick])
        for k in keys:
            draws[k][b] = A[k][ii].mean()
    return point, draws, G


def est(point_val, draw_vec):
    return {"est": (float(point_val) if point_val is not None and np.isfinite(point_val) else None),
            "ci95": ci95(draw_vec), "p_two_sided": two_sided_p(draw_vec)}


def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(a, dtype=float) / np.asarray(b, dtype=float)
    return r


def assert_pairing(B, A, qs):
    """the design is matched question by question; refuse to pair if any perturbation differs."""
    bad = []
    for q in qs:
        for k in PAIR_KEYS:
            if k in B[q] or k in A[q]:
                if B[q].get(k, "<absent>") != A[q].get(k, "<absent>"):
                    bad.append((q, k)); break
    return bad


def three_way(rows, qs, op):
    hit = np.array([float(rows[q]["%s_hit" % op]) for q in qs])
    par = np.array([float(rows[q]["%s_parsed" % op]) for q in qs])
    return {"hit": hit, "parsed": par, "parsed_wrong": par * (1.0 - hit), "refused": 1.0 - par}


def block(arrays_point, draws, ops, tag):
    """derive every per-product quantity from the (already paired) condition means."""
    P = lambda k: arrays_point[k]
    D = lambda k: draws[k]
    out = {"conditions": {}}
    for o in ops:
        out["conditions"][o] = {
            "R@0.5": est(P("%s.%s.hit" % (tag, o)), D("%s.%s.hit" % (tag, o))),
            "parse_rate": P("%s.%s.parsed" % (tag, o)),
            "split": {"hit": P("%s.%s.hit" % (tag, o)),
                      "parsed_wrong": P("%s.%s.parsed_wrong" % (tag, o)),
                      "refused": P("%s.%s.refused" % (tag, o))}}
    for ax, (ev, rd) in AXES.items():
        if ev not in ops or rd not in ops:
            out[ax] = {"note": "condition absent (n/a, not 0)"}; continue
        pe, pr = "%s.%s.hit" % (tag, ev), "%s.%s.hit" % (tag, rd)
        a = {"contrast": est(P(pr) - P(pe), D(pr) - D(pe)),
             "normalised_contrast": est(1.0 - P(pe) / P(pr) if P(pr) else None,
                                        1.0 - safe_div(D(pe), D(pr)))}
        if "black" in ops:
            pb = "%s.black.hit" % tag
            a["floor_margin_evid_minus_black"] = est(P(pe) - P(pb), D(pe) - D(pb))
            den = P(pr) - P(pb)
            a["floor_corrected_ratio"] = est((P(pr) - P(pe)) / den if den else None,
                                             safe_div(D(pr) - D(pe), D(pr) - D(pb)))
            a["R_black"] = P(pb)
        out[ax] = a
    if all(o in ops for o in ("mask_evid", "mask_rand", "swap_evid", "swap_rand")):
        me, mr = "%s.mask_evid.hit" % tag, "%s.mask_rand.hit" % tag
        se, sr = "%s.swap_evid.hit" % tag, "%s.swap_rand.hit" % tag
        out["mask_minus_swap"] = est((P(mr) - P(me)) - (P(sr) - P(se)),
                                     (D(mr) - D(me)) - (D(sr) - D(se)))
    return out


def paired_block(P, D, ops):
    out = {}
    for ax, (ev, rd) in AXES.items():
        if ev not in ops or rd not in ops:
            out[ax] = {"note": "condition absent (n/a, not 0)"}; continue
        be, br = "base.%s.hit" % ev, "base.%s.hit" % rd
        ae, ar = "arm.%s.hit" % ev, "arm.%s.hit" % rd
        did_p = (P(ar) - P(ae)) - (P(br) - P(be)); did_d = (D(ar) - D(ae)) - (D(br) - D(be))
        nb = 1.0 - P(be) / P(br) if P(br) else None
        na = 1.0 - P(ae) / P(ar) if P(ar) else None
        out[ax] = {"DiD_contrast_arm_minus_base": est(did_p, did_d),
                   "base_contrast": est(P(br) - P(be), D(br) - D(be)),
                   "arm_contrast": est(P(ar) - P(ae), D(ar) - D(ae)),
                   "normalised_contrast_diff": est((na - nb) if (na is not None and nb is not None) else None,
                                                   safe_div(D(ar) - D(ae), D(ar)) - safe_div(D(br) - D(be), D(br)))}
    if all(o in ops for o in ("mask_evid", "mask_rand", "swap_evid", "swap_rand")):
        f = lambda t, g: (g("%s.mask_rand.hit" % t) - g("%s.mask_evid.hit" % t)) - \
                         (g("%s.swap_rand.hit" % t) - g("%s.swap_evid.hit" % t))
        out["mask_minus_swap_diff"] = est(f("arm", P) - f("base", P), f("arm", D) - f("base", D))
    return out


def collect(base, arm, qs, ops, tag_ops=None):
    arrays = {}
    for tag, rows in (("base", base), ("arm", arm)):
        for o in ops:
            tw = three_way(rows, qs, o)
            for k, v in tw.items():
                arrays["%s.%s.%s" % (tag, o, k)] = v
    return arrays


def analyse(base_rows, arm_rows, ops, B, seed):
    qb, qa = set(base_rows), set(arm_rows)
    Ub = v1.usable(base_rows, ops); Ua = v1.usable(arm_rows, ops)
    qs = sorted(set(Ub) & set(Ua))
    info = {"n_base_rows": len(base_rows), "n_arm_rows": len(arm_rows),
            "n_shared_qid": len(qb & qa), "n_usable_base": len(Ub), "n_usable_arm": len(Ua),
            "n_paired": len(qs), "asserted_fields": list(PAIR_KEYS)}
    if not qs:
        return {"pairing": dict(info, note="no paired question (absence recorded, not silently 0)")}
    bad = assert_pairing(Ub, Ua, qs)
    if bad:
        raise SystemExit("PAIRING ASSERTION FAILED on %d question(s); first 5: %s -- refusing to report a "
                         "paired DiD over two different perturbations" % (len(bad), bad[:5]))
    info["pairing_mismatches"] = 0
    vids = [Ub[q]["video"] for q in qs]
    info["n_videos"] = len(set(vids))
    arrays = collect(Ub, Ua, qs, ops)
    pt, dr, G = boot_means(arrays, vids, B, seed)
    P = lambda k: pt[k]; D = lambda k: dr[k]
    out = {"pairing": info,
           "all_questions": {"n": len(qs), "n_videos": G,
                             "base": block(pt, dr, ops, "base"), "arm": block(pt, dr, ops, "arm"),
                             "paired": paired_block(P, D, ops)}}

    # ---- subset where EVERY requested condition parsed in BOTH products (removes the refusal component)
    sub = [q for q in qs if all(Ub[q]["%s_parsed" % o] and Ua[q]["%s_parsed" % o] for o in ops)]
    if sub:
        av = collect(Ub, Ua, sub, ops); vv = [Ub[q]["video"] for q in sub]
        p2, d2, G2 = boot_means(av, vv, B, seed)
        out["all_conditions_parsed_subset"] = {
            "n": len(sub), "n_videos": G2, "n_dropped": len(qs) - len(sub),
            "definition": "every requested condition parsed in BOTH products",
            "base": block(p2, d2, ops, "base"), "arm": block(p2, d2, ops, "arm"),
            "paired": paired_block(lambda k: p2[k], lambda k: d2[k], ops)}
    else:
        out["all_conditions_parsed_subset"] = {"n": 0, "note": "empty subset (recorded as n/a, not 0)"}

    # ---- capability-aligned: questions BOTH products answer correctly on the clean condition
    if "none" in ops:
        cap = [q for q in qs if Ub[q]["none_hit"] and Ua[q]["none_hit"]]
        out["capability_aligned"] = {"n": len(cap),
                                     "definition": "base clean-hit AND arm clean-hit; survival = R(evid) on "
                                                   "that subset; delta = base - arm"}
        if cap:
            cv = [Ub[q]["video"] for q in cap]
            ca = {}
            for ax, (ev, _rd) in AXES.items():
                if ev not in ops:
                    continue
                ca["base.%s" % ev] = np.array([float(Ub[q]["%s_hit" % ev]) for q in cap])
                ca["arm.%s" % ev] = np.array([float(Ua[q]["%s_hit" % ev]) for q in cap])
            if ca:
                p3, d3, G3 = boot_means(ca, cv, B, seed)
                for ax, (ev, _rd) in AXES.items():
                    kb, ka = "base.%s" % ev, "arm.%s" % ev
                    if kb not in p3:
                        out["capability_aligned"][ax] = {"note": "condition absent (n/a, not 0)"}; continue
                    out["capability_aligned"][ax] = {
                        "base_survival": p3[kb], "arm_survival": p3[ka],
                        "delta_base_minus_arm": est(p3[kb] - p3[ka], d3[kb] - d3[ka]), "n_videos": G3}
        else:
            out["capability_aligned"]["note"] = "no question answered correctly by both (n/a, not 0)"
    return out


# ---------------------------------------------------------------- selftest ----
def _row(q, v, vals, n_estar=5, extra=None):
    """vals: {op: (hit, parsed)}"""
    r = {"qid": q, "video": v, "degenerate": 0, "n_estar": n_estar, "nframes": 40,
         "swap_donor": "d%d" % (q % 3), "mask_rand_idx": [1, 2, 3], "swap_rand_idx": [4, 5, 6]}
    for op, (h, p) in vals.items():
        r["%s_hit" % op] = h; r["%s_parsed" % op] = p; r["%s_iou" % op] = float(h)
    if extra:
        r.update(extra)
    return r


def _mkset(spec, ops):
    rows = {}
    for q, (vid, vals) in enumerate(spec):
        rows[q] = _row(q, vid, {o: vals[o] for o in ops})
    return rows


def selftest():
    ok = True
    def chk(c, m):
        nonlocal ok; ok &= bool(c); print(("  PASS " if c else "  FAIL ") + m)
    ops = ["none", "mask_evid", "mask_rand", "swap_evid", "swap_rand", "black"]
    rng = np.random.default_rng(7)
    spec = []
    for i in range(120):
        vid = "v%d" % (i % 20)
        d = {"none": (int(rng.random() < .7), 1), "mask_evid": (int(rng.random() < .35), 1),
             "mask_rand": (int(rng.random() < .6), 1), "swap_evid": (int(rng.random() < .4), 1),
             "swap_rand": (int(rng.random() < .65), 1), "black": (int(rng.random() < .2), 1)}
        spec.append((vid, d))
    base = _mkset(spec, ops); arm = {q: dict(r) for q, r in base.items()}

    # (1) base == arm  ->  DiD exactly 0, CI contains 0, p ~ 1
    r1 = analyse(base, arm, ops, 400, 0)
    for ax in ("swap", "mask"):
        d = r1["all_questions"]["paired"][ax]["DiD_contrast_arm_minus_base"]
        chk(abs(d["est"]) < 1e-12 and d["ci95"][0] <= 0 <= d["ci95"][1] and d["p_two_sided"] >= 0.99,
            "base == arm -> %s DiD = 0, CI contains 0, p ~ 1" % ax)
    ca = r1["capability_aligned"]["swap"]
    chk(abs(ca["delta_base_minus_arm"]["est"]) < 1e-12,
        "base == arm -> capability-aligned delta = 0")

    # (2) inject REFUSALS into the arm's evid conditions on questions it had answered correctly:
    #     the all-question contrast must move, the all-parsed-subset contrast must not.
    arm2 = {q: dict(r) for q, r in base.items()}
    flipped = 0
    for q, r in arm2.items():
        if r["swap_evid_hit"] == 1 and flipped < 25:
            r["swap_evid_hit"] = 0; r["swap_evid_parsed"] = 0
            r["mask_evid_hit"] = 0; r["mask_evid_parsed"] = 0
            flipped += 1
    r2 = analyse(base, arm2, ops, 400, 0)
    full = r2["all_questions"]["paired"]["swap"]["DiD_contrast_arm_minus_base"]["est"]
    subd = r2["all_conditions_parsed_subset"]["paired"]["swap"]["DiD_contrast_arm_minus_base"]["est"]
    chk(flipped == 25 and abs(full) > 1e-6, "refusal-only injection moves the all-question DiD (%.4f)" % full)
    chk(abs(subd) < 1e-12,
        "refusal-only injection leaves the all-conditions-parsed-subset DiD at exactly 0 (refusal component "
        "is separated, not folded into the contrast)")
    chk(r2["all_questions"]["arm"]["conditions"]["swap_evid"]["split"]["refused"] > 0
        and r2["all_questions"]["base"]["conditions"]["swap_evid"]["split"]["refused"] == 0,
        "three-way split reports the injected refusals on the arm only")

    # (3) floor-corrected ratio == 1 exactly when R(evid) == R(black)
    spec3 = []
    for i in range(60):
        h = int(i % 3 == 0)
        spec3.append(("w%d" % (i % 10), {"none": (1, 1), "mask_evid": (h, 1), "mask_rand": (int(i % 2 == 0), 1),
                                         "swap_evid": (h, 1), "swap_rand": (int(i % 2 == 0), 1), "black": (h, 1)}))
    f = _mkset(spec3, ops)
    r3 = analyse(f, {q: dict(r) for q, r in f.items()}, ops, 200, 0)
    fr = r3["all_questions"]["base"]["swap"]["floor_corrected_ratio"]["est"]
    fm = r3["all_questions"]["base"]["swap"]["floor_margin_evid_minus_black"]["est"]
    chk(abs(fr - 1.0) < 1e-12 and abs(fm) < 1e-12,
        "R(evid) == R(black) -> floor margin 0 and floor-corrected ratio exactly 1")

    # (4) the pairing assertion actually fires (remove the mechanism and it must break)
    badarm = {q: dict(r) for q, r in base.items()}; badarm[3]["n_estar"] = badarm[3]["n_estar"] + 1
    try:
        analyse(base, badarm, ops, 50, 0); chk(False, "differing n_estar raises a pairing error")
    except SystemExit:
        chk(True, "differing n_estar raises a pairing error")
    badarm2 = {q: dict(r) for q, r in base.items()}; badarm2[5]["swap_donor"] = "OTHER"
    try:
        analyse(base, badarm2, ops, 50, 0); chk(False, "differing swap_donor raises a pairing error")
    except SystemExit:
        chk(True, "differing swap_donor raises a pairing error")

    # (5) contrast definition matches read_p5n_v1 exactly on the same rows (no silent redefinition)
    s1 = v1.summarize(base, ops, 200, 0)
    chk(abs(s1["swap_contrast"] - r1["all_questions"]["base"]["swap"]["contrast"]["est"]) < 1e-12
        and abs(s1["mask_contrast"] - r1["all_questions"]["base"]["mask"]["contrast"]["est"]) < 1e-12,
        "v2 contrast reproduces read_p5n_v1's contrast bit for bit")

    # (6) an arm that is genuinely more evidence-dependent gives a POSITIVE DiD (direction sanity)
    arm3 = {q: dict(r) for q, r in base.items()}; k = 0
    for q, r in arm3.items():
        if r["swap_evid_hit"] == 1 and k < 20:
            r["swap_evid_hit"] = 0; k += 1                       # still parsed -> not a refusal
    r4 = analyse(base, arm3, ops, 400, 0)
    d4 = r4["all_questions"]["paired"]["swap"]["DiD_contrast_arm_minus_base"]
    dsub = r4["all_conditions_parsed_subset"]["paired"]["swap"]["DiD_contrast_arm_minus_base"]["est"]
    chk(d4["est"] > 0 and d4["p_two_sided"] < 0.05 and abs(dsub - d4["est"]) < 1e-12,
        "parsed-but-wrong injection raises the DiD in BOTH versions (a real behavioural effect survives the "
        "parse-restricted subset)")
    print("[selftest]", "ALL PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, metavar=("BASE", "ARM"))
    ap.add_argument("--ops", default="none,mask_evid,mask_rand,swap_evid,swap_rand,black")
    ap.add_argument("--B", type=int, default=4000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.pair:
        raise SystemExit("need --pair BASE ARM")
    ops = [o for o in a.ops.split(",") if o]
    bfile, afile = a.pair
    brows = v1.load_rows([bfile]); arows = v1.load_rows([afile])
    have_b = set().union(*[set(r) for r in list(brows.values())[:5]])
    have_a = set().union(*[set(r) for r in list(arows.values())[:5]])
    missing = [o for o in ops if ("%s_hit" % o) not in have_b or ("%s_hit" % o) not in have_a]
    if missing:
        raise SystemExit("ops absent from one of the products: %s (absence is an error here, not n=0)" % missing)
    out = {"reader": READER, "reader_sha16": sha16(os.path.abspath(__file__)),
           "imports": {"read_p5n_v1": sha16(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "read_p5n_v1.py"))},
           "label": a.label, "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "params": {"ops": ops, "B": a.B, "seed": a.seed, "bootstrap_cluster": "video",
                      "p": "two-sided bootstrap: 2*min(P(draw<=0), P(draw>=0))"},
           "inputs": {"base": {"file": os.path.basename(bfile), "sha16": sha16(bfile)},
                      "arm": {"file": os.path.basename(afile), "sha16": sha16(afile)}}}
    out.update(analyse(brows, arows, ops, a.B, a.seed))
    AQ = out["all_questions"]; PR = AQ["paired"]
    print("[%s] paired n=%d (%d videos) | base %s | arm %s" %
          (a.label or "", out["pairing"]["n_paired"], out["pairing"].get("n_videos", -1),
           os.path.basename(bfile), os.path.basename(afile)))
    for ax in ("swap", "mask"):
        if "DiD_contrast_arm_minus_base" not in PR.get(ax, {}):
            print("  %-5s n/a" % ax); continue
        d = PR[ax]["DiD_contrast_arm_minus_base"]
        bb = AQ["base"][ax]; aa = AQ["arm"][ax]
        print("  %-5s base %+0.3f -> arm %+0.3f | DiD %+0.3f [%+0.3f,%+0.3f] p=%.4f | floor margin base "
              "%+0.3f%s arm %+0.3f | floor-corr base %s arm %s" %
              (ax, bb["contrast"]["est"], aa["contrast"]["est"], d["est"], d["ci95"][0], d["ci95"][1],
               d["p_two_sided"],
               bb.get("floor_margin_evid_minus_black", {}).get("est", float("nan")),
               "", aa.get("floor_margin_evid_minus_black", {}).get("est", float("nan")),
               ("%.3f" % bb["floor_corrected_ratio"]["est"]) if bb.get("floor_corrected_ratio", {}).get("est") is not None else "n/a",
               ("%.3f" % aa["floor_corrected_ratio"]["est"]) if aa.get("floor_corrected_ratio", {}).get("est") is not None else "n/a"))
    CA = out.get("capability_aligned", {})
    for ax in ("swap", "mask"):
        if isinstance(CA.get(ax), dict) and "delta_base_minus_arm" in CA[ax]:
            e = CA[ax]["delta_base_minus_arm"]
            print("  cap-aligned %-4s n=%d base %.3f arm %.3f delta %+0.3f [%+0.3f,%+0.3f] p=%.4f" %
                  (ax, CA["n"], CA[ax]["base_survival"], CA[ax]["arm_survival"], e["est"], e["ci95"][0],
                   e["ci95"][1], e["p_two_sided"]))
    S = out.get("all_conditions_parsed_subset", {})
    if S.get("n"):
        g = lambda blk, ax: (blk.get(ax) or {}).get("DiD_contrast_arm_minus_base", {}).get("est")
        f = lambda x: "n/a" if x is None else "%+0.3f" % x
        print("  all-parsed subset n=%d (dropped %d): swap DiD %s mask DiD %s | mask-swap diff full %s subset %s" %
              (S["n"], S["n_dropped"], f(g(S["paired"], "swap")), f(g(S["paired"], "mask")),
               f(PR.get("mask_minus_swap_diff", {}).get("est")),
               f(S["paired"].get("mask_minus_swap_diff", {}).get("est"))))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
