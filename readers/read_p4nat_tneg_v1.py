#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""read_p4nat_tneg_v1.py -- registered reader for the TRUE-NEGATIVE QUERY contrast.

Input: one pair of native-protocol grounding products from the SAME machine and the SAME arm --
  --clean    a --query-op none product (or any frozen r3/r5 product of that arm, which is the same thing)
  --foreign  a 12p4n_..._r8.py --query-op foreign product (every item asked with a caption from another video)
Rows are paired on the qid INTERSECTION (the clean product is usually the n=3720 full run, the foreign one a
slice), so nothing is compared across machines and nothing is compared across different questions.

Reported per side and as a PAIRED difference (foreign - clean):
  parse_rate; abstention rate, strict = the verbatim "There are none." bucket, wide = strict + natural-language
  refusal (both come from the frozen taxonomy in read_p4nat_selective_v1.classify_abstain, whose sha16 is
  recorded); R@0.5 against the ORIGINAL GT window; conditional R@0.5 on the answered subset.
Deltas carry a video-cluster bootstrap CI (B=2000, seed 0) and a two-sided bootstrap p.

Guards (each one has a negative control in --selftest): zero paired rows is an error, never a zero; the foreign
file must actually declare query_op=foreign and must actually carry partner videos different from the row video;
the video column must agree per qid; every unparsed row lands in exactly one bucket, and buckets that do not
occur are written as 0 with the bucket named (absence is a value, not a missing key).

usage:
  read_p4nat_tneg_v1.py --label L --clean C.json --foreign F.json [--B 2000] [--seed 0] --out OUT.json
  read_p4nat_tneg_v1.py --selftest
"""
import argparse, json, hashlib, os, sys, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_SEL = os.path.join(HERE, "read_p4nat_selective_v1.py")
_V1 = os.path.join(HERE, "read_p4nat_v1.py")
BUCKETS = ("literal_there_are_none", "nl_refusal", "empty_output", "empty_answer_tag", "gibberish", "other")
WIDE_BUCKETS = ("literal_there_are_none", "nl_refusal")


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def _load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SEL = _load_mod(_SEL, "read_p4nat_selective_v1_frozen")
classify_abstain = SEL.classify_abstain
iou_1d = SEL.iou_1d


def load_rows(path):
    d = json.load(open(path, encoding="utf-8"))
    rs = d["rows"] if isinstance(d, dict) and "rows" in d else d
    if not isinstance(rs, list) or not rs:
        raise SystemExit("%s: no rows" % path)
    by = {}
    for r in rs:
        for k in ("qid", "video", "gsA", "geA"):
            if k not in r or r[k] is None:
                raise SystemExit("%s: row missing %s: %s" % (path, k, str(r)[:120]))
        q = int(r["qid"])
        if q in by:
            raise SystemExit("%s: duplicate qid %d" % (path, q))
        by[q] = r
    return by


def check_foreign(rows, path):
    """the foreign product must SAY it is foreign and must SHOW it (partner video != row video)."""
    ops = sorted({str(r.get("query_op")) for r in rows})
    if ops != ["foreign"]:
        raise SystemExit("%s: query_op=%s, expected exactly ['foreign'] -- the mechanism was not applied" % (path, ops))
    have_partner = [r for r in rows if r.get("partner_video") is not None]
    if not have_partner:
        raise SystemExit("%s: no row carries partner_video (absent, not zero): cannot verify the partner is foreign" % path)
    same = sum(1 for r in have_partner if str(r["partner_video"]) == str(r["video"]))
    if same:
        raise SystemExit("%s: %d rows have a partner from their OWN video" % (path, same))
    hq = [r for r in rows if r.get("used_query") is not None and r.get("orig_query") is not None]
    changed = sum(1 for r in hq if str(r["used_query"]) != str(r["orig_query"]))
    frac = changed / len(hq) if hq else None
    if frac is not None and frac < 0.95:
        raise SystemExit("%s: only %.3f of rows actually changed the query text" % (path, frac))
    return {"n_rows_with_partner": len(have_partner), "frac_query_text_changed": frac,
            "reason_if_null": None if frac is not None else "rows carry no used_query/orig_query field"}


def side(rows):
    """vectors for one side of the pair, in the given (shared) row order"""
    iou = np.array([iou_1d(r.get("pred"), (r["gsA"], r["geA"])) for r in rows], dtype=float)
    ans = np.array([1 if r.get("pred") else 0 for r in rows], dtype=bool)
    strict = np.zeros(len(rows), dtype=bool)
    wide = np.zeros(len(rows), dtype=bool)
    buckets = {b: 0 for b in BUCKETS}
    for i, r in enumerate(rows):
        if r.get("pred"):
            continue  # abstention is only counted where there is no parsed window
        bw, bs = classify_abstain(r.get("raw") or "")
        b = bs if bs in buckets else "other"
        buckets[b] = buckets.get(b, 0) + 1
        strict[i] = (bs == "literal_there_are_none")
        wide[i] = (bw in WIDE_BUCKETS) or (bs in WIDE_BUCKETS)
    return {"iou": iou, "ans": ans, "strict": strict, "wide": wide, "buckets": buckets}


def side_summary(v, rows):
    n = len(rows)
    na = int(v["ans"].sum())
    nab = n - na
    return {"n": n, "parse_rate": float(v["ans"].mean()), "unparsed_rate": float(1 - v["ans"].mean()),
            "abstain_strict_rate": float(v["strict"].mean()), "abstain_wide_rate": float(v["wide"].mean()),
            "R@0.5": float((v["iou"] >= 0.5).mean()), "mIoU": float(v["iou"].mean()),
            "R@0.5_answered": float((v["iou"][v["ans"]] >= 0.5).mean()) if na else None,
            "R@0.5_answered_reason": None if na else "coverage 0: no answered row (absent, not zero)",
            "n_answered": na, "n_unparsed": nab,
            "buckets_strict_taxonomy": {b: v["buckets"].get(b, 0) for b in BUCKETS},
            "buckets_sum": int(sum(v["buckets"].get(b, 0) for b in BUCKETS))}


def clusters(vids):
    order = sorted(set(vids))
    idx = {v: i for i, v in enumerate(order)}
    g = [[] for _ in order]
    for i, v in enumerate(vids):
        g[idx[v]].append(i)
    return [np.array(x) for x in g]


def boot_delta(dvec, groups, B, seed):
    """cluster bootstrap of the paired mean difference; a degenerate (all-zero) vector gives CI [0,0], p=1"""
    rng = np.random.default_rng(seed)
    G = len(groups)
    draws = np.empty(B, dtype=float)
    for b in range(B):
        idx = np.concatenate([groups[k] for k in rng.integers(0, G, G)])
        draws[b] = dvec[idx].mean()
    lo = float((draws <= 0).mean()); hi = float((draws >= 0).mean())
    return {"point": float(dvec.mean()), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "p_two_sided": float(min(1.0, 2.0 * min(lo, hi))), "B": B, "seed": seed, "cluster": "video"}


def read_pair(clean_by, foreign_by, B=2000, seed=0, clean_path="", foreign_path=""):
    qs = sorted(set(clean_by) & set(foreign_by))
    if not qs:
        raise SystemExit("no paired qid between clean (n=%d) and foreign (n=%d): 0 pairs is an error, not a zero"
                         % (len(clean_by), len(foreign_by)))
    crows = [clean_by[q] for q in qs]
    frows = [foreign_by[q] for q in qs]
    bad = [q for q, c, f in zip(qs, crows, frows) if str(c["video"]) != str(f["video"])]
    if bad:
        raise SystemExit("video column differs on %d paired qids (first %s)" % (len(bad), bad[:5]))
    bad = [q for q, c, f in zip(qs, crows, frows) if (float(c["gsA"]), float(c["geA"])) != (float(f["gsA"]), float(f["geA"]))]
    if bad:
        raise SystemExit("GT window differs on %d paired qids (first %s)" % (len(bad), bad[:5]))
    cops = sorted({str(r.get("query_op")) for r in crows})
    if not set(cops) <= {"none", "None"}:
        raise SystemExit("clean file declares query_op=%s, expected none/absent" % cops)
    fchk = check_foreign(frows, foreign_path or "<foreign>")
    C, F = side(crows), side(frows)
    g = clusters([str(r["video"]) for r in crows])
    out = {"n_pairs": len(qs), "qid_min": int(qs[0]), "qid_max": int(qs[-1]), "n_videos": len(g),
           "n_clean_rows": len(clean_by), "n_foreign_rows": len(foreign_by),
           "clean": side_summary(C, crows), "foreign": side_summary(F, frows),
           "foreign_mechanism_check": fchk,
           "clean_query_op_field": cops,
           "delta": {}}
    for key, cv, fv in (("abstain_strict", C["strict"].astype(float), F["strict"].astype(float)),
                        ("abstain_wide", C["wide"].astype(float), F["wide"].astype(float)),
                        ("unparsed", (~C["ans"]).astype(float), (~F["ans"]).astype(float)),
                        ("R@0.5", (C["iou"] >= 0.5).astype(float), (F["iou"] >= 0.5).astype(float)),
                        ("mIoU", C["iou"], F["iou"])):
        out["delta"][key] = boot_delta(fv - cv, g, B, seed)
    return out


# ------------------------------------------------------------------ selftest
def _row(q, v, pred, raw, gs=0.0, ge=10.0, op=None, partner=None):
    r = {"qid": q, "video": v, "pred": pred, "raw": raw, "gsA": gs, "geA": ge}
    if op is not None:
        r["query_op"] = op
        r["orig_query"] = "own caption %d" % q
        r["used_query"] = ("foreign caption %d" % q) if op == "foreign" else r["orig_query"]
        r["partner_video"] = partner
    return r


def selftest():
    ok = True

    def chk(c, m):
        nonlocal ok
        ok &= bool(c)
        print(("  PASS " if c else "  FAIL ") + m)

    clean = {q: _row(q, "v%d" % (q // 2), [0.0, 10.0], "<answer> 0.00 to 10.00 </answer>", op="none") for q in range(4)}
    frows = [_row(0, "v0", None, "There are none.", op="foreign", partner="vX"),
             _row(1, "v0", None, "I cannot determine the time period for this event.", op="foreign", partner="vX"),
             _row(2, "v1", [0.0, 10.0], "<answer> 0.00 to 10.00 </answer>", op="foreign", partner="vX"),
             _row(3, "v1", [0.0, 5.0], "<answer> 0.00 to 5.00 </answer>", op="foreign", partner="vX")]
    foreign = {r["qid"]: r for r in frows}
    p = read_pair(dict(clean), dict(foreign), B=200, seed=0)
    chk(p["n_pairs"] == 4 and p["n_videos"] == 2, "positive: 4 paired qids over 2 video clusters")
    chk(abs(p["foreign"]["abstain_strict_rate"] - 0.25) < 1e-12 and abs(p["foreign"]["abstain_wide_rate"] - 0.5) < 1e-12,
        "positive: strict counts only the verbatim string; wide adds the natural-language refusal")
    chk(abs(p["delta"]["abstain_strict"]["point"] - 0.25) < 1e-12 and abs(p["delta"]["abstain_wide"]["point"] - 0.5) < 1e-12,
        "positive: paired delta = foreign - clean, exact on known rows")
    chk(abs(p["delta"]["R@0.5"]["point"] + 0.5) < 1e-12, "positive: delta R@0.5 = -0.5 (two rows lost the window)")
    chk(p["foreign"]["buckets_sum"] == p["foreign"]["n_unparsed"] and set(p["foreign"]["buckets_strict_taxonomy"]) == set(BUCKETS),
        "positive: every unparsed row lands in exactly one named bucket; absent buckets are written as 0")
    same = read_pair(dict(clean), {q: dict(r, query_op="foreign", partner_video="vX", used_query="x", orig_query="y") for q, r in clean.items()}, B=200, seed=0)
    d = same["delta"]
    chk(all(abs(d[k]["point"]) < 1e-15 and d[k]["ci95"] == [0.0, 0.0] and d[k]["p_two_sided"] == 1.0 for k in d),
        "negative control: clean == foreign gives delta exactly 0 with a degenerate CI and p=1")
    for kw, mut, name in (
            ("no pair", lambda: read_pair(dict(clean), {q + 100: dict(r) for q, r in foreign.items()}, B=50), "zero paired qids is an error, not a zero"),
            ("video", lambda: read_pair(dict(clean), {q: (dict(r, video="zz") if q == 0 else r) for q, r in foreign.items()}, B=50), "video mismatch on a paired qid is refused"),
            ("op", lambda: read_pair(dict(clean), {q: dict(r, query_op="none") for q, r in foreign.items()}, B=50), "a foreign file that declares query_op=none is refused"),
            ("partner", lambda: read_pair(dict(clean), {q: dict(r, partner_video=r["video"]) for q, r in foreign.items()}, B=50), "a partner taken from the row's own video is refused"),
            ("gt", lambda: read_pair(dict(clean), {q: (dict(r, geA=99.0) if q == 1 else r) for q, r in foreign.items()}, B=50), "a changed GT window is refused")):
        try:
            mut()
            chk(False, "negative: " + name)
        except SystemExit:
            chk(True, "negative: " + name)
    odd = {q: dict(r) for q, r in foreign.items()}
    odd[2] = _row(2, "v1", None, "addCriterion addCriterion addCriterion addCriterion", op="foreign", partner="vX")
    p2 = read_pair(dict(clean), odd, B=50, seed=0)
    chk(p2["foreign"]["n_unparsed"] == 3 and p2["foreign"]["abstain_strict_rate"] == 0.25 and p2["foreign"]["buckets_sum"] == 3,
        "negative: an unparsed row that is NOT a refusal stays out of the abstention rate but is still counted (gibberish/other)")
    ansraw = {q: dict(r) for q, r in foreign.items()}
    ansraw[3] = _row(3, "v1", [0.0, 5.0], "there are none of those, but <answer> 0.00 to 5.00 </answer>", op="foreign", partner="vX")
    p3 = read_pair(dict(clean), ansraw, B=50, seed=0)
    chk(p3["foreign"]["abstain_strict_rate"] == 0.25, "negative: a row that still produced a window is never counted as an abstention")
    print("[selftest]", "ALL PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--clean")
    ap.add_argument("--foreign")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.clean or not a.foreign:
        ap.error("--clean and --foreign are required unless --selftest")
    m = read_pair(load_rows(a.clean), load_rows(a.foreign), a.B, a.seed, a.clean, a.foreign)
    m["label"] = a.label
    m["inputs"] = [{"role": "clean", "file": os.path.basename(a.clean), "path": a.clean, "sha16": sha16(a.clean)},
                   {"role": "foreign", "file": os.path.basename(a.foreign), "path": a.foreign, "sha16": sha16(a.foreign)}]
    fm = a.foreign + ".meta.json"
    if os.path.exists(fm):
        try:
            m["foreign_prov"] = json.load(open(fm, encoding="utf-8")).get("prov")
        except Exception as e:
            m["foreign_prov"] = {"reason": "meta unreadable: %s" % e}
    else:
        m["foreign_prov"] = {"reason": "no <out>.meta.json beside the foreign product"}
    m["reader"] = "read_p4nat_tneg_v1"
    m["reader_sha16"] = sha16(os.path.abspath(__file__))
    m["imports"] = {"read_p4nat_selective_v1": sha16(_SEL), "read_p4nat_v1": sha16(_V1)}
    print(json.dumps({"label": m["label"], "n_pairs": m["n_pairs"],
                      "clean": {k: m["clean"][k] for k in ("parse_rate", "abstain_strict_rate", "abstain_wide_rate", "R@0.5")},
                      "foreign": {k: m["foreign"][k] for k in ("parse_rate", "abstain_strict_rate", "abstain_wide_rate", "R@0.5")},
                      "delta_abstain_strict": m["delta"]["abstain_strict"], "delta_abstain_wide": m["delta"]["abstain_wide"]},
                     ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(m, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
