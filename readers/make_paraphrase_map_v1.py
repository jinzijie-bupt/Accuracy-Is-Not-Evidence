#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_paraphrase_map_v1.py -- from Charades-STA test annotations, take another human-written caption of the same annotated span as a same-event paraphrase partner. 2026-09-11.

Row order matches 14L.load_test_drn exactly (same regex; verified all 3720 rows have videos present, so qid = matched row index).
Partner definition: same video, same (s, e), different text after normalization; among multiple candidates take the lowest word-set Jaccard (most different wording), ties broken by lower row index.
Output JSON: {"meta": {...}, "map": {qid: {caption, partner_caption, partner_kind: "sibling", partner_src_qid, jaccard, n_candidates}}}
Builds the map only for --n dev questions (default first 1000); questions without a partner are left out of map (probe records has_partner=0).
"""
import argparse, json, re, hashlib, collections, datetime, os, sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def load(anno):
    items = []
    for line in open(anno, encoding="utf-8"):
        m = re.match(r"^(\S+)\s+([\d.]+)\s+([\d.]+)##(.*)$", line.strip())
        if not m:
            continue
        items.append({"qid": len(items), "vid": m.group(1), "s": float(m.group(2)), "e": float(m.group(3)),
                      "caption": m.group(4).strip()})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = load(a.anno)
    byk = collections.defaultdict(list)
    for it in items:
        byk[(it["vid"], it["s"], it["e"])].append(it)
    M = {}
    jac = []
    for it in items[:a.n]:
        cands = [c for c in byk[(it["vid"], it["s"], it["e"])]
                 if c["qid"] != it["qid"] and norm(c["caption"]) != norm(it["caption"])]
        if not cands:
            continue
        A = set(norm(it["caption"]).split())
        best = None
        for c in cands:
            Bw = set(norm(c["caption"]).split())
            j = len(A & Bw) / max(1, len(A | Bw))
            key = (j, c["qid"])
            if best is None or key < best[0]:
                best = (key, c)
        j, c = best[0][0], best[1]
        M[str(it["qid"])] = {"caption": it["caption"], "partner_caption": c["caption"], "partner_kind": "sibling",
                             "partner_src_qid": c["qid"], "jaccard": round(j, 4), "n_candidates": len(cands)}
        jac.append(j)
    jac.sort()
    q = lambda p: jac[min(len(jac) - 1, int(p * len(jac)))] if jac else None
    meta = {"utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "anno": os.path.basename(a.anno), "anno_sha16": hashlib.sha256(open(a.anno, "rb").read()).hexdigest()[:16],
            "n_items_total": len(items), "n_dev": a.n, "n_with_partner": len(M),
            "partner_rule": "same video, same (s,e), normalized caption differs; min word-set Jaccard, tie -> smaller qid",
            "jaccard_p10_p50_p90": [q(0.1), q(0.5), q(0.9)],
            "builder_sha16": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]}
    json.dump({"meta": meta, "map": M}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False))
    print(f"-> {a.out} sha16={hashlib.sha256(open(a.out, 'rb').read()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
