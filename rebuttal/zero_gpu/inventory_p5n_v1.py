#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inventory_p5n_v1.py (rebuttal zero-GPU, 2026-11-06)
Locate every 12p5n behavioural-eval product under $LOCAL_ROOT/results, identify it by
content: sha16, row count, the instrument field in the rows, the ops present, and (when a .meta.json sidecar exists)
argv model / adapter / prompt_style / text_first / trainer_sha16.  Duplicate copies are grouped by sha16.
Output: inventory_p5n_v1.json (+ .tsv)."""
import os, json, hashlib, glob, sys
LOCAL_ROOT = os.environ.get("LOCAL_ROOT", ".")  # analysis directory holding the pulled result files (results/, rebuttal/)

ROOT = (LOCAL_ROOT + "/results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inventory_p5n_v1")


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def main():
    hits = []
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d != "_superseded"]
        for fn in fns:
            if not fn.endswith(".json") or fn.endswith(".meta.json"):
                continue
            if not (fn.startswith("p5n")):
                continue
            hits.append(os.path.join(dp, fn).replace("\\", "/"))
    groups = {}
    for p in sorted(hits):
        try:
            rows = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            print("unreadable", p, e); continue
        if not isinstance(rows, list) or not rows or "qid" not in rows[0]:
            continue
        s = sha16(p)
        g = groups.setdefault(s, {"sha16": s, "copies": [], "n_rows": len(rows)})
        g["copies"].append(p)
        insts = sorted({str(r.get("instrument")) for r in rows})
        ops = [o for o in ("none", "mask_evid", "mask_rand", "swap_evid", "swap_rand", "black") if ("%s_hit" % o) in rows[0] or any(("%s_hit" % o) in r for r in rows[:20])]
        g.update({"instrument_in_rows": insts, "ops": ops, "styles": sorted({str(r.get("style")) for r in rows}),
                  "qid_min": min(int(r["qid"]) for r in rows), "qid_max": max(int(r["qid"]) for r in rows),
                  "n_degenerate": sum(1 for r in rows if r.get("degenerate")),
                  "n_decode_error": sum(1 for r in rows if "decode_error" in r)})
        mp = p + ".meta.json"
        if os.path.exists(mp):
            m = json.load(open(mp, encoding="utf-8"))
            a = m.get("args", {})
            g.setdefault("meta", []).append({"meta_file": mp, "meta_sha16": sha16(mp), "instrument": m.get("instrument"),
                                             "trainer_sha16": m.get("trainer_sha16"), "model": a.get("model"),
                                             "adapter": a.get("adapter"), "prompt_style": a.get("prompt_style"),
                                             "text_first": a.get("text_first"), "ops": a.get("ops"), "n": a.get("n"),
                                             "offset": a.get("offset"), "seed": a.get("seed"), "out": a.get("out"),
                                             "hits": m.get("hits")})
    res = sorted(groups.values(), key=lambda g: g["copies"][0])
    json.dump({"script": os.path.basename(__file__), "script_sha16": sha16(os.path.abspath(__file__)), "root": ROOT,
               "products": res}, open(OUT + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    with open(OUT + ".tsv", "w", encoding="utf-8") as f:
        f.write("sha16\tn_rows\tinstrument\tstyles\tadapter\tmodel\tout\tcopies\n")
        for g in res:
            m = (g.get("meta") or [{}])[0]
            f.write("\t".join(str(x) for x in [g["sha16"], g["n_rows"], ",".join(g["instrument_in_rows"]), ",".join(g["styles"]),
                                                 m.get("adapter"), m.get("model"), m.get("out"),
                                                 " | ".join(os.path.relpath(c, ROOT) for c in g["copies"])]) + "\n")
    for g in res:
        m = (g.get("meta") or [{}])[0]
        print(g["sha16"], g["n_rows"], g["instrument_in_rows"], g["styles"], "adapter=", m.get("adapter"), "model=", os.path.basename(str(m.get("model"))), "\n    ", [os.path.relpath(c, ROOT) for c in g["copies"]])


if __name__ == "__main__":
    main()
