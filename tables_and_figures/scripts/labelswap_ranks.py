#!/usr/bin/env python3
"""labelswap_ranks.py -- rank agreement of configurations across the two label sets on the 870 shared questions,
and the spread relative to the constant-window score under each label set (review round 2026-09-24).
Reads the labelswap records of tables/paper_numbers_v2.json; prints and writes tables/labelswap_ranks.json."""
import json, os
from scipy.stats import spearmanr, kendalltau

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, "..", "tables", "paper_numbers_v2.json")
d = json.load(open(REG, encoding="utf-8"))
R = d["records"] if isinstance(d, dict) and "records" in d else d
rows = []
for r in R:
    if r.get("kind") != "labelswap":
        continue
    m = r["matrix_870"]
    rows.append({"label": r["label"], "family": r["family"], "sta": m["sta_run|STA_labels"], "tl": m["sta_run|TL_labels"]})
for x in sorted(rows, key=lambda x: -x["sta"]):
    print(f"{x['label']:28s} {x['family']:18s} STA {x['sta']:.3f}  TL {x['tl']:.3f}")
FIG = json.load(open(os.path.join(HERE, "..", "figures", "data_redesign_v4", "fig2_findings_manifest.json")))
shown = {(round(p["sta"], 6), round(p["tl"], 6)) for p in FIG["points"] if "sta" in p}
rows = [x for x in rows if (round(x["sta"], 6), round(x["tl"], 6)) in shown]   # the configurations drawn in Figure 2b
q7 = [x for x in rows if x["family"] != "TimeLens-8B"]
out = {}
for name, S in (("all", rows), ("7B", q7)):
    a, b = [x["sta"] for x in S], [x["tl"] for x in S]
    rho, p = spearmanr(a, b); tau, pt = kendalltau(a, b)
    out[name] = {"n": len(S), "spearman": rho, "spearman_p": p, "kendall": tau, "kendall_p": pt,
                 "spread_sta": max(a) - min(a), "spread_tl": max(b) - min(b), "mean_sta": sum(a) / len(a), "mean_tl": sum(b) / len(b),
                 "cv_sta": (max(a) - min(a)) / (sum(a) / len(a)), "cv_tl": (max(b) - min(b)) / (sum(b) / len(b))}
    print(name, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in out[name].items()}))
json.dump({"rows": rows, "summary": out}, open(os.path.join(HERE, "..", "tables", "labelswap_ranks.json"), "w"), indent=1)
