# -*- coding: utf-8 -*-
"""A5: selectivity to mismatched (foreign) queries.  Source list = the ten S4 (tneg) products in
results/judgment_20260921/tneg; their clean/foreign raw files are re-read here.  Per model, on the paired
qids 0..999: clean abstention rate, foreign abstention rate, J = P(abstain|foreign) - P(abstain|clean)
(for a binary abstain/answer score, AUROC = (1+J)/2), and the same with the strict refusal taxonomy
(literal 'There are none.' + explicit inability statements, from read_p4nat_selective_v1).
CIs: video-cluster bootstrap over the 1000 paired qids, B=2000, seed 4."""
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..", "..")).replace("\\", "/")
import os, sys, glob, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *

SEL = _SUPP + "/readers/read_p4nat_selective_v1.py"
spec = importlib.util.spec_from_file_location("sel", SEL); sel = importlib.util.module_from_spec(spec); spec.loader.exec_module(sel)
TN = sorted(glob.glob(RES + "/judgment_20260921/tneg/tneg_*.json"))
inputs = [SEL] + TN
res = []


def strict_abst(r):
    if r.get("pred"):
        return 0
    b = sel.classify_abstain(r.get("raw") or "")[1]
    return int(b in ("literal_there_are_none", "nl_refusal"))


for t in TN:
    T = load(t)
    paths = {i["role"]: _SUPP + "/" + i["path"] for i in T["inputs"]}
    inputs += list(paths.values())
    C = {int(r["qid"]): r for r in load(paths["clean"])}
    F = {int(r["qid"]): r for r in load(paths["foreign"])}
    qs = sorted(q for q in F if q in C)
    vids = [C[q]["video"] for q in qs]
    assert all(F[q]["video"] == C[q]["video"] for q in qs)
    ca = np.array([0 if C[q].get("pred") else 1 for q in qs]); fa = np.array([0 if F[q].get("pred") else 1 for q in qs])
    cs = np.array([strict_abst(C[q]) for q in qs]); fs = np.array([strict_abst(F[q]) for q in qs])
    bt = Boot(vids, B=B_DEFAULT, seed=4)
    J = float(fa.mean() - ca.mean()); Js = float(fs.mean() - cs.mean())
    row = {"label": T["label"], "tneg_product": t, "clean": paths["clean"], "foreign": paths["foreign"], "n": len(qs),
           "model": T.get("foreign_prov", {}).get("model"), "adapter": T.get("foreign_prov", {}).get("adapter"),
           "text_first": T.get("foreign_prov", {}).get("text_first"),
           "clean_abstain": float(ca.mean()), "foreign_abstain": float(fa.mean()),
           "J_unparsed": J, "J_unparsed_ci95": bt.ci(lambda ix: float(fa[ix].mean() - ca[ix].mean())),
           "AUROC_unparsed": (1 + J) / 2,
           "clean_abstain_strict": float(cs.mean()), "foreign_abstain_strict": float(fs.mean()),
           "J_strict": Js, "J_strict_ci95": bt.ci(lambda ix: float(fs[ix].mean() - cs[ix].mean())),
           "AUROC_strict": (1 + Js) / 2,
           "tneg_reader_delta_unparsed": T["delta"]["unparsed"]["point"],
           "matches_tneg_reader": abs(T["delta"]["unparsed"]["point"] - J) < 1e-9}
    row["_vecs"] = (qs, vids, ca, fa)
    res.append(row)

# paired Delta J (arm - base) within family and environment, same qids
PAIRS = [("tneg_a05_node2", "tneg_qwenbase_node2"), ("tneg_a05_b003_node2", "tneg_qwenbase_node2"), ("tneg_sft_node2", "tneg_qwenbase_node2"),
         ("tneg_tz_a05_b003_node3", "tneg_tzbase_node3"), ("tneg_tz_a10_node3", "tneg_tzbase_node3")]
by = {r["label"]: r for r in res}
deltas = []
for a, b in PAIRS:
    qa, va, caa, faa = by[a]["_vecs"]; qb, vb, cab, fab = by[b]["_vecs"]
    assert qa == qb and va == vb
    bt = Boot(va, B=B_DEFAULT, seed=5)
    d = float((faa.mean() - caa.mean()) - (fab.mean() - cab.mean()))
    deltas.append({"arm": a, "base": b, "delta_J": d,
                   "delta_J_ci95": bt.ci(lambda ix: float((faa[ix].mean() - caa[ix].mean()) - (fab[ix].mean() - cab[ix].mean()))),
                   "delta_clean_abstain": float(caa.mean() - cab.mean()),
                   "delta_clean_abstain_ci95": bt.ci(lambda ix: float(caa[ix].mean() - cab[ix].mean()))})
for r in res:
    r.pop("_vecs")

notes = ["Time-R1 base: only a clean-query file exists (p4nat_tr1base_timer1_default_cleanq_tf5meta_node3_n1000.json); "
         "no foreign-query run -> base-vs-trained selectivity for Time-R1 not computable.",
         "Environments differ across rows (Qwen rows env A/node2, TimeZero/Time-R1 rows env B/node3); compare within a family only.",
         "tzsft_tf row is text-first; its TimeZero base row is video-first."]
out = write_out("a5_foreign_selectivity", __file__, {"models": res, "paired_delta_J": deltas, "notes": notes,
                "bootstrap": {"B": B_DEFAULT, "seed": 4, "cluster": "video"}}, inputs)
for r in res:
    print(f'{r["label"]:26s} n{r["n"]} clean {r["clean_abstain"]:.3f} foreign {r["foreign_abstain"]:.3f} J {f3(r["J_unparsed"])} {fci(r["J_unparsed_ci95"])} '
          f'AUROC {r["AUROC_unparsed"]:.3f} | strict {r["clean_abstain_strict"]:.3f}/{r["foreign_abstain_strict"]:.3f} J {f3(r["J_strict"])} {fci(r["J_strict_ci95"])} match {r["matches_tneg_reader"]} tf {r["text_first"]}')
for d in deltas:
    print("DELTA", d["arm"], "-", d["base"], f3(d["delta_J"]), fci(d["delta_J_ci95"]), "dClean", f3(d["delta_clean_abstain"]), fci(d["delta_clean_abstain_ci95"]))
print(out)
