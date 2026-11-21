"""a6b_pair_v1.py -- A6b: RA-RFT vs its base Time-R1-7B under RA-RFT's NATIVE prompt (12p4n tf451 r6 3910c888fe96f3ed,
style rarft_native, full 3720, node7 tf451+FA2). Paired by qid; video-cluster bootstrap (B=2000, seed 0) for the differences.
Per model: R@0.3/0.5/0.7, mIoU, no-interval rate (= said 'no relevant segment' or no parsable interval; every Charades-STA
query is a true positive, so every no-interval answer is a false rejection). IoU as in analysis/read_p4nat_v1.iou_1d (GT A).
Writes rebuttal/a6/read/A6b_rarft_vs_tr1base_native.json with input sha16s."""
import glob, hashlib, json, os, sys
import numpy as np
LOCAL_ROOT = os.environ.get("LOCAL_ROOT", ".")  # analysis directory holding the pulled result files (results/, rebuttal/)

sys.path.insert(0, (LOCAL_ROOT + "/analysis"))
from read_p4nat_v1 import iou_1d  # noqa: E402

D = (LOCAL_ROOT + "/results/node7_pull/a6")
OUT = (LOCAL_ROOT + "/rebuttal/a6/read/A6b_rarft_vs_tr1base_native.json")


def load(tag):
    fs = sorted(glob.glob(f"{D}/p4nat_{tag}_rarft_native_tf451fa2_o*n465.json"))
    assert len(fs) == 8, (tag, len(fs))
    rows, shas = {}, {}
    for f in fs:
        b = open(f, "rb").read(); shas[os.path.basename(f)] = hashlib.sha256(b).hexdigest()[:16]
        for r in json.loads(b.decode("utf-8")):
            assert r["style"] == "rarft_native"
            rows[int(r["qid"])] = r
    assert len(rows) == 3720
    return rows, shas


A, sa = load("rarft"); B, sb = load("tr1base")
q = sorted(A); assert q == sorted(B)
for k in q:
    assert A[k]["video"] == B[k]["video"] and (A[k]["gsA"], A[k]["geA"]) == (B[k]["gsA"], B[k]["geA"])
vids = np.array([A[k]["video"] for k in q])


def feats(R):
    iou = np.array([iou_1d(R[k].get("pred"), (R[k]["gsA"], R[k]["geA"])) for k in q])
    nopred = np.array([R[k].get("pred") is None for k in q], float)
    return {"R@0.3": (iou >= 0.3).astype(float), "R@0.5": (iou >= 0.5).astype(float), "R@0.7": (iou >= 0.7).astype(float),
            "mIoU": iou, "no_interval": nopred}


fa, fb = feats(A), feats(B)
uv = np.unique(vids); idx = {v: np.where(vids == v)[0] for v in uv}
rng = np.random.default_rng(0); boots = [np.concatenate([idx[v] for v in rng.choice(uv, len(uv))]) for _ in range(2000)]
res = {}
for m in fa:
    d = fa[m] - fb[m]
    bs = np.array([d[b].mean() for b in boots])
    res[m] = {"rarft": float(fa[m].mean()), "tr1base": float(fb[m].mean()), "diff": float(d.mean()),
              "diff_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
# accuracy among answered questions (both gave an interval)
both = (fa["no_interval"] == 0) & (fb["no_interval"] == 0)
res["R@0.5_both_answered"] = {"n": int(both.sum()), "rarft": float(fa["R@0.5"][both].mean()), "tr1base": float(fb["R@0.5"][both].mean())}
out = {"script_sha16": hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16], "n": len(q), "n_videos": int(len(uv)),
       "instrument": "12p4n_native_acc_eval_tf451_r6 3910c888fe96f3ed rarft_native", "machine": "node7 tf4.51.3 FA2",
       "inputs": {"rarft": sa, "tr1base": sb}, "metrics": res}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
for m, v in res.items():
    print(m, {k: (round(x, 4) if isinstance(x, float) else ([round(y, 4) for y in x] if isinstance(x, list) else x)) for k, x in v.items()})
