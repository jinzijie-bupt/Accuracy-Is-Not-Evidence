# paired_r50.py (11-11): paired ΔR@0.5 (arm − base) on the full Charades-STA test (3,720 q) with a video-cluster bootstrap
# (10,000 resamples of videos, seed 20261111). Hit = IoU(pred, (gsA, geA)) >= 0.5, iou_1d copied from analysis/read_p4nat_v1.py.
# usage: python paired_r50.py LABEL BASE.json ARM.json [LABEL BASE ARM ...]  -> PAIRED_R50.json next to this script (appends)
import json, sys, os, hashlib, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'analysis'))
from read_p4nat_v1 import iou_1d
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, 'PAIRED_R50.json')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
a = sys.argv[1:]; assert len(a) % 3 == 0
for i in range(0, len(a), 3):
    lab, bp, ap = a[i:i + 3]
    B = {r['qid']: r for r in json.load(open(bp))}; A = {r['qid']: r for r in json.load(open(ap))}
    assert set(B) == set(A) and len(B) == 3720, (len(B), len(A))
    q = sorted(B); assert all(B[k]['video'] == A[k]['video'] and B[k]['gsA'] == A[k]['gsA'] for k in q)
    hb = np.array([iou_1d(B[k].get('pred'), (B[k]['gsA'], B[k]['geA'])) >= 0.5 for k in q], float)
    ha = np.array([iou_1d(A[k].get('pred'), (A[k]['gsA'], A[k]['geA'])) >= 0.5 for k in q], float)
    vids = sorted({B[k]['video'] for k in q}); vi = {v: j for j, v in enumerate(vids)}
    g = np.array([vi[B[k]['video']] for k in q]); d = ha - hb
    sd = np.bincount(g, d, len(vids)); sn = np.bincount(g, None, len(vids))
    rng = np.random.default_rng(20261111); bs = []
    for _ in range(10000):
        idx = rng.integers(0, len(vids), len(vids)); bs.append(sd[idx].sum() / sn[idx].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    res[lab] = {'base': bp, 'base_sha16': sha(bp), 'arm': ap, 'arm_sha16': sha(ap), 'n': len(q), 'n_videos': len(vids),
                'R50_base': hb.mean(), 'R50_arm': ha.mean(), 'delta_pts': 100 * d.mean(), 'ci95_pts': [100 * lo, 100 * hi]}
    print(lab, 'R50 %.3f -> %.3f  delta %+.1f [%+.1f, %+.1f] pts' % (hb.mean(), ha.mean(), 100 * d.mean(), 100 * lo, 100 * hi))
json.dump(res, open(OUT, 'w'), indent=1)
print('->', OUT, sha(OUT))
