# Family-neutral true-negative table (11-07). The registered reader's abstention taxonomy is Qwen-centric (RA-RFT refuses
# as "<answer>...no such activity...</answer><correction>...</correction>", which lands in 'other'), so this table uses
# the rule a6/a6b_pair_v1.py already uses: NO INTERVAL = rejection. Per arm, on the reader's paired qids (0..999):
#   clean R@0.5 | clean no-interval (= false rejection, every clean query is a true positive)
#   foreign no-interval (= correct rejection) | foreign answered (= hallucinated localization)
#   mixed50 = 0.5 * clean R@0.5 + 0.5 * foreign no-interval  (balanced positive/negative set: accuracy that also credits correct rejections)
# with a video-cluster bootstrap CI on mixed50 (B=2000, seed 0) computed from the per-row products.
import json, os, glob, hashlib, random, collections
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, 'read', 'TNEG_SUMMARY.json'), encoding='utf-8'))
P = os.path.normpath(os.path.join(HERE, '..', '..', 'results', 'node7_pull'))
M = os.path.normpath(os.path.join(HERE, '..', '..', 'results', 'node7_merged', 'tneg_clean'))


def iou(p, s, e):
    if p is None:
        return 0.0
    inter = max(0, min(p[1], e) - max(p[0], s)); uni = max(p[1], e) - min(p[0], s)
    return inter / uni if uni > 0 else 0.0


out = {'source_summary_sha16': hashlib.sha256(open(os.path.join(HERE, 'read', 'TNEG_SUMMARY.json'), 'rb').read()).hexdigest()[:16],
       'rule': 'no interval = rejection (a6b_pair_v1 rule)', 'arms': {}}
for arm, v in S['arms'].items():
    if v.get('status') != 'OK':
        continue
    clean = {r['qid']: r for r in json.load(open(os.path.join(M, f'{arm}_clean_q0-1394.json'), encoding='utf-8'))}
    ff = [f for f in glob.glob(os.path.join(P, 'tneg', arm, 'p4nat_*foreignq_s0_*_o0n1000.json'))][0]
    foreign = {r['qid']: r for r in json.load(open(ff, encoding='utf-8'))}
    q = sorted(set(clean) & set(foreign))
    rows = [(clean[k]['video'], float(iou(clean[k]['pred'], clean[k]['gsA'], clean[k]['geA']) >= .5),
             float(clean[k]['pred'] is None), float(foreign[k]['pred'] is None)) for k in q]
    n = len(rows)
    r50 = sum(r[1] for r in rows) / n; crej = sum(r[2] for r in rows) / n; frej = sum(r[3] for r in rows) / n
    mixed = .5 * r50 + .5 * frej
    byv = collections.defaultdict(list)
    for r in rows:
        byv[r[0]].append(r)
    vids = sorted(byv); rng = random.Random(0); bs = []
    for _ in range(2000):
        smp = [x for _ in vids for x in byv[rng.choice(vids)]]
        bs.append(.5 * sum(x[1] for x in smp) / len(smp) + .5 * sum(x[3] for x in smp) / len(smp))
    bs.sort()
    out['arms'][arm] = {'n': n, 'n_videos': len(vids), 'clean_R50': r50, 'clean_no_interval': crej,
                        'foreign_no_interval': frej, 'foreign_answered': 1 - frej, 'mixed50': mixed,
                        'mixed50_ci95': [bs[50], bs[1949]], 'foreign_sha16': hashlib.sha256(open(ff, 'rb').read()).hexdigest()[:16]}
    print(f"{arm:26s} n={n} clean R50 {r50:.3f} false-rej {crej:.3f} | TN reject {frej:.3f} | mixed50 {mixed:.3f} [{bs[50]:.3f},{bs[1949]:.3f}]")
p = os.path.join(HERE, 'read', 'TNEG_TABLE.json')
json.dump(out, open(p, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(p, hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16])
