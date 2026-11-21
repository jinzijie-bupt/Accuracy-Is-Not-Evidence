# charades_cd.py (11-09): R2 W1 (annotation prior / overfitting the training split), R3 Q1 (generalisation benefit).
# Charades-CD (Yuan et al., 2021) re-splits Charades-STA so that test-ood moment positions differ from the training
# distribution. Its annotation files (github yytzsy/grounding_changing_distribution, results/charades_cd/) contain 742
# test-ood and 210 test-iid moments that are, field for field (video, start, end, sentence), questions of our Charades-STA
# test set; their predictions already exist in the p4nat products (natural video, 3,720 questions). This script reads
# R@0.5 on those subsets for every arm against the base of the same stack. Paired, video-cluster bootstrap, 2,000 draws, seed 0.
# Unparsed predictions count as misses (coverage is ~1 for all arms).
import hashlib, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(os.path.dirname(HERE), 'results')
M = os.path.join(RES, 'node7_merged', 'newarm'); CD = os.path.join(RES, 'charades_cd'); OUT = os.path.join(HERE, 'charades_cd')
os.makedirs(OUT, exist_ok=True); sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
B451 = os.path.join(M, '_base_tf451', 'p4nat_qwenbase_tf451fa2_node7_o0n3720.json')
B5 = os.path.join(M, '_base_tf5', 'p4nat_qwenbase_tf5_node7_o0n3720.json')
ARMS = [('SFT451', B451, 'ref_qwensft', 'tf451fa2'), ('MDPO451', B451, 'ref_qwenmdpo', 'tf451fa2'),
        ('ours_a05_s0', B451, 'ref_qwena05_s0', 'tf451fa2'), ('ours_a05_s2', B451, 'B1a_qwen_a05_s2', 'tf451fa2'),
        ('B3_tsonly', B451, 'B3_tsonly_a05_s0', 'tf451fa2'), ('A2_iougrpo2fps', B451, 'A2_grpo2fps_s0', 'tf451fa2'),
        ('B5_grpo_placebo', B451, 'B5_grpo_placebo_lam001_s0', 'tf451fa2'), ('public_iougrpo16f', B5, 'ref_iougrpo16f', 'tf5'),
        ('B4_sft4800', B5, 'B4_sft_l4800_s0', 'tf5'), ('ours_l4800', B5, 'ref_l4800', 'tf5')]
key = lambda v, s, e, c: (v.replace('.mp4', ''), round(float(s), 1), round(float(e), 1), c.strip().rstrip('.').lower())


def iou(s, e, S, E):
    i = max(0.0, min(e, E) - max(s, S)); u = max(e, E) - min(s, S)
    return i / u if u > 0 else 0.0


def table(path):
    out = {}
    for r in json.load(open(path, encoding='utf-8')):
        pr = r.get('pred'); ok = isinstance(pr, list) and len(pr) == 2 and None not in pr
        out[int(r['qid'])] = (float(ok and iou(float(pr[0]), float(pr[1]), float(r['gsA']), float(r['geA'])) >= 0.5), r['video'])
    assert len(out) == 3720
    return out


base_rows = json.load(open(B451, encoding='utf-8')); qk = {key(r['video'], r['gsA'], r['geA'], r['caption']): int(r['qid']) for r in base_rows}
splits = {}
for sp in ('test_ood', 'test_iid'):
    f = os.path.join(CD, f'charades_{sp}.json'); d = json.load(open(f, encoding='utf-8'))
    q = sorted({qk[key(v, s, e, c)] for v, x in d.items() for (s, e), c in zip(x['timestamps'], x['sentences']) if key(v, s, e, c) in qk})
    splits[sp] = {'file_sha16': sha(f), 'qids': q}
splits['all_sta_test'] = {'file_sha16': None, 'qids': sorted(qk.values())}
for sp, v in splits.items():
    json.dump(v['qids'], open(os.path.join(OUT, f'cohort_{sp}.json'), 'w'))
rng = np.random.default_rng(0); summ = {'splits': {k: {'n': len(v['qids']), 'file_sha16': v['file_sha16']} for k, v in splits.items()},
                                         'bootstrap': {'draws': 2000, 'seed': 0, 'cluster': 'video'}, 'arms': {}}
bases = {}
for lab, bp, arm, tag in ARMS:
    ap = os.path.join(M, arm, f'p4nat_{arm}_{tag}_node7_o0n3720.json')
    bt = bases.setdefault(bp, table(bp)); at = table(ap); row = {'base_sha16': sha(bp), 'arm_sha16': sha(ap)}
    for sp, v in splits.items():
        qs = v['qids']; vids = sorted({bt[q][1] for q in qs}); byv = {u: [] for u in vids}
        for q in qs:
            byv[bt[q][1]].append(at[q][0] - bt[q][0])
        sums = np.array([sum(byv[u]) for u in vids]); cnts = np.array([len(byv[u]) for u in vids])
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(vids), len(vids)); bs.append(sums[i].sum() / cnts[i].sum())
        row[sp] = {'base_R50': float(np.mean([bt[q][0] for q in qs])), 'arm_R50': float(np.mean([at[q][0] for q in qs])),
                   'delta': float(sums.sum() / cnts.sum()), 'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                   'n': len(qs), 'n_videos': len(vids)}
    summ['arms'][lab] = row
    g = lambda d: f"{d['base_R50']:.3f}->{d['arm_R50']:.3f} {100 * d['delta']:+5.1f} [{100 * d['ci95'][0]:+5.1f},{100 * d['ci95'][1]:+5.1f}]"
    print(f"{lab:18s} STA {g(row['all_sta_test'])} | OOD {g(row['test_ood'])} | IID {g(row['test_iid'])}")
o = os.path.join(OUT, 'CHARADES_CD_SUMMARY.json'); json.dump(summ, open(o, 'w', encoding='utf-8'), indent=1)
print({k: v['n'] for k, v in summ['splits'].items()}); print(o, sha(o))
