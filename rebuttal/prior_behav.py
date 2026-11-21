# prior_behav.py (11-09): behavioural side of R2 W1 (annotation prior). On all 3,720 Charades-STA test questions (p4nat
# natural-video products, one per arm, paired with the base of the same stack):
#   (a) R@0.5 on the two prior strata from prior_strata.py (prior_iou >= 0.5 "prior answers", < 0.3 "prior fails") -> does
#       an arm's accuracy gain come from questions the training prior alone could answer?
#   (b) prior pull = mean IoU(prediction, prior window [0, 0.47] x duration) -> does the arm move its predictions towards
#       the prior window?
# Unparsed predictions count as misses in (a) and are excluded from (b). Paired bootstrap over questions, 2,000 draws, seed 0.
import hashlib, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.join(os.path.dirname(HERE), 'results', 'node7_merged', 'newarm')
OUT = os.path.join(HERE, 'prior_strata'); sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
PS = json.load(open(os.path.join(OUT, 'PRIOR_STRATA_SUMMARY.json'), encoding='utf-8'))['prior']; A, B = PS['window_frac']
PIOU_F = os.path.join(OUT, 'prior_iou_test3720.json'); piou = {int(k): v for k, v in json.load(open(PIOU_F)).items()}
p = lambda arm, tag: os.path.join(M, arm, f'p4nat_{arm if not arm.startswith("_") else ("qwenbase")}_{tag}_node7_o0n3720.json')
B451 = os.path.join(M, '_base_tf451', 'p4nat_qwenbase_tf451fa2_node7_o0n3720.json')
B5 = os.path.join(M, '_base_tf5', 'p4nat_qwenbase_tf5_node7_o0n3720.json')
ARMS = [('SFT451', B451, 'ref_qwensft', 'tf451fa2'), ('MDPO451', B451, 'ref_qwenmdpo', 'tf451fa2'),
        ('ours_a05_s0', B451, 'ref_qwena05_s0', 'tf451fa2'), ('ours_a05_s2', B451, 'B1a_qwen_a05_s2', 'tf451fa2'),
        ('B3_tsonly', B451, 'B3_tsonly_a05_s0', 'tf451fa2'), ('A2_iougrpo2fps', B451, 'A2_grpo2fps_s0', 'tf451fa2'),
        ('B5_grpo_placebo', B451, 'B5_grpo_placebo_lam001_s0', 'tf451fa2'), ('public_iougrpo16f', B5, 'ref_iougrpo16f', 'tf5'),
        ('B4_sft4800', B5, 'B4_sft_l4800_s0', 'tf5'), ('ours_l4800', B5, 'ref_l4800', 'tf5')]


def iou(s, e, S, E):
    i = max(0.0, min(e, E) - max(s, S)); u = max(e, E) - min(s, S)
    return i / u if u > 0 else 0.0


def table(path):
    out = {}
    for r in json.load(open(path, encoding='utf-8')):
        pr, d = r.get('pred'), float(r['dur'])
        ok = isinstance(pr, list) and len(pr) == 2 and pr[0] is not None and pr[1] is not None
        hit = ok and iou(float(pr[0]), float(pr[1]), float(r['gsA']), float(r['geA'])) >= 0.5
        pull = iou(float(pr[0]), float(pr[1]), A * d, B * d) if ok else np.nan
        out[int(r['qid'])] = (float(hit), pull)
    assert len(out) == 3720
    return out


rng = np.random.default_rng(0)
strata = {'all': sorted(piou), 'prior_answers': sorted(q for q in piou if piou[q] >= 0.5),
          'prior_fails': sorted(q for q in piou if piou[q] < 0.3)}
IDX = {k: [rng.integers(0, len(v), len(v)) for _ in range(2000)] for k, v in strata.items()}


def paired(a, b, qs, k, col):
    x = np.array([a[q][col] for q in qs]); y = np.array([b[q][col] for q in qs]); m = ~(np.isnan(x) | np.isnan(y))
    dlt = x - y; pt = float(np.nanmean(dlt[m]) if col else dlt.mean())
    bs = [np.nanmean(np.where(m[i], dlt[i], np.nan)) for i in IDX[k]]
    return {'base': float(np.nanmean(y[m])), 'arm': float(np.nanmean(x[m])), 'delta': pt,
            'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], 'n': int(m.sum())}


summ = {'prior_window_frac': [A, B], 'prior_iou_file_sha16': sha(PIOU_F), 'strata_n': {k: len(v) for k, v in strata.items()},
        'bootstrap': {'draws': 2000, 'seed': 0}, 'arms': {}}
bases = {}
for lab, bp, arm, tag in ARMS:
    ap = os.path.join(M, arm, f'p4nat_{arm}_{tag}_node7_o0n3720.json')
    bt = bases.setdefault(bp, table(bp)); at = table(ap)
    row = {'base_sha16': sha(bp), 'arm_sha16': sha(ap)}
    for k, qs in strata.items():
        row[f'R50_{k}'] = paired(at, bt, qs, k, 0)
    row['prior_pull_all'] = paired(at, bt, strata['all'], 'all', 1)
    summ['arms'][lab] = row
    g = lambda d: f"{100 * d['delta']:+5.1f} [{100 * d['ci95'][0]:+5.1f},{100 * d['ci95'][1]:+5.1f}]"
    pp = row['prior_pull_all']
    print(f"{lab:18s} dR50 all {g(row['R50_all'])} | prior_ans {g(row['R50_prior_answers'])} | prior_fail {g(row['R50_prior_fails'])}"
          f" | pull {pp['base']:.3f}->{pp['arm']:.3f} d{pp['delta']:+.3f} [{pp['ci95'][0]:+.3f},{pp['ci95'][1]:+.3f}]")
o = os.path.join(OUT, 'PRIOR_BEHAV_SUMMARY.json'); json.dump(summ, open(o, 'w', encoding='utf-8'), indent=1)
print('strata n', summ['strata_n']); print(o, sha(o))
