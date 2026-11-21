# prior_strata.py (11-09): R2 W1 "the audit cannot separate a model that exploits an annotation prior from one that reasons
# over the sequence". Annotation prior = the single normalised window [a,b] (fraction of video duration) that maximises the
# mean IoU with the Charades-STA TRAIN annotations (grid step 0.01; uses no video and no query). Each test question gets
# prior_iou = IoU(prior window * duration, GT). Strata on indep600: "prior answers" (prior_iou >= 0.5) and "prior fails"
# (prior_iou < 0.3). Every indep600 pair is re-read on each stratum with analysis/read_14Ln_v1.py (unchanged, --fixed-cohort).
# GT/duration per test qid come from the base p4nat product (3,720 rows, gsA/geA/dur).
import csv, hashlib, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
A = os.path.join(ROOT, 'results', 'charades_anno'); M = os.path.join(ROOT, 'results', 'node7_merged')
READER = os.path.join(ROOT, 'analysis', 'read_14Ln_v1.py'); OUT = os.path.join(HERE, 'prior_strata'); os.makedirs(OUT, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
dur = {r['id']: float(r['length']) for r in csv.DictReader(open(os.path.join(A, 'Charades_v1_train.csv'), encoding='utf-8'))}
tr = []
for line in open(os.path.join(A, 'charades_sta_train.txt'), encoding='utf-8'):
    v, s, e = line.split('##')[0].split()
    d = dur.get(v)
    if d and d > 0:
        tr.append((min(float(s), d) / d, min(float(e), d) / d))
tr = np.array(tr); g = np.round(np.arange(0, 1.0001, 0.01), 2)


def miou(a, b, S, E):
    inter = np.clip(np.minimum(b, E) - np.maximum(a, S), 0, None); uni = np.maximum(b, E) - np.minimum(a, S)
    return inter / np.where(uni > 0, uni, 1)


best = max(((miou(a, b, tr[:, 0], tr[:, 1]).mean(), a, b) for a in g for b in g if b > a), key=lambda t: t[0])
prior = {'window_frac': [float(best[1]), float(best[2])], 'train_mIoU': float(best[0]), 'n_train': len(tr),
         'train_R50': float((miou(best[1], best[2], tr[:, 0], tr[:, 1]) >= 0.5).mean())}
base = json.load(open(os.path.join(M, 'newarm', '_base_tf451', 'p4nat_qwenbase_tf451fa2_node7_o0n3720.json'), encoding='utf-8'))
piou = {}
for r in base:
    d = float(r['dur']); piou[int(r['qid'])] = float(miou(best[1] * d, best[2] * d, float(r['gsA']), float(r['geA'])))
qids = [int(q) for q in json.load(open(os.path.join(HERE, 'indep600_qids.json')))]
strata = {'prior_answers': sorted(q for q in qids if piou[q] >= 0.5), 'prior_fails': sorted(q for q in qids if piou[q] < 0.3)}
prior['test_R50_all3720'] = float(np.mean([v >= 0.5 for v in piou.values()]))
prior['indep600_counts'] = {k: len(v) for k, v in strata.items()}
for k, v in strata.items():
    json.dump(v, open(os.path.join(OUT, f'cohort_{k}.json'), 'w'))
json.dump({str(q): piou[q] for q in sorted(piou)}, open(os.path.join(OUT, 'prior_iou_test3720.json'), 'w'))
print('prior', prior)
a1 = lambda c: os.path.join(M, 'a1', c, f'14Ln_{c}_node7_o0n600.json')
na = lambda arm, sfx: os.path.join(M, 'newarm', arm, f'14Ln_indep600_{arm}_{sfx}_node7_o0n600R3.json')
B451, B5 = a1('A1-01_qwen_base_t1d_tf451'), a1('A1-07_qwen_base_t1d_tf5')
PAIRS = [('SFT451', B451, a1('A1-02_qwen_sft_tf451'), '4.51'), ('MDPO451', B451, a1('A1-03_qwen_mdpo_tf451'), '4.51'),
         ('ours_a05_s0', B451, a1('A1-04_qwen_a05_tf451'), '4.51'), ('ours_a05_s2', B451, na('B1a_qwen_a05_s2', 'tf451fa2'), '4.51'),
         ('B3_tsonly', B451, na('B3_tsonly_a05_s0', 'tf451fa2'), '4.51'), ('A2_iougrpo2fps', B451, na('A2_grpo2fps_s0', 'tf451fa2'), '4.51'),
         ('B5_grpo_placebo', B451, na('B5_grpo_placebo_lam001_s0', 'tf451fa2'), '4.51'),
         ('public_iougrpo', B5, a1('A1-08_qwen_iougrpo_tf5'), '5.12'), ('B4_sft4800', B5, na('B4_sft_l4800_s0', 'tf5'), '5.12'),
         ('ours_l4800', B5, na('ref_l4800', 'tf5'), '5.12')]
summ = {'reader': [os.path.basename(READER), sha(READER)], 'prior': prior,
        'cohorts': {k: sha(os.path.join(OUT, f'cohort_{k}.json')) for k in strata}, 'pairs': {}}
for lab, b, a, tf in PAIRS:
    if not (os.path.exists(a) and os.path.exists(b)):
        summ['pairs'][lab] = 'missing'; print(lab, 'missing'); continue
    row = {'base_sha16': sha(b), 'arm_sha16': sha(a)}
    for k in strata:
        o = os.path.join(OUT, f'{lab}__{k}.json')
        r = subprocess.run([sys.executable, READER, '--base', b, '--arm', a, '--expect-n', '600', '--ops', 'mask,swap', '--machine', 'node7',
                            '--transformers', tf, '--label', f'{lab}__{k}', '--out', o, '--fixed-cohort', os.path.join(OUT, f'cohort_{k}.json')],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode:
            row[k] = 'rc=%d %s' % (r.returncode, r.stderr[-200:]); continue
        po = json.load(open(o, encoding='utf-8'))['per_op']
        row[k] = {op: {'gamma': po[op]['gamma'], 'ci95': po[op]['gamma_ci95'], 'n': po[op].get('n_cohort')} for op in ('mask', 'swap')}
        row[k]['read_sha16'] = sha(o)
    summ['pairs'][lab] = row
    f = lambda d: f"{d['mask']['gamma']:+.3f} [{d['mask']['ci95'][0]:+.3f},{d['mask']['ci95'][1]:+.3f}] n={d['mask']['n']}" if isinstance(d, dict) else str(d)[:60]
    print(f'{lab:16s} ' + ' | '.join(f'{k} {f(row[k])}' for k in strata))
p = os.path.join(OUT, 'PRIOR_STRATA_SUMMARY.json'); json.dump(summ, open(p, 'w', encoding='utf-8'), indent=1)
print(p, sha(p))
