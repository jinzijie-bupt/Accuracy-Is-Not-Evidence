# read_wording_sweep.py (11-10): reads the para2 follow-ups (mk_wsweep_jobs.py, mk_para2x_jobs.py) with the registered
# readers (analysis/read_14Ln_v1.py, analysis/read_p5n_v2.py, unchanged). Inputs from results/node7_pull. Missing cells
# are reported as 'missing', never filled.
#  SWEEP  clean answers, first 200 questions (p5n 'none'): no-interval rate and R@0.5 per model x wording; default wording
#         from the first 200 rows of the existing default p5n shard o0 (a7 / newarm), the 8 others from wsweep/.
#  FORCED forced-answer p5n (interval grammar), first 500: base vs arm under para2, vchat and the default wording (a3f).
#  B8/B9  14Ln Gamma vs base: default wording on indep600 (vs A1-01, and vs the same-seed single-wording arm), para1 /
#         para2 first 200; p5n default / para2 first 500 vs base.
# Outputs rebuttal/a1/read_wording_sweep/<label>.json and WORDING_SWEEP_SUMMARY.json.
import glob, hashlib, json, os, subprocess, sys
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.normpath(os.path.join(H, '..', '..'))
PULL = os.path.join(ROOT, 'results', 'node7_pull'); MERGED = os.path.join(ROOT, 'results', 'node7_merged', 'wording_sweep')
MERGE = os.path.join(ROOT, 'rebuttal', 'merge_shards.py')
R14 = os.path.join(ROOT, 'analysis', 'read_14Ln_v1.py'); P5 = os.path.join(ROOT, 'analysis', 'read_p5n_v2.py')
OUT = os.path.join(H, 'read_wording_sweep'); os.makedirs(OUT, exist_ok=True); os.makedirs(MERGED, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
B8, B9 = 'B8_qwen_a05_mixfmt_s0', 'B9_qwen_a05_mixw_s2'
WORD = ['para1', 'para2', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6']
NEWTAG = {'a2grpo_s0': 'A2_grpo2fps_s0', 'a05_s2': 'B1a_qwen_a05_s2', 'b3tsonly': 'B3_tsonly_a05_s0', 'B5': 'B5_grpo_placebo_lam001_s0'}
A7TAG = {'qwenbase': 'qwenbase', 'sft': 'qwensft', 'a05_s0': 'qwena05'}


def files(pattern):
    return sorted(f for f in glob.glob(os.path.join(PULL, pattern)) if not f.endswith(('.meta.json', '.attn.json', '.partial.jsonl')))


def merged(pattern, nshard, tag):
    sh = files(pattern)
    if len(sh) != nshard:
        return None
    out = os.path.join(MERGED, f'{tag}.json')
    subprocess.run([sys.executable, MERGE, out] + sh, check=True, capture_output=True)
    return out


def default_p5n_o0(t):
    if t in A7TAG:
        return files(f'a7/p5n_{A7TAG[t]}_timer1_default_tf451fa2_o0n250.json')
    if t in NEWTAG:
        return files(f'newarm/{NEWTAG[t]}/p5n_{NEWTAG[t]}_tf451fa2_o0n250.json')
    if t == B9:
        return files(f'newarm/{B9}/p5n_px_{B9}_timer1_default_tf451fa2_o0n250.json')
    return []


summ = {'readers': {'R14': sha(R14), 'P5': sha(P5)}, 'script_sha16': sha(os.path.abspath(__file__)), 'sweep': {}, 'forced': {},
        'r14': {}, 'p5n': {}}
# ---------------- SWEEP
for t in ['qwenbase', 'sft', 'a2grpo_s0', 'a05_s0', 'a05_s2', 'b3tsonly', 'B5', B8, B9]:
    row = {}
    for st in ['default'] + WORD:
        if st == 'default':
            fs = default_p5n_o0(t); rows = json.load(open(fs[0], encoding='utf-8'))[:200] if len(fs) == 1 else None
        else:
            d = f'newarm/{t}' if t in (B8, B9) else f'wsweep/{t}'
            fs = files(f'{d}/p5n_ws_{t}_{st}_tf451fa2_o0n200.json'); rows = json.load(open(fs[0], encoding='utf-8')) if len(fs) == 1 else None
        if rows is None:
            row[st] = 'missing'; continue
        assert len(rows) == 200 and [str(r['qid']) for r in rows] == [str(i) for i in range(200)], (t, st)
        row[st] = {'sha16': sha(fs[0]), 'no_interval': sum(1 for r in rows if not r['none_parsed']) / 200,
                   'R50': sum(1 for r in rows if r['none_hit']) / 200}
    got = [row[s] for s in WORD if isinstance(row[s], dict)]
    if len(got) == len(WORD):
        row['mean8'] = {k: sum(g[k] for g in got) / 8 for k in ('no_interval', 'R50')}
        row['worst8_no_interval'] = max(g['no_interval'] for g in got)
    summ['sweep'][t] = row


# ---------------- p5n pair helper
def p5pair(lab, bp, ap, into):
    if not (bp and ap):
        into[lab] = 'missing'; return
    out = os.path.join(OUT, f'{lab}.json')
    r = subprocess.run([sys.executable, P5, '--pair', bp, ap, '--ops', 'none,mask_evid,mask_rand', '--label', lab, '--out', out],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode:
        into[lab] = f'reader rc={r.returncode} {r.stderr[-200:]}'; return
    d = json.load(open(out, encoding='utf-8')); m = d['all_questions']['paired']['mask']
    into[lab] = {'read_sha16': sha(out), 'base_sha16': sha(bp), 'arm_sha16': sha(ap), 'DiD': m['DiD_contrast_arm_minus_base'],
                 'base_contrast': m['base_contrast']['est'], 'arm_contrast': m['arm_contrast']['est'],
                 'capability_aligned_mask': d.get('capability_aligned', {}).get('mask'),
                 'base_split': {k: d['all_questions']['base']['conditions'][k]['split'] for k in ('none', 'mask_evid', 'mask_rand')},
                 'arm_split': {k: d['all_questions']['arm']['conditions'][k]['split'] for k in ('none', 'mask_evid', 'mask_rand')}}


# ---------------- FORCED (first 500)
FB = {}
for t in ('qwenbase', 'sft', 'a05_s0', 'a05_s2'):
    for st in ('para2', 'vchat'):
        FB[(st, t)] = merged(f'wforced/{t}/p5nF_{t}_{st}_tf451fa2_o*n250.json', 2, f'F_{st}_{t}')
FB[('para2', B9)] = merged(f'newarm/{B9}/p5nF_{B9}_para2_tf451fa2_o*n250.json', 2, f'F_para2_{B9}')
for t, a3 in (('qwenbase', 'qwenbase'), ('sft', 'qwensft'), ('a05_s0', 'qwena05')):
    FB[('timer1_default', t)] = merged(f'a3f/p5nF_{a3}_timer1_default_tf451fa2_o[02]*n250.json', 2, f'F_default_{t}')
for st, arms in (('para2', ('sft', 'a05_s0', 'a05_s2', B9)), ('vchat', ('sft', 'a05_s0', 'a05_s2')), ('timer1_default', ('sft', 'a05_s0'))):
    for a in arms:
        p5pair(f'forced_{st}__{a}_vs_qwenbase', FB.get((st, 'qwenbase')), FB.get((st, a)), summ['forced'])

# ---------------- B8 / B9: 14Ln and p5n
C = {'ind_base': merged('a1/A1-01_qwen_base_t1d_tf451/A1-01_qwen_base_t1d_tf451__o*n100.json', 6, 'ind_qwenbase'),
     'ind_s2': merged('newarm/B1a_qwen_a05_s2/14Ln_indep600_B1a_qwen_a05_s2_tf451fa2_o*n100R3.json', 6, 'ind_a05_s2')}
for tg in (B8, B9):
    C[f'ind_{tg}'] = merged(f'newarm/{tg}/14Ln_indep600_{tg}_tf451fa2_o*n100R3.json', 6, f'ind_{tg}')
    for st in ('para1', 'para2'):
        C[f'{st}_{tg}'] = merged(f'newarm/{tg}/14Ln_px_{tg}_{st}_tf451fa2_o*n100R3.json', 2, f'{st}_{tg}')
for st in ('para1', 'para2'):
    for t in ('qwenbase', 'a05_s2'):
        C[f'{st}_{t}'] = merged(f'promptx/{t}/14Ln_px_{t}_{st}_tf451fa2_o*n100R3.json', 2, f'{st}_{t}')
R14P = [(f'indep600__{tg}_vs_qwenbase', 'ind_base', f'ind_{tg}', '600') for tg in (B8, B9)]
R14P += [(f'indep600__{B9}_vs_a05_s2', 'ind_s2', f'ind_{B9}', '600')]
R14P += [(f'{st}__{tg}_vs_qwenbase', f'{st}_qwenbase', f'{st}_{tg}', '200') for st in ('para1', 'para2') for tg in (B8, B9)]
R14P += [(f'{st}__{B9}_vs_a05_s2', f'{st}_a05_s2', f'{st}_{B9}', '200') for st in ('para1', 'para2')]
for lab, b, a, n in R14P:
    bp, ap = C.get(b), C.get(a)
    if not (bp and ap):
        summ['r14'][lab] = 'missing'; continue
    out = os.path.join(OUT, f'{lab}.json')
    r = subprocess.run([sys.executable, R14, '--base', bp, '--arm', ap, '--expect-n', n, '--ops', 'mask,swap', '--machine', 'node7',
                        '--transformers', '4.51', '--label', lab, '--out', out], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode:
        summ['r14'][lab] = f'reader rc={r.returncode} {r.stderr[-200:]}'; continue
    d = json.load(open(out, encoding='utf-8')); po = d['per_op']['mask']
    rb = json.load(open(bp, encoding='utf-8')); ra = json.load(open(ap, encoding='utf-8'))
    noint = lambda R: sum(1 for x in R if not x.get('clean_parsed')) / len(R)
    summ['r14'][lab] = {'read_sha16': sha(out), 'base_sha16': sha(bp), 'arm_sha16': sha(ap), 'n_cohort': po['n_cohort'],
                        'gamma': po['gamma'], 'ci95': po['gamma_ci95'], 'clean_R50_base': d.get('clean_R50_base'),
                        'clean_R50_arm': d.get('clean_R50_arm'), 'no_interval_base': noint(rb), 'no_interval_arm': noint(ra)}
PB = {('timer1_default', 'qwenbase'): merged('a7/p5n_qwenbase_timer1_default_tf451fa2_o[02]*n250.json', 2, 'p5n_default_qwenbase'),
      ('para2', 'qwenbase'): merged('promptx/qwenbase/p5n_px_qwenbase_para2_tf451fa2_o*n250.json', 2, 'p5n_para2_qwenbase')}
for st in ('timer1_default', 'para2'):
    PB[(st, B9)] = merged(f'newarm/{B9}/p5n_px_{B9}_{st}_tf451fa2_o*n250.json', 2, f'p5n_{st}_{B9}')
    p5pair(f'p5n_{st}__{B9}_vs_qwenbase', PB.get((st, 'qwenbase')), PB.get((st, B9)), summ['p5n'])

p = os.path.join(OUT, 'WORDING_SWEEP_SUMMARY.json'); json.dump(summ, open(p, 'w', encoding='utf-8'), indent=1)
f3 = lambda x: f'{x:.3f}'
print('SWEEP  no-interval / R@0.5 (first 200)')
print(f"{'model':28s}" + ''.join(f'{s:>14s}' for s in ['default'] + WORD) + '   mean8 / worst')
for t, row in summ['sweep'].items():
    cells = ''.join(f"{'missing':>14s}" if isinstance(row[s], str) else f"{row[s]['no_interval']:>6.3f}/{row[s]['R50']:.3f}" for s in ['default'] + WORD)
    tail = f"   {row['mean8']['no_interval']:.3f}/{row['mean8']['R50']:.3f} / {row['worst8_no_interval']:.3f}" if 'mean8' in row else ''
    print(f'{t:28s}{cells}{tail}')
for sec in ('forced', 'p5n'):
    for lab, v in summ[sec].items():
        print(lab, v if isinstance(v, str) else f"DiD {v['DiD']['est']:+.3f} [{v['DiD']['ci95'][0]:+.3f},{v['DiD']['ci95'][1]:+.3f}] "
              f"none hit base/arm {v['base_split']['none']['hit']:.3f}/{v['arm_split']['none']['hit']:.3f} "
              f"evid hit {v['base_split']['mask_evid']['hit']:.3f}/{v['arm_split']['mask_evid']['hit']:.3f} "
              f"rand hit {v['base_split']['mask_rand']['hit']:.3f}/{v['arm_split']['mask_rand']['hit']:.3f}")
for lab, v in summ['r14'].items():
    print(f'{lab:44s}', v if isinstance(v, str) else f"G {v['gamma']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] n={v['n_cohort']} "
          f"R50 {v['clean_R50_base']:.3f}/{v['clean_R50_arm']:.3f} noint {v['no_interval_base']:.3f}/{v['no_interval_arm']:.3f}")
print(p, sha(p))
