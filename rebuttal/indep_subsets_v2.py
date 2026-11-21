# indep_subsets_v2.py (11-11): v1 + the 10 indep600 pairs of the R1 table that v1 did not cover (public TimeZero/Time-R1, ours on VideoChat-R1/TimeZero/Time-R1, alpha 0.25/1.0, TimeLens public + three alphas). Same seed, cohorts and reader. v1 header:
# R3 Q4 "Does the conclusion hold on other randomly sampled subsets?" The independent sample
# indep600 (600 q / 484 videos, disjoint from the paper's first 200) is split into three disjoint random subsets of 200
# questions (numpy seed 20261109), and every indep600 pair is re-read on each subset with the registered reader
# analysis/read_14Ln_v1.py (unchanged) via --fixed-cohort. Output: rebuttal/indep_subsets/<pair>__S<k>.json + SUMMARY.
import glob, hashlib, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
M = os.path.join(ROOT, 'results', 'node7_merged'); READER = os.path.join(ROOT, 'analysis', 'read_14Ln_v1.py')
OUT = os.path.join(HERE, 'indep_subsets'); os.makedirs(OUT, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
qids = [int(q) for q in json.load(open(os.path.join(HERE, 'indep600_qids.json')))]
assert len(qids) == 600 == len(set(qids))
perm = np.random.default_rng(20261109).permutation(len(qids))
subsets = {f'S{k + 1}': sorted(qids[i] for i in perm[k * 200:(k + 1) * 200]) for k in range(3)}
for k, s in subsets.items():
    json.dump(s, open(os.path.join(OUT, f'cohort_{k}.json'), 'w'))
a1 = lambda c: os.path.join(M, 'a1', c, f'14Ln_{c}_node7_o0n600.json')
na = lambda arm, sfx: os.path.join(M, 'newarm', arm, f'14Ln_indep600_{arm}_{sfx}_node7_o0n600R3.json')
B451, B5 = a1('A1-01_qwen_base_t1d_tf451'), a1('A1-07_qwen_base_t1d_tf5')
PAIRS = [('SFT451', B451, a1('A1-02_qwen_sft_tf451'), '4.51'), ('MDPO451', B451, a1('A1-03_qwen_mdpo_tf451'), '4.51'),
         ('ours_a05_s0', B451, a1('A1-04_qwen_a05_tf451'), '4.51'),
         ('ours_a05_s2', B451, na('B1a_qwen_a05_s2', 'tf451fa2'), '4.51'), ('B3_tsonly', B451, na('B3_tsonly_a05_s0', 'tf451fa2'), '4.51'),
         ('A2_iougrpo2fps', B451, na('A2_grpo2fps_s0', 'tf451fa2'), '4.51'),
         ('B5_grpo_placebo', B451, na('B5_grpo_placebo_lam001_s0', 'tf451fa2'), '4.51'),
         ('public_iougrpo', B5, a1('A1-08_qwen_iougrpo_tf5'), '5.12'), ('B4_sft4800', B5, na('B4_sft_l4800_s0', 'tf5'), '5.12'),
         ('ours_l4800', B5, na('ref_l4800', 'tf5'), '5.12'),
         ('RA-RFT', os.path.join(M, 'a6', '14Ln_indep600_tr1base_timer1_default_tf451fa2_node7_o0n600R3.json'),
          os.path.join(M, 'a6', '14Ln_indep600_rarft_timer1_default_tf451fa2_node7_o0n600R3.json'), '4.51'),
         ('public_timezero', B5, a1('A1-09_tz_base_t1d_tf5'), '5.12'), ('public_timer1', B5, a1('A1-10_tr1_base_t1d_tf5'), '5.12'),
         ('ours_vc_a05', a1('A1-05_vc_base_vchat_tf451'), a1('A1-06_vc_a05_tf451'), '4.51'),
         ('ours_tz_a05', a1('A1-09_tz_base_t1d_tf5'), a1('A1-13_tz_a05_tf5'), '5.12'),
         ('ours_tr1_a05', a1('A1-10_tr1_base_t1d_tf5'), a1('A1-14_tr1_a05_tf5'), '5.12'),
         ('ours_a025', B5, a1('A1-15_qwen_a025_tf5'), '5.12'), ('ours_a10', B5, a1('A1-16_qwen_a10_tf5'), '5.12'),
         ('public_timelens', a1('A1-17_q3vl8b_base_timelens_tf5'), a1('A1-18_timelens8b_base_tf5'), '5.12'),
         ('ours_tl_a05', a1('A1-18_timelens8b_base_tf5'), a1('A1-19_timelens8b_a05_tf5'), '5.12'),
         ('ours_tl_a025', a1('A1-18_timelens8b_base_tf5'), a1('A1-20_timelens8b_a025_tf5'), '5.12'),
         ('ours_tl_a10', a1('A1-18_timelens8b_base_tf5'), a1('A1-21_timelens8b_a10_tf5'), '5.12')]
summ = {'reader': [os.path.basename(READER), sha(READER)], 'seed': 20261109,
        'cohorts': {k: sha(os.path.join(OUT, f'cohort_{k}.json')) for k in subsets}, 'pairs': {}}
for lab, b, a, tf in PAIRS:
    if not (os.path.exists(a) and os.path.exists(b)):
        summ['pairs'][lab] = 'missing'; print(lab, 'missing'); continue
    row = {'base_sha16': sha(b), 'arm_sha16': sha(a)}
    for k in ['all'] + list(subsets):
        o = os.path.join(OUT, f'{lab}__{k}.json')
        cmd = [sys.executable, READER, '--base', b, '--arm', a, '--expect-n', '600', '--ops', 'mask,swap', '--machine', 'node7',
               '--transformers', tf, '--label', f'{lab}__{k}', '--out', o]
        if k != 'all':
            cmd += ['--fixed-cohort', os.path.join(OUT, f'cohort_{k}.json')]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode:
            row[k] = 'rc=%d %s' % (r.returncode, r.stderr[-200:]); continue
        t = json.load(open(o, encoding='utf-8')); po = t['per_op']
        row[k] = {op: {'gamma': po[op]['gamma'], 'ci95': po[op]['gamma_ci95'], 'n': po[op].get('n_cohort')} for op in ('mask', 'swap')}
        row[k]['read_sha16'] = sha(o)
    summ['pairs'][lab] = row
    f = lambda d: f"{d['mask']['gamma']:+.3f} [{d['mask']['ci95'][0]:+.3f},{d['mask']['ci95'][1]:+.3f}] n={d['mask']['n']}" if isinstance(d, dict) else str(d)[:60]
    print(f'{lab:16s} all {f(row["all"])} | ' + ' | '.join(f'{k} {f(row[k])}' for k in subsets))
p = os.path.join(OUT, 'INDEP_SUBSETS_SUMMARY_v2.json'); json.dump(summ, open(p, 'w', encoding='utf-8'), indent=1)
print(p, sha(p))
