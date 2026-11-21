# master_table.py (11-09): one row per trained arm for the general response / AC: accuracy gain, evidence dependence, and
# "hits without evidence", each copied from an on-disk product (no new computation):
#   dR50   : PRIOR_BEHAV_SUMMARY (paired vs the same-stack base, 3,720 Charades-STA test questions, natural video)
#   Gamma  : INDEP_SUBSETS_SUMMARY 'all' (indep600, mask, read_14Ln_v1)
#   evfree : P5N_CAPALIGNED mask capal_survival (among questions both base and arm answer correctly on clean video, share
#            still answered correctly after the event frames are masked; base -> arm)
#   A100 h : TRAIN_COMPUTE elapsed_s_trainer / 3600 where the trainer logged it
import hashlib, json, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
F = {'beh': os.path.join(HERE, 'prior_strata', 'PRIOR_BEHAV_SUMMARY.json'), 'sub': os.path.join(HERE, 'indep_subsets', 'INDEP_SUBSETS_SUMMARY.json'),
     'cap': os.path.join(HERE, 'P5N_CAPALIGNED.json'), 'cmp': os.path.join(ROOT, 'results', 'node7_pull', 'compute', 'TRAIN_COMPUTE.json'),
     'cmp6': os.path.join(ROOT, 'results', 'node9_pull', 'compute', 'TRAIN_COMPUTE.json')}
D = {k: json.load(open(v, encoding='utf-8')) for k, v in F.items()}
D['cmp'] = {**D['cmp6'], **D['cmp']}  # node9 trained B5/B5s1, node8 the rest; same A100-80G type; names do not collide
# A100 h is comparable only within one training stack (our recipe: 7.4-7.6 h on tf4.51+FA2, 14.8 h on tf5 for B1b)
ROWS = [  # label, stack, behav/subsets key, capaligned key, compute key, method family
    ('SFT (paper)', 'tf4.51', 'SFT451', None, None, 'SFT'),
    ('MDPO (paper)', 'tf4.51', 'MDPO451', None, None, 'preference'),
    ('IoU-GRPO 2 fps (A2)', 'tf4.51', 'A2_iougrpo2fps', 'NA_A2_grpo2fps_s0_p5n', 'A2_grpo2fps_s0', 'accuracy RL'),
    ('ours a0.5 s0', 'tf4.51', 'ours_a05_s0', 'A7_qwena05_vs_qwenbase', None, 'ours'),
    ('ours a0.5 s2 (B1a)', 'tf4.51', 'ours_a05_s2', 'NA_B1a_qwen_a05_s2_p5n', None, 'ours'),
    ('ours timestamp-only (B3)', 'tf4.51', 'B3_tsonly', 'NA_B3_tsonly_a05_s0_p5n', 'B3_tsonly_a05_s0', 'ours'),
    ('IoU-GRPO + placebo (B5)', 'tf4.51', 'B5_grpo_placebo', 'NA_B5_grpo_placebo_lam001_s0_p5n', 'B5_grpo_placebo_lam001_s0', 'accuracy RL + ours'),
    ('public IoU-GRPO 16f', 'tf5', 'public_iougrpo16f', None, None, 'accuracy RL'),
    ('SFT 4,800 (B4)', 'tf5', 'B4_sft4800', 'NA_B4_sft_l4800_s0_p5n', 'B4_sft_l4800_s0', 'SFT'),
    ('ours l4800', 'tf5', 'ours_l4800', 'E1m_ref_l4800_p5n', None, 'ours'),
]
out = {'sources': {k: [os.path.relpath(v, ROOT), sha(v)] for k, v in F.items()}, 'rows': []}
fmt = lambda x, c, s=1: f"{s * x:+.1f} [{s * c[0]:+.1f}, {s * c[1]:+.1f}]" if s != 1 else f"{x:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]"
md = ['| arm | stack | ΔR@0.5 (pp) | Γmask indep600 | hits without evidence (base→arm) | A100 h |', '|---|---|---|---|---|---|']
for lab, st, kb, kc, kt, fam in ROWS:
    r = {'arm': lab, 'stack': st, 'family': fam}
    b = D['beh']['arms'].get('public_iougrpo16f' if kb == 'public_iougrpo16f' else kb)
    if b: r['dR50'] = {'est': b['R50_all']['delta'] * 100, 'ci95': [c * 100 for c in b['R50_all']['ci95']]}
    s = D['sub']['pairs'].get('public_iougrpo' if kb == 'public_iougrpo16f' else kb)
    if isinstance(s, dict): r['gamma_mask'] = {'est': s['all']['mask']['gamma'], 'ci95': s['all']['mask']['ci95']}
    if kc: c = D['cap'][kc]['mask']; r['evfree'] = {'base': c['capal_survival_base'], 'arm': c['capal_survival_arm']}
    if kt and D['cmp'].get(kt, {}).get('elapsed_s_trainer'): r['a100_h'] = D['cmp'][kt]['elapsed_s_trainer'] / 3600
    out['rows'].append(r)
    g = lambda k, f: f(r[k]) if k in r else '—'
    md.append(f"| {lab} | {st} | {g('dR50', lambda d: f'{d[chr(101)+chr(115)+chr(116)]:+.1f} [{d[chr(99)+chr(105)+chr(57)+chr(53)][0]:+.1f}, {d[chr(99)+chr(105)+chr(57)+chr(53)][1]:+.1f}]')} | "
              f"{g('gamma_mask', lambda d: f'{d[chr(101)+chr(115)+chr(116)]:+.3f} [{d[chr(99)+chr(105)+chr(57)+chr(53)][0]:+.3f}, {d[chr(99)+chr(105)+chr(57)+chr(53)][1]:+.3f}]')} | "
              f"{g('evfree', lambda d: f'{d[chr(98)+chr(97)+chr(115)+chr(101)]:.2f} → {d[chr(97)+chr(114)+chr(109)]:.2f}')} | {g('a100_h', lambda h: f'{h:.1f}')} |")
p = os.path.join(HERE, 'MASTER_TABLE.json'); json.dump(out, open(p, 'w', encoding='utf-8'), indent=1)
q = os.path.join(HERE, 'MASTER_TABLE.md'); open(q, 'w', encoding='utf-8').write('\n'.join(md) + '\n')
print('\n'.join(md)); print(p, sha(p)); print(q, sha(q))
