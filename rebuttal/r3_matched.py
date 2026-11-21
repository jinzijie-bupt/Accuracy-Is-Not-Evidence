# r3_matched.py (11-10): R3 Q1 "how does increasing Gamma improve ... reliability, particularly when compared with models
# at similar accuracy or answer coverage?" -> accuracy-matched pairs, every cell copied from an on-disk read (sha16 kept).
# Pairs: (1) IoU-GRPO 2 fps (A2) vs IoU-GRPO + our two terms (B5 s0, B5 s1): same base, data, trainer, stack.
#        (2) SFT-4800 checkpoint at 800 pairs (E1 c800) vs ours on 4,800 pairs (l4800): same tf5 stack and base.
#        (3) SFT / mDPO-style (tf4.51) vs ours l4800 (tf5): accuracy gains within 1 point; each vs its own stack's base.
# Columns: dR50 (full test), Gamma mask, "hits without evidence" = capability-aligned survival under the evidence mask
# (base -> arm, both-correct questions), mismatched-query rejection (no interval) and mixed50, Charades-CD ood / iid
# gain, gain on questions the fixed prior window cannot answer. Missing cells stay None (never filled by hand).
import hashlib, json, os
H = os.path.dirname(os.path.abspath(__file__))
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
J = lambda p: json.load(open(os.path.join(H, p), encoding='utf-8'))
SRC = {}


def src(p):
    SRC[p] = sha(os.path.join(H, p)); return J(p)


MT = {r['arm']: r for r in src('MASTER_TABLE.json')['rows']}
TN = src('tneg/read/TNEG_TABLE.json')['arms']; TE1 = src('tneg/read/TNEG_E1.json')
CD = src('charades_cd/CHARADES_CD_SUMMARY.json')['arms']; PB = src('prior_strata/PRIOR_BEHAV_SUMMARY.json')['arms']
E1 = src('newarm/read/E1/E1_SUMMARY.json')['rows']['E1_c800']


def capal(p):
    d = src(p)['capability_aligned']['mask']; return [d['base_survival'], d['arm_survival']]


def row(label, mt=None, tn=None, cd=None, pb=None, ev=None, **over):
    r = {'arm': label}
    if mt:
        m = MT[mt]; r['dR50'] = m['dR50']['est'] / 100; r['gamma_mask'] = m['gamma_mask']['est']
        r['gamma_src'] = 'indep600'
    if tn:
        t = TN[tn] if isinstance(tn, str) else tn
        r['mismatch_reject'] = t['foreign_no_interval']; r['mixed50'] = t['mixed50']
    if cd:
        c = CD[cd]; r['cd_ood'] = c['test_ood']['delta']; r['cd_iid'] = c['test_iid']['delta']
    if pb:
        r['prior_fails_dR50'] = PB[pb]['R50_prior_fails']['delta']
    if ev:
        r['hits_wo_evidence'] = capal(ev)
    r.update(over); return r


L48_FIRST200 = src('newarm/read/E1/l4800_ours_tokc.json')['per_op']['mask']['subsets']['all']['gamma']  # same read as E1 c800
TN_BASE = {'tf451': TN['qwenbase_tf451'], 'tf5': TN['qwenbase_tf5']}
pairs = {
    'IoU-GRPO vs IoU-GRPO + ours (tf4.51)': [
        row('IoU-GRPO 2 fps (A2)', 'IoU-GRPO 2 fps (A2)', 'A2_grpo2fps_s0', 'A2_iougrpo2fps', 'A2_iougrpo2fps', 'newarm/read/NA_A2_grpo2fps_s0_p5n.json'),
        row('IoU-GRPO + ours, seed 0 (B5)', 'IoU-GRPO + placebo (B5)', 'B5_grpo_placebo_lam001_s0', 'B5_grpo_placebo', 'B5_grpo_placebo', 'newarm/read/NA_B5_grpo_placebo_lam001_s0_p5n.json'),
        row('IoU-GRPO + ours, seed 1 (B5s1)', None, None, None, None, 'newarm/read/NA_B5s1_grpo_placebo_lam001_s1_p5n.json',
            dR50=0.6655913978494624 - 0.5067204301075269, gamma_mask=src('newarm/read/NA_B5s1_grpo_placebo_lam001_s1_14Ln_indep600.json')['per_op']['mask']['gamma'], gamma_src='indep600'),
    ],
    'SFT checkpoint vs ours, same accuracy (tf5)': [
        row('SFT-4800 at 800 pairs (E1 c800)', None, TE1, None, None, 'newarm/read/E1m_E1_sftckpt_c800_p5n.json',
            dR50=E1['dR50'], gamma_mask=E1['gamma_mask'], gamma_src='first200 tokc'),
        row('ours, 4,800 pairs (l4800)', 'ours l4800', 'l4800', 'ours_l4800', 'ours_l4800', 'newarm/read/E1m_ref_l4800_p5n.json',
            gamma_mask=L48_FIRST200, gamma_src='first200 tokc'),
    ],
    'SFT / mDPO-style (tf4.51) vs ours l4800 (tf5)': [
        row('SFT', 'SFT (paper)', 'qwensft', 'SFT451', 'SFT451', 'a7/read/A7_qwensft_vs_qwenbase.json'),
        row('mDPO-style', 'MDPO (paper)', 'qwenmdpo', 'MDPO451', 'MDPO451', 'a7/read/A7_qwenmdpo_vs_qwenbase.json'),
        row('ours, 4,800 pairs (l4800)', 'ours l4800', 'l4800', 'ours_l4800', 'ours_l4800', 'newarm/read/E1m_ref_l4800_p5n.json'),
    ],
}
out = {'sources_sha16': SRC, 'base_mismatch_reject': {k: v['foreign_no_interval'] for k, v in TN_BASE.items()},
       'note': 'pair 2 Gamma: both on the first-200 token-level read (E1_SUMMARY / l4800_ours_tokc, same reader)',
       'pairs': pairs}
p = os.path.join(H, 'R3_MATCHED.json'); json.dump(out, open(p, 'w', encoding='utf-8'), indent=1)
f = lambda x, d=3: '—' if x is None else (f'{x:+.{d}f}' if isinstance(x, float) else str(x))
lines = ['| arm | ΔR@0.5 | Γ mask | hits w/o evidence (base→arm) | mismatch reject | mixed50 | CD ood / iid | ΔR prior-fails |', '|' + '---|' * 8]
for k, rs in pairs.items():
    lines.append(f'| **{k}** |' + ' |' * 7)
    for r in rs:
        hw = r.get('hits_wo_evidence'); cd = (r.get('cd_ood'), r.get('cd_iid'))
        lines.append(f"| {r['arm']} | {f(r.get('dR50'))} | {f(r.get('gamma_mask'))} ({r.get('gamma_src','')}) | "
                     f"{'—' if not hw else f'{hw[0]:.2f}→{hw[1]:.2f}'} | {f(r.get('mismatch_reject'))} | {f(r.get('mixed50'))} | "
                     f"{f(cd[0])} / {f(cd[1])} | {f(r.get('prior_fails_dR50'))} |")
md = os.path.join(H, 'R3_MATCHED.md'); open(md, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines)); print(p, sha(p), md, sha(md))
