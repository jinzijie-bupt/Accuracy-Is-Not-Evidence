# E1m (11-08): complete the accuracy-matched pair of E1 (R3 Q1 "compared with models at similar accuracy or answer
# coverage"): SFT checkpoint c800 (dR +7.1, Gamma -0.13) vs ours l4800 (dR +8.1, Gamma +2.77), both tf5, same base.
# For both: indep600 audit (14Ln r8), behaviour under edits (12p5n r3, 1000), and for c800 the foreign-query true-negative
# test (l4800 already has it) and the QVH pair (l4800 already queued). Argument lists are copied from existing jobs
# (B4 suite / tneg l4800 / qvh B4) with the adapter and label swapped.
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__)); RB = os.path.dirname(HERE)
R = WORK_ROOT
B4AD = f'{R}/results/train/adapter_nat_sft_l4800_s0'; C800 = f'{B4AD}__ckpt/c800'
L48 = (DATASETS_ROOT + '/ds-1/node3__adapter_nat_base_a05_g1_l4800_s0')
rd = lambda p: [json.loads(l) for l in open(os.path.join(RB, p), encoding='utf-8')]
jobs = []


def swap(j, ad_old, ad_new, lab_old, lab_new, prio=9):
    s = json.dumps(j)
    assert s.count(ad_old + '"') == 1, (j['name'], ad_old)
    k = json.loads(s.replace(ad_old + '"', ad_new + '"').replace(lab_old, lab_new)); k['prio'] = prio
    assert lab_new in k['out'] and k['out'] == k['argv'][k['argv'].index('--out') + 1]
    return k


b4 = rd('newarm/B4_sft_l4800_s0_jobs.jsonl')
for j in b4:
    if j['name'].startswith('na__B4_sft_l4800_s0__') and ('indep600' in j['name'] or '__p5n_' in j['name']):
        jobs.append(swap(j, B4AD, C800, 'B4_sft_l4800_s0', 'E1_sftckpt_c800'))
        jobs.append(swap(j, B4AD, L48, 'B4_sft_l4800_s0', 'ref_l4800'))
for j in rd('tneg/tneg_main_jobs.jsonl'):
    if j['name'] == 'tneg__l4800__foreignq_s0_o0n1000':
        jobs.append(swap(j, L48, C800, 'l4800', 'E1_sftckpt_c800'))
for j in rd('qvh/qvh_jobs.jsonl'):
    if j['name'].startswith('qvh__B4_sft4800__'):
        jobs.append(swap(j, B4AD, C800, 'B4_sft4800', 'E1_c800'))
assert len(jobs) == 6 + 6 + 4 + 4 + 1 + 5, len(jobs)
assert len({j['name'] for j in jobs}) == len(jobs) and len({j['out'] for j in jobs}) == len(jobs)
p = os.path.join(HERE, 'E1m_jobs.jsonl')
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    for j in jobs:
        f.write(json.dumps(j) + '\n')
print(p, len(jobs)); print(sorted(j['name'] for j in jobs)[:30])
