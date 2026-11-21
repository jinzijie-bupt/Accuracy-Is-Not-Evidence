# B1b eval suite (11-08, R2 Q2 "the audit's resolution comes from two pairs of seeds of one model"): TimeZero-7B a0.5,
# seed 1 (tf5, same args as local__nat_tz_a05_g1_s0 except --seed), paired with the seed-0 arm on the same probes:
#   14Ln r10 --tok-lp --cands first200 (base partner a5/tzbase; seed-0 partner a5/tza05)
#   14Ln r8 indep600 (base partner A1-09; seed-0 partner A1-13)
#   12p4n tf5meta r8 full 3720 for base / seed 0 / seed 1 (accuracy of the seed pair on this machine)
# Argument lists are copied from the a5 / a1 jobs of the seed-0 arm, with the adapter and output label swapped.
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__)); RB = os.path.dirname(HERE)
R = WORK_ROOT; TAG = 'B1b_tz_a05_s1'; O = f'{R}/results/newarm/{TAG}'
AD0 = (DATASETS_ROOT + '/ds-1/local__nat_tz_a05_g1_s0'); AD1 = f'{R}/results/train/adapter_nat_tz_a05_g1_s1'
MODEL = f'{R}/models/TimeZero-Charades-7B'
jobs = []
for l in open(os.path.join(RB, 'a5', 'a5_jobs.jsonl'), encoding='utf-8'):
    j = json.loads(l)
    if j['name'].startswith('a5__tza05__'):
        s = json.dumps(j).replace(AD0, AD1).replace('/results/a5/tza05/', f'/results/newarm/{TAG}/') \
            .replace('14Ln_tza05_', f'14Ln_{TAG}_').replace('a5__tza05__', f'na__{TAG}__14Ln_tokc_')
        k = json.loads(s); k['prio'] = 12; k['mkdir'] = O; jobs.append(k)
for l in open(os.path.join(RB, 'a1', 'a1_jobs.jsonl'), encoding='utf-8'):
    j = json.loads(l)
    if j['name'].startswith('a1__A1-13_tz_a05_tf5__'):
        off = j['name'].rsplit('__o', 1)[1]
        s = json.dumps(j).replace(AD0, AD1).replace('/results/a1/A1-13_tz_a05_tf5/', f'/results/newarm/{TAG}/') \
            .replace('A1-13_tz_a05_tf5__', f'14Ln_indep600_{TAG}_tf5_')
        k = json.loads(s); k['name'] = f'na__{TAG}__14Ln_indep600_o{off}'; k['prio'] = 13; k['mkdir'] = O; jobs.append(k)
for lab, ad, od in (('tzbase', None, f'{R}/results/newarm/_base_tz_tf5'), ('tza05_s0', AD0, f'{R}/results/newarm/ref_tza05_s0'),
                    (TAG, AD1, O)):
    for off in range(0, 3720, 465):
        out = f'{od}/p4nat_{lab}_tf5_o{off}n465.json'
        argv = ['12p4n_native_acc_eval_tf5meta_r8.py', '--model', MODEL] + (['--adapter', ad] if ad else []) + \
               ['--prompt-style', 'timer1_default', '--n', '465', '--offset', str(off), '--out', out]
        nm = f'na__{TAG}__p4nat_o{off}' if lab == TAG else f'na__{lab}__p4nat_o{off}'
        jobs.append({'name': nm, 'env': 'tf5', 'cwd': f'{R}/probe3_selector_evidence', 'prio': 16, 'argv': argv,
                     'out': out, 'mem_gb': 30.0, 'mkdir': od})
for k in jobs:
    s = json.dumps(k)
    assert AD0 not in s or k['name'].startswith('na__tza05_s0'), k['name']
    assert k['out'] == k['argv'][k['argv'].index('--out') + 1] and k['out'].startswith('/') and '\\' not in s
assert len(jobs) == 2 + 6 + 24, len(jobs)
assert len({k['name'] for k in jobs}) == len(jobs) and len({k['out'] for k in jobs}) == len(jobs)
p = os.path.join(HERE, f'{TAG}_jobs.jsonl')
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    for k in jobs:
        f.write(json.dumps(k) + '\n')
print(p, len(jobs))
