# mk_b3x_jobs.py (11-10): B3 (our objective on timestamp tokens only, a0.5 s0) keeps Gamma +0.60 under the vchat answer
# format against +0.10 / +0.20 for the full-string arm. Same audits for B3 under the instruction wordings para1 / para2
# (timer1 format, promptx_launch) and the held-out TimeLens format; first 200, 14Ln r8, tf451 FA2. 6 jobs, prio 9.
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__))
R = WORK_ROOT; A05 = (DATASETS_ROOT + '/ds-1/local__nat_base_a05_g1_s0')
AD3 = f'{R}/results/train/adapter_nat_base_a05_tsonly_s0'; T = 'b3tsonly'
src = [json.loads(l) for l in open(os.path.join(HERE, 'promptx_jobs.jsonl'), encoding='utf-8')
       if l and json.loads(l)['name'].startswith(('px__14Ln_a05_s0_para1__', 'px__14Ln_a05_s0_para2__'))]
src += [json.loads(l) for l in open(os.path.join(HERE, 'fmt3_jobs.jsonl'), encoding='utf-8') if '/fmt3/a05_s0/' in l]
jobs = []
for j in src:
    k = json.loads(json.dumps(j).replace(A05, AD3).replace('/promptx/a05_s0/', f'/promptx/{T}/').replace('/fmt3/a05_s0/', f'/fmt3/{T}/')
                   .replace('_a05_s0_', f'_{T}_'))
    k['prio'] = 9; k['mkdir'] = os.path.dirname(k['out']); jobs.append(k)
assert len(jobs) == 6 and all(AD3 in j['argv'] and A05 not in json.dumps(j) for j in jobs), len(jobs)
p = os.path.join(HERE, 'b3x_jobs.jsonl')
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    for j in jobs:
        f.write(json.dumps(j) + '\n')
print(p, [(j['name'], j['out'].rsplit('/', 2)[-2:]) for j in jobs])
