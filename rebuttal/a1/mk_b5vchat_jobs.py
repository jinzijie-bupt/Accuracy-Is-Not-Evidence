# mk_b5vchat_jobs.py (11-11 06Z): B5 (lam 0.01 s0) and B5s1 (seed 1) under the VideoChat-R1 answer format, 14Ln r8 first 200,
# argv = s2p__a05_s0__o{0,100} (s2p_jobs.jsonl) with the adapter and out swapped. Completes B5 x {default, para1, para2, vchat}
# and gives the B5 seed pair under vchat. node9 only (adapters trained there).
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__)); R = WORK_ROOT
A05 = (DATASETS_ROOT + '/ds-1/local__nat_base_a05_g1_s0')
AD = {'B5': f'{R}/results/train/b5_grpo_plus_placebo_lam001_s0', 'B5s1': f'{R}/results/train/b5_grpo_plus_placebo_lam001_s1'}
src = [json.loads(l) for l in open(os.path.join(HERE, 's2p_jobs.jsonl'), encoding='utf-8') if '"s2p__a05_s0__o' in l]
assert len(src) == 2
out = []
for t, ad in AD.items():
    for j in src:
        k = json.loads(json.dumps(j).replace(A05, ad).replace('/s2p/a05_s0/', f'/s2p/{t}/').replace('_s2p_a05_s0_', f'_s2p_{t}_')
                       .replace('s2p__a05_s0__', f's2p__{t}__'))
        k['prio'] = 6; k['mkdir'] = f'{R}/results/s2p/{t}'; out.append(k)
assert all(AD[j['name'].split('__')[1]] in j['argv'] and A05 not in json.dumps(j) for j in out) and len({j['out'] for j in out}) == 4
with open(os.path.join(HERE, 'b5vchat_jobs.jsonl'), 'w', encoding='utf-8', newline='\n') as f:
    for j in out:
        f.write(json.dumps(j) + '\n')
for j in out:
    print(j['name'], j['out'].split('/results/')[1])
