# mk_b5px_jobs.py (11-11): B5 (IoU-GRPO + our evidence term, lam 0.01 s0, node9-only adapter) under the two paraphrased
# wordings. The wording sweep (WORDING_SWEEP_SUMMARY 0dae07201f77e463) shows B5 gives an interval under all eight wordings
# (worst no-interval 0.010); this measures whether its evidence dependence holds there too:
#   14Ln r8 para1/para2 first 200 (same argv as px__14Ln_a05_s0_para*, adapter swapped)      4 jobs
#   12p5n r3 para2 first 500 (same argv as the px__p5n_a05_s0_para1 jobs, style -> para2)     2 jobs
# Base side = the existing qwenbase promptx / para2x shards on node8 (node8<->node9 byte-equivalence: xm_eq, 11-08).
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__))
R = WORK_ROOT; A05 = (DATASETS_ROOT + '/ds-1/local__nat_base_a05_g1_s0')
T, ADB = 'B5', f'{R}/results/train/b5_grpo_plus_placebo_lam001_s0'
PX = [json.loads(l) for l in open(os.path.join(HERE, 'promptx_jobs.jsonl'), encoding='utf-8')]
out = []
for j in PX:
    if j['name'].startswith(('px__14Ln_a05_s0_para1__', 'px__14Ln_a05_s0_para2__')):
        k = json.loads(json.dumps(j).replace(A05, ADB).replace('/promptx/a05_s0/', f'/promptx/{T}/')
                       .replace('_px_a05_s0_', f'_px_{T}_').replace('px__14Ln_a05_s0_', f'px__14Ln_{T}_'))
        k['prio'] = 6; k['mkdir'] = f'{R}/results/promptx/{T}'; out.append(k)
    if j['name'].startswith('px__p5n_a05_s0_para1__'):
        k = json.loads(json.dumps(j).replace(A05, ADB).replace('/promptx/a05_s0/', f'/promptx/{T}/')
                       .replace('_px_a05_s0_para1_', f'_px_{T}_para2_').replace('px__p5n_a05_s0_para1__', f'px__p5n_{T}_para2__'))
        a = k['argv']; a[a.index('--prompt-style') + 1] = 'para2'
        k['prio'] = 6; k['mkdir'] = f'{R}/results/promptx/{T}'; out.append(k)
assert len(out) == 6, len(out)
assert all(ADB in j['argv'] and A05 not in json.dumps(j) for j in out)
assert len({j['out'] for j in out}) == 6
with open(os.path.join(HERE, 'b5px_jobs.jsonl'), 'w', encoding='utf-8', newline='\n') as f:
    for j in out:
        f.write(json.dumps(j) + '\n')
for j in out:
    a = j['argv']
    print(j['name'], '| style', a[a.index('--prompt-style') + 1], '| n', a[a.index('--n') + 1], '| off', a[a.index('--offset') + 1],
          '|', j['out'].split('/results/')[1])
