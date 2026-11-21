# mk_para2x_jobs.py (11-10): follow-up to promptx. Under wording para2 our models leave many clean questions without an
# interval (14Ln first 200, greedy: s0 0.18, s2 0.65; base 0.005, SFT 0.035). These jobs measure it on regenerated
# answers (p5n, first 500, ops none / mask_evid / mask_rand) and check B8 (mixed answer formats) under para2.
#   (a) p5n para2: qwenbase, a05_s0, a05_s2, sft (copied from the para1 p5n jobs, style swapped)   8 jobs, now
#   (b) 14Ln para2 for B8 (copied from px__14Ln_a05_s0_para2, adapter/tag swapped)                 2 jobs, on B8 .done
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__))
R = WORK_ROOT; TAG = 'B8_qwen_a05_mixfmt_s0'; AD8 = f'{R}/results/train/adapter_nat_base_a05_mixfmt_s0'
A05 = (DATASETS_ROOT + '/ds-1/local__nat_base_a05_g1_s0')
PX = [json.loads(l) for l in open(os.path.join(HERE, 'promptx_jobs.jsonl'), encoding='utf-8')]
A2 = [j for j in PX if j['name'] == 'px__14Ln_a05_s2_para1__o0'][0]['argv']
AS2 = A2[A2.index('--adapter') + 1]
now, b8 = [], []
for j in PX:
    n = j['name']
    if n.startswith('px__p5n_') and '_para1__' in n:
        k = json.loads(json.dumps(j).replace('para1', 'para2')); k['prio'] = 9; now.append(k)
        if n.startswith('px__p5n_a05_s0_para1__'):
            s2 = json.loads(json.dumps(k).replace(A05, AS2).replace('/promptx/a05_s0/', '/promptx/a05_s2/')
                            .replace('_a05_s0_', '_a05_s2_'))
            s2['mkdir'] = f'{R}/results/promptx/a05_s2'; now.append(s2)
    if n.startswith('px__14Ln_a05_s0_para2__'):
        k = json.loads(json.dumps(j).replace(A05, AD8).replace('/promptx/a05_s0/', f'/newarm/{TAG}/')
                       .replace('_px_a05_s0_', f'_px_{TAG}_').replace('px__14Ln_a05_s0_', f'b8__14Ln_px_{TAG}_'))
        k['prio'] = 9; k['mkdir'] = f'{R}/results/newarm/{TAG}'; b8.append(k)
assert len(now) == 8 and len(b8) == 2, (len(now), len(b8))
assert all(AD8 in j['argv'] for j in b8) and sum(AS2 in j['argv'] for j in now) == 2
assert len({j['out'] for j in now + b8}) == 10 and all('para2' in j['out'] for j in now + b8)
for p, js in [(os.path.join(HERE, 'para2x_p5n_jobs.jsonl'), now), (os.path.join(HERE, 'para2x_b8_jobs.jsonl'), b8)]:
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        for j in js:
            f.write(json.dumps(j) + '\n')
    print(p, len(js), [j['name'] for j in js])
