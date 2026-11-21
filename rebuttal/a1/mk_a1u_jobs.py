# A1u (11-08, R1 W2a "gains under unseen noise / Gaussian blur are 0.07-0.16 nats per token, within the stated
# resolution"): rerun the three training-unseen ops of 14Ln r7/r8 (noise, gblur, pscram; disjoint from the trainer's
# SUPPORTED_V2, r7 selftest) on the independent 600-question sample (indep600, 484 videos, 3x the first-200 sample),
# with the same argv as the A1 cells except --ops. Pairs (same stack each):
#   tf451+FA2: A1-01 qwen base | A1-02 SFT | A1-04 ours a05     tf5: A1-07 qwen base | l4800 (ours, 4,800 pairs)
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
HERE = os.path.dirname(os.path.abspath(__file__))
L4800 = (DATASETS_ROOT + '/ds-1/node3__adapter_nat_base_a05_g1_l4800_s0')
src = [json.loads(l) for l in open(os.path.join(HERE, 'a1_jobs.jsonl'), encoding='utf-8')]
pick = {'A1-01_qwen_base_t1d_tf451': 'U01_qwen_base_tf451', 'A1-02_qwen_sft_tf451': 'U02_qwen_sft_tf451',
        'A1-04_qwen_a05_tf451': 'U04_qwen_a05_tf451', 'A1-07_qwen_base_t1d_tf5': 'U07_qwen_base_tf5'}
out = []
for j in src:
    cell = j['name'].split('__')[1]
    tgt = [(pick[cell], None)] if cell in pick else []
    if cell == 'A1-07_qwen_base_t1d_tf5':
        tgt.append(('U22_qwen_l4800_tf5', L4800))
    for new, adapter in tgt:
        s = json.dumps(j).replace(cell, new).replace('"a1__', '"a1u__')
        k = json.loads(s)
        a = k['argv']
        i = a.index('--ops'); assert a[i + 1] == 'mask,swap'; a[i + 1] = 'noise,gblur,pscram'
        if adapter:
            assert '--adapter' not in a
            m = a.index('--model'); a[m + 2:m + 2] = ['--adapter', adapter]
        k['out'] = k['out'].replace('/a1/', '/a1u/'); a[a.index('--out') + 1] = k['out']
        k['mkdir'] = os.path.dirname(k['out']).replace('\\', '/')
        k['prio'] = 11
        assert '/a1u/' in k['out'] and new in k['out'] and k['argv'][a.index('--out') + 1] == k['out']
        out.append(k)
assert len(out) == 30, len(out)
p = os.path.join(HERE, 'a1u_jobs.jsonl')
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    for k in out:
        f.write(json.dumps(k) + '\n')
print(p, len(out))
