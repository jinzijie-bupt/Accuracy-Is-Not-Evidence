# mk_a1s_jobs.py (11-09): two additions to A1u for R1 W2a ("gains under unseen noise and Gaussian blur are within the
# mask-audit resolution"). Same instrument and args as a1u_jobs.jsonl (14Ln r8 on indep600), only these change:
#  (1) A1s severity: noise/gblur at stronger settings via the instrument's own flags (default sigma 40 / 3, under which the
#      base's psi is only 0.06 / 0.02, i.e. the op hardly removes evidence): L2 = --noise-sigma 80 --gblur-sigma 6,
#      L3 = --noise-sigma 120 --gblur-sigma 10. Cells U01 base451, U02 SFT451, U04 ours a05 (tf451) at L2+L3; U07 base5,
#      U22 l4800 (tf5) at L3.
#  (2) A1u seed pair: U04 with the seed-2 adapter (B1a) at the default severities, ops noise,gblur,pscram -> an
#      op-specific seed-to-seed resolution next to the mask/swap one in the paper.
import json, os
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__)); R = WORK_ROOT
J = [json.loads(l) for l in open(os.path.join(HERE, 'a1u_jobs.jsonl'), encoding='utf-8')]
LV = {'L2': ('80', '6'), 'L3': ('120', '10')}
CELLS = {'U01_qwen_base_tf451': ('L2', 'L3'), 'U02_qwen_sft_tf451': ('L2', 'L3'), 'U04_qwen_a05_tf451': ('L2', 'L3'),
         'U07_qwen_base_tf5': ('L3',), 'U22_qwen_l4800_tf5': ('L3',)}
out = []
for j in J:
    cell = j['name'].split('__')[1]
    for lv in CELLS.get(cell, ()):
        k = json.loads(json.dumps(j)); a = k['argv']
        a[a.index('--ops') + 1] = 'noise,gblur'
        i = a.index('--out'); a[i:i] = ['--noise-sigma', LV[lv][0], '--gblur-sigma', LV[lv][1]]
        new = f'{cell}_{lv}'
        o = a[a.index('--out') + 1].replace('/results/a1u/', '/results/a1s/').replace(cell, new)
        a[a.index('--out') + 1] = o; k['out'] = o; k['name'] = j['name'].replace('a1u__', 'a1s__').replace(cell, new)
        k['prio'] = 10
        if 'mkdir' in k: k['mkdir'] = os.path.dirname(o)
        out.append(k)
S2 = f'{R}/results/train/adapter_nat_base_a05_g1_s2'
for j in J:
    if j['name'].startswith('a1u__U04_qwen_a05_tf451__'):
        k = json.loads(json.dumps(j)); a = k['argv']; i = a.index('--adapter')
        a[i + 1] = S2
        o = a[a.index('--out') + 1].replace('U04_qwen_a05_tf451', 'U05_qwen_a05s2_tf451')
        a[a.index('--out') + 1] = o; k['out'] = o; k['name'] = j['name'].replace('U04_qwen_a05_tf451', 'U05_qwen_a05s2_tf451')
        k['prio'] = 10
        if 'mkdir' in k: k['mkdir'] = os.path.dirname(o)
        out.append(k)
assert len(out) == 6 * (2 + 2 + 2 + 1 + 1) + 6, len(out)
assert len({k['out'] for k in out}) == len(out) and len({k['name'] for k in out}) == len(out)
p = os.path.join(HERE, 'a1s_jobs.jsonl')
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    for k in out:
        f.write(json.dumps(k) + '\n')
print(p, len(out))
for k in out[:1] + out[-1:]:
    print(k['name'], ' '.join(k['argv'])[-420:])
