# mk_flank_jobs.py (11-09): R2 W1/W3 ("the audit credits only one route"; "what did the model come to rely on").
# Same instrument and args as the A1u indep600 cells (14Ln r8, a1u_jobs.jsonl), ops back to mask,swap, plus the r8 flag
# --jitter shiftl|shiftr --jitter-frac 1.0: the edited window E* is the annotated interval moved by one full event length,
# i.e. the frames directly before / after the event (clipped at the video ends; the placebo still matches |E*|, the scored
# answer is still the GT interval). Gamma on these flank windows vs Gamma on the event itself (A1 indep600) says whether the
# trained dependence sits on the event frames or spreads to the surrounding context.
# Cells: base451 (U01 template), SFT451 (U02), ours a05 s0 (U04), B5 IoU-GRPO+placebo (U04 template, adapter swapped).
import json, os
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__)); R = WORK_ROOT
J = [json.loads(l) for l in open(os.path.join(HERE, 'a1u_jobs.jsonl'), encoding='utf-8')]
CELLS = {'F01_qwen_base_tf451': ('U01_qwen_base_tf451', None), 'F02_qwen_sft_tf451': ('U02_qwen_sft_tf451', None),
         'F04_qwen_a05_tf451': ('U04_qwen_a05_tf451', None),
         'F05_qwen_b5_tf451': ('U04_qwen_a05_tf451', f'{R}/results/train/b5_grpo_plus_placebo_lam001_s0')}
out = []
for cell, (tmpl, adapter) in CELLS.items():
    for j in J:
        if j['name'].split('__')[1] != tmpl:
            continue
        for d in ('shiftl', 'shiftr'):
            k = json.loads(json.dumps(j)); a = k['argv']
            a[a.index('--ops') + 1] = 'mask,swap'
            if adapter:
                a[a.index('--adapter') + 1] = adapter
            i = a.index('--out'); a[i:i] = ['--jitter', d, '--jitter-frac', '1.0']
            new = f'{cell}_{d}1'
            o = a[a.index('--out') + 1].replace('/results/a1u/', '/results/flank/').replace(tmpl, new)
            a[a.index('--out') + 1] = o; k['out'] = o
            k['name'] = j['name'].replace('a1u__', 'fl__').replace(tmpl, new); k['prio'] = 9
            if 'mkdir' in k: k['mkdir'] = os.path.dirname(o)
            out.append(k)
assert len(out) == 4 * 2 * 6, len(out)
assert len({k['out'] for k in out}) == len(out) and len({k['name'] for k in out}) == len(out)
assert all('/a1u/' not in k['out'] and 'U0' not in k['out'] for k in out)
p = os.path.join(HERE, 'flank_jobs.jsonl')
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    for k in out:
        f.write(json.dumps(k) + '\n')
print(p, len(out))
for k in (out[0], out[-1]):
    print(k['name'], ' '.join(k['argv'])[-400:])
