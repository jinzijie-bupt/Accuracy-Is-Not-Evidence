# mk_b8_jobs.py (11-10, R2 Q2 "... under one prompt"; follow-up to s2p). B8 = our objective (a0.5, seed 0, B1a's args)
# trained with two answer formats mixed per example (27_papo_train_qcdro_v3seln_mix.py --style-mix
# timer1_default:timer1,vchat:vchat, stable hash, 1220/1180 split). Audits (14Ln r8, tf451 FA2):
#   (a) timer1 format, indep600 (B3's indep600 lines, adapter/tag swapped; pairs with A1-01)           6 jobs
#   (b) vchat format, first 200 (s2p args; pairs with s2p/qwenbase)                                      2 jobs
#   (c) held-out TimeLens format (--prompt-style timelens --answer-style timelens), first 200, for the base,
#       ours s0 (timer1 only) and B8 -> neither model saw this format in training                          6 jobs
#   (d) instruction wording para1, timer1 format, first 200 (promptx_launch.py; pairs with promptx base)   2 jobs
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__)); RB = os.path.dirname(HERE)
R = WORK_ROOT; TAG = 'B8_qwen_a05_mixfmt_s0'; AD8 = f'{R}/results/train/adapter_nat_base_a05_mixfmt_s0'
O = f'{R}/results/newarm/{TAG}'
A05 = (DATASETS_ROOT + '/ds-1/local__nat_base_a05_g1_s0')
jobs = []
AD3, T3 = f'{R}/results/train/adapter_nat_base_a05_tsonly_s0', 'B3_tsonly_a05_s0'
for l in open(os.path.join(HERE, f'{T3}_jobs.jsonl'), encoding='utf-8'):
    j = json.loads(l)
    if 'indep600' in j['out']:
        k = json.loads(json.dumps(j).replace(AD3, AD8).replace(T3, TAG)); k['prio'] = 9; jobs.append(k)
s2p = [json.loads(l) for l in open(os.path.join(RB, 'a1', 's2p_jobs.jsonl'), encoding='utf-8') if '__qwenbase__' in l]


def setv(a, f, v):
    a[a.index(f) + 1] = v


for tag, ad, style, outdir in [(TAG, AD8, 'vchat', f'newarm/{TAG}'),
                               ('qwenbase', None, 'timelens', 'fmt3/qwenbase'), ('a05_s0', A05, 'timelens', 'fmt3/a05_s0'),
                               (TAG, AD8, 'timelens', f'newarm/{TAG}')]:
    for j in s2p:
        a = list(j['argv']); setv(a, '--prompt-style', style); setv(a, '--answer-style', style)
        if ad:
            i = a.index('--prompt-style'); a[i:i] = ['--adapter', ad]
        off = a[a.index('--offset') + 1]
        o = f'{R}/results/{outdir}/14Ln_{style}_{tag}_tf451fa2_o{off}n100R3.json'; setv(a, '--out', o)
        jobs.append({**j, 'name': f'b8__14Ln_{tag}_{style}__o{off}', 'prio': 9, 'argv': a, 'out': o, 'mkdir': os.path.dirname(o)})
for l in open(os.path.join(RB, 'a1', 'promptx_jobs.jsonl'), encoding='utf-8'):
    j = json.loads(l)
    if j['name'].startswith('px__14Ln_a05_s0_para1__'):
        k = json.loads(json.dumps(j).replace(A05, AD8).replace('/promptx/a05_s0/', f'/newarm/{TAG}/')
                       .replace('_px_a05_s0_', f'_px_{TAG}_').replace('px__14Ln_a05_s0_', f'b8__14Ln_px_{TAG}_'))
        k['prio'] = 9; k['mkdir'] = O; jobs.append(k)
assert len(jobs) == 16, len(jobs)
b8 = [j for j in jobs if '/fmt3/' not in j['out']]; now = [j for j in jobs if '/fmt3/' in j['out']]
assert len(b8) == 12 and len(now) == 4 and all(AD8 in j['argv'] for j in b8)
for p, js in [(os.path.join(HERE, f'{TAG}_jobs.jsonl'), b8), (os.path.join(RB, 'a1', 'fmt3_jobs.jsonl'), now)]:
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        for j in js:
            f.write(json.dumps(j) + '\n')
    print(p, len(js))
