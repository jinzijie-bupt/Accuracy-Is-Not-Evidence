# mk_wsweep_jobs.py (11-10): follow-up to the para2 finding (14Ln first 200, clean greedy: no interval s0 0.18, s2 0.65,
# base 0.005, SFT 0.035). Three groups, all tf451 FA2, instruments unchanged, extra wordings via promptx2_launch.py
# (d713477328601b9a; para1/para2 byte-identical to promptx_launch.py, w1..w6 written before any w-result):
#  (a) wording sweep, clean answers only (12p5n r3 --ops none), first 200, eight wordings (para1, para2, w1..w6) x seven
#      models (base, SFT, IoU-GRPO 2 fps, ours a0.5 s0 / s2, B3 timestamp-only, B5 IoU-GRPO + our terms)      56 jobs, prio 9
#      -> no-interval rate and R@0.5 per wording: is para2 one bad wording or the typical case?
#  (b) forced answer (12p5n forced --forced prefix, ops none / mask_evid / mask_rand), first 500, base / SFT / s0 / s2,
#      under para2 and under vchat                                                                           16 jobs, prio 8
#      -> accuracy when an interval must be given (abstention vs lost localisation) and the evidence-vs-placebo
#         difference without the refusal channel
#  (c) B9 = our objective, a0.5, SEED 2 (the seed that refuses most), trained with five instruction wordings mixed per
#      example (timer1_default + tw1..tw4, trainer 27_papo_train_qcdro_v3seln_mixw.py), hooked on B9 .done:
#      sweep (8), 14Ln para1/para2 first 200 (4), 14Ln default indep600 (6), p5n default first 500 (2),
#      p5n para2 first 500 (2), forced para2 first 500 (2)                                                    24 jobs, prio 8
#      and B8 (answer formats mixed) on the sweep, hooked on B8 .done                                          8 jobs, prio 9
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__)); RB = os.path.dirname(HERE)
R = WORK_ROOT; L1 = f'{R}/setup/promptx_launch.py'; L2 = f'{R}/setup/promptx2_launch.py'
A05 = (DATASETS_ROOT + '/ds-1/local__nat_base_a05_g1_s0')
AD = {'qwenbase': None, 'sft': (DATASETS_ROOT + '/ds-1/local__nat_sft_b0_s0'),
      'a2grpo_s0': f'{R}/results/grpo/adapter_grpo_native_iou_s0', 'a05_s0': A05,
      'a05_s2': f'{R}/results/train/adapter_nat_base_a05_g1_s2', 'b3tsonly': f'{R}/results/train/adapter_nat_base_a05_tsonly_s0',
      'B5': f'{R}/results/train/b5_grpo_plus_placebo_lam001_s0'}
B8, AD8 = 'B8_qwen_a05_mixfmt_s0', f'{R}/results/train/adapter_nat_base_a05_mixfmt_s0'
B9, AD9 = 'B9_qwen_a05_mixw_s2', f'{R}/results/train/adapter_nat_base_a05_mixw_s2'
WORD = ['para1', 'para2', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6']
PX = [json.loads(l) for l in open(os.path.join(HERE, 'promptx_jobs.jsonl'), encoding='utf-8')]
F3 = [json.loads(l) for l in open(os.path.join(RB, 'a3f', 'a3f_jobs.jsonl'), encoding='utf-8')]


def setv(a, f, v):
    a[a.index(f) + 1] = v


def with_adapter(a, ad):
    a = list(a)
    if '--adapter' in a:
        i = a.index('--adapter'); del a[i:i + 2]
    if ad:
        i = a.index('--model') + 2; a[i:i] = ['--adapter', ad]
    return a


def job(name, argv, out, prio, mem=30.0):
    return {'name': name, 'env': 'tf451', 'cwd': f'{R}/probe3_selector_evidence', 'prio': prio, 'argv': argv, 'out': out,
            'mem_gb': mem, 'mkdir': os.path.dirname(out)}


P5 = [j for j in PX if j['name'] == 'px__p5n_qwenbase_para1__o0'][0]['argv']   # promptx_launch + 12p5n r3, para1
assert P5[2] == L1 and '12p5n_native_perturb_eval_r3.py' in P5


def sweep(tag, ad, style, prio, outdir):
    a = with_adapter(P5, ad); a[2] = L2; setv(a, '--prompt-style', style); setv(a, '--ops', 'none'); setv(a, '--n', '200')
    setv(a, '--offset', '0'); o = f'{R}/results/{outdir}/p5n_ws_{tag}_{style}_tf451fa2_o0n200.json'; setv(a, '--out', o)
    return job(f'ws__p5n_{tag}_{style}__o0', a, o, prio)


FT = [j for j in F3 if j['name'] == 'a3f__qwena05__o0'][0]['argv']   # attn_launch + 12p5n forced, timer1_default


def forced(tag, ad, style, off, prio, outdir):
    a = with_adapter(FT, ad); setv(a, '--prompt-style', style); setv(a, '--ops', 'none,mask_evid,mask_rand')
    setv(a, '--offset', str(off)); setv(a, '--n', '250')
    if style.startswith(('para', 'w')):
        a[2:2] = [L2]
    o = f'{R}/results/{outdir}/p5nF_{tag}_{style}_tf451fa2_o{off}n250.json'; setv(a, '--out', o)
    return job(f'wf__p5nF_{tag}_{style}__o{off}', a, o, prio)


A = [sweep(t, AD[t], s, 9, f'wsweep/{t}') for t in AD for s in WORD]
B = [forced(t, AD[t], s, off, 8, f'wforced/{t}') for t in ('qwenbase', 'sft', 'a05_s0', 'a05_s2') for s in ('para2', 'vchat')
     for off in (0, 250)]
C = [sweep(B9, AD9, s, 8, f'newarm/{B9}') for s in WORD]
for j in PX:   # 14Ln para1/para2, first 200 (promptx_launch, as the other arms)
    if j['name'].startswith(('px__14Ln_a05_s0_para1__', 'px__14Ln_a05_s0_para2__')):
        k = json.loads(json.dumps(j).replace(A05, AD9).replace('/promptx/a05_s0/', f'/newarm/{B9}/')
                       .replace('_px_a05_s0_', f'_px_{B9}_').replace('px__14Ln_a05_s0_', f'b9__14Ln_px_{B9}_'))
        k['prio'] = 8; k['mkdir'] = f'{R}/results/newarm/{B9}'; C.append(k)
for l in open(os.path.join(RB, 'newarm', f'{B8}_jobs.jsonl'), encoding='utf-8'):   # default wording, indep600
    j = json.loads(l)
    if 'indep600' in j['out']:
        k = json.loads(json.dumps(j).replace(AD8, AD9).replace(B8, B9)); k['prio'] = 8; C.append(k)
for j in PX:   # p5n default first 500 (r3, timer1_default) and para2 first 500
    if j['name'].startswith('px__p5n_a05_s0_para1__'):
        for st in ('timer1_default', 'para2'):
            a = list(j['argv']); setv(a, '--adapter', AD9); setv(a, '--prompt-style', st)
            if st == 'timer1_default':
                del a[2]
            off = a[a.index('--offset') + 1]
            o = f'{R}/results/newarm/{B9}/p5n_px_{B9}_{st}_tf451fa2_o{off}n250.json'; setv(a, '--out', o)
            C.append(job(f'b9__p5n_{B9}_{st}__o{off}', a, o, 8))
C += [forced(B9, AD9, 'para2', off, 8, f'newarm/{B9}') for off in (0, 250)]
D = [sweep(B8, AD8, s, 9, f'newarm/{B8}') for s in WORD]
assert (len(A), len(B), len(C), len(D)) == (56, 16, 24, 8), (len(A), len(B), len(C), len(D))
allj = A + B + C + D
assert len({j['out'] for j in allj}) == len(allj) and len({j['name'] for j in allj}) == len(allj)
assert all(AD9 in j['argv'] for j in C) and all(AD8 in j['argv'] for j in D)
assert all(j['argv'][0].endswith('attn_launch.py') for j in allj)
for p, js in [('wsweep_jobs.jsonl', A + B), ('wsweep_b9_jobs.jsonl', C), ('wsweep_b8_jobs.jsonl', D)]:
    with open(os.path.join(HERE, p), 'w', encoding='utf-8', newline='\n') as f:
        for j in js:
            f.write(json.dumps(j) + '\n')
    print(p, len(js))
for j in (A[1], B[0], B[2], C[8], C[12], C[18], C[19], C[22]):
    print(j['name'], '|', ' '.join(j['argv'][1:4]), '|', ' '.join(x for x in j['argv'] if x.startswith((DATASETS_ROOT, f'{R}/results/train', f'{R}/results/grpo'))),
          '| style', j['argv'][j['argv'].index('--prompt-style') + 1], '| ops', j['argv'][j['argv'].index('--ops') + 1] if '--ops' in j['argv'] else None,
          '| n', j['argv'][j['argv'].index('--n') + 1], '|', j['out'].split('/results/')[1])
