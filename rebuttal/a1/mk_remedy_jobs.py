# mk_remedy_jobs.py (11-11 04Z): follow-ups after B8/B9 came back negative (training-time format / wording mixing raises
# abstention). Clean answers only (12p5n r3 --ops none, first 200), same argv as the wording sweep (mk_wsweep_jobs.py),
# on the four wordings that matter: para1 (control) and the three where our arms abstain (para2, w2, w3).
#  (a) alpha axis, tf5 stack (adapters were trained on tf5; tf4.51 would silently no-op them): Qwen tf5 base,
#      alpha 0.25 (node1), alpha 0.5 (l4800, 4800 questions), alpha 1.0 (node2) + timer1_default        20 jobs node8 prio 10
#      -> does the clean-answer NLL weight control abstention?
#  (b) B6 = B5 recipe from TimeZero (tf451) + TimeZero tf451 base                                           8 jobs node8 prio 10
#  (c) B5s1 (lam 0.01 seed 1) and B5c (lam 0.03 seed 0), node9-only adapters                                 8 jobs node9 prio 12
#      -> is "B5 never abstains" a property of the recipe or of one run?
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__))
R = WORK_ROOT; L2 = f'{R}/setup/promptx2_launch.py'
QW = f'{R}/models/Qwen2.5-VL-7B-Instruct'; TZ = f'{R}/models/TimeZero-Charades-7B'
PX = [json.loads(l) for l in open(os.path.join(HERE, 'promptx_jobs.jsonl'), encoding='utf-8')]
P5 = [j for j in PX if j['name'] == 'px__p5n_qwenbase_para1__o0'][0]['argv']
assert P5[0].endswith('attn_launch.py') and P5[3] == '12p5n_native_perturb_eval_r3.py' and '--adapter' not in P5
W4 = ['para1', 'para2', 'w2', 'w3']


def setv(a, f, v):
    a[a.index(f) + 1] = v


def cell(tag, model, ad, style, env, prio, outdir):
    a = list(P5); a[2] = L2
    if env == 'tf5':
        a = a[2:]
    setv(a, '--model', model)
    if ad:
        i = a.index('--model') + 2; a[i:i] = ['--adapter', ad]
    setv(a, '--prompt-style', style); setv(a, '--ops', 'none'); setv(a, '--n', '200'); setv(a, '--offset', '0')
    o = f'{R}/results/{outdir}/p5n_ws_{tag}_{style}_{"tf5" if env == "tf5" else "tf451fa2"}_o0n200.json'; setv(a, '--out', o)
    return {'name': f'ws__p5n_{tag}_{style}__o0', 'env': env, 'cwd': f'{R}/probe3_selector_evidence', 'prio': prio, 'argv': a,
            'out': o, 'mem_gb': 30.0, 'mkdir': os.path.dirname(o)}


ALPHA = {'qwenbase_tf5': None, 'a025_tf5': (DATASETS_ROOT + '/ds-2/node1__adapter_nat_base_a025_g1_s0'),
         'l4800_tf5': (DATASETS_ROOT + '/ds-1/node3__adapter_nat_base_a05_g1_l4800_s0'),
         'a10_tf5': (DATASETS_ROOT + '/ds-2/node2__adapter_nat_base_a10_g1_s0')}
A = [cell(t, QW, ad, s, 'tf5', 10, f'wsweep/{t}') for t, ad in ALPHA.items() for s in ['timer1_default'] + W4]
B = [cell(t, TZ, ad, s, 'tf451', 10, f'wsweep/{t}') for t, ad in
     {'tzbase451': None, 'B6': f'{R}/results/train/b6_tz_grpo_plus_placebo_lam001_s0'}.items() for s in W4]
C = [cell(t, QW, ad, s, 'tf451', 12, f'wsweep/{t}') for t, ad in
     {'B5s1': f'{R}/results/train/b5_grpo_plus_placebo_lam001_s1',
      'B5c': f'{R}/results/train/b5_grpo_plus_placebo_lam003_s0'}.items() for s in W4]
assert (len(A), len(B), len(C)) == (20, 8, 8)
allj = A + B + C
assert len({j['out'] for j in allj}) == 36 and len({j['name'] for j in allj}) == 36
for p, js in [('remedy_node8_jobs.jsonl', A + B), ('remedy_node9_jobs.jsonl', C)]:
    with open(os.path.join(HERE, p), 'w', encoding='utf-8', newline='\n') as f:
        for j in js:
            f.write(json.dumps(j) + '\n')
    print(p, len(js))
for j in (A[0], A[7], B[5], C[2]):
    print(j['env'], j['name'], '|', ' '.join(j['argv']))
