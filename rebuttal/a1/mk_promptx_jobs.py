# mk_promptx_jobs.py (11-10, R2 Q2 "... under one prompt"). s2p's vchat template differs from timer1_default in the
# ANSWER FORMAT only (instruction wording is the same), so it moved the scored string, not the prompt. Two lists:
#  (1) 14Ln r8, first 200, instruction wordings para1 / para2 (promptx_launch.py; answer format kept = timer1):
#      base, ours a0.5 s0, ours a0.5 s2 (B1a), SFT, IoU-GRPO 2 fps s0 -> 20 jobs.
#  (2) 12p5n r3 regenerated answers (ops none,mask_evid,mask_rand), first 500: under vchat (base, s0, s2, SFT) and under
#      para1 (base, s0, SFT) -> behaviour does not depend on the likelihood of a fixed string -> 14 jobs.
# --smoke writes promptx_smoke_jobs.jsonl (n=2, one job per instrument per style) instead.
import json, os, sys
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__)); RB = os.path.dirname(HERE)
R = WORK_ROOT; PX = f'{R}/setup/promptx_launch.py'
AD = {'qwenbase': None, 'a05_s0': (DATASETS_ROOT + '/ds-1/local__nat_base_a05_g1_s0'),
      'a05_s2': f'{R}/results/train/adapter_nat_base_a05_g1_s2', 'sft': (DATASETS_ROOT + '/ds-1/local__nat_sft_b0_s0'),
      'a2grpo_s0': f'{R}/results/grpo/adapter_grpo_native_iou_s0'}
SMOKE = '--smoke' in sys.argv
s2p = [json.loads(l) for l in open(os.path.join(HERE, 's2p_jobs.jsonl'), encoding='utf-8') if '__qwenbase__' in l]
a7 = [json.loads(l) for l in open(os.path.join(RB, 'a7', 'a7_jobs.jsonl'), encoding='utf-8') if 'p5n_qwenbase__' in l]
assert len(s2p) == 2 and len(a7) >= 2


def setv(a, flag, v):
    a[a.index(flag) + 1] = v


def with_adapter(a, ad):
    if ad:
        i = a.index('--prompt-style'); a[i:i] = ['--adapter', ad]
    return a


jobs = []
for style in ('para1', 'para2'):
    for tag in ('qwenbase', 'a05_s0', 'a05_s2', 'sft', 'a2grpo_s0'):
        for j in s2p:
            a = list(j['argv']); assert a[2] == '14Ln_native_sentinel_probe_r8.py'
            a = a[:2] + [PX] + a[2:]
            setv(a, '--prompt-style', style); setv(a, '--answer-style', 'timer1'); a = with_adapter(a, AD[tag])
            off = a[a.index('--offset') + 1]
            o = f'{R}/results/promptx/{tag}/14Ln_px_{tag}_{style}_tf451fa2_o{off}n100R3.json'
            if SMOKE:
                if tag != 'a05_s0' or off != '0':
                    continue
                setv(a, '--n', '2'); o = o.replace('/promptx/', '/promptx_smoke/').replace('n100R3', 'n2R3')
            setv(a, '--out', o)
            jobs.append({**j, 'name': f'px__14Ln_{tag}_{style}__o{off}' + ('_smoke' if SMOKE else ''), 'prio': 1 if SMOKE else 7,
                         'argv': a, 'out': o, 'mkdir': os.path.dirname(o)})
for style, tags in (('vchat', ('qwenbase', 'a05_s0', 'a05_s2', 'sft')), ('para1', ('qwenbase', 'a05_s0', 'sft'))):
    for tag in tags:
        for j in a7[:2]:
            a = list(j['argv']); assert a[2] == '12p5n_native_perturb_eval_r3.py'
            if style != 'vchat':
                a = a[:2] + [PX] + a[2:]
            setv(a, '--prompt-style', style); setv(a, '--ops', 'none,mask_evid,mask_rand'); a = with_adapter(a, AD[tag])
            off = a[a.index('--offset') + 1]
            o = f'{R}/results/promptx/{tag}/p5n_px_{tag}_{style}_tf451fa2_o{off}n250.json'
            if SMOKE:
                if tag != 'a05_s0' or off != '0':
                    continue
                setv(a, '--n', '2'); o = o.replace('/promptx/', '/promptx_smoke/').replace('n250', 'n2')
            setv(a, '--out', o)
            jobs.append({**j, 'name': f'px__p5n_{tag}_{style}__o{off}' + ('_smoke' if SMOKE else ''), 'prio': 1 if SMOKE else 7,
                         'argv': a, 'out': o, 'mkdir': os.path.dirname(o)})
p = os.path.join(HERE, 'promptx_smoke_jobs.jsonl' if SMOKE else 'promptx_jobs.jsonl')
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    for j in jobs:
        f.write(json.dumps(j) + '\n')
print(p, len(jobs))
