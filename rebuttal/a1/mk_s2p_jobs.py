# S2P (11-08, R2 Q2 "the audit's resolution comes from two pairs of seeds of one model under ONE PROMPT"): the
# Qwen a0.5 seed pair (s0 = local__nat_base_a05_g1_s0, s2 = B1a) and the base, audited under a second template
# (--prompt-style vchat --answer-style vchat, the VideoChat-R1 template the paper already uses for A1-11/12), first 200,
# 14Ln r8, tf451+FA2 (the stack both seeds were trained on). Seed-vs-seed Gamma under this template = a resolution
# estimate from a second prompt; each seed vs base = Gamma under the second template.
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
R = WORK_ROOT; P = f'{R}/probe3_selector_evidence'
ARMS = [('qwenbase', None), ('a05_s0', (DATASETS_ROOT + '/ds-1/local__nat_base_a05_g1_s0')),
        ('a05_s2', f'{R}/results/train/adapter_nat_base_a05_g1_s2')]
C14 = ['--ops', 'mask,swap', '--R', '3', '--thr', '0.5', '--seed', '1000', '--fps', '2.0', '--total-pixels', '2809856',
       '--min-pixels', '12544', '--max-frames', '768', '--max-new-tokens', '128', '--gpu', '0']
jobs = []
for tag, ad in ARMS:
    O = f'{R}/results/s2p/{tag}'
    for off in (0, 100):
        out = f'{O}/14Ln_s2p_{tag}_vchat_tf451fa2_o{off}n100R3.json'
        argv = [f'{R}/setup/attn_launch.py', 'flash_attention_2', '14Ln_native_sentinel_probe_r8.py',
                '--model', f'{R}/models/Qwen2.5-VL-7B-Instruct'] + (['--adapter', ad] if ad else []) + \
               ['--prompt-style', 'vchat', '--answer-style', 'vchat'] + C14 + ['--n', '100', '--offset', str(off), '--out', out]
        jobs.append({'name': f's2p__{tag}__o{off}', 'env': 'tf451', 'cwd': P, 'prio': 12, 'argv': argv, 'out': out,
                     'mem_gb': 30.0, 'mkdir': O})
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 's2p_jobs.jsonl')
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    for j in jobs:
        f.write(json.dumps(j) + '\n')
print(p, len(jobs))
