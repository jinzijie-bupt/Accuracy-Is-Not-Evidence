# True-negative (foreign-query) eval jobs, 11-07: every item of qids 0..999 is asked with a caption from a different
# video (partner map from the evaluated slice o0n1000, seed 0 -- the same slice/seed as the 09-21 tf5 tneg products).
# tf451 arms use 12p4n tf451 r7 (f0094466f5c48a93, = r6 + foreign); tf5 arms use tf5meta r8 (74aa4be8cb1f48d9).
# Clean partners for the same qids are the existing p4nat full products (newarm/, a6/).
# usage: python mk_tneg_jobs.py [smoke|main|B4|B3|B5]  -> writes tneg_<set>_jobs.jsonl next to this file
import json, os, sys
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
R = WORK_ROOT
M = R + '/models/Qwen2.5-VL-7B-Instruct'
DS1 = (DATASETS_ROOT + '/ds-1')
T451 = [R + '/setup/attn_launch.py', 'flash_attention_2', '12p4n_native_acc_eval_tf451_r7.py']
T5 = ['12p4n_native_acc_eval_tf5meta_r8.py']
# name, env, tool, model, adapter, prompt-style
ARMS = {
    'main': [
        ('qwenbase_tf451', 'tf451', T451, M, None, 'timer1_default'),
        ('qwena05_s0', 'tf451', T451, M, DS1 + '/local__nat_base_a05_g1_s0', 'timer1_default'),
        ('B1a_qwen_a05_s2', 'tf451', T451, M, R + '/results/train/adapter_nat_base_a05_g1_s2', 'timer1_default'),
        ('qwensft', 'tf451', T451, M, DS1 + '/local__nat_sft_b0_s0', 'timer1_default'),
        ('qwenmdpo', 'tf451', T451, M, DS1 + '/local__adapter_nat_mdpo_a05_s0', 'timer1_default'),
        ('A2_grpo2fps_s0', 'tf451', T451, M, R + '/results/grpo/adapter_grpo_native_iou_s0', 'timer1_default'),
        ('rarft_native', 'tf451', T451, R + '/models/RA-RFT-Qwen2.5-VL-7B', None, 'rarft_native'),
        ('tr1base_native', 'tf451', T451, R + '/models/Time-R1-7B', None, 'rarft_native'),
        ('qwenbase_tf5', 'tf5', T5, M, None, 'timer1_default'),
        ('l4800', 'tf5', T5, M, DS1 + '/node3__adapter_nat_base_a05_g1_l4800_s0', 'timer1_default'),
        ('iougrpo16f', 'tf5', T5, M, DS1 + '/local__x0_s0_iou700', 'timer1_default'),
    ],
    'B4': [('B4_sft_l4800_s0', 'tf5', T5, M, R + '/results/train/adapter_nat_sft_l4800_s0', 'timer1_default')],
    'B3': [('B3_tsonly_a05_s0', 'tf451', T451, M, R + '/results/train/adapter_nat_base_a05_tsonly_s0', 'timer1_default')],
    'B5': [('B5_grpo_placebo_lam001_s0', 'tf451', T451, M, R + '/results/train/b5_grpo_plus_placebo_lam001_s0', 'timer1_default')],
}


def job(name, env, tool, model, adapter, style, n, off, qop, prio, tag):
    stack = 'tf451fa2' if env == 'tf451' else 'tf5'
    d = f'{R}/results/tneg/{name}'
    out = f'{d}/p4nat_{name}_{tag}_{stack}_o{off}n{n}.json'
    argv = tool + ['--model', model] + (['--adapter', adapter] if adapter else []) + [
        '--prompt-style', style, '--n', str(n), '--offset', str(off), '--query-op', qop, '--query-seed', '0', '--out', out]
    for a in argv:
        assert not a.startswith('C:') and ':/' not in a and '\\' not in a, a  # MSYS mangling guard
    return {'name': f'tneg__{name}__{tag}_o{off}n{n}', 'env': env, 'cwd': R + '/probe3_selector_evidence', 'prio': prio,
            'argv': argv, 'out': out, 'mem_gb': 30.0, 'mkdir': d}


which = sys.argv[1] if len(sys.argv) > 1 else 'main'
if which == 'smoke':
    # landing gate: r7 with defaults must reproduce the frozen r5 base product rows 0..2; foreign n=3 shows the partner path
    jobs = [job('qwenbase_tf451', 'tf451', T451, M, None, 'timer1_default', 3, 0, 'none', 1, 'smokenone'),
            job('qwenbase_tf451', 'tf451', T451, M, None, 'timer1_default', 6, 0, 'foreign', 1, 'smokeforeign')]  # qids 0-3 share one video: n=3 has no partner (guard)
else:
    jobs = [job(*a, 1000, 0, 'foreign', 8, 'foreignq_s0') for a in ARMS[which]]
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'tneg_{which}_jobs.jsonl')
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    for j in jobs:
        f.write(json.dumps(j, ensure_ascii=False) + '\n')
print(p, len(jobs))
