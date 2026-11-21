# QVH (11-08, R1 W2b "on QVHighlights clean accuracy is flat or lower", R3 W1 "+0.1 vs SFT +6.8"): the paper's two
# QVHighlights instruments, unchanged, on the rebuttal arms, each arm paired with the base on its own stack:
#   accuracy  12p6n_tlbench_eval_r4.py (85e1cd3f195e6d72), all 1,541 items of TimeLens-Bench qvhighlights-timelens.json
#             (sha16 a7f70291a531eaee, = the paper's file), 4 shards of 386
#   Gamma     14Ln_native_sentinel_probe_r5.py (0d4636d4fd2e8fba), first 200 items, --ops mask,swap --R 3, ONE job per arm
#             (r5 draws placebo frames from one global stream, so the paper ran it unsharded; we keep that)
# Args = chain_node2_qvh.sh / chain_node2_qvh14Ln.sh verbatim except paths. tf451 arms run under attn_launch FA2 like every
# other tf451 job here. usage: mk_qvh_jobs.py [--smoke] -> rebuttal/qvh/qvh_jobs.jsonl (or qvh_smoke_jobs.jsonl)
import json, os, sys
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
R = WORK_ROOT; P = f'{R}/probe3_selector_evidence'; D = f'{R}/data/qvh_tlbench'
ANNO = f'{D}/qvhighlights-timelens.json'; VD = f'{D}/videos'
MQ = f'{R}/models/Qwen2.5-VL-7B-Instruct'; MT = f'{R}/models/TimeZero-Charades-7B'
DS1 = (DATASETS_ROOT + '/ds-1')
ARMS = [  # tag, env, model, adapter
    ('base451', 'tf451', MQ, None),
    ('a05_s0', 'tf451', MQ, f'{DS1}/local__nat_base_a05_g1_s0'),
    ('a05_s2', 'tf451', MQ, f'{R}/results/train/adapter_nat_base_a05_g1_s2'),
    ('sft451', 'tf451', MQ, f'{DS1}/local__nat_sft_b0_s0'),
    ('A2_grpo2fps', 'tf451', MQ, f'{R}/results/grpo/adapter_grpo_native_iou_s0'),
    ('B3_tsonly', 'tf451', MQ, f'{R}/results/train/adapter_nat_base_a05_tsonly_s0'),
    ('base5', 'tf5', MQ, None),
    ('l4800', 'tf5', MQ, f'{DS1}/node3__adapter_nat_base_a05_g1_l4800_s0'),
    ('B4_sft4800', 'tf5', MQ, f'{R}/results/train/adapter_nat_sft_l4800_s0'),
    ('iougrpo16f', 'tf5', MQ, f'{DS1}/local__x0_s0_iou700'),
    ('tzbase', 'tf5', MT, None),
    ('tza05_s0', 'tf5', MT, f'{DS1}/local__nat_tz_a05_g1_s0'),
]
smoke = '--smoke' in sys.argv
jobs = []
for tag, env, model, ad in ARMS:
    if smoke and tag not in ('base451', 'a05_s0', 'base5', 'l4800'):
        continue
    FA = [f'{R}/setup/attn_launch.py', 'flash_attention_2'] if env == 'tf451' else []
    sfx = 'tf451fa2' if env == 'tf451' else 'tf5'
    M = ['--model', model] + (['--adapter', ad] if ad else [])
    O = f'{R}/results/qvh{"_smoke" if smoke else ""}/{tag}'
    shards = [(0, 3)] if smoke else [(o, 386) for o in (0, 386, 772, 1158)]
    for off, n in shards:
        out = f'{O}/p6n_qvh_{tag}_{sfx}_o{off}n{n}.json'
        jobs.append({'name': f'qvh{"s" if smoke else ""}__{tag}__p6n_o{off}', 'env': env, 'cwd': P, 'prio': 10 if smoke else 15,
                     'argv': FA + ['12p6n_tlbench_eval_r4.py'] + M + ['--anno', ANNO, '--video-dir', VD, '--prompt-style', 'timer1_default',
                                                                     '--n', str(n), '--offset', str(off), '--gpu', '0', '--out', out],
                     'out': out, 'mem_gb': 30.0, 'mkdir': O})
    n = 4 if smoke else 200
    out = f'{O}/14Ln_qvh_{tag}_{sfx}_o0n{n}R3.json'
    jobs.append({'name': f'qvh{"s" if smoke else ""}__{tag}__14Ln', 'env': env, 'cwd': P, 'prio': 10 if smoke else 15,
                 'argv': FA + ['14Ln_native_sentinel_probe_r5.py'] + M + ['--anno', ANNO, '--video-dir', VD, '--ext', '.mp4',
                                                                        '--prompt-style', 'timer1_default', '--answer-style', 'timer1',
                                                                        '--n', str(n), '--offset', '0', '--ops', 'mask,swap', '--R', '3',
                                                                        '--gpu', '0', '--out', out],
                 'out': out, 'mem_gb': 34.0, 'mkdir': O})
for j in jobs:
    s = json.dumps(j)
    assert '\\' not in s and ':/' not in s.replace('https:/', ''), j['name']
    assert j['out'] == j['argv'][j['argv'].index('--out') + 1]
assert len({j['name'] for j in jobs}) == len(jobs)
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qvh_smoke_jobs.jsonl' if smoke else 'qvh_jobs.jsonl')
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    for j in jobs:
        f.write(json.dumps(j) + '\n')
print(p, len(jobs))
