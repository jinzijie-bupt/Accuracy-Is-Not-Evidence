# mk_b6_jobs.py (11-09): B6 = IoU-GRPO + placebo term (lam 0.01, seed 0; = B5 recipe) started from TimeZero-Charades-7B
# instead of Qwen2.5-VL-7B-Instruct. R2 W2: "applied on top of the strongest public checkpoint it is negative in domain".
# Trained and evaluated on tf451+FA2 like B5, so it needs a TimeZero base on tf451 (all earlier TimeZero products are tf5):
#   base_jobs  (prio 10, run now): tokc 2, indep600 6, p4nat 8, p5n 4, QVH 5, tneg 1   -- copied from the Qwen tf451 base
#              jobs (a5 qwenbase451 / A1-01 / na__base_tf451 / a7 qwenbase / qvh base451 / tneg qwenbase_tf451) with the
#              model path and labels swapped
#   arm_jobs   (hooked on B6 .done): the B5 arm-only list (20) + QVH B5 (5) + tneg B5 (1), model/adapter/labels swapped;
#              tokc arm jobs wait for and pair with the tzbase451 tokc shards.
import json, os
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__)); RB = os.path.dirname(HERE)
SK = os.environ.get('SETUP_DIR', os.path.join(RB, 'setup'))  # local staging folder of job files pushed to the GPU machine
R = WORK_ROOT; MQ = f'{R}/models/Qwen2.5-VL-7B-Instruct'; MT = f'{R}/models/TimeZero-Charades-7B'
rd = lambda p: [json.loads(l) for l in open(p, encoding='utf-8')]


def sw(j, reps, name, prio):
    s = json.dumps(j)
    for a, b in reps:
        assert a in s, (j['name'], a)
        s = s.replace(a, b)
    k = json.loads(s); k['name'] = name; k['prio'] = prio
    assert MQ not in s and MT in k['argv'] and k['out'] == k['argv'][k['argv'].index('--out') + 1], name
    assert 'qwenbase' not in k['out'] and 'B5' not in k['out'] and 'A1-01' not in k['out'], k['out']
    return k


base = []
for j in rd(os.path.join(RB, 'a5', 'a5_jobs.jsonl')):
    if j['name'].startswith('a5__qwenbase451__'):
        base.append(sw(j, [(MQ, MT), ('qwenbase451', 'tzbase451')], j['name'].replace('qwenbase451', 'tzbase451'), 10))
for j in rd(os.path.join(RB, 'a1', 'a1_jobs.jsonl')):
    if j['name'].startswith('a1__A1-01_qwen_base_t1d_tf451__'):
        base.append(sw(j, [(MQ, MT), ('A1-01_qwen_base_t1d_tf451', 'TZ451_tz_base_t1d_tf451')],
                       j['name'].replace('A1-01_qwen_base_t1d_tf451', 'TZ451_tz_base_t1d_tf451'), 10))
seen = set()
for j in rd(os.path.join(HERE, 'B5_grpo_placebo_lam001_s0_jobs.jsonl')):
    if j['name'].startswith('na__base_tf451__p4nat_') and j['name'] not in seen:
        seen.add(j['name'])
        base.append(sw(j, [(MQ, MT), ('_base_tf451/p4nat_qwenbase_tf451fa2', '_base_tz_tf451/p4nat_tzbase_tf451fa2')],
                       j['name'].replace('na__base_tf451__', 'na__base_tz_tf451__'), 10))
for j in rd(os.path.join(RB, 'a7', 'a7_jobs.jsonl')):
    if j['name'].startswith('a7__p5n_qwenbase__'):
        base.append(sw(j, [(MQ, MT), ('results/a7/p5n_qwenbase_', 'results/newarm/_base_tz_tf451/p5n_tzbase_')],
                       j['name'].replace('a7__p5n_qwenbase__', 'na__base_tz_tf451__p5n_'), 10))
for j in rd(os.path.join(RB, 'qvh', 'qvh_jobs.jsonl')):
    if j['name'].startswith('qvh__base451__'):
        base.append(sw(j, [(MQ, MT), ('base451', 'tzbase451')], j['name'].replace('base451', 'tzbase451'), 10))
for j in rd(os.path.join(RB, 'tneg', 'tneg_main_jobs.jsonl')):
    if j['name'].startswith('tneg__qwenbase_tf451__foreignq_s0'):
        base.append(sw(j, [(MQ, MT), ('qwenbase_tf451', 'tzbase_tf451')], j['name'].replace('qwenbase_tf451', 'tzbase_tf451'), 10))
assert len(base) == 2 + 6 + 8 + 4 + 5 + 1, len(base)

AD5 = f'{R}/results/train/b5_grpo_plus_placebo_lam001_s0'; AD6 = f'{R}/results/train/b6_tz_grpo_plus_placebo_lam001_s0'
arm = []
for j in rd(os.path.join(SK, 'node9', 'B5_arm_only_jobs.jsonl')):
    reps = [(MQ, MT), (AD5 + '"', AD6 + '"'), ('B5_grpo_placebo_lam001_s0', 'B6_tzgrpo_placebo_lam001_s0')]
    if 'tokc' in j['name']:
        reps.append(('a5/qwenbase451/14Ln_qwenbase451_', 'a5/tzbase451/14Ln_tzbase451_'))
    arm.append(sw(j, reps, j['name'].replace('B5_grpo_placebo_lam001_s0', 'B6_tzgrpo_placebo_lam001_s0'), 11))
for j in rd(os.path.join(RB, 'qvh', 'qvh_B5_jobs.jsonl')):
    arm.append(sw(j, [(MQ, MT), (AD5 + '"', AD6 + '"'), ('B5_grpo_placebo', 'B6_tzgrpo_placebo')],
                  j['name'].replace('B5_grpo_placebo', 'B6_tzgrpo_placebo'), 11))
for j in rd(os.path.join(RB, 'tneg', 'tneg_B5_jobs.jsonl')):
    arm.append(sw(j, [(MQ, MT), (AD5 + '"', AD6 + '"'), ('B5_grpo_placebo_lam001_s0', 'B6_tzgrpo_placebo_lam001_s0')],
                  j['name'].replace('B5_grpo_placebo_lam001_s0', 'B6_tzgrpo_placebo_lam001_s0'), 11))
assert len(arm) == 26, len(arm)
for J in (base, arm):
    assert len({k['name'] for k in J}) == len(J) and len({k['out'] for k in J}) == len(J)
for k in arm:
    if 'tokc' in k['name']:
        assert 'tzbase451' in json.dumps(k) and 'qwenbase451' not in json.dumps(k)
    assert AD6 in k['argv']
for fn, J in (('B6_tz_base451_jobs.jsonl', base), ('B6_tzgrpo_placebo_lam001_s0_jobs.jsonl', arm)):
    with open(os.path.join(HERE, fn), 'w', encoding='utf-8', newline='\n') as f:
        for k in J:
            f.write(json.dumps(k) + '\n')
    print(fn, len(J))
for k in base[:1] + base[8:9] + arm[:1]:
    print(k['name'], '|', ' '.join(k['argv'])[:260], '\n   ->', k['out'])
