# mk_node9b_jobs.py (11-08): node9 follow-ups to B5 (IoU-GRPO + placebo, lam 0.01, s0; run on node9, paired with node8 bases).
#  (1) xm_eq: two node8 base jobs re-run verbatim on node9 (argv unchanged, --out redirected) -> byte/record comparison
#      with the node8 products decides whether B5(node9) vs base(node8) pairs are within-machine-equivalent.
#  (2) B5s1 = same recipe, seed 1 (replication); (3) B5c = lam 0.03, seed 0 (dose point). Arm-only eval lists are
#      node9/B5_arm_only_jobs.jsonl with adapter dir + label swapped.
import json, os
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__)); RB = os.path.dirname(HERE)
S = os.path.join(os.environ.get('SETUP_DIR', os.path.join(RB, 'setup')), 'node9')  # local staging folder of job files
R = WORK_ROOT
rd = lambda p: [json.loads(l) for l in open(p, encoding='utf-8')]
def w(p, J):
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        for j in J: f.write(json.dumps(j) + '\n')
    print(p, len(J))
eq = []
for j in rd(os.path.join(HERE, 'B5_grpo_placebo_lam001_s0_jobs.jsonl')):
    if j['name'] == 'na__base_tf451__p4nat_o0': eq.append(j)
for j in rd(os.path.join(RB, 'a1', 'a1_jobs.jsonl')):
    if j['name'] == 'a1__A1-01_qwen_base_t1d_tf451__o0': eq.append(j)
assert len(eq) == 2
out = []
for j in eq:
    k = json.loads(json.dumps(j)); i = k['argv'].index('--out')
    new = f"{R}/results/xm_eq/" + os.path.basename(j['out']); k['argv'][i + 1] = new; k['out'] = new
    k['name'] = 'xmeq__' + j['name']; k['prio'] = 1
    if 'mkdir' in k: k['mkdir'] = f'{R}/results/xm_eq'
    out.append(k)
w(os.path.join(S, 'xm_eq_jobs.jsonl'), out); w(os.path.join(HERE, 'node9_xm_eq_jobs.jsonl'), out)
arm = rd(os.path.join(S, 'B5_arm_only_jobs.jsonl')); assert len(arm) == 20
for lab, ad in [('B5s1_grpo_placebo_lam001_s1', 'b5_grpo_plus_placebo_lam001_s1'), ('B5c_grpo_placebo_lam003_s0', 'b5_grpo_plus_placebo_lam003_s0')]:
    J = []
    for j in arm:
        s = json.dumps(j); assert s.count('b5_grpo_plus_placebo_lam001_s0"') == 1
        k = json.loads(s.replace('b5_grpo_plus_placebo_lam001_s0"', ad + '"').replace('B5_grpo_placebo_lam001_s0', lab))
        assert lab in k['out'] and k['out'] == k['argv'][k['argv'].index('--out') + 1]; J.append(k)
    assert len({k['out'] for k in J}) == 20
    w(os.path.join(S, f'{lab}_jobs.jsonl'), J); w(os.path.join(HERE, f'{lab}_jobs.jsonl'), J)
