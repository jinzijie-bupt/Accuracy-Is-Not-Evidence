# read_a1s.py (11-09): reads the A1s jobs (a1s_jobs.jsonl, generator mk_a1s_jobs.py) with the registered reader
# analysis/read_14Ln_v1.py (unchanged), same merge path as read_a1u.py.
#  (1) severity sweep, ops noise,gblur: L2 (sigma 80 / 6) and L3 (120 / 10); every arm is paired with the base at the SAME
#      severity on the same stack: U02 SFT / U04 ours vs U01 base (tf451, L2+L3); U22 l4800 vs U07 base (tf5, L3).
#  (2) seed pair at the default severities, ops noise,gblur,pscram: U05 (ours, seed-2 adapter) vs U01 base, next to the
#      existing U04 (seed 0) vs U01 read -> |Gamma(s0) - Gamma(s2)| per op = op-specific seed resolution.
# Outputs rebuttal/a1/read_a1s/<pair>.json and A1S_SUMMARY.json.
import glob, hashlib, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
PULL = os.path.join(ROOT, 'results', 'node7_pull'); MERGED = os.path.join(ROOT, 'results', 'node7_merged')
MERGE = os.path.join(ROOT, 'rebuttal', 'merge_shards.py'); READER = os.path.join(ROOT, 'analysis', 'read_14Ln_v1.py')
OUT = os.path.join(HERE, 'read_a1s'); os.makedirs(OUT, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]


def merged(exp, cell):
    sh = sorted(f for f in glob.glob(os.path.join(PULL, exp, cell, f'{cell}__o*n100.json')) if not f.endswith(('.meta.json', '.attn.json')))
    if len(sh) != 6:
        return None
    os.makedirs(os.path.join(MERGED, exp), exist_ok=True)
    out = os.path.join(MERGED, exp, f'{cell}_o0n600.json')
    subprocess.run([sys.executable, MERGE, out] + sh, check=True, capture_output=True)
    return out


PAIRS = [('A1S_sft451_L2', 'a1s', 'U01_qwen_base_tf451_L2', 'U02_qwen_sft_tf451_L2', '4.51', 'noise,gblur'),
         ('A1S_a05_451_L2', 'a1s', 'U01_qwen_base_tf451_L2', 'U04_qwen_a05_tf451_L2', '4.51', 'noise,gblur'),
         ('A1S_sft451_L3', 'a1s', 'U01_qwen_base_tf451_L3', 'U02_qwen_sft_tf451_L3', '4.51', 'noise,gblur'),
         ('A1S_a05_451_L3', 'a1s', 'U01_qwen_base_tf451_L3', 'U04_qwen_a05_tf451_L3', '4.51', 'noise,gblur'),
         ('A1S_l4800_L3', 'a1s', 'U07_qwen_base_tf5_L3', 'U22_qwen_l4800_tf5_L3', '5.12', 'noise,gblur'),
         ('A1U_a05s2_451_vs_base451', 'a1u', 'U01_qwen_base_tf451', 'U05_qwen_a05s2_tf451', '4.51', 'noise,gblur,pscram')]
summ = {'reader': os.path.basename(READER), 'reader_sha16': sha(READER), 'pairs': {}}
for pid, exp, b, a, tf, ops in PAIRS:
    mb, ma = merged(exp, b), merged(exp, a)
    if not (mb and ma):
        summ['pairs'][pid] = 'missing shards'; print(pid, 'missing shards'); continue
    out = os.path.join(OUT, f'{pid}.json')
    r = subprocess.run([sys.executable, READER, '--base', mb, '--arm', ma, '--expect-n', '600', '--ops', ops, '--machine', 'node7',
                        '--transformers', tf, '--label', pid, '--out', out], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0:
        summ['pairs'][pid] = 'reader rc=%d %s' % (r.returncode, r.stderr[-300:]); print(pid, 'FAIL', r.stderr[-300:]); continue
    po = json.load(open(out, encoding='utf-8'))['per_op']
    summ['pairs'][pid] = {'read': os.path.basename(out), 'read_sha16': sha(out), 'base_sha16': sha(mb), 'arm_sha16': sha(ma),
                          'per_op': {op: {'gamma': po[op]['gamma'], 'ci95': po[op]['gamma_ci95'], 'n': po[op].get('n_cohort')}
                                     for op in ops.split(',')}}
    print(pid, ' | '.join(f"{op} {d['gamma']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] n={d['n']}"
                          for op, d in summ['pairs'][pid]['per_op'].items()))
s0 = os.path.join(HERE, 'read_a1u', 'A1U_a05_451_vs_base451.json')
if isinstance(summ['pairs'].get('A1U_a05s2_451_vs_base451'), dict) and os.path.exists(s0):
    p0 = json.load(open(s0, encoding='utf-8'))['per_op']; p2 = summ['pairs']['A1U_a05s2_451_vs_base451']['per_op']
    summ['seed_resolution'] = {'s0_read_sha16': sha(s0), **{op: abs(p0[op]['gamma'] - p2[op]['gamma']) for op in p2}}
    print('seed |s0-s2|', summ['seed_resolution'])
p = os.path.join(OUT, 'A1S_SUMMARY.json'); json.dump(summ, open(p, 'w', encoding='utf-8'), indent=1)
print(p, sha(p))
