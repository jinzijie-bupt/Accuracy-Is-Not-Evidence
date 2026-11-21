# read_b1b_s2p.py (11-10): R2 Q2 resolution rows, with the registered reader analysis/read_14Ln_v1.py (unchanged).
#  B1b: TimeZero a0.5 seed 1 (mk_b1b_jobs.py), indep600, paired with its base A1-09 and directly with seed 0 (A1-13).
#  s2p: second prompt template (vchat) on the first 200 questions (Qwen tf451 FA2): seed 0 and seed 2 of our Qwen
#       model vs the base under the same template, and seed 2 vs seed 0 directly (seed resolution under the 2nd prompt).
# Merged shards go to results/node7_merged/<same rel dir>/; outputs rebuttal/newarm/read_b1b_s2p/<pair>.json + SUMMARY.
import glob, hashlib, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
PULL = os.path.join(ROOT, 'results', 'node7_pull'); MERGED = os.path.join(ROOT, 'results', 'node7_merged')
MERGE = os.path.join(ROOT, 'rebuttal', 'merge_shards.py'); READER = os.path.join(ROOT, 'analysis', 'read_14Ln_v1.py')
OUT = os.path.join(HERE, 'read_b1b_s2p'); os.makedirs(OUT, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]


def merged(reldir, stem, nshard, total, R=''):
    sh = sorted(f for f in glob.glob(os.path.join(PULL, reldir, f'{stem}_o*n100{R}.json')))
    if len(sh) != nshard:
        return None
    os.makedirs(os.path.join(MERGED, reldir), exist_ok=True)
    out = os.path.join(MERGED, reldir, f'{stem}_node7_o0n{total}{R}.json')
    subprocess.run([sys.executable, MERGE, out] + sh, check=True, capture_output=True)
    return out


A1 = lambda c: os.path.join(MERGED, 'a1', c, f'14Ln_{c}_node7_o0n600.json')
B1B = merged('newarm/B1b_tz_a05_s1', '14Ln_indep600_B1b_tz_a05_s1_tf5', 6, 600)
S2P = {k: merged(f's2p/{k}', f'14Ln_s2p_{k}_vchat_tf451fa2', 2, 200, 'R3') for k in ('qwenbase', 'a05_s0', 'a05_s2')}
PAIRS = [  # label, base, arm, n, transformers
    ('B1b_tz_s1_vs_tzbase', A1('A1-09_tz_base_t1d_tf5'), B1B, 600, '5.12'),
    ('B1b_tz_s1_vs_tz_s0', A1('A1-13_tz_a05_tf5'), B1B, 600, '5.12'),
    ('A1-13_tz_s0_vs_tzbase', A1('A1-09_tz_base_t1d_tf5'), A1('A1-13_tz_a05_tf5'), 600, '5.12'),
    ('s2p_a05_s0_vs_base', S2P['qwenbase'], S2P['a05_s0'], 200, '4.51'),
    ('s2p_a05_s2_vs_base', S2P['qwenbase'], S2P['a05_s2'], 200, '4.51'),
    ('s2p_a05_s2_vs_s0', S2P['a05_s0'], S2P['a05_s2'], 200, '4.51'),
]
summ = {'reader': [os.path.basename(READER), sha(READER)], 'script_sha16': sha(os.path.abspath(__file__)), 'pairs': {}}
for lab, b, a, n, tf in PAIRS:
    if not (b and a and os.path.exists(b) and os.path.exists(a)):
        summ['pairs'][lab] = 'missing'; print(lab, 'missing'); continue
    out = os.path.join(OUT, f'{lab}.json')
    r = subprocess.run([sys.executable, READER, '--base', b, '--arm', a, '--expect-n', str(n), '--ops', 'mask,swap',
                        '--machine', 'node7', '--transformers', tf, '--label', lab, '--out', out],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode:
        summ['pairs'][lab] = f'reader rc={r.returncode} {r.stderr[-300:]}'; print(lab, summ['pairs'][lab]); continue
    po = json.load(open(out, encoding='utf-8'))['per_op']
    summ['pairs'][lab] = {'read_sha16': sha(out), 'base_sha16': sha(b), 'arm_sha16': sha(a),
                          **{op: {'gamma': po[op]['gamma'], 'ci95': po[op]['gamma_ci95'], 'n': po[op]['n_cohort']}
                             for op in ('mask', 'swap')}}
    x = summ['pairs'][lab]
    print(f"{lab:24s} " + ' | '.join(f"{op} {x[op]['gamma']:+.3f} [{x[op]['ci95'][0]:+.3f},{x[op]['ci95'][1]:+.3f}] n={x[op]['n']}" for op in ('mask', 'swap')))
p = os.path.join(OUT, 'B1B_S2P_SUMMARY.json'); json.dump(summ, open(p, 'w', encoding='utf-8'), indent=1)
print(p, sha(p))
