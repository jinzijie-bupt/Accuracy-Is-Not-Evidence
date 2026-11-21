# read_flank.py (11-09): reads the flank jobs (flank_jobs.jsonl from mk_flank_jobs.py; F05 re-run on node9 from
# flank_F05_node9_jobs.jsonl, node8 == node9 byte-identical) with the registered reader analysis/read_14Ln_v1.py (unchanged).
# The edited window is the annotated interval moved one event length left (shiftl1) or right (shiftr1); every arm is paired
# with the base read under the same shift. Next to it: the same pair with the edit on the event itself (A1 indep600 /
# newarm indep600 products, identical instrument and args except --jitter), so each row reads
#   Gamma(event) | Gamma(frames before) | Gamma(frames after), plus the base's own psi in each window.
# Outputs rebuttal/a1/read_flank/<pair>.json and FLANK_SUMMARY.json.
import glob, hashlib, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
PULLS = [os.path.join(ROOT, 'results', 'node7_pull'), os.path.join(ROOT, 'results', 'node9_pull')]
MERGED = os.path.join(ROOT, 'results', 'node7_merged'); MERGE = os.path.join(ROOT, 'rebuttal', 'merge_shards.py')
READER = os.path.join(ROOT, 'analysis', 'read_14Ln_v1.py'); OUT = os.path.join(HERE, 'read_flank'); os.makedirs(OUT, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]


def merged(cell):
    for pull in PULLS:
        sh = sorted(f for f in glob.glob(os.path.join(pull, 'flank', cell, f'{cell}__o*n100.json')) if not f.endswith(('.meta.json', '.attn.json')))
        if len(sh) == 6:
            os.makedirs(os.path.join(MERGED, 'flank'), exist_ok=True)
            out = os.path.join(MERGED, 'flank', f'{cell}_o0n600.json')
            subprocess.run([sys.executable, MERGE, out] + sh, check=True, capture_output=True)
            return out
    return None


a1 = lambda c: os.path.join(MERGED, 'a1', c, f'14Ln_{c}_node7_o0n600.json')
EVENT = {'F02': a1('A1-02_qwen_sft_tf451'), 'F04': a1('A1-04_qwen_a05_tf451'),
         'F05': os.path.join(MERGED, 'newarm', 'B5_grpo_placebo_lam001_s0', '14Ln_indep600_B5_grpo_placebo_lam001_s0_tf451fa2_node7_o0n600R3.json')}
EBASE = a1('A1-01_qwen_base_t1d_tf451')
ARMS = {'F02': 'F02_qwen_sft_tf451', 'F04': 'F04_qwen_a05_tf451', 'F05': 'F05_qwen_b5_tf451'}
LAB = {'F02': 'SFT', 'F04': 'ours_a05_s0', 'F05': 'B5_grpo_placebo'}


def read(lab, b, a):
    out = os.path.join(OUT, f'{lab}.json')
    r = subprocess.run([sys.executable, READER, '--base', b, '--arm', a, '--expect-n', '600', '--ops', 'mask,swap', '--machine', 'node7',
                        '--transformers', '4.51', '--label', lab, '--out', out], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode:
        return 'reader rc=%d %s' % (r.returncode, r.stderr[-300:])
    po = json.load(open(out, encoding='utf-8'))['per_op']
    return {'read_sha16': sha(out), 'base_sha16': sha(b), 'arm_sha16': sha(a),
            **{op: {'gamma': po[op]['gamma'], 'ci95': po[op]['gamma_ci95'], 'n': po[op]['n_cohort'],
                    'base_psi': po[op]['base_on_cohort']['psi'], 'arm_psi': po[op]['arm_on_cohort']['psi']} for op in ('mask', 'swap')}}


summ = {'reader': [os.path.basename(READER), sha(READER)], 'pairs': {}}
for k, arm in ARMS.items():
    row = {'event': read(f'{LAB[k]}__event', EBASE, EVENT[k])}
    for d in ('shiftl1', 'shiftr1'):
        mb, ma = merged(f'F01_qwen_base_tf451_{d}'), merged(f'{arm}_{d}')
        row[d] = read(f'{LAB[k]}__{d}', mb, ma) if (mb and ma) else 'missing shards'
    summ['pairs'][LAB[k]] = row
    f = lambda x, op: f"{x[op]['gamma']:+.3f} [{x[op]['ci95'][0]:+.3f},{x[op]['ci95'][1]:+.3f}] n={x[op]['n']} psi b/a {x[op]['base_psi']:+.3f}/{x[op]['arm_psi']:+.3f}" if isinstance(x, dict) else str(x)[:50]
    for op in ('mask', 'swap'):
        print(f"{LAB[k]:16s} {op:4s} event {f(row['event'], op)} | before {f(row['shiftl1'], op)} | after {f(row['shiftr1'], op)}")
p = os.path.join(OUT, 'FLANK_SUMMARY.json'); json.dump(summ, open(p, 'w', encoding='utf-8'), indent=1)
print(p, sha(p))
