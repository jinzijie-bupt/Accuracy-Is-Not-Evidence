# read_a1u.py (11-08): A1u = training-unseen ops (noise, gblur, pscram) on indep600. Merges each cell's 6 shards with
# rebuttal/merge_shards.py and runs the registered reader analysis/read_14Ln_v1.py (unchanged) on three pairs:
#   U02 SFT vs U01 base (tf451) | U04 ours a0.5 vs U01 base (tf451) | U22 ours l4800 vs U07 base (tf5)
# Outputs rebuttal/a1/read_a1u/<pair>.json and A1U_SUMMARY.json (per op: Gamma + CI, n, and the base's own psi level,
# i.e. how much the op removes from the base at all).
import glob, hashlib, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
PULL = os.path.join(ROOT, 'results', 'node7_pull', 'a1u'); MERGED = os.path.join(ROOT, 'results', 'node7_merged', 'a1u')
MERGE = os.path.join(ROOT, 'rebuttal', 'merge_shards.py'); READER = os.path.join(ROOT, 'analysis', 'read_14Ln_v1.py')
OUT = os.path.join(HERE, 'read_a1u'); os.makedirs(OUT, exist_ok=True); os.makedirs(MERGED, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
OPS = 'noise,gblur,pscram'


def merged(cell):
    sh = sorted(f for f in glob.glob(os.path.join(PULL, cell, f'{cell}__o*n100.json')) if not f.endswith(('.meta.json', '.attn.json')))
    if len(sh) != 6:
        return None
    out = os.path.join(MERGED, f'{cell}_o0n600.json')
    subprocess.run([sys.executable, MERGE, out] + sh, check=True, capture_output=True)
    return out


PAIRS = [('A1U_sft451_vs_base451', 'U01_qwen_base_tf451', 'U02_qwen_sft_tf451', '4.51'),
         ('A1U_a05_451_vs_base451', 'U01_qwen_base_tf451', 'U04_qwen_a05_tf451', '4.51'),
         ('A1U_l4800_vs_base5', 'U07_qwen_base_tf5', 'U22_qwen_l4800_tf5', '5.12')]
summ = {'reader': os.path.basename(READER), 'reader_sha16': sha(READER), 'ops': OPS, 'pairs': {}}
for pid, b, a, tf in PAIRS:
    mb, ma = merged(b), merged(a)
    if not (mb and ma):
        summ['pairs'][pid] = 'missing shards'; print(pid, 'missing shards'); continue
    out = os.path.join(OUT, f'{pid}.json')
    r = subprocess.run([sys.executable, READER, '--base', mb, '--arm', ma, '--expect-n', '600', '--ops', OPS, '--machine', 'node7',
                        '--transformers', tf, '--label', pid, '--out', out], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0:
        summ['pairs'][pid] = 'reader rc=%d %s' % (r.returncode, r.stderr[-300:]); print(pid, 'FAIL', r.stderr[-300:]); continue
    summ['pairs'][pid] = {'read': os.path.basename(out), 'read_sha16': sha(out), 'base_sha16': sha(mb), 'arm_sha16': sha(ma)}
    print(pid, 'OK', sha(out))
p = os.path.join(OUT, 'A1U_SUMMARY.json'); json.dump(summ, open(p, 'w', encoding='utf-8'), indent=1)
print(p, sha(p))
