# run_tneg_read_e1.py (11-10): the E1 c800 row (SFT-4800 checkpoint at 800 pairs, the accuracy-matched SFT of R2 W2 /
# R3 Q1) for the true-negative table, with the same registered reader and the same clean-shard rule as run_tneg_read.py
# (shards o0/o465/o930 of the arm's full p4nat, qids 0..1394). Table quantities as tneg_table.py:
# foreign_no_interval = 1 - foreign parse_rate; mixed50 = (clean R@0.5 + foreign_no_interval) / 2.
# Writes read/E1_sftckpt_c800.json and read/TNEG_E1.json; does not touch TNEG_SUMMARY.json / TNEG_TABLE.json.
import glob, hashlib, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.normpath(os.path.join(HERE, '..', '..', 'results', 'node7_pull'))
READER = os.path.normpath(os.path.join(HERE, '..', '..', 'analysis', 'read_p4nat_tneg_v1.py'))
MERGED = os.path.normpath(os.path.join(HERE, '..', '..', 'results', 'node7_merged', 'tneg_clean'))
OUT = os.path.join(HERE, 'read'); arm = 'E1_sftckpt_c800'
sha16 = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
offs = {int(os.path.basename(f).rsplit('_o', 1)[1].split('n')[0]): f
        for f in glob.glob(os.path.join(P, 'newarm', arm, f'p4nat_{arm}_tf5_o*n465.json'))}
fr = [f for f in glob.glob(os.path.join(P, 'tneg', arm, 'p4nat_*_foreignq_s0_*_o0n1000.json')) if not f.endswith(('.meta.json', '.attn.json'))]
assert {0, 465, 930} <= set(offs) and len(fr) == 1, (sorted(offs), fr)
rows = []
for o in (0, 465, 930):
    rows += json.load(open(offs[o], encoding='utf-8'))
os.makedirs(MERGED, exist_ok=True)
cpath = os.path.join(MERGED, f'{arm}_clean_q0-1394.json'); json.dump(rows, open(cpath, 'w', encoding='utf-8'), ensure_ascii=False)
out = os.path.join(OUT, f'{arm}.json')
subprocess.run([sys.executable, READER, '--label', arm, '--clean', cpath, '--foreign', fr[0], '--out', out], check=True,
               capture_output=True)
d = json.load(open(out, encoding='utf-8')); c, f = d['clean'], d['foreign']
fni = 1 - f['parse_rate']
res = {'reader': [os.path.basename(READER), sha16(READER)], 'read_sha16': sha16(out), 'foreign_sha16': sha16(fr[0]),
       'clean_shards_sha16': {o: sha16(offs[o]) for o in (0, 465, 930)}, 'n': f['n'], 'clean_R50': c['R@0.5'],
       'foreign_no_interval': fni, 'mixed50': (c['R@0.5'] + fni) / 2}
p = os.path.join(OUT, 'TNEG_E1.json'); json.dump(res, open(p, 'w', encoding='utf-8'), indent=1)
print(json.dumps(res), p, sha16(p))
