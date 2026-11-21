# e1_read.py (11-08): E1 = SFT-4800 (B4) checkpoints c800/c1600/c2400/c3200 (+ final c4800 = B4) on the tf5 stack, the
# SFT accuracy-vs-Gamma trajectory for R3 Q1 ("compare with models at similar accuracy").
# Accuracy: paired dR@0.5 vs _base_tf5 on all 3,720 qids (same estimator as p4nat_paired_parse.py: iid bootstrap, B=2000,
# seed 0). Gamma: analysis/read_14Ln_tok_v1.py (unchanged) vs the a5 qwenbase5 tokc shards, first 200 -- the same
# partner as NA_B4 tokc and A5_l4800. Reference rows: l4800 (ours, 4,800 pairs) and B4 final.
import glob, hashlib, json, os, random, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
NA = os.path.join(ROOT, 'results', 'node7_pull', 'newarm'); A5 = os.path.join(ROOT, 'results', 'node7_pull', 'a5')
TOK = os.path.join(ROOT, 'analysis', 'read_14Ln_tok_v1.py'); OUT = os.path.join(HERE, 'read', 'E1'); os.makedirs(OUT, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]


def load(d):
    R = {}
    for f in sorted(glob.glob(os.path.join(NA, d, 'p4nat_*_o*n465.json'))):
        if f.endswith('.meta.json'):
            continue
        for r in json.load(open(f, encoding='utf-8')):
            R[r['qid']] = r
    return R


def hit(r):
    p = r['pred']
    if p is None:
        return 0
    s, e = r['gsA'], r['geA']; inter = max(0, min(p[1], e) - max(p[0], s)); uni = max(p[1], e) - min(p[0], s)
    return int(uni > 0 and inter / uni >= .5)


base = load('_base_tf5'); assert len(base) == 3720
bt = sorted(f for f in glob.glob(os.path.join(A5, 'qwenbase5', '14Ln_qwenbase5_tf5_tokc_o*n100R3.json')) if not f.endswith('.meta.json'))
assert len(bt) == 2, bt
ROWS = [('E1_c800', 'E1_sftckpt_c800', 800), ('E1_c1600', 'E1_sftckpt_c1600', 1600), ('E1_c2400', 'E1_sftckpt_c2400', 2400),
        ('E1_c3200', 'E1_sftckpt_c3200', 3200), ('B4_c4800', 'B4_sft_l4800_s0', 4800), ('l4800_ours', 'ref_l4800', 4800)]
res = {'base_p4nat': '_base_tf5', 'tok_reader_sha16': sha(TOK), 'base_tokc_sha16': [sha(f) for f in bt], 'rows': {}}
for lab, d, pairs in ROWS:
    A = load(d); row = {'pairs_seen': pairs}
    if len(A) == 3720:
        q = sorted(A); dd = [hit(A[k]) - hit(base[k]) for k in q]
        rng = random.Random(0); bs = sorted(sum(rng.choice(dd) for _ in dd) / len(dd) for _ in range(2000))
        row.update({'R50': sum(hit(A[k]) for k in q) / 3720, 'R50_base': sum(hit(base[k]) for k in q) / 3720,
                    'dR50': sum(dd) / len(dd), 'dR50_ci95': [bs[50], bs[1949]], 'parse': sum(A[k]['pred'] is not None for k in q) / 3720})
    else:
        row['p4nat'] = f'incomplete {len(A)}'
    at = sorted(f for f in glob.glob(os.path.join(NA, d, f'14Ln_*tokc_o*n100R3.json')) if not f.endswith('.meta.json'))
    if lab == 'l4800_ours':
        at = sorted(f for f in glob.glob(os.path.join(A5, 'l4800', '14Ln_l4800_tf5_tokc_o*n100R3.json')) if not f.endswith('.meta.json'))
    if len(at) == 2:
        o = os.path.join(OUT, f'{lab}_tokc.json')
        r = subprocess.run([sys.executable, TOK, '--base'] + bt + ['--arm'] + at + ['--expect-n', '200', '--label', lab, '--out', o],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode == 0:
            t = json.load(open(o, encoding='utf-8'))
            m = t['per_op']['mask']['subsets']['all'] if 'per_op' in t else None
            row.update({'tok_read': os.path.basename(o), 'tok_read_sha16': sha(o)})
            if m:
                row.update({'gamma_mask': m['gamma'], 'gamma_mask_ci95': m['gamma_ci95'], 'n_cohort': t['per_op']['mask'].get('n_cohort')})
        else:
            row['tok'] = 'reader rc=%d %s' % (r.returncode, r.stderr[-300:])
    res['rows'][lab] = row
    g = row.get('gamma_mask'); ci = row.get('gamma_mask_ci95') or [float('nan')] * 2
    print(f"{lab:11s} pairs {pairs:5d} R50 {row.get('R50', float('nan')):.3f} dR {row.get('dR50', float('nan')):+.3f} "
          f"[{(row.get('dR50_ci95') or [float('nan')]*2)[0]:+.3f},{(row.get('dR50_ci95') or [float('nan')]*2)[1]:+.3f}] | "
          f"Gamma mask {g if g is not None else float('nan'):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] n={row.get('n_cohort')} {row.get('tok', '')}")
p = os.path.join(OUT, 'E1_SUMMARY.json'); json.dump(res, open(p, 'w', encoding='utf-8'), indent=1)
print(p, sha(p))
