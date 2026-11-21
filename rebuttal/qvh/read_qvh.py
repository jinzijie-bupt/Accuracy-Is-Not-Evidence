# read_qvh.py (11-08): QVHighlights (TimeLens-Bench qvhighlights-timelens.json, 1,541 items) for the rebuttal arms,
# each paired with the base on its own stack (R1 W2b, R3 W1). Registered readers only, unchanged:
#   accuracy  analysis/read_p4nat_selective_v1.py --pair BASE ARM (marginal R@0.5 delta, video-cluster bootstrap B=4000;
#             unparsed = IoU 0 for the marginal numbers; also conditional/coverage)  on 4 merged shards (n=1541)
#   Gamma     analysis/read_14Ln_v1.py (14Ln r5 products, first 200, mask+swap, unsharded)
# Products: results/node7_pull/qvh/<tag>/ (B5 from node9, copied there). Merged shards -> results/node7_merged/qvh/.
# Output: rebuttal/qvh/read/QVH_<arm>_vs_<base>_{acc,gamma}.json + QVH_SUMMARY.json
import glob, hashlib, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
PULL = os.path.join(ROOT, 'results', 'node7_pull', 'qvh'); MERGED = os.path.join(ROOT, 'results', 'node7_merged', 'qvh')
MERGE = os.path.join(ROOT, 'rebuttal', 'merge_shards.py')
SEL = os.path.join(ROOT, 'analysis', 'read_p4nat_selective_v1.py'); G14 = os.path.join(ROOT, 'analysis', 'read_14Ln_v1.py')
OUT = os.path.join(HERE, 'read'); os.makedirs(OUT, exist_ok=True); os.makedirs(MERGED, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
SFX = {'tf451': 'tf451fa2', 'tf5': 'tf5'}; TF = {'tf451': '4.51', 'tf5': '5.12'}
PAIRS = [  # arm, base, stack
    ('a05_s0', 'base451', 'tf451'), ('a05_s2', 'base451', 'tf451'), ('sft451', 'base451', 'tf451'),
    ('A2_grpo2fps', 'base451', 'tf451'), ('B3_tsonly', 'base451', 'tf451'), ('B5_grpo_placebo', 'base451', 'tf451'),
    ('l4800', 'base5', 'tf5'), ('B4_sft4800', 'base5', 'tf5'), ('iougrpo16f', 'base5', 'tf5'), ('E1_c800', 'base5', 'tf5'),
    ('tza05_s0', 'tzbase', 'tf5')]


def p6n(tag, st):
    sh = sorted(f for f in glob.glob(os.path.join(PULL, tag, f'p6n_qvh_{tag}_{SFX[st]}_o*n386.json')))
    if len(sh) != 4:
        return None, f'{len(sh)}/4 shards'
    out = os.path.join(MERGED, f'p6n_qvh_{tag}_{SFX[st]}_o0n1541.json')
    r = subprocess.run([sys.executable, MERGE, out] + sh, capture_output=True, text=True)
    if r.returncode:
        return None, 'merge rc=%d %s' % (r.returncode, r.stderr[-200:])
    n = len(json.load(open(out, encoding='utf-8')))
    return (out, None) if n == 1541 else (None, f'merged n={n}')


def g14(tag, st):
    f = os.path.join(PULL, tag, f'14Ln_qvh_{tag}_{SFX[st]}_o0n200R3.json')
    return f if os.path.exists(f) else None


summ = {'readers': {'acc': [os.path.basename(SEL), sha(SEL)], 'gamma': [os.path.basename(G14), sha(G14)]}, 'pairs': {}}
for arm, base, st in PAIRS:
    pid = f'QVH_{arm}_vs_{base}'; row = {'stack': st}
    ma, ea = p6n(arm, st); mb, eb = p6n(base, st)
    if ma and mb:
        o = os.path.join(OUT, pid + '_acc.json')
        r = subprocess.run([sys.executable, SEL, '--label', pid, '--pair', mb, ma, '--out', o], capture_output=True, text=True,
                           encoding='utf-8', errors='replace')
        if r.returncode == 0:
            t = json.load(open(o, encoding='utf-8')); pt = t['point']; bm = t['delta_marginal_boot']
            row['acc'] = {'n': t['n'], 'R50_base': pt['ref_marginal_R@0.5'], 'R50_arm': pt['arm_marginal_R@0.5'],
                          'dR50': pt['delta_marginal'], 'dR50_ci95': bm['ci95'], 'p': bm['p_two_sided'],
                          'mIoU_base': t['ref_single']['marginal']['mIoU'], 'mIoU_arm': t['arm_single']['marginal']['mIoU'],
                          'coverage_base': t['coverage_ref'], 'coverage_arm': t['coverage_arm'],
                          'read': os.path.basename(o), 'read_sha16': sha(o), 'base_sha16': sha(mb), 'arm_sha16': sha(ma)}
        else:
            row['acc'] = 'reader rc=%d %s' % (r.returncode, r.stderr[-300:])
    else:
        row['acc'] = f'missing: arm {ea or "ok"} / base {eb or "ok"}'
    ga, gb = g14(arm, st), g14(base, st)
    if ga and gb:
        o = os.path.join(OUT, pid + '_gamma.json')
        r = subprocess.run([sys.executable, G14, '--base', gb, '--arm', ga, '--expect-n', '200', '--ops', 'mask,swap',
                            '--machine', 'node7', '--transformers', TF[st], '--label', pid, '--out', o],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        row['gamma'] = {'read': os.path.basename(o), 'read_sha16': sha(o), 'base_sha16': sha(gb), 'arm_sha16': sha(ga)} \
            if r.returncode == 0 else 'reader rc=%d %s' % (r.returncode, r.stderr[-300:])
    else:
        row['gamma'] = f'missing: arm {"ok" if ga else "absent"} / base {"ok" if gb else "absent"}'
    summ['pairs'][pid] = row
    a = row['acc']
    print(f"{pid:32s}", f"R50 {a['R50_base']:.3f}->{a['R50_arm']:.3f} dR {a['dR50']:+.3f} [{a['dR50_ci95'][0]:+.3f},{a['dR50_ci95'][1]:+.3f}]"
          if isinstance(a, dict) else a, '|', 'gamma ' + (row['gamma']['read_sha16'] if isinstance(row['gamma'], dict) else row['gamma']))
p = os.path.join(OUT, 'QVH_SUMMARY.json'); json.dump(summ, open(p, 'w', encoding='utf-8'), indent=1)
print(p, sha(p))
