# p4nat paired readout (11-07): per arm vs the base of ITS OWN stack (_base_tf451 or _base_tf5) on the same 3720 qids --
# overall R@.5, parse rate, paired dR with bootstrap CI, dR on both-parsed qids, and the
# share of the gap from arm parse failures where base hit. Writes newarm/read/p4nat_paired_parse.json.
import json, glob, os, random, hashlib, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'node7_pull', 'newarm')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'read', 'p4nat_paired_parse.json')
ARMS = ['ref_qwena05_s0', 'B1a_qwen_a05_s2', 'ref_qwensft', 'ref_qwenmdpo', 'A2_grpo2fps_s0', 'ref_l4800', 'B4_sft_l4800_s0', 'ref_iougrpo16f']
TF5 = {'ref_l4800', 'B4_sft_l4800_s0', 'ref_iougrpo16f'}  # evaluated with the tf5meta instrument -> paired with _base_tf5

def load(arm):
    fs = sorted(glob.glob(os.path.join(ROOT, arm, 'p4nat_*_o*n465.json')))
    R = {}
    for f in fs:
        for r in json.load(open(f, encoding='utf-8')):
            R[r['qid']] = r
    sha = {os.path.basename(f): hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16] for f in fs}
    return R, sha

def hit(r):
    p = r['pred']
    if p is None:
        return 0
    s, e = r['gsA'], r['geA']
    inter = max(0, min(p[1], e) - max(p[0], s)); uni = max(p[1], e) - min(p[0], s)
    return int(uni > 0 and inter / uni >= .5)

BASES = {s: load(s) for s in ('_base_tf451', '_base_tf5')}
for s, (b, _) in BASES.items():
    assert len(b) in (0, 3720), (s, len(b))
out = {'bases': {s: v[1] for s, v in BASES.items()}, 'n_boot': 2000, 'seed': 0, 'arms': {}}
for arm in ARMS:
    bname = '_base_tf5' if arm in TF5 else '_base_tf451'
    base = BASES[bname][0]
    if len(base) != 3720:
        out['arms'][arm] = {'status': f'base {bname} incomplete n={len(base)}'}; continue
    A, sha = load(arm)
    if len(A) != 3720:
        out['arms'][arm] = {'status': f'incomplete n={len(A)}'}; continue
    q = sorted(A)
    d = [hit(A[k]) - hit(base[k]) for k in q]
    both = [k for k in q if A[k]['pred'] is not None and base[k]['pred'] is not None]
    rng = random.Random(0); bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    lost = sum(1 for k in q if A[k]['pred'] is None and hit(base[k]))
    un = [k for k in q if A[k]['pred'] is None]
    out['arms'][arm] = {
        'stack': 'tf5' if arm in TF5 else 'tf451fa2', 'base': bname,
        'R50_arm': sum(hit(A[k]) for k in q) / len(q), 'R50_base': sum(hit(base[k]) for k in q) / len(q),
        'parse_arm': sum(A[k]['pred'] is not None for k in q) / len(q),
        'paired_dR50': sum(d) / len(d), 'ci95': [bs[50], bs[1949]],
        'n_both_parsed': len(both), 'dR50_both_parsed': sum(hit(A[k]) - hit(base[k]) for k in both) / len(both),
        'arm_unparsed_base_hit': lost, 'arm_unparsed_base_hit_frac': lost / len(q),
        # unparsed = abstentions? (no digit in raw); base accuracy on the abstained vs answered subsets
        'n_unparsed': len(un), 'unparsed_no_digit': sum(1 for k in un if not any(ch.isdigit() for ch in A[k]['raw'])),
        'base_R50_on_arm_unparsed': (sum(hit(base[k]) for k in un) / len(un)) if un else None,
        'base_R50_on_arm_answered': sum(hit(base[k]) for k in q if k not in set(un)) / (len(q) - len(un)),
        'arm_R50_on_arm_answered': sum(hit(A[k]) for k in q if k not in set(un)) / (len(q) - len(un)),
        'shards': sha}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(OUT, hashlib.sha256(open(OUT, 'rb').read()).hexdigest()[:16])
for a, v in out['arms'].items():
    if 'paired_dR50' in v:
        print(f"{a:16s} {v['stack']:9s} base {v['R50_base']:.3f} R50 {v['R50_arm']:.3f} parse {v['parse_arm']:.3f} dR {v['paired_dR50']:+.3f} "
              f"[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] both-parsed {v['dR50_both_parsed']:+.3f} (n={v['n_both_parsed']}) "
              f"unparsed&base-hit {v['arm_unparsed_base_hit_frac']:.3f}")
        if v['n_unparsed']:
            print(f"   unparsed n={v['n_unparsed']} no-digit {v['unparsed_no_digit']} | base R50 on arm-abstained "
                  f"{v['base_R50_on_arm_unparsed']:.3f} vs on arm-answered {v['base_R50_on_arm_answered']:.3f} | "
                  f"arm R50 on answered {v['arm_R50_on_arm_answered']:.3f}")
    else:
        print(a, v)
