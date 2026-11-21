# p5n_capaligned_table.py (11-08): one table over every read_p5n_v2 paired read present (behaviour under evidence
# edits, 12p5n r3, 1,000 q). For each pair: raw DiD (all questions; R1/R2 asked "does the arm lose more when evidence is
# removed"), the normalised-contrast difference [1-R(evid)/R(rand)] and the capability-aligned survival delta (questions
# BOTH answer correctly clean; positive = arm loses more) -- the two estimators that remove the clean-accuracy headroom
# confound (an arm with higher clean R@.5 has more to lose). Values are copied from the reads; nothing is re-estimated.
import glob, hashlib, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
reads = sorted(f for f in glob.glob(os.path.join(HERE, '*', 'read', '*p5n*.json')) if not f.endswith('.stamp.json'))
reads += sorted(f for f in glob.glob(os.path.join(HERE, '*', 'read', 'A7_*.json')) if not f.endswith('.stamp.json'))
rows = {}
for f in reads:
    t = json.load(open(f, encoding='utf-8'))
    if t.get('reader') != 'read_p5n_v2':
        continue
    A = t['all_questions']; cond = lambda s, c: A[s]['conditions'][c]['R@0.5']['est']
    r = {'read': os.path.relpath(f, HERE).replace('\\', '/'), 'read_sha16': sha(f), 'inputs': t['inputs'], 'n': A['n'],
         'clean_R50_base': cond('base', 'none'), 'clean_R50_arm': cond('arm', 'none')}
    for ax in ('mask', 'swap'):
        P = A['paired'][ax]; C = t['capability_aligned'][ax]
        r[ax] = {'DiD': P['DiD_contrast_arm_minus_base'], 'norm_diff': P['normalised_contrast_diff'],
                 'capal_survival_base': C['base_survival'], 'capal_survival_arm': C['arm_survival'],
                 'capal_delta_base_minus_arm': C['delta_base_minus_arm']}
    r['capal_n'] = t['capability_aligned']['n']
    rows[t['label']] = r
f3 = lambda e: f"{e['est']:+.3f} [{e['ci95'][0]:+.3f},{e['ci95'][1]:+.3f}]"
for k, r in rows.items():
    print(f"{k:34s} clean {r['clean_R50_base']:.3f}->{r['clean_R50_arm']:.3f} | mask DiD {f3(r['mask']['DiD'])} norm {f3(r['mask']['norm_diff'])} "
          f"capal {f3(r['mask']['capal_delta_base_minus_arm'])} (n={r['capal_n']}) | swap capal {f3(r['swap']['capal_delta_base_minus_arm'])}")
p = os.path.join(HERE, 'P5N_CAPALIGNED.json'); json.dump(rows, open(p, 'w', encoding='utf-8'), indent=1)
print(p, sha(p))
