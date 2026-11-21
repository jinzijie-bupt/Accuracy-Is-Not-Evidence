# r2q12_answered_subset.py (11-08): R2-Q1-2 "For the questions that are still answered with an interval under the mask,
# does the score still hold?"  Gamma (likelihood audit, tokc products, first 200) restricted to the questions on which the
# ARM still returns a parsable interval when the evidence frames are edited (12p5n r3 behaviour product, same qids, same
# stack, same machine).  Registered reader analysis/read_14Ln_tok_v1.py, unchanged, via --fixed-cohort (the reader still
# intersects with the clean-hit cohort of both sides).  Per op o in {mask, swap}: cohort = {q < 200 : arm o_evid_parsed=1}
# ("arm answers"), and {.. and base o_evid_parsed=1} ("both answer"); the unrestricted read is the reference row.
# Output: rebuttal/r2q12/<label>__<op>__<cohort>.json + cohort files + R2Q12_SUMMARY.json
import glob, hashlib, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PULL = os.path.join(ROOT, 'results', 'node7_pull'); MER = os.path.join(ROOT, 'results', 'node7_merged')
TOK = os.path.join(ROOT, 'analysis', 'read_14Ln_tok_v1.py'); OUT = os.path.join(HERE, 'r2q12'); os.makedirs(OUT, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
g = lambda pat: sorted(f for f in glob.glob(os.path.join(PULL, pat)) if not f.endswith(('.meta.json', '.attn.json')))
BASE = {'tf451': (g('a5/qwenbase451/14Ln_qwenbase451_tf451fa2_tokc_o*n100R3.json'),
                  os.path.join(MER, 'a7', 'p5n_qwenbase_timer1_default_tf451fa2_node7_o0n1000.json')),
        'tf5': (g('a5/qwenbase5/14Ln_qwenbase5_tf5_tokc_o*n100R3.json'),
                os.path.join(MER, 'newarm', '_base_tf5', 'p5n_qwenbase_tf5_node7_o0n1000.json'))}
ARMS = [  # label, stack, tokc shards, p5n merged
    ('ours_a05_s0', 'tf451', g('a5/qwena05/14Ln_qwena05_tf451fa2_tokc_o*n100R3.json'),
     os.path.join(MER, 'a7', 'p5n_qwena05_timer1_default_tf451fa2_node7_o0n1000.json')),
    ('ours_a05_s2', 'tf451', g('newarm/B1a_qwen_a05_s2/14Ln_B1a_qwen_a05_s2_tf451fa2_tokc_o*n100R3.json'),
     os.path.join(MER, 'newarm', 'B1a_qwen_a05_s2', 'p5n_B1a_qwen_a05_s2_tf451fa2_node7_o0n1000.json')),
    ('B3_tsonly', 'tf451', g('newarm/B3_tsonly_a05_s0/14Ln_B3_tsonly_a05_s0_tf451fa2_tokc_o*n100R3.json'),
     os.path.join(MER, 'newarm', 'B3_tsonly_a05_s0', 'p5n_B3_tsonly_a05_s0_tf451fa2_node7_o0n1000.json')),
    ('A2_iougrpo', 'tf451', g('newarm/A2_grpo2fps_s0/14Ln_A2_grpo2fps_s0_tf451fa2_tokc_o*n100R3.json'),
     os.path.join(MER, 'newarm', 'A2_grpo2fps_s0', 'p5n_A2_grpo2fps_s0_tf451fa2_node7_o0n1000.json')),
    ('B5_grpo_placebo', 'tf451', g('newarm/B5_grpo_placebo_lam001_s0/14Ln_B5_grpo_placebo_lam001_s0_tf451fa2_tokc_o*n100R3.json'),
     os.path.join(MER, 'newarm', 'B5_grpo_placebo_lam001_s0', 'p5n_B5_grpo_placebo_lam001_s0_tf451fa2_node7_o0n1000.json')),
    ('sft451', 'tf451', g('a5/qwensft/14Ln_qwensft_tf451fa2_tokc_o*n100R3.json'),
     os.path.join(MER, 'a7', 'p5n_qwensft_timer1_default_tf451fa2_node7_o0n1000.json')),
    ('ours_l4800', 'tf5', g('a5/l4800/14Ln_l4800_tf5_tokc_o*n100R3.json'),
     os.path.join(MER, 'newarm', 'ref_l4800', 'p5n_ref_l4800_tf5_node7_o0n1000.json')),
    ('B4_sft4800', 'tf5', g('newarm/B4_sft_l4800_s0/14Ln_B4_sft_l4800_s0_tf5_tokc_o*n100R3.json'),
     os.path.join(MER, 'newarm', 'B4_sft_l4800_s0', 'p5n_B4_sft_l4800_s0_tf5_node7_o0n1000.json')),
]


def parsed(p5, op):
    return {int(r['qid']) for r in json.load(open(p5, encoding='utf-8')) if int(r['qid']) < 200 and int(r.get(f'{op}_evid_parsed') or 0) == 1}


def read(lab, bt, at, op, cohort_file):
    o = os.path.join(OUT, f'{lab}__{op}__{os.path.basename(cohort_file)[:-5] if cohort_file else "all"}.json')
    cmd = [sys.executable, TOK, '--base'] + bt + ['--arm'] + at + ['--expect-n', '200', '--ops', op, '--label', lab, '--out', o]
    if cohort_file:
        cmd += ['--fixed-cohort', cohort_file]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode:
        return {'err': 'rc=%d %s' % (r.returncode, r.stderr[-300:])}
    t = json.load(open(o, encoding='utf-8')); m = t['per_op'][op]
    s = m['subsets']['all']
    return {'gamma': s['gamma'], 'ci95': s['gamma_ci95'], 'n': m.get('n_cohort'), 'read': os.path.basename(o), 'sha16': sha(o)}


summ = {'reader': [os.path.basename(TOK), sha(TOK)], 'rows': {}}
for lab, st, at, p5a in ARMS:
    bt, p5b = BASE[st]
    if len(at) != 2 or len(bt) != 2 or not (os.path.exists(p5a) and os.path.exists(p5b)):
        summ['rows'][lab] = f'missing: tokc arm {len(at)}/2, p5n arm {os.path.exists(p5a)}, p5n base {os.path.exists(p5b)}'
        print(lab, summ['rows'][lab]); continue
    row = {'p5n_arm_sha16': sha(p5a), 'p5n_base_sha16': sha(p5b)}
    for op in ('mask', 'swap'):
        A, B = parsed(p5a, op), parsed(p5b, op)
        cf = {}
        for nm, qs in (('arm_answers', A), ('both_answer', A & B)):
            f = os.path.join(OUT, f'cohort_{lab}_{op}_{nm}.json'); json.dump(sorted(qs), open(f, 'w')); cf[nm] = f
        row[op] = {'n_arm_answers_first200': len(A), 'n_both_answer_first200': len(A & B),
                   'all': read(lab, bt, at, op, None),
                   'arm_answers': read(lab, bt, at, op, cf['arm_answers']),
                   'both_answer': read(lab, bt, at, op, cf['both_answer'])}
        f3 = lambda d: f"{d['gamma']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] n={d['n']}" if 'gamma' in d else d['err'][:80]
        print(f"{lab:16s} {op:4s} all {f3(row[op]['all'])} | arm-answers({len(A)}) {f3(row[op]['arm_answers'])} | both({len(A & B)}) {f3(row[op]['both_answer'])}")
    summ['rows'][lab] = row
p = os.path.join(OUT, 'R2Q12_SUMMARY.json'); json.dump(summ, open(p, 'w', encoding='utf-8'), indent=1)
print(p, sha(p))
