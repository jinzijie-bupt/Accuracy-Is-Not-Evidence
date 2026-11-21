# read_template_axis.py (11-10): R2 Q2 "... under one prompt". Every read on the prompt / answer-format axis, with the
# registered readers (analysis/read_14Ln_v1.py, analysis/read_p5n_v2.py, unchanged). Inputs from results/node7_pull
# (s2p, s2p2 ctrl, promptx, fmt3, newarm/B8...). Missing cells are reported as 'missing', never filled.
#  14Ln, first 200 (Gamma, token mean; plus sequence sum and clean log-prob of the scored string):
#    second answer format (vchat): base vs ours s0 / s2 / SFT / mDPO / IoU-GRPO / B3 / B8; s2 vs s0
#    instruction wordings para1, para2 (timer1 format): base vs ours s0 / s2 / SFT / IoU-GRPO / B8(para1); s2 vs s0
#    held-out TimeLens format: base vs ours s0 / B3 / B8   (11-10 add: B3 under para1/para2/TimeLens, B8 under para2,
#    p5n under para2)
#  p5n regenerated answers, first 500 (ops none, mask_evid, mask_rand): base vs s0 / s2 / SFT under vchat, vs s0 / SFT
#    under para1 -> behavioural evidence-vs-placebo difference independent of a scored string.
# Outputs rebuttal/a1/read_template_axis/<label>.json and TEMPLATE_AXIS_SUMMARY.json.
import glob, hashlib, json, os, subprocess, sys
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.normpath(os.path.join(H, '..', '..'))
PULL = os.path.join(ROOT, 'results', 'node7_pull'); MERGED = os.path.join(ROOT, 'results', 'node7_merged', 'template_axis')
MERGE = os.path.join(ROOT, 'rebuttal', 'merge_shards.py')
R14 = os.path.join(ROOT, 'analysis', 'read_14Ln_v1.py'); P5 = os.path.join(ROOT, 'analysis', 'read_p5n_v2.py')
OUT = os.path.join(H, 'read_template_axis'); os.makedirs(OUT, exist_ok=True); os.makedirs(MERGED, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
B8 = 'B8_qwen_a05_mixfmt_s0'


def merged(pattern, nshard, tag):
    sh = sorted(f for f in glob.glob(os.path.join(PULL, pattern)) if not f.endswith(('.meta.json', '.attn.json', '.partial.jsonl')))
    if len(sh) != nshard:
        return None
    out = os.path.join(MERGED, f'{tag}.json')
    subprocess.run([sys.executable, MERGE, out] + sh, check=True, capture_output=True)
    return out


C = {}
for t in ('qwenbase', 'a05_s0', 'a05_s2', 'sft', 'mdpo', 'a2grpo_s0', 'b3tsonly', 'B5', 'B5s1'):   # B5/B5s1: node9, 11-11
    C[('vchat', t)] = merged(f's2p/{t}/14Ln_s2p_{t}_vchat_tf451fa2_o*n100R3.json', 2, f'vchat_{t}')
C[('vchat', 'B8')] = merged(f'newarm/{B8}/14Ln_s2p_{B8}_vchat_tf451fa2_o*n100R3.json', 2, 'vchat_B8')  # name from mk_b8 (s2p-like)
if C[('vchat', 'B8')] is None:
    C[('vchat', 'B8')] = merged(f'newarm/{B8}/14Ln_vchat_{B8}_tf451fa2_o*n100R3.json', 2, 'vchat_B8')
for st in ('para1', 'para2'):
    for t in ('qwenbase', 'a05_s0', 'a05_s2', 'sft', 'a2grpo_s0', 'b3tsonly'):
        C[(st, t)] = merged(f'promptx/{t}/14Ln_px_{t}_{st}_tf451fa2_o*n100R3.json', 2, f'{st}_{t}')
C[('para1', 'B8')] = merged(f'newarm/{B8}/14Ln_px_{B8}_para1_tf451fa2_o*n100R3.json', 2, 'para1_B8')
C[('para2', 'B8')] = merged(f'newarm/{B8}/14Ln_px_{B8}_para2_tf451fa2_o*n100R3.json', 2, 'para2_B8')
for st in ('para1', 'para2'):   # 11-11: B5 (IoU-GRPO + our terms, s0) and B5s1, node9 products copied into node7_pull
    for t in ('B5', 'B5s1'):
        C[(st, t)] = merged(f'promptx/{t}/14Ln_px_{t}_{st}_tf451fa2_o*n100R3.json', 2, f'{st}_{t}')
for t in ('qwenbase', 'a05_s0', 'b3tsonly'):
    C[('timelens', t)] = merged(f'fmt3/{t}/14Ln_timelens_{t}_tf451fa2_o*n100R3.json', 2, f'timelens_{t}')
C[('timelens', 'B8')] = merged(f'newarm/{B8}/14Ln_timelens_{B8}_tf451fa2_o*n100R3.json', 2, 'timelens_B8')

PAIRS = [(st, 'qwenbase', a) for st, arms in [('vchat', ('a05_s0', 'a05_s2', 'sft', 'mdpo', 'a2grpo_s0', 'b3tsonly', 'B8')),
                                              ('para1', ('a05_s0', 'a05_s2', 'sft', 'a2grpo_s0', 'b3tsonly', 'B8')),
                                              ('para2', ('a05_s0', 'a05_s2', 'sft', 'a2grpo_s0', 'b3tsonly', 'B8')),
                                              ('timelens', ('a05_s0', 'b3tsonly', 'B8'))] for a in arms]
PAIRS += [(st, 'a05_s0', 'a05_s2') for st in ('vchat', 'para1', 'para2')]
PAIRS += [(st, 'qwenbase', t) for st in ('para1', 'para2', 'vchat') for t in ('B5', 'B5s1')] + [(st, 'B5', 'B5s1') for st in ('para1', 'para2', 'vchat')]
summ = {'readers': {'R14': sha(R14), 'P5': sha(P5)}, 'script_sha16': sha(os.path.abspath(__file__)), 'r14': {}, 'p5n': {}}
for st, b, a in PAIRS:
    lab = f'{st}__{a}_vs_{b}'; bp, ap = C.get((st, b)), C.get((st, a))
    if not (bp and ap):
        summ['r14'][lab] = 'missing'; continue
    out = os.path.join(OUT, f'{lab}.json')
    r = subprocess.run([sys.executable, R14, '--base', bp, '--arm', ap, '--expect-n', '200', '--ops', 'mask,swap', '--machine', 'node7',
                        '--transformers', '4.51', '--label', lab, '--out', out], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode:
        summ['r14'][lab] = f'reader rc={r.returncode} {r.stderr[-200:]}'; continue
    d = json.load(open(out, encoding='utf-8')); po = d['per_op']['mask']; bo, ao = po['base_on_cohort'], po['arm_on_cohort']
    rows_b = {x['qid']: x for x in json.load(open(bp, encoding='utf-8'))}; rows_a = {x['qid']: x for x in json.load(open(ap, encoding='utf-8'))}
    lp = lambda R: sum(x['lp_clean_sum'] for x in R.values() if x.get('lp_clean_sum') is not None) / max(1, sum(1 for x in R.values() if x.get('lp_clean_sum') is not None))
    refuse = lambda R: sum(1 for x in R.values() if not x.get('clean_parsed')) / len(R)
    summ['r14'][lab] = {'read_sha16': sha(out), 'base_sha16': sha(bp), 'arm_sha16': sha(ap), 'n_cohort': po['n_cohort'],
                        'gamma': po['gamma'], 'ci95': po['gamma_ci95'], 'gamma_seqsum': ao['psi_seqsum'] - bo['psi_seqsum'],
                        'swap_gamma': d['per_op']['swap']['gamma'], 'clean_R50_base': d.get('clean_R50_base'), 'clean_R50_arm': d.get('clean_R50_arm'),
                        'clean_lp_sum_mean_base': lp(rows_b), 'clean_lp_sum_mean_arm': lp(rows_a),
                        'no_interval_base': refuse(rows_b), 'no_interval_arm': refuse(rows_a)}

P = {}
for st, tags in (('vchat', ('qwenbase', 'a05_s0', 'a05_s2', 'sft')), ('para1', ('qwenbase', 'a05_s0', 'sft')),
                 ('para2', ('qwenbase', 'a05_s0', 'a05_s2', 'sft', 'B5'))):
    for t in tags:
        P[(st, t)] = merged(f'promptx/{t}/p5n_px_{t}_{st}_tf451fa2_o*n250.json', 2, f'p5n_{st}_{t}')
for st, tags in (('vchat', ('a05_s0', 'a05_s2', 'sft')), ('para1', ('a05_s0', 'sft')), ('para2', ('a05_s0', 'a05_s2', 'sft', 'B5'))):
    for t in tags:
        lab = f'p5n_{st}__{t}_vs_qwenbase'; bp, ap = P.get((st, 'qwenbase')), P.get((st, t))
        if not (bp and ap):
            summ['p5n'][lab] = 'missing'; continue
        out = os.path.join(OUT, f'{lab}.json')
        r = subprocess.run([sys.executable, P5, '--pair', bp, ap, '--ops', 'none,mask_evid,mask_rand', '--label', lab, '--out', out],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode:
            summ['p5n'][lab] = f'reader rc={r.returncode} {r.stderr[-200:]}'; continue
        d = json.load(open(out, encoding='utf-8'))
        summ['p5n'][lab] = {'read_sha16': sha(out), 'all_questions_paired': d['all_questions'].get('paired'),
                            'capability_aligned_mask': d.get('capability_aligned', {}).get('mask'), 'base_sha16': sha(bp), 'arm_sha16': sha(ap),
                            'base_split': {k: d['all_questions']['base']['conditions'][k]['split'] for k in ('none', 'mask_evid', 'mask_rand')},
                            'arm_split': {k: d['all_questions']['arm']['conditions'][k]['split'] for k in ('none', 'mask_evid', 'mask_rand')}}
p = os.path.join(OUT, 'TEMPLATE_AXIS_SUMMARY.json'); json.dump(summ, open(p, 'w', encoding='utf-8'), indent=1)
for lab, v in summ['r14'].items():
    print(f"{lab:34s}", v if isinstance(v, str) else f"G {v['gamma']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] seq {v['gamma_seqsum']:+.2f} n={v['n_cohort']} R50 {v['clean_R50_base']:.3f}/{v['clean_R50_arm']:.3f} lp {v['clean_lp_sum_mean_base']:.1f}/{v['clean_lp_sum_mean_arm']:.1f} noint {v['no_interval_base']:.3f}/{v['no_interval_arm']:.3f}")
for lab, v in summ['p5n'].items():
    print(lab, v if isinstance(v, str) else json.dumps({'base': v['base_split'], 'arm': v['arm_split']}))
print(p, sha(p))
