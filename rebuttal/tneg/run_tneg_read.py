# Drive the registered true-negative reader (analysis/read_p4nat_tneg_v1.py) over the 11-07 tneg products.
# Clean side = the arm's existing full p4nat shards on the SAME machine family (node7/node8 are bitwise-equivalent),
# concatenated (shards o0/o465/o930 cover qids 0..1394 >= the foreign slice 0..999); the reader pairs on the qid
# intersection. Writes rebuttal/tneg/read/<arm>.json and a summary table rebuttal/tneg/read/TNEG_SUMMARY.json.
import json, glob, os, subprocess, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.normpath(os.path.join(HERE, '..', '..', 'results', 'node7_pull'))
READER = os.path.normpath(os.path.join(HERE, '..', '..', 'analysis', 'read_p4nat_tneg_v1.py'))
MERGED = os.path.normpath(os.path.join(HERE, '..', '..', 'results', 'node7_merged', 'tneg_clean'))
OUT = os.path.join(HERE, 'read')
# arm -> (clean shard glob relative to P, foreign glob relative to P)
ARMS = {
    'qwenbase_tf451': ('newarm/_base_tf451/p4nat_qwenbase_tf451fa2_o*n465.json', 'tneg/qwenbase_tf451/p4nat_qwenbase_tf451_foreignq_s0_*_o0n1000.json'),
    'qwena05_s0': ('newarm/ref_qwena05_s0/p4nat_*_o*n465.json', 'tneg/qwena05_s0/p4nat_*_foreignq_s0_*_o0n1000.json'),
    'B1a_qwen_a05_s2': ('newarm/B1a_qwen_a05_s2/p4nat_*_o*n465.json', 'tneg/B1a_qwen_a05_s2/p4nat_*_foreignq_s0_*_o0n1000.json'),
    'qwensft': ('newarm/ref_qwensft/p4nat_*_o*n465.json', 'tneg/qwensft/p4nat_*_foreignq_s0_*_o0n1000.json'),
    'qwenmdpo': ('newarm/ref_qwenmdpo/p4nat_*_o*n465.json', 'tneg/qwenmdpo/p4nat_*_foreignq_s0_*_o0n1000.json'),
    'A2_grpo2fps_s0': ('newarm/A2_grpo2fps_s0/p4nat_*_o*n465.json', 'tneg/A2_grpo2fps_s0/p4nat_*_foreignq_s0_*_o0n1000.json'),
    'rarft_native': ('a6/p4nat_rarft_rarft_native_tf451fa2_o*n465.json', 'tneg/rarft_native/p4nat_*_foreignq_s0_*_o0n1000.json'),
    'tr1base_native': ('a6/p4nat_tr1base_rarft_native_tf451fa2_o*n465.json', 'tneg/tr1base_native/p4nat_*_foreignq_s0_*_o0n1000.json'),
    'qwenbase_tf5': ('newarm/_base_tf5/p4nat_qwenbase_tf5_o*n465.json', 'tneg/qwenbase_tf5/p4nat_*_foreignq_s0_*_o0n1000.json'),
    'l4800': ('newarm/ref_l4800/p4nat_*_o*n465.json', 'tneg/l4800/p4nat_*_foreignq_s0_*_o0n1000.json'),
    'iougrpo16f': ('newarm/ref_iougrpo16f/p4nat_*_o*n465.json', 'tneg/iougrpo16f/p4nat_*_foreignq_s0_*_o0n1000.json'),
    'B4_sft_l4800_s0': ('newarm/B4_sft_l4800_s0/p4nat_*_o*n465.json', 'tneg/B4_sft_l4800_s0/p4nat_*_foreignq_s0_*_o0n1000.json'),
    'B3_tsonly_a05_s0': ('newarm/B3_tsonly_a05_s0/p4nat_*_o*n465.json', 'tneg/B3_tsonly_a05_s0/p4nat_*_foreignq_s0_*_o0n1000.json'),
    'B5_grpo_placebo_lam001_s0': ('newarm/B5_grpo_placebo_lam001_s0/p4nat_*_o*n465.json', 'tneg/B5_grpo_placebo_lam001_s0/p4nat_*_foreignq_s0_*_o0n1000.json'),
}
NEED = {0, 465, 930}  # shards covering qids 0..1394


def sha16(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]


os.makedirs(OUT, exist_ok=True); os.makedirs(MERGED, exist_ok=True)
summary = {'reader': os.path.basename(READER), 'reader_sha16': sha16(READER), 'arms': {}}
for arm, (cg, fg) in ARMS.items():
    shards = [f for f in glob.glob(os.path.join(P, cg)) if not f.endswith(('.attn.json', '.meta.json'))]
    offs = {int(os.path.basename(f).rsplit('_o', 1)[1].split('n')[0]): f for f in shards}
    fr = [f for f in glob.glob(os.path.join(P, fg)) if not f.endswith(('.attn.json', '.meta.json'))]
    if not NEED <= set(offs) or len(fr) != 1:
        summary['arms'][arm] = {'status': f'missing: clean offsets {sorted(offs)} need {sorted(NEED)}; foreign files {len(fr)}'}
        print(arm, summary['arms'][arm]['status']); continue
    rows = []
    for o in sorted(NEED):
        rows += json.load(open(offs[o], encoding='utf-8'))
    cpath = os.path.join(MERGED, f'{arm}_clean_q0-1394.json')
    json.dump(rows, open(cpath, 'w', encoding='utf-8'), ensure_ascii=False)
    out = os.path.join(OUT, f'{arm}.json')
    r = subprocess.run([sys.executable, READER, '--label', arm, '--clean', cpath, '--foreign', fr[0], '--out', out],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0:
        summary['arms'][arm] = {'status': 'reader rc=%d' % r.returncode, 'err': r.stderr[-600:]}
        print(arm, 'READER FAIL', r.stderr[-400:]); continue
    d = json.load(open(out, encoding='utf-8'))
    summary['arms'][arm] = {'status': 'OK', 'read': os.path.relpath(out, HERE), 'read_sha16': sha16(out),
                            'clean_shards_sha16': {o: sha16(offs[o]) for o in sorted(NEED)}, 'foreign_sha16': sha16(fr[0]),
                            'clean': {k: d['clean'].get(k) for k in ('n', 'R@0.5', 'parse_rate', 'abstain_strict_rate', 'abstain_wide_rate', 'R@0.5_answered')},
                            'foreign': {k: d['foreign'].get(k) for k in ('n', 'R@0.5', 'parse_rate', 'abstain_strict_rate', 'abstain_wide_rate')}}
    c, f = d['clean'], d['foreign']
    print(f"{arm:28s} clean R50 {c['R@0.5']:.3f} abst(w) {c['abstain_wide_rate']:.3f} | foreign abst strict {f['abstain_strict_rate']:.3f} wide {f['abstain_wide_rate']:.3f} parse {f['parse_rate']:.3f}")
sp = os.path.join(OUT, 'TNEG_SUMMARY.json')
json.dump(summary, open(sp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(sp, sha16(sp))
