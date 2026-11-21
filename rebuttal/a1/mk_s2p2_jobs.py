# mk_s2p2_jobs.py (11-10): follow-up to s2p (R2 Q2 "... under one prompt").
# s2p showed our Qwen a0.5 seeds, trained under timer1_default, have Gamma +0.10 / +0.20 under the VideoChat-R1 template
# (vs +1.5 under their own). Two job lists:
#  (1) s2p2_ctrl_jobs.jsonl, prio 8: the accuracy-oriented / other Qwen arms under the same second template, first 200,
#      args identical to s2p_jobs.jsonl except --adapter and the output label (SFT, mDPO-style, IoU-GRPO 2 fps s0, B3).
#  (2) newarm/B7_qwen_a05_vchat_s0_jobs.jsonl, prio 9: B7 = our objective trained under the vchat template
#      (27_papo_train_qcdro_v3seln.py, B1a's args with --prompt-style vchat --answer-style vchat, seed 0), audited
#      (a) under vchat, first 200 (s2p args, paired with s2p/qwenbase) and (b) under timer1_default, indep600
#      (B3's indep600 job lines with adapter / tag swapped, paired with A1-01) -> train-template x audit-template 2x2.
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__)); RB = os.path.dirname(HERE)
R = WORK_ROOT
CTRL = [('sft', (DATASETS_ROOT + '/ds-1/local__nat_sft_b0_s0')),
        ('mdpo', (DATASETS_ROOT + '/ds-1/local__adapter_nat_mdpo_a05_s0')),
        ('a2grpo_s0', f'{R}/results/grpo/adapter_grpo_native_iou_s0'),
        ('b3tsonly', f'{R}/results/train/adapter_nat_base_a05_tsonly_s0')]
B7 = 'B7_qwen_a05_vchat_s0'; AD7 = f'{R}/results/train/adapter_nat_base_a05_vchat_s0'
base = [json.loads(l) for l in open(os.path.join(HERE, 's2p_jobs.jsonl'), encoding='utf-8') if '__qwenbase__' in l]
assert len(base) == 2 and all('--adapter' not in j['argv'] for j in base)


def s2p_like(tag, ad, prio, outdir):
    out = []
    for j in base:
        a = list(j['argv']); i = a.index('--prompt-style')
        a[i:i] = ['--adapter', ad]
        o = j['out'].replace('/s2p/qwenbase/', f'/{outdir}/').replace('_s2p_qwenbase_', f'_s2p_{tag}_')
        a[a.index('--out') + 1] = o
        out.append({**j, 'name': j['name'].replace('s2p__qwenbase__', f's2p__{tag}__'), 'prio': prio, 'argv': a, 'out': o,
                    'mkdir': os.path.dirname(o)})
    return out


ctrl = [k for t, ad in CTRL for k in s2p_like(t, ad, 8, f's2p/{t}')]
b7 = s2p_like(B7, AD7, 9, f'newarm/{B7}')
AD3 = f'{R}/results/train/adapter_nat_base_a05_tsonly_s0'; T3 = 'B3_tsonly_a05_s0'
for l in open(os.path.join(RB, 'newarm', f'{T3}_jobs.jsonl'), encoding='utf-8'):
    j = json.loads(l)
    if 'indep600' in j['out']:
        assert AD3 in j['argv']
        k = json.loads(json.dumps(j).replace(AD3, AD7).replace(T3, B7)); k['prio'] = 9; b7.append(k)
assert len(b7) == 8, len(b7)
for p, js in [(os.path.join(HERE, 's2p2_ctrl_jobs.jsonl'), ctrl), (os.path.join(RB, 'newarm', f'{B7}_jobs.jsonl'), b7)]:
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        for j in js:
            f.write(json.dumps(j) + '\n')
    print(p, len(js))
