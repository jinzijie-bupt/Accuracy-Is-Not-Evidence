# mk_a2s1_jobs.py (11-09): R1 Q1 "compare IoU-GRPO and the proposed objective with matched training frames, data, and
# compute". A2 (2 fps IoU-GRPO, seed 0) already matches frames and data; its 600 steps took 54,221 s on one A100, our recipe
# 300 steps ~27,000 s. A2s1 = the same trainer and args with seed 1 (no schedule fingerprint, as B5s1), and
# ckpt_copier_grpo.py archives the step-300 save (~27,100 s at ~90 s/step: the compute-matched point; it has seen 1,200 of
# the 2,400 questions). Eval jobs = A2's arm jobs (indep600 14Ln, tokc, p4nat, p5n; the shared base jobs are dropped) with
# adapter and label swapped, for the final adapter (A2s1) and the step-300 snapshot (A2s1c300).
import json, os
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__)); R = WORK_ROOT
J = [json.loads(l) for l in open(os.path.join(HERE, 'A2_grpo2fps_s0_jobs.jsonl'), encoding='utf-8')]
A0 = f'{R}/results/grpo/adapter_grpo_native_iou_s0'
ARMS = {'A2_grpo2fps_s1': f'{R}/results/grpo/adapter_grpo_native_iou_s1',
        'A2_grpo2fps_s1c300': f'{R}/results/grpo/adapter_grpo_native_iou_s1__ckpt/s300'}
for lab in ARMS:
    out = []
    for j in J:
        if not j['name'].startswith('na__A2_grpo2fps_s0__'):
            continue
        k = json.loads(json.dumps(j)); a = k['argv']
        assert a[a.index('--adapter') + 1] == A0
        a[a.index('--adapter') + 1] = ARMS[lab]
        for i, x in enumerate(a):
            if '/newarm/A2_grpo2fps_s0/' in x:
                a[i] = x.replace('A2_grpo2fps_s0', lab)
        k['out'] = k['out'].replace('A2_grpo2fps_s0', lab); k['name'] = k['name'].replace('A2_grpo2fps_s0', lab)
        if 'mkdir' in k: k['mkdir'] = k['mkdir'].replace('A2_grpo2fps_s0', lab)
        k['prio'] = 9
        assert a[a.index('--adapter') + 1] == ARMS[lab] and a[a.index('--out') + 1] == k['out'] and lab in k['out']
        assert 'A2_grpo2fps_s0' not in ' '.join(a) + k['name']
        out.append(k)
    assert len(out) == 20, len(out)
    p = os.path.join(HERE, f'{lab}_jobs.jsonl')
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        for k in out:
            f.write(json.dumps(k) + '\n')
    print(p, len(out)); print('  ', ' '.join(out[0]['argv'])[-330:])
