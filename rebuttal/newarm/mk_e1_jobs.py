# E1 (11-08, R3 Q1 "compare with models at similar accuracy"): the SFT-4800 arm (B4) saved LoRA checkpoints every 400
# pairs (results/train/adapter_nat_sft_l4800_s0__ckpt/cNNNN, prov.json gives the source sha and committed pair count).
# Evaluate c800/c1600/c2400/c3200 with exactly the B4 eval argv (tf5 stack, same base partner files):
#   full p4nat (8x465)  -> paired dR vs _base_tf5, locates the checkpoint whose accuracy matches l4800 (+8.1)
#   14Ln tokc first200  -> Gamma (mask) all-token / timestamp-token, same partner as NA_B4 tokc and A5 l4800
# Together with B4 final (c4800) this gives the SFT accuracy-vs-Gamma trajectory. Output label E1_sftckpt_cNNNN.
import json, os
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'B4_sft_l4800_s0_jobs.jsonl')
AD = (WORK_ROOT + '/results/train/adapter_nat_sft_l4800_s0')
CK = ['c800', 'c1600', 'c2400', 'c3200']
base = [json.loads(l) for l in open(SRC, encoding='utf-8')]
keep = [j for j in base if '__base_tf5__' not in j['name'] and ('__p4nat_' in j['name'] or '__14Ln_tokc_' in j['name'])]
assert len(keep) == 10, len(keep)
out = []
for c in CK:
    tag = f'E1_sftckpt_{c}'
    for j in keep:
        s = json.dumps(j)
        assert s.count(AD + '"') == 1, j['name']          # exactly one --adapter value
        s = s.replace(AD + '"', f'{AD}__ckpt/{c}"')
        s = s.replace('B4_sft_l4800_s0', tag)               # names, output dirs, output files
        k = json.loads(s)
        assert f'__ckpt/{c}' in k['argv'][k['argv'].index('--adapter') + 1]
        assert tag in k['out'] and k['out'].startswith('/') and '\\' not in s and 'Program Files' not in s
        k['prio'] = 9 if '__p4nat_' in k['name'] else 10
        out.append(k)
p = os.path.join(HERE, 'E1_sftckpt_jobs.jsonl')
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    for k in out:
        f.write(json.dumps(k) + '\n')
print(p, len(out))
