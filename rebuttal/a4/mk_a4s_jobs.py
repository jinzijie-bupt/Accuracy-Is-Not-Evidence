# mk_a4s_jobs.py (11-10): A4 control row. The SFT model (A1-02, args identical to A1-04 except the adapter) under the
# same five A4 conditions (jitter shrink/expand/shiftl/shiftr 10%, ops freeze,occlude), first 200, so that R2 Q2's
# "does a ten-percent jitter change which values count" is read for an accuracy-oriented model too. Job lines are the
# qwen_a05 lines of a4_jobs.jsonl with the adapter and tag swapped; prio 8. Pairs with a4/qwen_base__<cond>__tf451.
import json, os
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")  # read-only dataset mount (adapters and videos staged as datasets)
HERE = os.path.dirname(os.path.abspath(__file__))
A05, SFT = (DATASETS_ROOT + '/ds-1/local__nat_base_a05_g1_s0'), (DATASETS_ROOT + '/ds-1/local__nat_sft_b0_s0')
out = []
for l in open(os.path.join(HERE, 'a4_jobs.jsonl'), encoding='utf-8'):
    j = json.loads(l)
    if j['name'].startswith('a4__qwen_a05__'):
        assert A05 in j['argv']
        k = json.loads(json.dumps(j).replace(A05, SFT).replace('qwen_a05__', 'qwen_sft__')); k['prio'] = 8; out.append(k)
assert len(out) == 10
p = os.path.join(HERE, 'a4s_jobs.jsonl')
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    for j in out:
        f.write(json.dumps(j) + '\n')
print(p, len(out))
