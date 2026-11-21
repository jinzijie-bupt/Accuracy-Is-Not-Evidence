# read_remedy_sweep.py (11-11): clean answers of the remedy sweep (mk_remedy_jobs.py), first 200 questions, p5n 'none'.
# Same counting as the SWEEP block of read_wording_sweep.py: no-interval rate = share of rows with none_parsed false;
# R@0.5 = share with none_hit. Inputs results/node7_pull/wsweep/<tag>/p5n_ws_<tag>_<style>_<env>_o0n200.json
# (node9 products copied in from node9_pull). Missing cells are reported as 'missing'.
# Output rebuttal/a1/read_wording_sweep/REMEDY_SWEEP_SUMMARY.json
import glob, hashlib, json, os
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.normpath(os.path.join(H, '..', '..'))
PULL = os.path.join(ROOT, 'results', 'node7_pull', 'wsweep')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
TAGS = {'qwenbase_tf5': 'tf5', 'a025_tf5': 'tf5', 'l4800_tf5': 'tf5', 'a10_tf5': 'tf5',
        'tzbase451': 'tf451fa2', 'B6': 'tf451fa2', 'B5s1': 'tf451fa2', 'B5c': 'tf451fa2'}
STYLES = ['timer1_default', 'para1', 'para2', 'w2', 'w3']
summ = {'script_sha16': sha(os.path.abspath(__file__)), 'cells': {}}
for t, env in TAGS.items():
    row = {}
    for st in STYLES:
        fs = [f for f in glob.glob(os.path.join(PULL, t, f'p5n_ws_{t}_{st}_{env}_o0n200.json'))]
        if len(fs) != 1:
            row[st] = 'missing'; continue
        rows = json.load(open(fs[0], encoding='utf-8'))
        assert len(rows) == 200 and [str(r['qid']) for r in rows] == [str(i) for i in range(200)], (t, st)
        row[st] = {'file': os.path.relpath(fs[0], ROOT), 'sha16': sha(fs[0]),
                   'no_interval': sum(1 for r in rows if not r['none_parsed']) / 200,
                   'R50': sum(1 for r in rows if r['none_hit']) / 200}
    summ['cells'][t] = row
p = os.path.join(H, 'read_wording_sweep', 'REMEDY_SWEEP_SUMMARY.json')
json.dump(summ, open(p, 'w', encoding='utf-8'), indent=1)
print(f"{'model':14s}" + ''.join(f'{s:>16s}' for s in STYLES))
for t, row in summ['cells'].items():
    print(f'{t:14s}' + ''.join(f"{'missing':>16s}" if isinstance(row[s], str) else f"{row[s]['no_interval']:>8.3f}/{row[s]['R50']:.3f}" for s in STYLES))
print(p, sha(p))
