#!/usr/bin/env python
"""navmr_read.py -- reader for navmr_audit.py products -> NAVMR_SUMMARY.json
usage: python navmr_read.py <dir with navmr_{base,na}_charades_test{.json,.meta.json}> [out.json]
"""
import sys, os, json, hashlib
import numpy as np

D = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, 'NAVMR_SUMMARY.json')
TAG = 'charades_test'
B = 2000


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


inputs = {}
prod, meta = {}, {}
for m in ('base', 'na'):
    p = os.path.join(D, f'navmr_{m}_{TAG}.json'); pm = p.replace('.json', '.meta.json')
    inputs[os.path.basename(p)] = sha256(p); inputs[os.path.basename(pm)] = sha256(pm)
    prod[m] = {r['qid']: r for r in json.load(open(p))['rows']}
    meta[m] = json.load(open(pm))
    assert meta[m]['product_sha256'] == inputs[os.path.basename(p)], f'{m} product sha mismatch with meta'
R = int(meta['base']['args']['R'])
assert meta['base']['script_sha256'] == meta['na']['script_sha256']

qids = [q for q in prod['base'] if q in prod['na']]
inc = [q for q in qids if prod['base'][q].get('excluded') is None and 'cond' in prod['base'][q]
       and prod['na'][q].get('excluded') is None and 'cond' in prod['na'][q]]
excl = {}
for q in qids:
    e = prod['base'][q].get('excluded')
    if e is not None:
        excl[e] = excl.get(e, 0) + 1
for q in inc:  # same index sets for both models
    a, b = prod['base'][q], prod['na'][q]
    assert a['E'] == b['E'] and a['rand'] == b['rand'] and a['donor'] == b['donor'] and a['donor_idx'] == b['donor_idx']

hit = lambda m, q, c: prod[m][q]['cond'][c]['hit5']
cohort_main = [q for q in inc if hit('base', q, 'clean') and hit('na', q, 'clean')]


def boot(vals, vids, seed=0):
    """video-cluster bootstrap of the mean of per-query vals (resample videos w/ replacement)."""
    vals = np.asarray(vals, float); vids = np.asarray(vids)
    uv, inv = np.unique(vids, return_inverse=True)
    sums = np.bincount(inv, weights=vals, minlength=len(uv)); cnts = np.bincount(inv, minlength=len(uv)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uv), size=(B, len(uv)))
    st = sums[idx].sum(1) / cnts[idx].sum(1)
    return dict(mean=float(vals.mean()), ci95=[float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))],
                n_q=int(len(vals)), n_vid=int(len(uv)))


def psi_q(m, q, op):
    c = prod[m][q]['cond']
    f0 = c['clean']['fg_gt']
    return (f0 - c[f'{op}_evid']['fg_gt']) - np.mean([f0 - c[f'{op}_rand{r}']['fg_gt'] for r in range(R)])


def block(Q, name):
    vids = [prod['base'][q]['vid'] for q in Q]
    out = dict(cohort=name, n_q=len(Q), n_vid=len(set(vids)))
    for op in ('swap', 'mask'):
        o = {}
        pb = np.array([psi_q('base', q, op) for q in Q]); pn = np.array([psi_q('na', q, op) for q in Q])
        o['psi_base'] = boot(pb, vids); o['psi_na'] = boot(pn, vids)
        o['Gamma_na_minus_base'] = boot(pn - pb, vids)
        # components (raw fg_gt drops)
        for m in ('base', 'na'):
            o[f'fg_drop_evid_{m}'] = float(np.mean([prod[m][q]['cond']['clean']['fg_gt'] - prod[m][q]['cond'][f'{op}_evid']['fg_gt'] for q in Q]))
            o[f'fg_drop_rand_{m}'] = float(np.mean([np.mean([prod[m][q]['cond']['clean']['fg_gt'] - prod[m][q]['cond'][f'{op}_rand{r}']['fg_gt'] for r in range(R)]) for q in Q]))
        # behaviour
        beh = {}
        dd = {}
        for m in ('base', 'na'):
            hc = np.array([hit(m, q, 'clean') for q in Q], float)
            he = np.array([hit(m, q, f'{op}_evid') for q in Q], float)
            hr = np.array([np.mean([hit(m, q, f'{op}_rand{r}') for r in range(R)]) for q in Q], float)
            beh[m] = {'R@0.5_clean': float(hc.mean() * 100), 'R@0.5_evid': float(he.mean() * 100), 'R@0.5_rand': float(hr.mean() * 100),
                      'EO_drop_rand_minus_evid': boot((hr - he) * 100, vids)}
            dd[m] = hr - he
        beh['DiD_EOdrop_na_minus_base'] = boot((dd['na'] - dd['base']) * 100, vids)
        o['behaviour'] = beh
        # NA rejection
        rc = np.array([prod['na'][q]['cond']['clean']['reject'] for q in Q], float)
        re_ = np.array([prod['na'][q]['cond'][f'{op}_evid']['reject'] for q in Q], float)
        rr = np.array([np.mean([prod['na'][q]['cond'][f'{op}_rand{r}']['reject'] for r in range(R)]) for q in Q], float)
        pc = {c: float(np.mean([prod['na'][q]['cond'][c]['pred_class'] for q in Q])) for c in ('clean', f'{op}_evid')}
        pc['rand_mean'] = float(np.mean([np.mean([prod['na'][q]['cond'][f'{op}_rand{r}']['pred_class'] for r in range(R)]) for q in Q]))
        o['na_reject'] = {'thr': meta['na']['args']['na_thr'], 'rate_clean': float(rc.mean() * 100), 'rate_evid': float(re_.mean() * 100),
                          'rate_rand': float(rr.mean() * 100), 'evid_minus_rand_pp': boot((re_ - rr) * 100, vids),
                          'evid_minus_clean_pp': boot((re_ - rc) * 100, vids), 'mean_pred_class': pc}
        out[op] = o
    return out


san = meta['base']['sanity']
assert san == meta['na']['sanity']
ab = san['a_clean_nopad_base']; an = san['a_clean_nopad_na']
abr = san['a_clean_repo_batched_base']; anr = san['a_clean_repo_batched_na']
checks = {
    'a_base_R1@0.3_vs_zoo_72.63': dict(nopad=ab['R1@0.3'], repo=abr['R1@0.3'], ok=abs(ab['R1@0.3'] - 72.63) <= 3 and abs(abr['R1@0.3'] - 72.63) <= 3),
    'a_base_mIoU_vs_zoo_52.17': dict(nopad=ab['mIoU'], repo=abr['mIoU'], ok=abs(ab['mIoU'] - 52.17) <= 3 and abs(abr['mIoU'] - 52.17) <= 3),
    'a_base_R1@0.5_vs_NAVMR_60.22': dict(nopad=ab['R1@0.5'], repo=abr['R1@0.5'], ok=abs(ab['R1@0.5'] - 60.22) <= 3 and abs(abr['R1@0.5'] - 60.22) <= 3),
    'a_na_R1@0.5_negaware_vs_NAVMR_55.54': dict(nopad=san['a_clean_nopad_na_negaware']['R1@0.5_strict_gt_rejected_as_miss'],
                                                 repo=san['a_clean_repo_batched_na_negaware']['R1@0.5_strict_gt_rejected_as_miss'],
                                                 ok=abs(san['a_clean_repo_batched_na_negaware']['R1@0.5_strict_gt_rejected_as_miss'] - 55.54) <= 3),
    'b_identity_edits': dict(ok=bool(san['b_pass']), detail=san['b_identity_edits']),
    'c_na_reject_clean_pos': dict(rate=san['c_na_reject_rate_clean_pos_nopad'], ok=san['c_na_reject_rate_clean_pos_nopad'] < 20),
    'c_idneg_rejection_acc': dict(value=san.get('c_idneg', {}).get('rejection_accuracy'), paper=64.11),
}
summary = dict(
    inputs_sha256=inputs, reader_sha256=sha256(os.path.abspath(__file__)), script_sha256=meta['base']['script_sha256'],
    test_file=meta['base']['args']['pos_jsonl'], test_file_sha256=meta['base']['pos_jsonl_sha256'], n_test=len(qids),
    excluded=excl, n_included=len(inc), n_cohort_bothhit=len(cohort_main), R=R, bootstrap=dict(B=B, seed=0, unit='video'),
    sanity_checks=checks, sanity_all_pass=bool(all(v.get('ok', True) for v in checks.values())), sanity_raw=san,
    main_bothhit=block(cohort_main, 'both models hit clean IoU>=0.5'),
    all_included=block(inc, 'all included queries (0<k<=L/2)'),
)
with open(OUT + '.tmp', 'w') as f:
    json.dump(summary, f, indent=1)
os.replace(OUT + '.tmp', OUT)
print('wrote', OUT, sha256(OUT)[:16])
