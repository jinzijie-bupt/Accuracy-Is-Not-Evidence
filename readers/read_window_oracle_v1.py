#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""read_window_oracle_v1.py — registered reader for the "window-length oracle" diagnostic.

Question it answers (paper Finding 2, "accuracy is a property of the annotation"): how much of an arm's Recall@IoU gain on
Charades-STA labels is reproduced by a *span prior alone*?  We take a model's own predictions, KEEP THE PREDICTED CENTRE, and
FORCE THE WINDOW LENGTH to a constant (default 8.0 s) or to the ground-truth length of that query (--gt-len), then re-score
against the same labels.  Everything else (parsing, unparsed-counts-as-miss, IoU thresholds) follows read_p4nat_v1.

This is a DIAGNOSTIC quantity: the constant-length variant uses only the prediction; the --gt-len variant uses the GT length
and is an upper bound, never a model score.  Both are reported with that label in the output.

Input: a p4nat full-test product (list of rows: qid, video, dur, pred=[s,e]|None, gsA, geA, raw, ...).
Output: JSON with inputs[].sha16, reader_sha16, params, R@{0.3,0.5,0.7}, mIoU, n, n_parsed for ORIGINAL and ORACLE predictions,
        plus per-qid IoUs.  Unparsed rows stay unparsed (miss) in both.

Usage:
  python analysis/read_window_oracle_v1.py --product <p4nat_full.json> --window-len 8.0 --label <name> --out <json>
  python analysis/read_window_oracle_v1.py --product <p4nat_full.json> --gt-len --label <name> --out <json>
  python analysis/read_window_oracle_v1.py --selftest
"""
import argparse, hashlib, json, os, sys, datetime

THRS = (0.3, 0.5, 0.7)


def sha16(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:16]


def iou(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def load_rows(path):
    d = json.load(open(path, encoding='utf-8'))
    rows = d['rows'] if isinstance(d, dict) and 'rows' in d else d
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f'[error] no rows in {path}')
    return rows


def oracle_window(pred, dur, length):
    """Keep the predicted centre, force the window length; clip to [0, dur]."""
    c = 0.5 * (pred[0] + pred[1])
    s = max(0.0, c - 0.5 * length); e = min(float(dur), c + 0.5 * length)
    if e < s:
        s, e = e, s
    return [s, e]


def score(rows, mode, window_len):
    per = []
    hits = {t: [0, 0] for t in THRS}  # [orig, oracle]
    miou = [0.0, 0.0]
    n_parsed = 0
    for r in rows:
        for k in ('qid', 'gsA', 'geA'):
            if k not in r:
                raise SystemExit(f'[error] row lacks required field {k!r}: qid={r.get("qid")}')
        gt = [float(r['gsA']), float(r['geA'])]
        pred = r.get('pred')
        rec = {'qid': r['qid'], 'video': r.get('video'), 'parsed': bool(pred)}
        if pred:
            if 'dur' not in r or r['dur'] is None:
                raise SystemExit(f'[error] parsed row without duration (needed for clipping): qid={r["qid"]}')
            n_parsed += 1
            pred = [float(pred[0]), float(pred[1])]
            L = (gt[1] - gt[0]) if mode == 'gt' else window_len
            orc = oracle_window(pred, r['dur'], L)
            io, ir = iou(pred, gt), iou(orc, gt)
            rec.update({'iou_orig': io, 'iou_oracle': ir, 'oracle_pred': orc, 'pred': pred})
            miou[0] += io; miou[1] += ir
            for t in THRS:
                hits[t][0] += int(io >= t); hits[t][1] += int(ir >= t)
        else:
            rec.update({'iou_orig': 0.0, 'iou_oracle': 0.0})
        per.append(rec)
    n = len(rows)
    out = {}
    for j, name in enumerate(('original', 'oracle')):
        out[name] = {f'R@{t}': hits[t][j] / n for t in THRS}
        out[name]['mIoU'] = miou[j] / n
        out[name]['n'] = n
        out[name]['n_parsed'] = n_parsed
        out[name]['parse_rate'] = n_parsed / n
    return out, per


def run(args):
    rows = load_rows(args.product)
    if args.expect_n and len(rows) != args.expect_n:
        raise SystemExit(f'[error] expected {args.expect_n} rows, got {len(rows)}')
    mode = 'gt' if args.gt_len else 'const'
    res, per = score(rows, mode, args.window_len)
    out = {
        'reader': os.path.basename(__file__), 'reader_sha16': sha16(os.path.abspath(__file__)),
        'generated_utc': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'label': args.label,
        'diagnostic': ('window-length oracle: predicted centre kept, window length forced to %.2f s (uses only the prediction)' % args.window_len)
                      if mode == 'const' else 'GT-length oracle: predicted centre kept, window length set to the GT length of the query (uses GT length; upper bound, not a model score)',
        'params': {'mode': mode, 'window_len_s': None if mode == 'gt' else args.window_len, 'thresholds': list(THRS), 'unparsed': 'counted as miss in both'},
        'inputs': [{'path': args.product, 'sha16': sha16(args.product), 'n_rows': len(rows)}],
        'results': res,
        'delta_oracle_minus_original': {k: res['oracle'][k] - res['original'][k] for k in res['original'] if k.startswith('R@') or k == 'mIoU'},
        'per_qid': per,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    o, r = res['original'], res['oracle']
    print(f"[window_oracle] {args.label} n={o['n']} parsed={o['n_parsed']} | original R@0.5={o['R@0.5']:.4f} mIoU={o['mIoU']:.4f} | oracle({out['params']['mode']}) R@0.5={r['R@0.5']:.4f} mIoU={r['mIoU']:.4f} -> {args.out}")


def selftest():
    import tempfile
    ok = True
    def chk(name, cond):
        nonlocal ok
        print(('  PASS ' if cond else '  FAIL ') + name); ok = ok and cond
    # rows: GT [10,18] (8 s) in a 30 s video; prediction centred at 14 with 2 s length -> IoU 0.25 (miss); oracle 8 s -> IoU 1.0 (hit)
    rows = [
        {'qid': 0, 'video': 'a', 'dur': 30.0, 'pred': [13.0, 15.0], 'gsA': 10.0, 'geA': 18.0},
        {'qid': 1, 'video': 'b', 'dur': 30.0, 'pred': None, 'gsA': 5.0, 'geA': 9.0},           # unparsed stays a miss
        {'qid': 2, 'video': 'c', 'dur': 10.0, 'pred': [8.0, 10.0], 'gsA': 4.0, 'geA': 10.0},   # clipping at dur: centre 9, 8 s -> [5,10] vs GT [4,10] -> IoU 5/6
    ]
    res, per = score(rows, 'const', 8.0)
    chk('positive: 2 s prediction at the GT centre misses (IoU .25) and the 8 s oracle hits (IoU 1.0)', per[0]['iou_orig'] == 0.25 and per[0]['iou_oracle'] == 1.0)
    # row 2 original: [8,10] vs GT [4,10] -> IoU 2/6 (miss); so original R@0.5 = 0/3 and the oracle lifts rows 0 and 2 -> 2/3
    chk('positive: original R@0.5 = 0/3 (row 2 IoU 1/3 misses), oracle R@0.5 = 2/3', res['original']['R@0.5'] == 0.0 and abs(res['oracle']['R@0.5'] - 2/3) < 1e-9)
    chk('positive: window is clipped to [0, dur] (row 2 oracle = [5,10], IoU 5/6)', per[2]['oracle_pred'] == [5.0, 10.0] and abs(per[2]['iou_oracle'] - 5/6) < 1e-9)
    chk('positive: unparsed row stays unparsed and counts as a miss in both', per[1]['parsed'] is False and per[1]['iou_oracle'] == 0.0 and res['oracle']['n_parsed'] == 2)
    res4, per4 = score(rows, 'const', 3.9)
    chk('negative: changing the forced length (3.9 s) changes the result (row 0 IoU .4875, a miss)', abs(per4[0]['iou_oracle'] - 3.9/8) < 1e-9 and res4['oracle']['R@0.5'] < res['oracle']['R@0.5'])
    resg, perg = score(rows, 'gt', 8.0)
    chk('positive: --gt-len uses each query\'s GT length (row 2: GT 6 s -> [6,10] vs [4,10] -> IoU 4/6)', abs(perg[2]['iou_oracle'] - 4/6) < 1e-9)
    try:
        score([{'qid': 9, 'video': 'z', 'pred': [1.0, 2.0], 'gsA': 0.0, 'geA': 3.0}], 'const', 8.0); chk('negative: a parsed row without duration raises', False)
    except SystemExit:
        chk('negative: a parsed row without duration raises', True)
    try:
        score([{'qid': 9, 'video': 'z', 'dur': 5.0, 'pred': [1.0, 2.0], 'gsA': 0.0}], 'const', 8.0); chk('negative: a row without GT end raises', False)
    except SystemExit:
        chk('negative: a row without GT end raises', True)
    # end-to-end through run(): output carries input sha16 and the diagnostic label
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'prod.json'); json.dump(rows, open(p, 'w'))
        o = os.path.join(td, 'out.json')
        ns = argparse.Namespace(product=p, window_len=8.0, gt_len=False, label='selftest', out=o, expect_n=3)
        run(ns); d = json.load(open(o, encoding='utf-8'))
        chk('positive: output carries inputs[].sha16, reader_sha16 and the diagnostic label', d['inputs'][0]['sha16'] == sha16(p) and len(d['reader_sha16']) == 16 and d['diagnostic'].startswith('window-length oracle'))
        ns2 = argparse.Namespace(product=p, window_len=8.0, gt_len=False, label='selftest', out=o, expect_n=4)
        try:
            run(ns2); chk('negative: --expect-n mismatch raises', False)
        except SystemExit:
            chk('negative: --expect-n mismatch raises', True)
    print('[selftest] ALL PASS' if ok else '[selftest] FAIL'); sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--product'); ap.add_argument('--window-len', type=float, default=8.0)
    ap.add_argument('--gt-len', action='store_true'); ap.add_argument('--label', default='')
    ap.add_argument('--out'); ap.add_argument('--expect-n', type=int, default=None)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if not (a.product and a.out):
        ap.error('--product and --out are required (or --selftest)')
    run(a)


if __name__ == '__main__':
    main()
