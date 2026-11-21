#!/usr/bin/env python
"""navmr_audit.py -- feature-level evidence-dependence audit of UniVTG (BASE) vs UniVTG-NA (NA)
on Charades-STA test positives.

Imports ONLY the model-definition modules of each repo (model/univtg*.py, transformer_encoder_droppath,
position_encoding, matcher (scipy stubbed, matcher unused at inference), utils/span_utils).
Feature loading re-implements DatasetMR (UniVTG-NA main/dataset.py == UniVTG main/dataset.py for this path):
  * video: per feature dir in opt.v_feat_dirs order [vid_slowfast, vid_clip]: np.load(npz)['features'].float32,
    l2-normalise rows (eps 1e-5) separately, truncate all to min length, concat along dim 1  (no max_v_l truncation:
    the repo's [:max_v_l] is commented out)
  * text: np.load(npz)['last_hidden_state'].float32, l2-normalise rows (no max_q_l truncation, also commented out)
  * ctx_mode video_tef: append [i/L, (i+1)/L] columns
  * timestamp = (arange(L)+clip_len/2)/L repeated twice
  * decoding (compute_mr_results, l1 dense regression): span = (timestamp + pred_spans) * duration, clamp [0,duration],
    score = pred_logits[...,0] (already sigmoid inside the model), top-1 = highest score; rows rounded to 4 decimals
  * IoU as eval/utils.compute_temporal_iou_batch_paired (union = hull), hit = IoU>=thd
Edits act on the first v_dim (=2816) columns only (normalised features), never on the 2 TEF columns.
Main analysis forward = one query per forward with its 9 variants batched (same L, no padding => no padding
artefacts in Conv1d head / LSTM relevance head).
"""
import sys, os, json, time, hashlib, argparse, types, importlib
import numpy as np
import torch
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)

ROOT = (WORK_ROOT + '/navmr')
P = argparse.ArgumentParser()
P.add_argument('--pos_jsonl', default=f'{ROOT}/work/MomentofUntruth-main/UniVTG-NA/data/charades/metadata/charades_pos_test.jsonl')
P.add_argument('--idneg_jsonl', default=f'{ROOT}/work/MomentofUntruth-main/UniVTG-NA/data/charades/metadata/charades_id_neg_test.jsonl')
P.add_argument('--oodneg_jsonl', default=f'{ROOT}/work/MomentofUntruth-main/UniVTG-NA/data/charades/metadata/charades_ood_pos_neg_test.jsonl')
P.add_argument('--feat_root', default=f'{ROOT}/feat/charades')
P.add_argument('--base_code', default=f'{ROOT}/work/UniVTG-main')
P.add_argument('--na_code', default=f'{ROOT}/work/MomentofUntruth-main/UniVTG-NA/main')
P.add_argument('--base_ckpt_dir', default=f'{ROOT}/dl/ckpt/univtg_charades')
P.add_argument('--na_ckpt_dir', default=f'{ROOT}/dl/ckpt/univtg_na_charades')
P.add_argument('--out_dir', default=f'{ROOT}/out')
P.add_argument('--tag', default='charades_test')
P.add_argument('--limit', type=int, default=0)
P.add_argument('--R', type=int, default=3)
P.add_argument('--na_thr', type=float, default=0.5)
P.add_argument('--device', default='cuda')
P.add_argument('--threads', type=int, default=8)
P.add_argument('--n_sanity_b', type=int, default=100)
args = P.parse_args()

torch.set_num_threads(args.threads)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
T0 = time.time()


def sha256(p, n=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(f'[{time.time()-T0:7.1f}s]', *a, flush=True)


# ---------------------------------------------------------------- model import (two repos, same module names)
_scipy = types.ModuleType('scipy'); _opt = types.ModuleType('scipy.optimize')
def _lsa(*a, **k):
    raise RuntimeError('matcher not used at inference')
_opt.linear_sum_assignment = _lsa; _scipy.optimize = _opt
sys.modules.setdefault('scipy', _scipy); sys.modules.setdefault('scipy.optimize', _opt)


def import_model_module(code_root, modname):
    for k in list(sys.modules):
        if k in ('model', 'utils') or k.startswith('model.') or k.startswith('utils.'):
            del sys.modules[k]
    sys.path.insert(0, code_root)
    try:
        mod = importlib.import_module(f'model.{modname}')
    finally:
        sys.path.remove(code_root)
    for k in list(sys.modules):
        if k in ('model', 'utils') or k.startswith('model.') or k.startswith('utils.'):
            del sys.modules[k]
    return mod


def build(code_root, modname, ckpt_dir):
    opt = json.load(open(f'{ckpt_dir}/opt.json'))
    ns = argparse.Namespace(**opt)
    ns.v_feat_dim = opt['v_feat_dim'] + (2 if 'tef' in opt['ctx_mode'] else 0)  # BaseOptions.parse adds 2 for tef
    ns.device = 'cpu'
    ns.pos_train_path = opt.get('pos_train_path', opt.get('train_path', ['x']))
    if not hasattr(ns, 'c_loss_coef'):
        ns.c_loss_coef = 1.0
    mod = import_model_module(code_root, modname)
    model, _crit = mod.build_model(ns)
    ck = torch.load(f'{ckpt_dir}/model_best.ckpt', map_location='cpu', weights_only=False)
    sd = {k.replace('module.', ''): v for k, v in ck['model'].items()}
    res = model.load_state_dict(sd, strict=False)
    info = dict(missing=list(res.missing_keys), unexpected=list(res.unexpected_keys), epoch=ck.get('epoch'),
                n_keys=len(sd), module_file=mod.__file__, module_sha256=sha256(mod.__file__),
                ckpt_sha256=sha256(f'{ckpt_dir}/model_best.ckpt'), opt_sha256=sha256(f'{ckpt_dir}/opt.json'))
    if res.missing_keys or res.unexpected_keys:
        log('KEY MISMATCH', modname, info['missing'], info['unexpected'])
        raise SystemExit(f'non-trivial key mismatch for {modname}')
    model.load_state_dict(sd, strict=True)
    model.eval().to(DEV)
    return model, opt, info


base, opt_b, info_b = build(args.base_code, 'univtg', args.base_ckpt_dir)
na, opt_n, info_n = build(args.na_code, 'univtg_na', args.na_ckpt_dir)
log('models built', info_b['epoch'], info_n['epoch'], 'device', DEV)
for k in ['clip_length', 'ctx_mode', 'v_feat_types', 'no_norm_vfeat', 'no_norm_tfeat', 'max_v_l', 'max_q_l', 'span_loss_type', 'round_multiple']:
    assert opt_b[k] == opt_n[k], (k, opt_b[k], opt_n[k])
CLIP_LEN = float(opt_b['clip_length'])
assert opt_b['ctx_mode'] == 'video_tef' and opt_b['span_loss_type'] == 'l1' and opt_b['round_multiple'] <= 0
VDIRS = [os.path.join(args.feat_root, os.path.basename(d)) for d in opt_b['v_feat_dirs']]  # [vid_slowfast, vid_clip]
TDIR = os.path.join(args.feat_root, 'txt_clip')
NORM_V = not opt_b['no_norm_vfeat']; NORM_T = not opt_b['no_norm_tfeat']


# ---------------------------------------------------------------- data (DatasetMR re-implementation)
def l2n(a, eps=1e-5):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l.strip('\n')) for l in f.readlines()]


_vcache = {}
def vfeat(vid):
    if vid not in _vcache:
        fl = []
        for d in VDIRS:
            x = np.load(os.path.join(d, f'{vid}.npz'))['features'].astype(np.float32)
            if NORM_V:
                x = l2n(x)
            fl.append(x)
        m = min(len(e) for e in fl)
        _vcache[vid] = np.concatenate([e[:m] for e in fl], axis=1)
    return _vcache[vid]


def tfeat(qid):
    p = os.path.join(TDIR, f'{qid}.npz')
    q = np.load(p)['last_hidden_state'].astype(np.float32)
    if NORM_T:
        q = l2n(q)
    return q


def tef(L):
    st = np.arange(0, L, 1.0, dtype=np.float32) / L
    return np.stack([st, st + 1.0 / L], 1).astype(np.float32)


def timestamp(L):
    t = ((torch.arange(0, L) + CLIP_LEN / 2) / L).unsqueeze(1).repeat(1, 2)
    return t


def iou(p, g):
    inter = max(0.0, min(p[1], g[1]) - max(p[0], g[0]))
    union = max(p[1], g[1]) - min(p[0], g[0])
    return inter / union if union != 0 else 0.0


@torch.no_grad()
def forward(model, vids, txt, Ls=None):
    """vids: (B,L,D+2) float32 np (same L); txt: (Lq,512). returns dict of np arrays (no padding)."""
    B, L, _ = vids.shape
    sv = torch.from_numpy(vids).to(DEV)
    st = torch.from_numpy(np.repeat(txt[None], B, 0)).to(DEV)
    out = model(src_txt=st, src_txt_mask=torch.ones(B, txt.shape[0], device=DEV),
                src_vid=sv, src_vid_mask=torch.ones(B, L, device=DEV))
    r = dict(prob=out['pred_logits'][..., 0].float().cpu().numpy(),      # sigmoid already applied in model
             spans=out['pred_spans'].float().cpu().numpy())
    if 'pred_class' in out:
        r['pred_class'] = out['pred_class'][:, 0].float().cpu().numpy()
    return r


def decode(prob_row, span_row, L, duration):
    ts = timestamp(L).numpy()
    sp = (torch.from_numpy(ts + span_row) * duration)
    sp = torch.clamp(sp, 0, duration).numpy()  # same op order as repo (float32 tensors)
    j = int(np.argmax(prob_row))  # sorted(..., reverse=True) is stable -> first max == argmax
    row = [float(f'{e:.4f}') for e in [float(sp[j, 0]), float(sp[j, 1]), float(prob_row[j])]]
    return row


# ---------------------------------------------------------------- index sets
OP_RAND, OP_DONOR = 1, 2


def qnum(qid):
    return int(str(qid).replace('test', ''))


def evid_set(gt, L):
    s, e = gt
    return [i for i in range(L) if (i * CLIP_LEN < e) and ((i + 1) * CLIP_LEN > s)]


def make_sets(meta, L, all_vids, vid_len):
    E = evid_set(meta['relevant_windows'][0], L)
    k = len(E)
    rec = dict(k=k, E=E)
    if k == 0 or k > L / 2:
        rec['excluded'] = 'k0' if k == 0 else 'k_gt_half'
        return rec
    rec['excluded'] = None
    q = qnum(meta['qid'])
    nonE = np.array([i for i in range(L) if i not in set(E)])
    rand = []
    for r in range(args.R):
        g = np.random.default_rng(np.random.SeedSequence([1000, q, OP_RAND, r]))
        rand.append(sorted(int(x) for x in g.choice(nonE, size=k, replace=False)))
    g = np.random.default_rng(np.random.SeedSequence([1000, q, OP_DONOR, 0]))
    pool = [v for v in all_vids if v != meta['vid']]
    dv = pool[int(g.integers(len(pool)))]
    Ld = vid_len[dv]
    didx = sorted(int(x) for x in g.choice(Ld, size=k, replace=(Ld < k)))
    rec.update(rand=rand, donor=dv, donor_idx=didx, donor_L=Ld)
    return rec


def apply_edit(X, idx, kind, donor_feat=None, donor_idx=None):
    """X: (L, D+2) clean input. edits columns [:D] only."""
    Y = X.copy()
    if len(idx) == 0:
        return Y
    D = X.shape[1] - 2
    if kind == 'mask':
        Y[idx, :D] = 0.0
    elif kind == 'swap':
        Y[idx, :D] = donor_feat[donor_idx, :D]
    return Y


# ---------------------------------------------------------------- main
data = load_jsonl(args.pos_jsonl)
if args.limit:
    data = data[:args.limit]
all_vids = sorted(set(m['vid'] for m in load_jsonl(args.pos_jsonl)))
vid_len = {}
missing_v = []
for v in all_vids:
    try:
        vid_len[v] = vfeat(v).shape[0]
    except Exception as e:
        missing_v.append(v)
all_vids = [v for v in all_vids if v in vid_len]
log('videos', len(all_vids), 'missing', len(missing_v))

COND = ['clean', 'swap_evid'] + [f'swap_rand{r}' for r in range(args.R)] + ['mask_evid'] + [f'mask_rand{r}' for r in range(args.R)]
rows = {'base': [], 'na': []}
missing_t = []
for qi, meta in enumerate(data):
    qid, vid, dur = meta['qid'], meta['vid'], float(meta['duration'])
    gt = [float(x) for x in meta['relevant_windows'][0]]
    if vid not in vid_len:
        for m in rows:
            rows[m].append(dict(qid=qid, vid=vid, excluded='missing_video'))
        continue
    try:
        txt = tfeat(qid)
    except Exception:
        missing_t.append(qid)
        for m in rows:
            rows[m].append(dict(qid=qid, vid=vid, excluded='missing_text'))
        continue
    V = vfeat(vid); L = V.shape[0]
    X = np.concatenate([V, tef(L)], 1).astype(np.float32)
    S = make_sets(meta, L, all_vids, vid_len)
    variants = [X]
    if S['excluded'] is None:
        DV = vfeat(S['donor'])
        variants.append(apply_edit(X, S['E'], 'swap', DV, S['donor_idx']))
        for r in range(args.R):
            variants.append(apply_edit(X, S['rand'][r], 'swap', DV, S['donor_idx']))
        variants.append(apply_edit(X, S['E'], 'mask'))
        for r in range(args.R):
            variants.append(apply_edit(X, S['rand'][r], 'mask'))
        # guard: TEF columns untouched
        for Y in variants:
            assert np.array_equal(Y[:, -2:], X[:, -2:])
    B = np.stack(variants, 0)
    conds = COND[:len(variants)]
    for mname, model in (('base', base), ('na', na)):
        o = forward(model, B, txt)
        row = dict(qid=qid, vid=vid, duration=dur, gt=gt, L=L, excluded=S['excluded'], k=S['k'], E=S['E'])
        if S['excluded'] is None:
            row.update(rand=S['rand'], donor=S['donor'], donor_idx=S['donor_idx'], donor_L=S['donor_L'])
        res = {}
        for ci, c in enumerate(conds):
            top = decode(o['prob'][ci], o['spans'][ci], L, dur)
            iu = iou(top[:2], gt)
            d = dict(top1=top, iou=iu, hit5=int(iu >= 0.5), hit7=int(iu >= 0.7), hit3=int(iu >= 0.3))
            if S['k'] > 0:
                d['fg_gt'] = float(np.mean(np.log(np.clip(o['prob'][ci][S['E']], 1e-6, None))))
            d['fg_mean_all'] = float(np.mean(o['prob'][ci]))
            if 'pred_class' in o:
                d['pred_class'] = float(o['pred_class'][ci])
                d['reject'] = int(o['pred_class'][ci] < args.na_thr)
            res[c] = d
        row['cond'] = res
        rows[mname].append(row)
    if qi % 500 == 0:
        log('q', qi, qid, 'L', L, 'k', S['k'], S['excluded'])
log('main loop done', len(data), 'missing_text', len(missing_t))


# ---------------------------------------------------------------- sanity (a): repo-protocol batched eval (padding as DataLoader eval_bsz)
@torch.no_grad()
def repo_batched_eval(model, opt, items, thr=None):
    """items: list of meta. Mimics start_end_collate_mr + compute_mr_results with batch size opt['eval_bsz'],
    file order, zero padding + masks. returns per-query (top1, pred_class)."""
    bsz = int(opt['eval_bsz'])
    outl = []
    for b0 in range(0, len(items), bsz):
        batch = items[b0:b0 + bsz]
        Xs, Ts = [], []
        for m in batch:
            V = vfeat(m['vid']); L = V.shape[0]
            Xs.append(np.concatenate([V, tef(L)], 1).astype(np.float32)); Ts.append(tfeat(m['qid']))
        Lv = max(x.shape[0] for x in Xs); Lq = max(t.shape[0] for t in Ts)
        sv = np.zeros((len(batch), Lv, Xs[0].shape[1]), np.float32); mv = np.zeros((len(batch), Lv), np.float32)
        stx = np.zeros((len(batch), Lq, Ts[0].shape[1]), np.float32); mt = np.zeros((len(batch), Lq), np.float32)
        tsp = torch.zeros(len(batch), Lv, 2)
        for i, (x, t) in enumerate(zip(Xs, Ts)):
            sv[i, :len(x)] = x; mv[i, :len(x)] = 1; stx[i, :len(t)] = t; mt[i, :len(t)] = 1
            tsp[i, :len(x)] = timestamp(len(x))
        out = model(src_txt=torch.from_numpy(stx).to(DEV), src_txt_mask=torch.from_numpy(mt).to(DEV),
                    src_vid=torch.from_numpy(sv).to(DEV), src_vid_mask=torch.from_numpy(mv).to(DEV))
        scores = out['pred_logits'][..., 0].clone()
        mask = torch.from_numpy(mv).to(DEV).bool()
        scores[~mask] = 0
        spans = tsp.to(DEV) + out['pred_spans']
        for i, m in enumerate(batch):
            dur = float(m['duration'])
            sp = torch.clamp(spans[i].cpu() * dur, 0, dur)
            sc = scores[i].cpu()
            ranked = sorted(torch.cat([sp, sc[:, None]], 1).tolist(), key=lambda x: x[2], reverse=True)
            top = [float(f'{e:.4f}') for e in ranked[0]]
            pc = float(out['pred_class'][i, 0]) if 'pred_class' in out else None
            outl.append(dict(qid=m['qid'], top1=top, pred_class=pc))
    return outl


def metrics_from(tops, metas):
    ious = np.array([iou(t['top1'][:2], [float(x) for x in m['relevant_windows'][0]]) for t, m in zip(tops, metas)])
    return {'R1@0.3': float(np.mean(ious >= 0.3) * 100), 'R1@0.5': float(np.mean(ious >= 0.5) * 100),
            'R1@0.7': float(np.mean(ious >= 0.7) * 100), 'mIoU': float(np.mean(ious) * 100), 'n': int(len(ious))}


valid_meta = [m for m in data if m['vid'] in vid_len and m['qid'] not in set(missing_t)]
sanity = {}
# (a1) our per-query (bsz-1-equivalent, no padding) clean metrics on the full test set
for mname in ('base', 'na'):
    rr = [r for r in rows[mname] if 'cond' in r]
    ious = np.array([r['cond']['clean']['iou'] for r in rr])
    sanity[f'a_clean_nopad_{mname}'] = {'R1@0.3': float(np.mean(ious >= 0.3) * 100), 'R1@0.5': float(np.mean(ious >= 0.5) * 100),
                                        'R1@0.7': float(np.mean(ious >= 0.7) * 100), 'mIoU': float(np.mean(ious) * 100), 'n': int(len(ious))}
# (a2) repo protocol (eval_bsz with padding, file order)
rb = repo_batched_eval(base, opt_b, valid_meta)
rn = repo_batched_eval(na, opt_n, valid_meta)
sanity['a_clean_repo_batched_base'] = dict(metrics_from(rb, valid_meta), eval_bsz=opt_b['eval_bsz'])
sanity['a_clean_repo_batched_na'] = dict(metrics_from(rn, valid_meta), eval_bsz=opt_n['eval_bsz'],
                                         reject_rate_pos=float(np.mean([t['pred_class'] < args.na_thr for t in rn]) * 100))


def na_aware_recall(tops, metas):
    """negative_aware_results.recall_scores: positives rejected (pred_class<thr) count as misses; strict IoU > thd."""
    ious = np.array([iou(t['top1'][:2], [float(x) for x in m['relevant_windows'][0]]) for t, m in zip(tops, metas)])
    acc = np.array([t['pred_class'] >= args.na_thr for t in tops])
    return {f'R1@{t}_strict_gt_rejected_as_miss': float(np.mean((ious > t) & acc) * 100) for t in (0.3, 0.5, 0.7)}


sanity['a_clean_repo_batched_na_negaware'] = na_aware_recall(rn, valid_meta)
_rr = [r for r in rows['na'] if 'cond' in r]
sanity['a_clean_nopad_na_negaware'] = na_aware_recall([dict(top1=r['cond']['clean']['top1'], pred_class=r['cond']['clean']['pred_class']) for r in _rr],
                                                      [dict(relevant_windows=[r['gt']]) for r in _rr])
sanity['a_reference'] = {'UniVTG_model_zoo_charades_wPT': {'R1@0.3': 72.63, 'mIoU': 52.17},
                         'NA-VMR_paper_arXiv2502.08544v2_Table2': {'UniVTG': {'R1@0.5': 60.22, 'R1@0.7': 38.55, 'ID_rej': 0.0},
                                                                   'UniVTG-NA': {'R1@0.5': 55.54, 'R1@0.7': 35.11, 'ID_rej': 64.11, 'OOD_rej': 100.0}}}
log('sanity a', json.dumps({k: v for k, v in sanity.items()}))


# (b) identity edits, bsz 1, deterministic
def outs_equal(model, X, Y, txt):
    o1 = forward(model, X[None], txt); o2 = forward(model, Y[None], txt)
    eq = all(np.array_equal(o1[k], o2[k]) for k in o1)
    md = max(float(np.max(np.abs(o1[k] - o2[k]))) for k in o1)
    return eq, md


sb = dict(n=0, empty_identical_base=0, empty_identical_na=0, self_swap_identical_base=0, self_swap_identical_na=0,
          inputs_identical_empty=0, inputs_identical_selfswap=0, max_absdiff=0.0,
          batch9_vs_bsz1_clean_maxabs_prob=0.0, batch9_vs_bsz1_top1_agree=0)
for meta in valid_meta[:args.n_sanity_b]:
    V = vfeat(meta['vid']); L = V.shape[0]; txt = tfeat(meta['qid'])
    X = np.concatenate([V, tef(L)], 1).astype(np.float32)
    E = evid_set(meta['relevant_windows'][0], L)
    Ye = apply_edit(X, [], 'swap', V, [])
    Ym = apply_edit(X, [], 'mask')
    Ys = apply_edit(X, E, 'swap', V, E)  # swap clip with itself
    sb['n'] += 1
    sb['inputs_identical_empty'] += int(np.array_equal(Ye, X) and np.array_equal(Ym, X))
    sb['inputs_identical_selfswap'] += int(np.array_equal(Ys, X))
    for mname, model in (('base', base), ('na', na)):
        e1, d1 = outs_equal(model, X, Ye, txt); e2, d2 = outs_equal(model, X, Ym, txt); e3, d3 = outs_equal(model, X, Ys, txt)
        sb[f'empty_identical_{mname}'] += int(e1 and e2)
        sb[f'self_swap_identical_{mname}'] += int(e3)
        sb['max_absdiff'] = max(sb['max_absdiff'], d1, d2, d3)
    # batch-of-9 (main loop) clean output vs bsz-1 clean output
    o1 = forward(na, X[None], txt)
    o9 = forward(na, np.stack([X] * len(COND), 0), txt)
    sb['batch9_vs_bsz1_clean_maxabs_prob'] = max(sb['batch9_vs_bsz1_clean_maxabs_prob'], float(np.max(np.abs(o9['prob'][0] - o1['prob'][0]))))
    rr = [r for r in rows['na'] if r['qid'] == meta['qid']][0]
    top = decode(o1['prob'][0], o1['spans'][0], L, float(meta['duration']))
    sb['batch9_vs_bsz1_top1_agree'] += int(top == rr['cond']['clean']['top1'])
sanity['b_identity_edits'] = sb
sanity['b_pass'] = bool(sb['n'] > 0 and sb['empty_identical_base'] == sb['n'] and sb['empty_identical_na'] == sb['n']
                        and sb['self_swap_identical_base'] == sb['n'] and sb['self_swap_identical_na'] == sb['n']
                        and sb['inputs_identical_empty'] == sb['n'] and sb['inputs_identical_selfswap'] == sb['n'])
log('sanity b', json.dumps(sb), sanity['b_pass'])


# (c) NA rejection on clean positives (no padding) and on shipped negative test sets
@torch.no_grad()
def na_pred_class_nopad(metas):
    pcs, miss = [], 0
    for m in metas:
        try:
            V = vfeat(m['vid']); txt = tfeat(m['qid'])
        except Exception:
            miss += 1; continue
        X = np.concatenate([V, tef(V.shape[0])], 1).astype(np.float32)
        pcs.append(float(forward(na, X[None], txt)['pred_class'][0]))
    return np.array(pcs), miss


pc_pos = np.array([r['cond']['clean']['pred_class'] for r in rows['na'] if 'cond' in r])
sanity['c_na_reject_rate_clean_pos_nopad'] = float(np.mean(pc_pos < args.na_thr) * 100)
sanity['c_na_threshold'] = args.na_thr
sanity['c_na_threshold_source'] = 'negative_aware_results.py: pred_threshold = 0.5; reject iff pred_class < 0.5'
if not args.limit:
    idn = load_jsonl(args.idneg_jsonl)
    idn = [m for m in idn if m['relevant_windows'] == [[0, 0]]]
    pcs, miss = na_pred_class_nopad(idn)
    sanity['c_idneg'] = dict(file=args.idneg_jsonl, file_sha256=sha256(args.idneg_jsonl), n=int(len(pcs)), n_missing_feat=miss,
                             rejection_accuracy=float(np.mean(pcs < args.na_thr) * 100) if len(pcs) else None)
    ood = [m for m in load_jsonl(args.oodneg_jsonl) if m['relevant_windows'] == [[0, 0]]]
    pcs, miss = na_pred_class_nopad(ood)
    sanity['c_oodneg'] = dict(file=args.oodneg_jsonl, file_sha256=sha256(args.oodneg_jsonl), n=int(len(pcs)), n_missing_feat=miss,
                              rejection_accuracy=float(np.mean(pcs < args.na_thr) * 100) if len(pcs) else None)
    # repo-protocol (padded batches) on the in-domain pos+neg file, as charades_inference_indomain.sh (eval_bsz=2)
    idpn = load_jsonl(args.idneg_jsonl.replace('charades_id_neg_test', 'charades_id_pos_neg_test'))
    o2 = dict(opt_n); o2['eval_bsz'] = 2
    try:
        rp = repo_batched_eval(na, o2, idpn)
        negm = [i for i, m in enumerate(idpn) if m['relevant_windows'] == [[0, 0]]]
        posm = [i for i, m in enumerate(idpn) if m['relevant_windows'] != [[0, 0]]]
        sanity['c_id_pos_neg_repo_bsz2'] = dict(
            n_neg=len(negm), n_pos=len(posm),
            rejection_accuracy=float(np.mean([rp[i]['pred_class'] < args.na_thr for i in negm]) * 100),
            pos_reject_rate=float(np.mean([rp[i]['pred_class'] < args.na_thr for i in posm]) * 100))
    except Exception as e:
        sanity['c_id_pos_neg_repo_bsz2'] = dict(error=repr(e))
log('sanity c', json.dumps({k: v for k, v in sanity.items() if k.startswith('c_')}))

# ---------------------------------------------------------------- write products
os.makedirs(args.out_dir, exist_ok=True)
script_sha = sha256(os.path.abspath(__file__))
feat_mani = {}
mf = f'{ROOT}/dl/MANIFEST.tsv'
if os.path.exists(mf):
    for line in open(mf):
        p = line.strip().split('\t')
        if len(p) == 3 and p[0].startswith('feat/'):
            feat_mani[p[0]] = p[2]
common = dict(script=os.path.abspath(__file__), script_sha256=script_sha, args=vars(args), device=str(DEV),
              torch=torch.__version__, numpy=np.__version__, pos_jsonl_sha256=sha256(args.pos_jsonl),
              n_queries=len(data), n_videos=len(all_vids), missing_videos=missing_v, missing_text=missing_t,
              conditions=COND, clip_length=CLIP_LEN, v_feat_dirs=VDIRS, t_feat_dir=TDIR, normalize_v=NORM_V,
              normalize_t=NORM_T, feature_archives_sha256=feat_mani, seeds='SeedSequence([1000, qnum, op, r]); op 1=rand index set (shared by swap_rand_r and mask_rand_r), op 2=donor (r=0; donor video+donor clip idx shared by swap_evid and swap_rand_r)',
              sanity=sanity, wall_s=time.time() - T0)
for mname, info, opt in (('base', info_b, opt_b), ('na', info_n, opt_n)):
    p = f'{args.out_dir}/navmr_{mname}_{args.tag}.json'
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(dict(model=mname, rows=rows[mname]), f)
    os.replace(tmp, p)
    meta = dict(common, model=mname, model_info=info, opt=opt, product=p, product_sha256=sha256(p),
                n_rows=len(rows[mname]), n_included=sum(1 for r in rows[mname] if r.get('excluded') is None and 'cond' in r))
    with open(p.replace('.json', '.meta.json') + '.tmp', 'w') as f:
        json.dump(meta, f, indent=1)
    os.replace(p.replace('.json', '.meta.json') + '.tmp', p.replace('.json', '.meta.json'))
    log('wrote', p, meta['product_sha256'][:16])
log('DONE')
