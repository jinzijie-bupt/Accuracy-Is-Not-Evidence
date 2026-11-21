# r8 (rebuttal 2026-11-06, mk_14Ln_r8.py): r7 c3a58c45bf342c6b + --qid-file + --jitter + ops freeze/occlude (defaults = r7).
# r7 (2026-09-21, mk_14Ln_r7.py): r4 db271b3f9c6e7855 + training-unseen ops noise/pscram/gblur + per-(qid,op,r) RNG (--legacy-rng = r4 stream) + --selftest; --ops default stays mask,swap.
# r4 (2026-09-20, mk_14Ln_r4.py): r3 c5c5ca0bb244ccfc + trainer r4 (timelens style, --image-patch-size) + '-' separator; defaults = r3.
# text-first variant (2026-09-20, mk_textfirst_probes.py): source b33f38c0ec599a3a + --text-first (message order text->video; trainer r3 729232059d642960 scoring path follows the same flag). Flag off = source behaviour.
# r2 (2026-09-18): use_cache=True in the clean generation (VideoChat-R1 config has use_cache=false); scoring path unchanged.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""14Ln_native_sentinel_probe.py (2026-09-17) — fixed-answer audit under the NATIVE protocol.

Same estimands and record fields as the registered 14Ls_sentinel_probe.py (264bd380db19c756, --target-mode gt,
ops mask,swap, R placebo draws of |E*| random non-evidence frames, token-mean drops), but the video input is the
native protocol shared with 12p4n_native_acc_eval_tf5meta.py and 27_papo_train_qcdro_v3seln.py:
  frames = qwen_vl_utils.fetch_video(fps, total_pixels, min_pixels, max_frames)  (T≈2·dur frames, smart-resized)
  prompt = Time-R1 'timer1_default' or VideoChat-R1 'vchat' template; clean answer generated greedily (max_new_tokens)
  target = GT window in the checkpoint's answer format, scored by teacher forcing on the same video input.
E* = frames whose timestamp lies in [gs, ge]; RAND = R draws of |E*| frames outside E* (rng seeded 1000+gpu, question order).
Sharding: --n/--offset (run several processes on several cards; readers concatenate rows by qid).
Output rows: qid, video, n_estar, clean_iou, hit, clean_parsed, target_mode, sentinel, dilate_k, gt_target_text,
  lp_clean_sum, lp_clean_mean, tgt_len, {op}_drop_evid, {op}_drop_rand, {op}_tokmean_drop_evid, {op}_tokmean_drop_rand,
  plus native extras: nframes, fps, style, clean_text (tail), pred.
"""
import argparse, os, sys, json, time, random, hashlib, importlib.util, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_T = os.path.join(HERE, "27_papo_train_qcdro_v3seln_r4.py")
_spec = importlib.util.spec_from_file_location("v3seln", _T)
V = importlib.util.module_from_spec(_spec); sys.modules["v3seln"] = V
try:
    _spec.loader.exec_module(V)
except Exception:                       # r7: --selftest runs offline (no torch); a real run still raises
    if "--selftest" not in sys.argv[1:]:
        raise
    V = None
TRAINER_SHA16 = hashlib.sha256(open(_T, "rb").read()).hexdigest()[:16]

_NUM = re.compile(r"[-+]?\d*\.?\d+")


def parse_native(text, dur):
    m = re.search(r"<answer>(.*?)</answer>", text, re.I | re.S)
    tgt = m.group(1) if m else text
    m2 = re.search(r"([-+]?\d*\.?\d+)\s*(?:to|and|,|-)\s*([-+]?\d*\.?\d+)", tgt, re.I)  # r4: TimeLens "<s> - <e> seconds"
    if not m2:
        nums = _NUM.findall(tgt)
        if len(nums) < 2:
            return None
        s, e = float(nums[0]), float(nums[1])
    else:
        s, e = float(m2.group(1)), float(m2.group(2))
    if e < s:
        s, e = e, s
    if dur > 0:
        s, e = max(0.0, min(s, dur)), max(0.0, min(e, dur))
    return (s, e)


def iou_1d(a, b):
    s1, e1 = a; s2, e2 = b
    inter = max(0.0, min(e1, e2) - max(s1, s2)); uni = max(e1, e2) - min(s1, s2)
    return inter / uni if uni > 0 else 0.0


def load_test(anno_path, n=None, offset=0):
    items = []
    with open(anno_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            left, caption = line.split("##", 1)
            vid, s, e = left.split()
            items.append({"qid": i, "video": vid + ".mp4", "gs": float(s), "ge": float(e), "caption": caption.strip()})
    return items[offset: (offset + n) if n else None]


OPS_ALL = ("mask", "swap", "noise", "pscram", "gblur", "freeze", "occlude")   # r8: + freeze/occlude (never in training)
OPS_DONOR = ("swap",)                                     # r7: the only op that needs a donor video decoded
_HO_SEED = 20260828                                       # r7: the constant of 14Lho_heldout_op_probe.py
_PERM_CACHE = {}


def rng_for(seed, qid, op, tag):
    """r7: per-(qid, op, tag) RNG stream; tag = replicate index r, or "donor"."""
    return random.Random(f"{seed}:{qid}:{op}:{tag}")


def draw_rand(nonev, k, seed, qid, op, r, rng=None):
    """r7: draw the |E*| RAND control frames. rng=None -> per-(qid,op,r) seeding (identical across arms);
    rng given -> r4's single global stream (--legacy-rng), which is consumed on hit rows only and therefore
    de-synchronises for every question after the first hit flip (verify_journal C15)."""
    return np.array((rng if rng is not None else rng_for(seed, qid, op, r)).sample(nonev, k))


def _ho_perm(nblocks):
    """r7: 14Lho's `_HO_PERM = np.random.default_rng(20260828).permutation(256)`, generalised to nblocks
    (nblocks == 256 reproduces 14Lho bit-for-bit)."""
    if nblocks not in _PERM_CACHE:
        _PERM_CACHE[nblocks] = np.random.default_rng(_HO_SEED).permutation(nblocks)
    return _PERM_CACHE[nblocks]


def _pscram_frame(img, grid):
    """r7: permute the gh x gw pixel blocks of one frame. 14Lho assumed 224x224 (16x16 blocks of 14x14);
    native frames are smart-resized, so the block grid is cropped to gh*bh x gw*bw the same way
    18_papo_train_operator_mix.corrupt_op's blur crops, and the remainder rows/cols stay untouched -> the
    pixel multiset of the frame is exactly preserved."""
    h, w = img.shape[:2]
    gh = max(1, min(int(grid), h)); gw = max(1, min(int(grid), w))
    bh, bw = max(1, h // gh), max(1, w // gw)
    hh, ww = gh * bh, gw * bw
    p = img[:hh, :ww].reshape(gh, bh, gw, bw, 3).transpose(0, 2, 1, 3, 4).reshape(gh * gw, bh, bw, 3)
    p = p[_ho_perm(gh * gw)]
    out = img.copy()
    out[:hh, :ww] = p.reshape(gh, gw, bh, bw, 3).transpose(0, 2, 1, 3, 4).reshape(hh, ww, 3)
    return out


def _gauss1d(sigma):
    r = max(1, int(round(3.0 * float(sigma))))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * float(sigma) ** 2))
    return k / k.sum()


def _gblur_frame(img, sigma):
    """r7: separable Gaussian blur with edge padding, numpy only (no cv2 -> identical on every machine).
    This is NOT the trainer's `blur`, which is 16x16 block-mean pooling."""
    k = _gauss1d(sigma); r = len(k) // 2
    a = img.astype(np.float64)
    p = np.pad(a, ((r, r), (0, 0), (0, 0)), mode="edge")
    a = sum(k[i] * p[i:i + img.shape[0]] for i in range(len(k)))
    p = np.pad(a, ((0, 0), (r, r), (0, 0)), mode="edge")
    a = sum(k[i] * p[:, i:i + img.shape[1]] for i in range(len(k)))
    return np.clip(np.rint(a), 0, 255).astype(img.dtype)


def _freeze_src(T, idx):
    """r8: for each selected index, the nearest EARLIER unselected index (else nearest later)."""
    sel = set(int(i) for i in idx); src = {}
    for i in sorted(sel):
        j = i - 1
        while j >= 0 and j in sel:
            j -= 1
        if j < 0:
            j = i + 1
            while j < T and j in sel:
                j += 1
        if j >= T:
            raise ValueError("freeze: every frame selected")
        src[i] = j
    return src


def apply_op(frames, idx, op, distractor, noise_sigma=40.0, gblur_sigma=3.0, pscram_grid=16, occ_side=0.6):
    """r4 semantics for mask/swap (unchanged) plus three operators that never appear in training:
      noise  = additive Gaussian noise (sigma in 0..255 pixel units), semantics copied from 14Lho
      pscram = pixel-block permutation (14Lho _HO_PERM), pixel multiset preserved
      gblur  = separable Gaussian blur; NOT the trainer's `blur`, which is 16x16 block-mean pooling
    Every op keeps the frame count and the frame positions: only the frames in `idx` change."""
    f = frames.copy(); e = np.asarray(idx, int)
    if op == "mask":
        f[e] = 0
    elif op == "swap":
        m = min(len(e), len(distractor)); f[e[:m]] = distractor[:m]
    elif op == "noise":
        nrng = np.random.default_rng(_HO_SEED)          # 14Lho: re-seeded per call -> deterministic
        for i in e:
            f[i] = np.clip(f[i].astype(np.int16) + nrng.normal(0, float(noise_sigma), f[i].shape), 0, 255).astype(f.dtype)
    elif op == "pscram":
        for i in e:
            f[i] = _pscram_frame(f[i], pscram_grid)
    elif op == "gblur":
        for i in e:
            f[i] = _gblur_frame(f[i], gblur_sigma)
    elif op == "freeze":                                  # r8
        for i, j in _freeze_src(frames.shape[0], e).items():
            f[i] = frames[j]
    elif op == "occlude":                                 # r8
        h, w = frames.shape[1:3]; bh, bw = int(round(h * occ_side)), int(round(w * occ_side))
        y0, x0 = (h - bh) // 2, (w - bw) // 2
        for i in e:
            f[i, y0:y0 + bh, x0:x0 + bw] = 128
    else:
        raise ValueError(op)
    return f


def selftest_r8():
    ok = True
    T = 10; fr = np.arange(T * 2 * 2 * 3, dtype=np.uint8).reshape(T, 2, 2, 3)
    f = apply_op(fr, [4, 5, 6], "freeze", None)
    ok &= bool((f[4] == fr[3]).all() and (f[6] == fr[3]).all() and (f[7] == fr[7]).all())
    f = apply_op(fr, [0, 1], "freeze", None)
    ok &= bool((f[0] == fr[2]).all() and (f[1] == fr[2]).all())
    f = apply_op(fr, [2, 7], "freeze", None)                  # placebo-style scattered frames
    ok &= bool((f[2] == fr[1]).all() and (f[7] == fr[6]).all())
    big = np.zeros((3, 10, 10, 3), np.uint8); g = apply_op(big, [1], "occlude", None, occ_side=0.6)
    ok &= bool(g[1, 2:8, 2:8].min() == 128 and g[1, 0, 0].max() == 0 and g[0].max() == 0 and g[2].max() == 0)
    print("[14Ln r8 selftest]", "PASS" if ok else "FAIL")
    return ok


def selftest():
    """r7 offline bench. Positives: the three training-unseen ops do what they claim. Counter-examples (each
    fails if the mechanism is removed): same seed -> identical rand_idx; a flipped hit must NOT de-synchronise
    later questions (with --legacy-rng it demonstrably does); an unknown op must raise instead of silently
    returning the clean frames (the 14Lho identity-fallback bug)."""
    ok = True

    def chk(c, m):
        nonlocal ok
        ok &= bool(c); print(("  PASS " if c else "  FAIL ") + m)

    rs = np.random.default_rng(7)
    fr = rs.integers(0, 256, (5, 64, 96, 3), dtype=np.uint8)
    zero = np.zeros((4, 32, 32, 3), np.uint8)
    const = np.full((4, 32, 32, 3), 100, np.uint8)
    dis = np.full((2, 64, 96, 3), 7, np.uint8)

    m = apply_op(fr, [1, 2], "mask", dis)
    chk(m[1].sum() == 0 and m[2].sum() == 0 and (m[0] == fr[0]).all() and fr[1].sum() > 0,
        "r4 regression: mask zeroes only the named frames, source array untouched")
    w = apply_op(fr, [1, 2], "swap", dis)
    chk((w[1] == 7).all() and (w[2] == 7).all() and (w[3] == fr[3]).all(), "r4 regression: swap writes donor frames")
    chk(parse_native("From 12.5 to 40 seconds.", 150.0) == (12.5, 40.0) and parse_native("no event", 10) is None,
        "r4 regression: parse_native")
    chk(abs(iou_1d((0.0, 10.0), (5.0, 15.0)) - 1.0 / 3.0) < 1e-9 and iou_1d((0.0, 1.0), (2.0, 3.0)) == 0.0,
        "r4 regression: iou_1d")

    n0 = apply_op(zero, [0, 1], "noise", None)
    chk(n0.shape == zero.shape and n0[0].var() > 0 and n0[1].var() > 0 and n0[2].var() == 0,
        "noise: variance > 0 on the named all-zero frames only")
    chk(int(n0.max()) <= 255 and int(n0.min()) >= 0 and n0.dtype == zero.dtype, "noise: clipped to 0..255, dtype kept")
    nc = apply_op(const, [0], "noise", None); d = nc[0].astype(float) - 100.0
    chk(abs(d.std() - 40.0) < 6.0 and abs(d.mean()) < 4.0, "noise: sigma ~ 40 pixel units on a mid-grey frame (obs sd %.1f)" % d.std())
    chk(np.array_equal(apply_op(const, [0], "noise", None)[0], nc[0]),
        "noise: deterministic (rng re-seeded 20260828 per call) -> same perturbation instance across arms")
    chk(not np.array_equal(apply_op(const, [0], "noise", None, noise_sigma=5.0)[0], nc[0]),
        "noise: --noise-sigma is wired (counter-example: sigma 5 gives a different frame)")

    p = apply_op(fr, [0], "pscram", None)
    chk(np.array_equal(np.sort(p[0].reshape(-1)), np.sort(fr[0].reshape(-1))), "pscram: pixel multiset unchanged")
    chk(not np.array_equal(p[0], fr[0]) and np.array_equal(p[1], fr[1]),
        "pscram: the named frame really changed, the others are untouched")
    chk(np.array_equal(apply_op(const, [0], "pscram", None)[0], const[0]),
        "pscram: identity on a constant frame (permuting equal blocks)")
    big = rs.integers(0, 256, (1, 224, 224, 3), dtype=np.uint8)
    ho = big[0].reshape(16, 14, 16, 14, 3).transpose(0, 2, 1, 3, 4).reshape(256, 14, 14, 3)
    ho = ho[np.random.default_rng(20260828).permutation(256)].reshape(16, 16, 14, 14, 3).transpose(0, 2, 1, 3, 4).reshape(224, 224, 3)
    chk(np.array_equal(apply_op(big, [0], "pscram", None)[0], ho),
        "pscram: bit-identical to 14Lho _HO_PERM on a 224x224 frame")

    g = apply_op(fr, [0], "gblur", None)
    chk(g[0].astype(float).var() < fr[0].astype(float).var() * 0.9, "gblur: variance drops on a noisy frame")
    chk(np.array_equal(apply_op(const, [0], "gblur", None)[0], const[0]),
        "gblur: constant frame preserved (kernel sums to 1, edge padding)")
    chk(apply_op(fr, [0], "gblur", None, gblur_sigma=8.0)[0].astype(float).var() < g[0].astype(float).var(),
        "gblur: larger sigma blurs more (--gblur-sigma wired)")
    chk(np.array_equal(apply_op(fr, [1], "gblur", None)[0], fr[0]), "gblur: frames outside idx untouched")
    chk(all(apply_op(fr, [1], o, dis).shape == fr.shape for o in OPS_ALL), "every op keeps frame count and frame shape")
    chk(set(("noise", "pscram", "gblur")).isdisjoint({"swap", "mask", "blur", "tshift", "shuffle"}),
        "the three added ops are disjoint from the trainer SUPPORTED_V2 (training-unseen)")
    try:
        apply_op(fr, [0], "mosaic", dis); raised = False
    except ValueError:
        raised = True
    chk(raised, "unknown op raises instead of silently returning clean frames (14Lho identity-fallback bug)")

    nonev = list(range(40)); k = 6
    a = [draw_rand(nonev, k, 1000, q, "noise", r) for q in (3, 9, 17) for r in range(3)]
    b = [draw_rand(nonev, k, 1000, q, "noise", r) for q in (3, 9, 17) for r in range(3)]
    chk(all(np.array_equal(x, y) for x, y in zip(a, b)), "same seed twice -> rand_idx bit-identical")
    chk(all(len(x) == k and len(set(x.tolist())) == k and set(x.tolist()) <= set(nonev) for x in a),
        "rand_idx: |E*| distinct indices, all outside E*")
    chk(not np.array_equal(draw_rand(nonev, k, 1000, 3, "noise", 0), draw_rand(nonev, k, 1000, 3, "pscram", 0)),
        "op is part of the key (different op -> different draw)")
    chk(not np.array_equal(draw_rand(nonev, k, 1000, 3, "noise", 0), draw_rand(nonev, k, 1000, 3, "noise", 1)),
        "replicate index is part of the key (R draws are distinct)")
    chk(not np.array_equal(draw_rand(nonev, k, 1000, 3, "noise", 0), draw_rand(nonev, k, 1001, 3, "noise", 0)),
        "--seed is wired into the key")
    cands = ["v%d.mp4" % i for i in range(50)]
    chk(rng_for(1000, 12, "swap", "donor").choice(cands) == rng_for(1000, 12, "swap", "donor").choice(cands),
        "donor video per (qid,op) reproducible across arms")

    def run(hits, rng=None):
        return [draw_rand(nonev, k, 1000, q, op, r, rng) for q in hits for op in ("mask", "swap") for r in range(3)]

    hits_a = [3, 9, 17, 25]; hits_b = [3, 17, 25]          # arm B misses question 9 (one flipped hit)
    ia = {q: run([q]) for q in hits_a}
    ib = {q: run([q]) for q in hits_b}
    chk(all(all(np.array_equal(x, y) for x, y in zip(ia[q], ib[q])) for q in hits_b),
        "a flipped hit does NOT de-synchronise the later questions (the r7 fix)")
    ra = random.Random(1000); la = {q: run([q], ra) for q in hits_a}
    rb = random.Random(1000); lb = {q: run([q], rb) for q in hits_b}
    desync = [q for q in hits_b if not all(np.array_equal(x, y) for x, y in zip(la[q], lb[q]))]
    chk(desync == [17, 25],
        "counter-example has power: with --legacy-rng the same flip de-synchronises every later question")
    chk(all(np.array_equal(x, y) for x, y in zip(la[3], lb[3])),
        "counter-example matches the C15 structure: only the questions before the flip still agree")
    print("[selftest]", "ALL PASS" if ok else "FAIL"); return ok


def main():
    if "--selftest" in sys.argv[1:]:    # r7: before argparse (--out stays required) and before torch
        sys.exit(0 if (selftest() & selftest_r8()) else 1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--adapter", default=None)
    ap.add_argument("--anno", default=None, help="charades_sta_test.txt (default: <VIDEO_DIR>/../anno/charades_sta_test.txt)")
    ap.add_argument("--n", type=int, default=None); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--ops", default="mask,swap"); ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompt-style", choices=sorted(V.NATIVE_PROMPTS), default="timer1_default")
    ap.add_argument("--answer-style", choices=["timer1", "vchat", "timelens"], default="timer1")
    ap.add_argument("--image-patch-size", type=int, default=14, help="14 Qwen2/2.5-VL (default), 16 Qwen3-VL")
    ap.add_argument("--fps", type=float, default=2.0); ap.add_argument("--total-pixels", type=int, default=3584 * 28 * 28)
    ap.add_argument("--min-pixels", type=int, default=16 * 28 * 28); ap.add_argument("--max-frames", type=int, default=768)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--legacy-rng", action="store_true", help="r7: r4 single global RNG stream (consumed on hit rows only); default = per-(qid,op,r) seeding")
    ap.add_argument("--noise-sigma", type=float, default=40.0, help="r7: --ops noise, sigma in 0..255 pixel units (14Lho value)")
    ap.add_argument("--gblur-sigma", type=float, default=3.0, help="r7: --ops gblur, Gaussian sigma in pixels")
    ap.add_argument("--pscram-grid", type=int, default=16, help="r7: --ops pscram, N x N block grid (14Lho used 16)")
    ap.add_argument("--text-first", action="store_true", help="text part before the video part (TimeZero/VideoChat-R1 order)")
    ap.add_argument("--qid-file", default=None, help="r8: run only these qids (json list or one per line)")
    ap.add_argument("--jitter", default="none", choices=["none", "shrink", "expand", "shiftl", "shiftr"])
    ap.add_argument("--jitter-frac", type=float, default=0.10)
    ap.add_argument("--occ-side", type=float, default=0.6, help="r8: occlude box side as a fraction of h and w")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    V._TEXT_FIRST["on"] = bool(args.text_first)
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    ops = [o for o in args.ops.split(",") if o]
    bad = [o for o in ops if o not in OPS_ALL]
    if bad:
        raise SystemExit(f"[14Ln r7] unknown op(s) {bad}; known = {list(OPS_ALL)}")
    anno = args.anno or os.path.join(os.path.dirname(V.base.VIDEO_DIR), "anno", "charades_sta_test.txt")
    if args.qid_file:                                    # r8
        _raw = open(args.qid_file, encoding="utf-8").read().strip()
        _q = [int(x) for x in (json.loads(_raw) if _raw.startswith("[") else _raw.split())]
        _by = {it["qid"]: it for it in load_test(anno)}
        _sel = [_by[q] for q in _q]
        items = _sel[args.offset: (args.offset + args.n) if args.n else None]
    else:
        items = load_test(anno, args.n, args.offset)
    all_videos = sorted({it["video"] for it in load_test(anno)})
    torch.cuda.set_device(args.gpu)
    proc = AutoProcessor.from_pretrained(args.model)
    model = V._fp_dtype_compat(AutoModelForImageTextToText, args.model, device_map={"": args.gpu})
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    rng = random.Random(args.seed + args.gpu) if args.legacy_rng else None   # r7: legacy single stream
    rng_mode = "legacy" if args.legacy_rng else "per_qid_op_r"
    print(f"[14Ln] n={len(items)} offset={args.offset} ops={ops} R={args.R} style={args.prompt_style}/{args.answer_style} "
          f"fps={args.fps} rng={rng_mode} trainer_sha16={TRAINER_SHA16} model={os.path.basename(args.model)} adapter={args.adapter}", flush=True)
    res = []; t0 = time.time(); nhit = 0
    for j, it in enumerate(items):
        vpath = os.path.join(V.base.VIDEO_DIR, it["video"])
        try:
            frames, times, dur, sfps = V.native_decode(vpath, args)
        except Exception as exc:
            res.append({"qid": it["qid"], "video": it["video"], "decode_error": f"{type(exc).__name__}: {exc}"}); continue
        T = frames.shape[0]
        _L = it["ge"] - it["gs"]; _f = args.jitter_frac      # r8: jittered window defines E* only
        _js, _je = {"none": (it["gs"], it["ge"]), "shrink": (it["gs"] + _f * _L, it["ge"] - _f * _L),
                    "expand": (it["gs"] - _f * _L, it["ge"] + _f * _L), "shiftl": (it["gs"] - _f * _L, it["ge"] - _f * _L),
                    "shiftr": (it["gs"] + _f * _L, it["ge"] + _f * _L)}[args.jitter]
        _js, _je = max(0.0, _js), min(float(dur), _je) if dur and dur > 0 else _je
        estar = np.asarray([i for i, t in enumerate(times) if _js - 1e-6 <= t <= _je + 1e-6], dtype=int)
        rec = {"qid": it["qid"], "video": it["video"], "n_estar": int(len(estar)), "nframes": int(T), "fps": float(sfps),
               "style": args.prompt_style, "target_mode": "gt", "sentinel": "none", "dilate_k": 0,
               "rng_mode": rng_mode, "jitter": args.jitter, "jitter_frac": args.jitter_frac}
        if len(estar) == 0 or len(estar) >= T:
            rec["degenerate"] = 1; res.append(rec); continue
        prompt_text = V.NATIVE_PROMPTS[args.prompt_style].format(it["caption"])
        cache = {}
        vid = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
        _vp = {"type": "video", "video": "native_placeholder"}; _tp = {"type": "text", "text": prompt_text}; msgs = [{"role": "user", "content": ([_tp, _vp] if args.text_first else [_vp, _tp])}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        cache["text"] = text
        inputs = V._native_inputs(proc, text, vid, sfps)
        cache["plen"] = int(inputs.input_ids.shape[1])
        inputs = inputs.to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)  # r2: see 12p4n r3
        clean_text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        pw = parse_native(clean_text, dur)
        clean = iou_1d(pw, (it["gs"], it["ge"])) if pw else 0.0
        rec.update({"clean_iou": float(clean), "hit": int(clean >= args.thr), "clean_parsed": int(pw is not None),
                    "pred": pw, "clean_text": clean_text[-120:]})
        if clean < args.thr:
            res.append(rec); continue
        nhit += 1
        tgt_txt = V.native_completion(it["gs"], it["ge"], args.answer_style)
        rec["gt_target_text"] = tgt_txt

        def score(fr):
            with torch.no_grad():
                s, n = V.native_completion_logprob(model, proc, fr, sfps, prompt_text, tgt_txt, model.device, cache)
            return float(s), int(n)

        lp_c_sum, tgt_len = score(frames)
        rec["lp_clean_sum"] = lp_c_sum; rec["lp_clean_mean"] = lp_c_sum / max(1, tgt_len); rec["tgt_len"] = tgt_len
        nonev = [i for i in range(T) if i not in set(estar.tolist())]
        okw = dict(noise_sigma=args.noise_sigma, gblur_sigma=args.gblur_sigma, pscram_grid=args.pscram_grid, occ_side=args.occ_side)
        for op in ops:
            dvid = None; distractor = None
            if args.legacy_rng or op in OPS_DONOR:      # r7: only swap uses the donor (r4 decoded one per op)
                drng = rng if args.legacy_rng else rng_for(args.seed, it["qid"], op, "donor")
                dvid = drng.choice([v for v in all_videos if v != it["video"]])
                dfr, _, _, _ = V.native_decode(os.path.join(V.base.VIDEO_DIR, dvid), args)
                if dfr.shape[1:3] != frames.shape[1:3]:
                    import cv2
                    dfr = np.stack([cv2.resize(np.ascontiguousarray(f), (frames.shape[2], frames.shape[1])) for f in dfr])
                didx = drng.sample(range(dfr.shape[0]), min(dfr.shape[0], len(estar)))
                distractor = dfr[didx]
            lp_e_sum, _ = score(apply_op(frames, estar, op, distractor, **okw))
            rand_sums = []; rand_idx = []
            for r_i in range(args.R):
                ridx = draw_rand(nonev, min(len(nonev), len(estar)), args.seed, it["qid"], op, r_i, rng)
                rand_idx.append([int(x) for x in ridx])
                rs, _ = score(apply_op(frames, ridx, op, distractor, **okw))
                rand_sums.append(rs)
            rec[f"{op}_drop_evid"] = lp_c_sum - lp_e_sum
            rec[f"{op}_drop_rand"] = lp_c_sum - float(np.mean(rand_sums))
            rec[f"{op}_tokmean_drop_evid"] = (lp_c_sum - lp_e_sum) / max(1, tgt_len)
            rec[f"{op}_tokmean_drop_rand"] = (lp_c_sum - float(np.mean(rand_sums))) / max(1, tgt_len)
            rec[f"{op}_rand_sums"] = rand_sums; rec[f"{op}_rand_idx"] = rand_idx; rec[f"{op}_donor"] = dvid
        res.append(rec)
        if (j + 1) % 25 == 0:
            print(f"  [14Ln] {j+1}/{len(items)} hits={nhit} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False)
    meta = {"instrument": "14Ln_native_sentinel_probe_v1r2_tf_q3_r8", "trainer_sha16": TRAINER_SHA16, "args": vars(args),
            "ops": ops, "rng_mode": rng_mode, "probe_sha16": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16],
            "n_rows": len(res), "n_hits": nhit, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=1)
    print(f"[14Ln] DONE n={len(res)} hits={nhit} elapsed={time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
