# r8: r5 93c38595d8d30bd7 + --query-op {none,foreign} / --query-seed (true-negative query control: each item is asked with a caption from a different video) + --selftest + <out>.meta.json prov sidecar; defaults reproduce r5 line for line.
# r5: r4 51656eed173f17c1 + prompt style tz_official (TimeZero/VideoChat-R1 official CoT template, 2048 tokens) + --text-first (official message order); rows carry text_first/max_new_tokens.
# r4: r3 + --video-op black (all-zero frames, same token layout) for the no-visual-content control.
# r3: r2 + use_cache=True in generate (VideoChat-R1_7B config/generation_config carry use_cache=false; greedy outputs unchanged).
# r2: + per-row partial dump to <out>.partial.jsonl and resume from it (skip scored qids); final json schema identical to r1 0c750a851c2b15ab.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""12p4n_native_acc_eval.py — Charades-STA grounding accuracy under the "native protocol" (
under our K=16 uniform frames + fixed prompt, each checkpoint scores far below its self-reported accuracy; is the accuracy ranking a protocol artifact?).
Reuses the dataset/scoring row structure of 12p4a (e4612b61) (load_set_a / parse_window / row keys); changes only two things:
  (1) video input uses fps sampling from qwen_vl_utils.process_vision_info (default fps=2.0, total_pixels=3584*28*28, min_pixels=16*28*28, matching the Time-R1 paper "2 FPS, ≈2.8M pixels" and
     VideoChat-R1 evaluate_chara.py; frame cap is the qwen_vl_utils default 768);
  (2) prompt chosen by --prompt-style: timer1_default / timer1_r1 / timer1_nocot (verbatim from the three blocks in Time-R1 config.py) / vchat (verbatim from VideoChat-R1 evaluate_chara.py, answer [(s, e)]). Parsing: "a to b" or "[(a, b)]" or a start/end number pair inside <answer>; on failure falls back to 12p4a.parse_window.
Output rows = same keys as 12p4a (qid/video/caption/dur/pred/gsA/geA/gsB/geB/raw) + style/fps/nframes, read directly by the p4acc_window_dev / native_acc_dev readers.
"""
import argparse, os, re, sys, json, time, hashlib, importlib.util
B = os.path.dirname(os.path.abspath(__file__))
_p4a = os.path.join(B, '12p4a_grounding_eval_adapter.py')
_sha = hashlib.sha256(open(_p4a, 'rb').read()).hexdigest(); assert _sha.startswith('1dbbd1ba'), f'12p4a sha mismatch {_sha[:16]}'
spec = importlib.util.spec_from_file_location('p4a', _p4a); p4a = importlib.util.module_from_spec(spec); sys.modules['p4a'] = p4a; spec.loader.exec_module(p4a)

PROMPTS = {
    'timer1_default': 'To accurately pinpoint the event "{}" in the video, determine the precise time period of the event.\n\nProvide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83".',
    'timer1_r1': 'To accurately pinpoint the event "{}" in the video, determine the precise time period of the event.\n\nOutput your thought process within the <think> </think> tags, including analysis with either specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.\n\nThen, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83".',
    'timer1_nocot': 'To accurately pinpoint the event "{}" in the video, determine the precise time period of the event.\n\nOutput your thought process within the <think> </think> tags, including analysis with either specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.\n\nThen, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83".\n<think>\n</think>',
    'tz_official': 'To accurately pinpoint the event "{}" in the video, determine the precise time period of the event.\n\nOutput your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.\n\nThen, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83".',
    'vchat': 'To accurately pinpoint the event "{}" in the video, determine the precise time period of the event. Provide the start and end times (in seconds, precise to two decimal places) in the format of [(s1, e1), (s2, e2), ...] within the <answer> </answer> tags. For example: <answer>[(12.54, 17.83)]</answer>',
}
_NUM = re.compile(r'[-+]?\d*\.?\d+')


def parse_native(text, dur):
    m = re.search(r'<answer>(.*?)</answer>', text, re.I | re.S); tgt = m.group(1) if m else text
    m2 = re.search(r'([-+]?\d*\.?\d+)\s*(?:to|and|,)\s*([-+]?\d*\.?\d+)', tgt, re.I)
    if m2: s, e = float(m2.group(1)), float(m2.group(2))
    else: return p4a.parse_window(text, dur)
    if e < s: s, e = e, s
    if dur > 0: s, e = max(0.0, min(s, dur)), max(0.0, min(e, dur))
    return (s, e)


def _partial_path(out): return out + '.partial.jsonl'


def load_partial(out):
    """r2: rows already scored in an earlier (killed) run, keyed by qid; missing/empty file -> {}."""
    p = _partial_path(out); done = {}
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: r = json.loads(line)
                except Exception: continue  # torn last line from a kill mid-write
                done[int(r['qid'])] = r
    return done


QUERY_OPS = ('none', 'foreign')


def _self_sha16():
    return hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:16]


def build_partner_map(items, seed):
    """r8: deterministic foreign-query partner for every item, drawn from INSIDE the evaluated slice.
    Items are ranked by qid; the partner of rank k is the first rank (k + off + d) % n (d = 0, 1, 2, ...) whose
    video differs, with off = 1 + seed % (n - 1) a fixed non-zero cyclic offset.  The map is therefore a pure
    function of (sorted qid list, their videos, seed): identical across runs, machines and row orders."""
    n = len(items)
    if n < 2:
        raise SystemExit('[p4n] foreign: need >= 2 items in the slice, got %d (absent, not zero)' % n)
    order = sorted(range(n), key=lambda i: int(items[i]['qid']))
    off = 1 + (int(seed) % (n - 1))
    pmap = {}
    for k, i in enumerate(order):
        for d in range(n):
            j = order[(k + off + d) % n]
            if str(items[j]['video']) != str(items[i]['video']):
                pmap[int(items[i]['qid'])] = int(items[j]['qid'])
                break
        else:
            raise SystemExit('[p4n] foreign: no different-video partner for qid %s' % items[i]['qid'])
    if len(pmap) != n:
        raise SystemExit('[p4n] foreign: partner map covers %d of %d items' % (len(pmap), n))
    return pmap


def partner_map_sha16(pmap):
    blob = json.dumps({str(k): int(v) for k, v in sorted(pmap.items())}, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def pick_query(it, query_op, pmap, by_qid):
    """-> (used_query, partner_qid, partner_video).  'none' returns the row's own caption and no partner."""
    if query_op == 'none':
        return it['caption'], None, None
    pj = pmap[int(it['qid'])]
    p = by_qid[pj]
    if str(p['video']) == str(it['video']):  # holds even for a forged map: a same-video partner is never used
        raise SystemExit('[p4n] foreign: partner qid %s shares the video of qid %s' % (pj, it['qid']))
    return p['caption'], int(pj), str(p['video'])


def _synth(qids, vids):
    return [{'qid': q, 'video': v, 'caption': 'cap%d' % q} for q, v in zip(qids, vids)]


def selftest():
    ok = True

    def chk(c, m):
        nonlocal ok
        ok &= bool(c)
        print(('  PASS ' if c else '  FAIL ') + m)

    it = _synth([5, 1, 3, 2, 4, 0], ['a.mp4', 'a.mp4', 'b.mp4', 'b.mp4', 'c.mp4', 'a.mp4'])
    by = {int(x['qid']): x for x in it}
    m0 = build_partner_map(it, 0)
    chk(all(by[p]['video'] != by[q]['video'] for q, p in m0.items()), 'positive: every partner comes from a different video')
    chk(len(m0) == len(it) and set(m0) == set(by), 'positive: map covers every qid exactly once')
    chk(partner_map_sha16(build_partner_map(it, 0)) == partner_map_sha16(m0), 'positive: map byte-identical across two runs')
    chk(partner_map_sha16(build_partner_map(list(reversed(it)), 0)) == partner_map_sha16(m0), 'positive: map keyed on qid order, not on row order')
    chk(partner_map_sha16(build_partner_map(it, 1)) != partner_map_sha16(m0), 'negative: a different --query-seed changes the map (the seed is wired, not decorative)')
    try:
        build_partner_map(_synth([0, 1, 2], ['a.mp4'] * 3), 0)
        chk(False, 'negative: single-video slice must be refused')
    except SystemExit:
        chk(True, 'negative: single-video slice refused instead of silently returning a same-video partner')
    u, pq, pv = pick_query(it[0], 'none', m0, by)
    chk(u == it[0]['caption'] and pq is None and pv is None, 'none: used_query == orig_query, no partner recorded')
    u2, pq2, pv2 = pick_query(it[0], 'foreign', m0, by)
    chk(u2 == by[pq2]['caption'] and pv2 != it[0]['video'] and u2 != it[0]['caption'], 'foreign: used_query is the partner caption and the partner video differs')
    try:
        pick_query(it[0], 'foreign', {5: 1}, by)  # qid 1 is the same video as qid 5
        chk(False, 'negative: forged same-video partner must be refused')
    except SystemExit:
        chk(True, 'negative: forged same-video partner refused at use time')
    ap = _build_ap()
    try:
        ap.parse_args(['--query-op', 'bogus', '--model', 'm', '--out', 'o'])
        chk(False, 'negative: unknown --query-op rejected')
    except SystemExit:
        chk(True, 'negative: unknown --query-op rejected by the real parser (argparse runs before "import torch")')
    a = _build_ap().parse_args(['--model', 'm', '--out', 'o'])
    chk(a.query_op == 'none' and a.query_seed == 0 and a.text_first is False and a.video_op == 'none', 'defaults reproduce r5 (query_op=none, seed=0)')
    src = open(os.path.abspath(__file__), encoding='utf-8').read()
    chk(0 < src.index('if args.selftest') < src.index('    import torch'), 'structure: the selftest dispatch precedes "import torch" in main()')
    chk("'used_query'" in src and "'partner_qid'" in src and "'partner_map_sha16'" in src, 'structure: rows carry used_query/partner_qid and meta carries partner_map_sha16')
    print('[selftest]', 'ALL PASS' if ok else 'FAIL')
    print('[selftest] SKIP (needs torch/GPU, run on the GPU node): --n 3 smoke and the landing gate against the frozen r5/r3 product')
    return ok


def _build_ap():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model'); ap.add_argument('--adapter', default=None)
    ap.add_argument('--n', type=int, default=None); ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--set', choices=['orig'], default='orig')
    ap.add_argument('--prompt-style', choices=sorted(PROMPTS), default='timer1_default')
    ap.add_argument('--fps', type=float, default=2.0); ap.add_argument('--total-pixels', type=int, default=3584 * 28 * 28)
    ap.add_argument('--min-pixels', type=int, default=16 * 28 * 28); ap.add_argument('--max-frames', type=int, default=768)
    ap.add_argument('--max-new-tokens', type=int, default=None); ap.add_argument('--out')
    ap.add_argument('--query-op', choices=QUERY_OPS, default='none', help="r8: foreign = ask every item with a caption taken from a different video (true-negative query control)")
    ap.add_argument('--query-seed', type=int, default=0, help='r8: picks the fixed cyclic offset of the partner map')
    ap.add_argument('--selftest', action='store_true', help='r8: positive + negative controls, no torch needed')
    ap.add_argument('--text-first', action='store_true', help='r5: text part before the video part (TimeZero evaluate.py order)')
    ap.add_argument('--video-op', choices=['none', 'black'], default='none', help='r4: black = zero every frame after decoding (same token count, no visual content)')
    return ap


def main():
    args = _build_ap().parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.model or not args.out:
        _build_ap().error('--model and --out are required unless --selftest')
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info
    items = p4a.load_set_a(None)[args.offset: (args.offset + args.n) if args.n else None]
    by_qid = {int(x['qid']): x for x in items}
    assert len(by_qid) == len(items), 'duplicate qid inside the slice'
    pmap, pmap_sha = {}, None
    if args.query_op == 'foreign':
        pmap = build_partner_map(items, args.query_seed); pmap_sha = partner_map_sha16(pmap)
        print('[p4n] r8 foreign: partner map n=%d sha16=%s seed=%d' % (len(pmap), pmap_sha, args.query_seed), flush=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    dm = os.environ.get('L14_DEVICE_MAP') or {'': 0}
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dm, trust_remote_code=True)
    if args.adapter:
        from peft import PeftModel; model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    mnt = args.max_new_tokens or (2048 if args.prompt_style == 'tz_official' else (1024 if args.prompt_style == 'timer1_r1' else 128))
    done = load_partial(args.out); res = []; nparse = 0; t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if done: print(f'[p4n] resume: {len(done)} rows from {_partial_path(args.out)}', flush=True)
    pf = open(_partial_path(args.out), 'a', encoding='utf-8')
    for j, it in enumerate(items):
        if int(it['qid']) in done:
            r = done[int(it['qid'])]; res.append(r); nparse += int(r.get('pred') is not None); continue
        vpath = os.path.join(p4a.VIDEO_DIR, it['video'])
        q, _pqid, _pvid = pick_query(it, args.query_op, pmap, by_qid); content_text = PROMPTS[args.prompt_style].format(q)
        _vpart = {'type': 'video', 'video': vpath, 'fps': args.fps, 'total_pixels': args.total_pixels, 'min_pixels': args.min_pixels, 'max_frames': args.max_frames}; _tpart = {'type': 'text', 'text': content_text}
        msgs = [{'role': 'user', 'content': ([_tpart, _vpart] if args.text_first else [_vpart, _tpart])}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        try:
            imgs, vids, vkw = process_vision_info(msgs, return_video_kwargs=True)
        except TypeError:
            imgs, vids = process_vision_info(msgs); vkw = {}
        if isinstance(vkw.get('fps'), list) and len(vkw['fps']) == 1: vkw['fps'] = vkw['fps'][0]  # tf5 strict typed-dict: fps must be scalar
        fps_v = vkw.get('fps', args.fps)
        if isinstance(fps_v, list): fps_v = fps_v[0]
        if args.video_op == 'black': vids = [torch.zeros_like(vids[0])]  # r4: same T/H/W, all-zero pixels
        nfr0 = int(vids[0].shape[0])
        meta = [{'fps': float(fps_v), 'total_num_frames': nfr0, 'duration': nfr0 / float(fps_v), 'video_backend': 'decord', 'frames_indices': list(range(nfr0))}]
        inputs = proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors='pt', video_metadata=meta, do_sample_frames=False)
        if j == 0: print('[tf5meta] second_per_grid_ts=', inputs.get('second_per_grid_ts'), 'video_grid_thw=', inputs.get('video_grid_thw'), flush=True)
        inputs = inputs.to(model.device)
        nfr = int(vids[0].shape[0]) if vids is not None and hasattr(vids[0], 'shape') else -1
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=mnt, do_sample=False, use_cache=True)  # r3: VideoChat-R1 config sets use_cache=false -> 12x slower greedy decode, same outputs
        ans = proc.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        dur = float(it.get('dur') or 0.0)
        if dur <= 0:
            try:
                import cv2; cap = cv2.VideoCapture(vpath); dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 24.0); cap.release()
            except Exception: dur = 0.0
        pw = parse_native(ans, dur)
        if pw is not None: nparse += 1
        res.append({'set': args.set, 'qid': it['qid'], 'video': it['video'], 'caption': it['caption'], 'dur': dur, 'pred': pw,
                    'gsA': it['gsA'], 'geA': it['geA'], 'gsB': it.get('gsB'), 'geB': it.get('geB'), 'raw': ans[-160:],
                    'style': args.prompt_style, 'fps': args.fps, 'nframes': nfr, 'total_pixels': args.total_pixels, 'video_op': args.video_op, 'text_first': int(args.text_first), 'max_new_tokens': mnt,
                    'query_op': args.query_op, 'query_seed': args.query_seed, 'partner_qid': _pqid, 'partner_video': _pvid,
                    'orig_query': it['caption'], 'used_query': q})
        pf.write(json.dumps(res[-1], ensure_ascii=False) + chr(10)); pf.flush()
        if (j + 1) % 50 == 0: print(f'  [p4n] {j+1}/{len(items)} parse={nparse/(j+1):.3f} nframes={nfr} ({time.time()-t0:.0f}s)', flush=True)
    pf.close()
    assert len(res) == len(items) and [r['qid'] for r in res] == [it['qid'] for it in items], 'row/qid order mismatch'
    json.dump(res, open(args.out, 'w'), ensure_ascii=False)
    _meta = {'instrument': os.path.basename(os.path.abspath(__file__)), 'instrument_sha16': _self_sha16(),
             'prov': {'instrument': os.path.basename(os.path.abspath(__file__)), 'instrument_sha16': _self_sha16(),
                      'query_op': args.query_op, 'query_seed': args.query_seed, 'partner_map_sha16': pmap_sha,
                      'partner_map_reason': None if pmap_sha else 'query_op=none: no partner map is built',
                      'model': args.model, 'adapter': args.adapter, 'prompt_style': args.prompt_style,
                      'text_first': int(args.text_first), 'video_op': args.video_op},
             'args': vars(args), 'n_rows': len(res), 'parse_rate': nparse / max(len(res), 1),
             'elapsed_s': round(time.time() - t0, 1),
             'partner_map': {str(k): int(v) for k, v in sorted(pmap.items())} if pmap else None}
    json.dump(_meta, open(args.out + '.meta.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'[p4n] DONE n={len(res)} parse_rate={nparse/max(len(res),1):.3f} style={args.prompt_style} fps={args.fps} elapsed={time.time()-t0:.0f}s -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
