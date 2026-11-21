# r5 (2026-09-20, mk_12p4n_r5.py): r4 2573fe50c8e671ef + prompt style tz_official (TimeZero/VideoChat-R1 official CoT template, 2048 tokens) + --text-first (official message order); rows carry text_first/max_new_tokens.
# r4 (2026-09-18): r3 + --video-op black (all-zero frames, same token layout) for the no-visual-content control.
# r3 (2026-09-18): r2 + use_cache=True in generate (VideoChat-R1_7B config/generation_config carry use_cache=false; greedy outputs unchanged).
# r2 (2026-09-18, mk_12p4n_r2.py): + per-row partial dump to <out>.partial.jsonl and resume from it (skip scored qids); final json schema identical to r1 33e94529686a5769.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""12p4n_native_acc_eval.py (2026-09-03, new lineage; measures clean accuracy only, does not touch the audit endpoints) — Charades-STA grounding accuracy under the "native protocol" (addresses review items B-2/Q6.1:
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
    'rarft_native': 'Given a video and a query "{}", determine whether the video contains a segment that is relevant to the query.\n\n**If a relevant segment exists**, output your thought process within the <think> </think> tags, including analysis with specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.\nThen, provide the start and end times (in seconds, precise to two decimal places) within the <answer> </answer> tags.\nFor example: <answer>12.54 to 17.83</answer>. Finally, provide "<correction>NIL</correction>".\n\n**If no relevant segment exists**, output your thought process within the <think> </think> tags, including analysis with specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.\nThen, state explicitly that the video does not contain a relevant segment and explain why you think so within the <answer> </answer> tags.\nFinally, based on the given video and query, output a corrected query that is likely to be relevant query to the video within the <correction> </correction> tags.\n',
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


def parse_native_rarft(text, dur):
    """r6: interval only from inside <answer>; none -> None (RA-RFT 'no relevant segment')."""
    m = re.search(r'<answer>(.*?)</answer>', text, re.I | re.S)
    if not m: return None
    m2 = re.search(r'([-+]?\d*\.?\d+)\s*(?:to|and|,)\s*([-+]?\d*\.?\d+)', m.group(1), re.I)
    if not m2: return None
    s, e = float(m2.group(1)), float(m2.group(2))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--adapter', default=None)
    ap.add_argument('--n', type=int, default=None); ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--set', choices=['orig'], default='orig')
    ap.add_argument('--prompt-style', choices=sorted(PROMPTS), default='timer1_default')
    ap.add_argument('--fps', type=float, default=2.0); ap.add_argument('--total-pixels', type=int, default=3584 * 28 * 28)
    ap.add_argument('--min-pixels', type=int, default=16 * 28 * 28); ap.add_argument('--max-frames', type=int, default=768)
    ap.add_argument('--max-new-tokens', type=int, default=None); ap.add_argument('--out', required=True)
    ap.add_argument('--text-first', action='store_true', help='r5: text part before the video part (TimeZero evaluate.py order)')
    ap.add_argument('--video-op', choices=['none', 'black'], default='none', help='r4: black = zero every frame after decoding (same token count, no visual content)')
    args = ap.parse_args()
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info
    items = p4a.load_set_a(None)[args.offset: (args.offset + args.n) if args.n else None]
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    dm = os.environ.get('L14_DEVICE_MAP') or {'': 0}
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=dm, trust_remote_code=True)
    if args.adapter:
        from peft import PeftModel; model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    mnt = args.max_new_tokens or (512 if args.prompt_style == 'rarft_native' else 2048 if args.prompt_style == 'tz_official' else (1024 if args.prompt_style == 'timer1_r1' else 128))
    done = load_partial(args.out); res = []; nparse = 0; t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if done: print(f'[p4n] resume: {len(done)} rows from {_partial_path(args.out)}', flush=True)
    pf = open(_partial_path(args.out), 'a', encoding='utf-8')
    for j, it in enumerate(items):
        if int(it['qid']) in done:
            r = done[int(it['qid'])]; res.append(r); nparse += int(r.get('pred') is not None); continue
        vpath = os.path.join(p4a.VIDEO_DIR, it['video'])
        q = it['caption']; content_text = PROMPTS[args.prompt_style].format(q)
        _vpart = {'type': 'video', 'video': vpath, 'fps': args.fps, 'total_pixels': args.total_pixels, 'min_pixels': args.min_pixels, 'max_frames': args.max_frames}; _tpart = {'type': 'text', 'text': content_text}
        msgs = [{'role': 'user', 'content': ([_tpart, _vpart] if args.text_first else [_vpart, _tpart])}]
        if args.prompt_style == 'rarft_native': msgs = [{'role': 'system', 'content': [{'type': 'text', 'text': 'You are a video analysis expert.'}]}] + msgs  # r6
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        try:
            imgs, vids, vkw = process_vision_info(msgs, return_video_kwargs=True)
        except TypeError:
            imgs, vids = process_vision_info(msgs); vkw = {}
        pass  # tf451 variant: keep fps as returned by qwen_vl_utils (list), as in TimeZero evaluate.py
        if args.video_op == 'black': vids = [torch.zeros_like(vids[0])]  # r4: same T/H/W, all-zero pixels
        inputs = proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors='pt', **vkw)
        if j == 0: print('[tf451] vkw=', vkw, 'second_per_grid_ts=', inputs.get('second_per_grid_ts'), 'video_grid_thw=', inputs.get('video_grid_thw'), flush=True)
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
        pw = parse_native_rarft(ans, dur) if args.prompt_style == 'rarft_native' else parse_native(ans, dur)
        if pw is not None: nparse += 1
        res.append({'set': args.set, 'qid': it['qid'], 'video': it['video'], 'caption': it['caption'], 'dur': dur, 'pred': pw,
                    'gsA': it['gsA'], 'geA': it['geA'], 'gsB': it.get('gsB'), 'geB': it.get('geB'), 'raw': ans[-160:],
                    'style': args.prompt_style, 'fps': args.fps, 'nframes': nfr, 'total_pixels': args.total_pixels, 'video_op': args.video_op, 'text_first': int(args.text_first), 'max_new_tokens': mnt})
        pf.write(json.dumps(res[-1], ensure_ascii=False) + chr(10)); pf.flush()
        if (j + 1) % 50 == 0: print(f'  [p4n] {j+1}/{len(items)} parse={nparse/(j+1):.3f} nframes={nfr} ({time.time()-t0:.0f}s)', flush=True)
    pf.close()
    assert len(res) == len(items) and [r['qid'] for r in res] == [it['qid'] for it in items], 'row/qid order mismatch'
    json.dump(res, open(args.out, 'w'), ensure_ascii=False)
    print(f'[p4n] DONE n={len(res)} parse_rate={nparse/max(len(res),1):.3f} style={args.prompt_style} fps={args.fps} elapsed={time.time()-t0:.0f}s -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
