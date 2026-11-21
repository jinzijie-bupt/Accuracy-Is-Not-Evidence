#!/usr/bin/env python3
"""Native-video causal provenance probe for Qwen2.5-VL-style processors.

This mirrors 14_causal_provenance_probe.py but sends the sampled frames through
the processor as one `video` item instead of K independent `image` items.
"""
from __future__ import annotations

import argparse
import multiprocessing as pymp
import importlib.util
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
MODEL = str(ROOT / "models" / "Qwen2.5-VL-7B-Instruct")
VIDEO_DIR = Path(ROOT / "data" / "charades_sta" / "Charades_v1_480")
NATIVE_CACHE = ROOT / "framecache_native_diag"

_P14 = BASE / "14_causal_provenance_probe.py"
_spec14 = importlib.util.spec_from_file_location("g14_native", _P14)
g14 = importlib.util.module_from_spec(_spec14)
sys.modules["g14_native"] = g14
assert _spec14.loader is not None
_spec14.loader.exec_module(g14)

decode_video = g14.decode_video
ensure_videos = g14.ensure_videos
predecode_all = g14.predecode_all
load_durations = g14.load_durations
load_test_drn = g14.load_test_drn
parse_window = g14.parse_window
iou_1d = g14.iou_1d
apply_op = g14.apply_op
spearman = g14.spearman
mcnemar = g14.mcnemar


def load_model_and_processor(model_path, rank, attn_implementation: str | None = None):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    proc = AutoProcessor.from_pretrained(model_path)
    # AWQ/GPTQ checkpoints: quantized GEMM kernels are fp16-only; bf16 activations crash Triton.
    load_dtype = torch.bfloat16
    try:
        import json as _json, os as _os
        with open(_os.path.join(model_path, "config.json")) as _f:
            if "quantization_config" in _json.load(_f):
                load_dtype = torch.float16
    except Exception:
        pass
    kwargs = {"dtype": load_dtype, "device_map": {"": rank}}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    except Exception as exc:
        from transformers import Qwen2_5_VLForConditionalGeneration

        print(
            "[model-load] AutoModelForImageTextToText failed; fallback "
            f"Qwen2_5_VLForConditionalGeneration: {type(exc).__name__}: {exc}",
            flush=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    attn = getattr(getattr(model, "config", None), "_attn_implementation", None)
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    print(f"[model-load] attn_implementation={attn} eos_token_id={eos}", flush=True)
    if load_dtype == torch.float16:
        model = model.to(torch.float16)  # unify stray bf16 modules left by the quantizer path
    return proc, model


def decode_video_size(video: str, K: int, size: int):
    key = NATIVE_CACHE / f"{video}.K{K}.S{size}.npz"
    if key.exists():
        d = np.load(key)
        return d["frames"], float(d["dur"]), d["times"]
    import cv2

    path = VIDEO_DIR / video
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = nf / fps if nf > 0 else 0.0
    idxs = np.linspace(0, max(nf - 1, 0), K).astype(int) if nf > 0 else np.zeros(K, int)
    frames, times = [], []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if not ok:
            fr = np.zeros((size, size, 3), np.uint8)
        else:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr = cv2.resize(fr, (size, size))
        frames.append(fr)
        times.append(float(idx) / fps)
    cap.release()
    arr = np.stack(frames)
    ts = np.asarray(times, dtype=float)
    NATIVE_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(key, frames=arr, dur=dur, times=ts)
    return arr, dur, ts


def _predecode_size(args):
    video, K, size = args
    try:
        decode_video_size(video, K, size)
        return None
    except Exception as exc:
        return f"{video}: {exc}"


def predecode_all_size(videos, K: int, size: int, workers: int):
    todo = sorted({v for v in videos if not (NATIVE_CACHE / f"{v}.K{K}.S{size}.npz").exists()})
    if not todo:
        print("[decode-native] all cache hits")
        return
    print(f"[decode-native] {len(todo)} videos, K={K}, size={size}, workers={workers}")
    t0 = time.time()
    errs = []
    with pymp.Pool(workers) as pool:
        for i, err in enumerate(pool.imap_unordered(_predecode_size, [(v, K, size) for v in todo], chunksize=4), 1):
            if err:
                errs.append(err)
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[decode-native] done {len(todo)}, elapsed={time.time()-t0:.0f}s, errors={len(errs)}")
    if errs:
        print("  sample errors:", errs[:3])


def hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def prompt_text(kind: str, times, dur: float, caption: str) -> str:
    if kind == "timestamps":
        tline = ", ".join(f"{float(t):.1f}s" for t in times)
        return (
            f"This is a {dur:.1f}-second video represented by {len(times)} uniformly sampled frames "
            f"at: {tline}. During which time interval does the following happen: \"{caption}\"? "
            f"Answer ONLY as: start=<sec> end=<sec>."
        )
    if kind == "official_sec":
        return (
            f"The video duration is {dur:.1f} seconds. Detect the start and end time of "
            f"the event: \"{caption}\". Return exactly: start=<sec> end=<sec>."
        )
    if kind == "official_hms":
        return (
            f"The video duration is {hms(dur)}. Detect the start and end time of the event: "
            f"\"{caption}\". Return exactly: start=HH:MM:SS end=HH:MM:SS."
        )
    raise ValueError(f"unknown native prompt kind: {kind}")


def parse_time_token(tok: str) -> float:
    tok = tok.strip()
    if ":" not in tok:
        return float(tok)
    parts = [float(x) for x in tok.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    return parts[0]


_TIME = r"[-+]?\d*\.?\d+(?::\d*\.?\d+){0,2}"
_NUM = re.compile(r"[-+]?\d*\.?\d+")


def parse_window_native(text: str, dur: float):
    m = re.search(rf"start\s*[=:]?\s*({_TIME}).*?end\s*[=:]?\s*({_TIME})", text, re.I | re.S)
    if m:
        s, e = parse_time_token(m.group(1)), parse_time_token(m.group(2))
    else:
        m = re.search(rf"({_TIME})\s*(?:-|to|,|;)\s*({_TIME})", text, re.I)
        if m:
            s, e = parse_time_token(m.group(1)), parse_time_token(m.group(2))
        else:
            nums = _NUM.findall(text)
            if len(nums) < 2:
                return None
            s, e = float(nums[0]), float(nums[1])
    if e < s:
        s, e = e, s
    if dur > 0:
        s, e = max(0.0, min(s, dur)), max(0.0, min(e, dur))
    return (s, e)


def build_native_prompt(proc, frames, times, dur, caption, resized: int, prompt_kind: str, sample_fps_arg: float | None):
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    pil = [Image.fromarray(f) for f in frames]
    sample_fps = float(sample_fps_arg) if sample_fps_arg else len(pil) / max(float(dur), 1e-6)
    q = prompt_text(prompt_kind, times, dur, caption)
    content = [
        {
            "type": "video",
            "video": pil,
            "sample_fps": sample_fps,
            "raw_fps": sample_fps,
            "resized_height": resized,
            "resized_width": resized,
        },
        {"type": "text", "text": q},
    ]
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(msgs, return_video_kwargs=True)
    if isinstance(video_kwargs.get("fps"), list) and len(video_kwargs["fps"]) == 1:
        video_kwargs["fps"] = video_kwargs["fps"][0]
    return text, image_inputs, video_inputs, video_kwargs


def generation_ids(proc, model):
    tok = getattr(proc, "tokenizer", None)
    pad_id = getattr(tok, "pad_token_id", None)
    eos_id = getattr(tok, "eos_token_id", None)
    gen_cfg = getattr(model, "generation_config", None)
    if eos_id is None:
        eos_id = getattr(gen_cfg, "eos_token_id", None)
    if pad_id is None:
        pad_id = eos_id
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0] if eos_id else None
    if isinstance(pad_id, (list, tuple)):
        pad_id = pad_id[0] if pad_id else None
    return pad_id, eos_id


def append_partial_jsonl(path: Path | None, records) -> None:
    if not path or not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def start_item_heartbeat(rank: int, idx: int, total: int, qid, video: str, after: float, interval: float):
    if after <= 0 or interval <= 0:
        return None
    done = threading.Event()
    t0 = time.time()

    def _beat():
        wait_s = after
        while not done.wait(wait_s):
            elapsed = time.time() - t0
            print(
                f"  [gpu{rank}] HEARTBEAT idx={idx}/{total} qid={qid} "
                f"video={video} elapsed={elapsed:.0f}s",
                flush=True,
            )
            wait_s = interval

    threading.Thread(target=_beat, daemon=True).start()
    return done


def predict_iou(model, proc, frames, times, dur, caption, gs, ge, device, resized, prompt_kind, sample_fps):
    import torch

    text, image_inputs, video_inputs, video_kwargs = build_native_prompt(
        proc, frames, times, dur, caption, resized, prompt_kind, sample_fps
    )
    inp = proc(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(device)
    pad_id, eos_id = generation_ids(proc, model)
    gen_kwargs = {"max_new_tokens": 32, "do_sample": False}
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id
    if eos_id is not None:
        gen_kwargs["eos_token_id"] = eos_id
    with torch.no_grad():
        out = model.generate(**inp, **gen_kwargs)
    ans = proc.batch_decode(out[:, inp.input_ids.shape[1] :], skip_special_tokens=True)[0]
    pred = parse_window_native(ans, dur)
    return (iou_1d(pred, (gs, ge)) if pred else 0.0, pred is not None)


def gpu_worker(
    rank,
    shard,
    all_videos,
    K,
    ops,
    R,
    thr,
    ret,
    model_path=MODEL,
    adapter=None,
    resized=224,
    decode_size=224,
    prompt_kind="timestamps",
    sample_fps=None,
    seed=20260705,
    out_path=None,
    partial_every=25,
    heartbeat_after=180.0,
    heartbeat_interval=60.0,
    attn_implementation=None,
):
    import torch

    try:
        torch.cuda.set_device(rank)
        proc, model = load_model_and_processor(model_path, rank, attn_implementation)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        model = model.eval()
        rng = random.Random(seed + rank)
        rows = []
        partial_buf = []
        partial_path = None
        if out_path:
            partial_path = Path(f"{out_path}.partial.jsonl")
            if rank != 0:
                partial_path = Path(f"{out_path}.partial.rank{rank}.jsonl")
            if partial_path.exists():
                partial_path.unlink()
        t0 = time.time()

        def _record(rec):
            rows.append(rec)
            partial_buf.append(rec)
            if len(partial_buf) >= max(1, int(partial_every)):
                append_partial_jsonl(partial_path, partial_buf)
                partial_buf.clear()

        for j, it in enumerate(shard):
            item_t0 = time.time()
            hb = start_item_heartbeat(
                rank,
                j + 1,
                len(shard),
                it.get("qid"),
                it.get("video", ""),
                heartbeat_after,
                heartbeat_interval,
            )
            gens = 0
            parsed_n = 0
            try:
                frames, dur, times = decode_video_size(it["video"], K, decode_size)
                estar = np.where((times >= it["gs"]) & (times <= it["ge"]))[0]
                if len(estar) == 0 or len(estar) >= K:
                    print(
                        f"  [gpu{rank}] item {j+1}/{len(shard)} qid={it['qid']} "
                        f"sec={time.time()-item_t0:.1f} n_estar={len(estar)} "
                        "gens=0 parsed=0 skip=degenerate",
                        flush=True,
                    )
                    continue
                nonev = np.array([i for i in range(K) if i not in set(estar.tolist())])
                clean, clean_p = predict_iou(
                    model, proc, frames, times, dur, it["caption"], it["gs"], it["ge"], rank, resized, prompt_kind, sample_fps
                )
                gens += 1
                parsed_n += int(clean_p)
                rec = {
                    "qid": it["qid"],
                    "video": it["video"],
                    "caption": it["caption"],
                    "gs": it["gs"],
                    "ge": it["ge"],
                    "dur": dur,
                    "n_estar": int(len(estar)),
                    "clean_iou": float(clean),
                    "hit": int(clean >= thr),
                    "clean_parsed": int(clean_p),
                }
                if clean >= thr:
                    for op in ops:
                        dvid = rng.choice([v for v in all_videos if v != it["video"]])
                        dfr, _, _ = decode_video_size(dvid, K, decode_size)
                        didx = rng.sample(range(K), min(K, len(estar)))
                        distractor = dfr[didx]

                        fe = apply_op(frames, estar, op, distractor)
                        evid_iou, evid_p = predict_iou(
                            model,
                            proc,
                            fe,
                            times,
                            dur,
                            it["caption"],
                            it["gs"],
                            it["ge"],
                            rank,
                            resized,
                            prompt_kind,
                            sample_fps,
                        )
                        gens += 1
                        parsed_n += int(evid_p)
                        rand_ious, rand_ps = [], []
                        rand_sets = [
                            np.array(rng.sample(list(nonev), min(len(nonev), len(estar)))) for _ in range(R)
                        ]
                        for ridx in rand_sets:
                            fr = apply_op(frames, ridx, op, distractor)
                            riou, rp = predict_iou(
                                model,
                                proc,
                                fr,
                                times,
                                dur,
                                it["caption"],
                                it["gs"],
                                it["ge"],
                                rank,
                                resized,
                                prompt_kind,
                                sample_fps,
                            )
                            gens += 1
                            parsed_n += int(rp)
                            rand_ious.append(riou)
                            rand_ps.append(rp)
                        rec[f"{op}_evid_iou"] = float(evid_iou)
                        rec[f"{op}_evid_parsed"] = int(evid_p)
                        rec[f"{op}_rand_iou_mean"] = float(np.mean(rand_ious))
                        rec[f"{op}_rand_parsed_mean"] = float(np.mean(rand_ps))
                        rec[f"{op}_rand_strategy"] = "random"
                        rec[f"{op}_rand_n"] = int(len(rand_sets))
                        rec[f"{op}_evid_flip"] = int(clean >= thr and evid_iou < thr)
                        rec[f"{op}_rand_flip"] = float(np.mean([clean >= thr and ri < thr for ri in rand_ious]))
                        rec[f"{op}_evid_flip_parsed"] = int(clean >= thr and evid_iou < thr and evid_p)
                rec["item_seconds"] = float(time.time() - item_t0)
                rec["gens"] = int(gens)
                rec["parsed_n"] = int(parsed_n)
                _record(rec)
                print(
                    f"  [gpu{rank}] item {j+1}/{len(shard)} qid={it['qid']} "
                    f"sec={rec['item_seconds']:.1f} n_estar={len(estar)} "
                    f"gens={gens} parsed={parsed_n}",
                    flush=True,
                )
                if rank == 0 and (j + 1) % 10 == 0:
                    print(f"  [gpu0] {j+1}/{len(shard)} ({time.time()-t0:.0f}s)", flush=True)
            finally:
                if hb is not None:
                    hb.set()
        append_partial_jsonl(partial_path, partial_buf)
        ret[rank] = rows
    except Exception as exc:
        import traceback

        ret[rank] = {"error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-1600:]}


def report(rows, ops, thr):
    S = [r for r in rows if r["hit"]]
    print(f"\n=== native-video causal probe (n={len(rows)}, clean-hit |S|={len(S)}) ===")
    for op in ops:
        A = [r for r in S if f"{op}_evid_flip" in r]
        if not A:
            print(f"[{op}] no clean-hit rows")
            continue
        ef = float(np.mean([r[f"{op}_evid_flip"] for r in A]))
        rf = float(np.mean([r[f"{op}_rand_flip"] for r in A]))
        survive = 1.0 - ef
        mc = mcnemar([r[f"{op}_evid_flip"] for r in A], [r[f"{op}_rand_flip"] >= 0.5 for r in A])
        print(
            f"[{op}] flip_EVID={ef:.3f} flip_RAND={rf:.3f} DeltaFlip={ef-rf:+.3f} "
            f"the audit={survive:.3f} McNemar chi2={mc['chi2_df1']:.1f} "
            f"(b={mc['b_evid_only']},c={mc['c_rand_only']})"
        )
    if len(S) >= 3:
        iou = [r["clean_iou"] for r in S]
        for op in ops:
            A = [r for r in S if f"{op}_evid_flip" in r]
            if len(A) < 3:
                continue
            causal = [r[f"{op}_evid_flip"] - r[f"{op}_rand_flip"] for r in A]
            print(f"[decouple:{op}] Spearman(clean_IoU, causal_dep)={spearman(iou[:len(A)], causal):+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--frames", "-K", type=int, default=32)
    ap.add_argument("--resized", type=int, default=224)
    ap.add_argument("--decode-size", type=int, default=224)
    ap.add_argument("--native-prompt", default="timestamps", choices=["timestamps", "official_sec", "official_hms"])
    ap.add_argument("--sample-fps", type=float, default=None)
    ap.add_argument("--ops", default="mask,swap")
    ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--attn-implementation", default=None, choices=[None, "sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--partial-every", type=int, default=25)
    ap.add_argument("--heartbeat-after", type=float, default=180.0)
    ap.add_argument("--heartbeat-interval", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ops = [x.strip() for x in args.ops.split(",") if x.strip()]

    import torch
    import torch.multiprocessing as mp

    ndev = torch.cuda.device_count()
    durs = load_durations()
    ensure_videos()
    all_items = load_test_drn(durs, None)
    end = args.offset + args.n if args.n else None
    items = all_items[args.offset:end]
    all_videos = sorted({it["video"] for it in items})
    print(
        f"[native-video] items={len(items)} ops={ops} R={args.R} K={args.frames} "
        f"decode_size={args.decode_size} resized={args.resized} prompt={args.native_prompt} "
        f"sample_fps={args.sample_fps} GPUs={ndev} adapter={args.adapter}"
    )
    if not items:
        print("[native-video] no items")
        return
    predecode_all_size([it["video"] for it in items], args.frames, args.decode_size, args.workers)

    ctx = mp.get_context("spawn")
    ret = ctx.Manager().dict()
    shards = [items[r::ndev] for r in range(ndev)]
    procs = [
        ctx.Process(
            target=gpu_worker,
            args=(
                r,
                shards[r],
                all_videos,
                args.frames,
                ops,
                args.R,
                args.thr,
                ret,
                args.model,
                args.adapter,
                args.resized,
                args.decode_size,
                args.native_prompt,
                args.sample_fps,
                20260705,
                args.out,
                args.partial_every,
                args.heartbeat_after,
                args.heartbeat_interval,
                args.attn_implementation,
            ),
        )
        for r in range(ndev)
    ]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    rows = []
    for r in range(ndev):
        val = ret.get(r)
        if isinstance(val, dict) and "error" in val:
            print(f"[GPU{r} ERR] {val['error']}\n{val.get('tb', '')}")
            continue
        rows.extend(val or [])
    if not rows:
        print("[native-video] no results")
        return
    rows.sort(key=lambda r: int(r["qid"]))
    print(f"[native-video] done rows={len(rows)} elapsed={time.time()-t0:.0f}s")
    report(rows, ops, args.thr)
    with open(args.out, "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"details -> {args.out}")


if __name__ == "__main__":
    main()
