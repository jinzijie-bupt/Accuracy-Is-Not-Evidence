#!/usr/bin/env python3
"""
12_charades_grounding.py — zero-shot temporal grounding R@1 baseline for base Qwen2.5-VL on Charades-STA.

First milestone for A-lever/B-safe: get the real R@1 of the base model first; only then does 11_transfer_killprobe
provenance-steering have a reference baseline.

Design against a known issue (CPU decoding starving the GPU):
  - Stage 1 [CPU, multiprocess]: sample K frames from every video used + measure duration, cache to framecache/*.npz.
  - Stage 2 [GPU, one process per card]: read frames from cache and run Qwen, sharded across two cards (≈8.8 min/full set, see infer_bench).
  Decoding is decoupled from inference -> GPU does not wait on CPU; the cache means zero decoding on reruns/prompt changes.

Usage:
  python 12_charades_grounding.py --n 50        # small batch first to check the pipeline
  python 12_charades_grounding.py               # full set 3720
"""
import argparse, os, glob, re, json, time
import numpy as np

ROOT = os.environ.get("DATA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA      = f"{ROOT}/data/charades_sta"
VIDEO_DIR = f"{DATA}/Charades_v1_480"
CACHE     = f"{ROOT}/framecache"
MODEL     = f"{ROOT}/models/Qwen2.5-VL-7B-Instruct"
IOU_THR   = (0.3, 0.5, 0.7)


# ---------------- data ----------------
def load_annotations(n=None):
    import pandas as pd
    f = glob.glob(f"{DATA}/data/*.parquet")[0]
    df = pd.read_parquet(f)
    items = []
    for i, r in enumerate(df.itertuples(index=False)):
        ts = r.timestamp
        items.append({"qid": i, "video": r.video, "caption": r.caption,
                      "gs": float(ts[0]), "ge": float(ts[1])})
    return items[:n] if n else items


def ensure_videos():
    """Extract finished zips into Charades_v1_480/ (python zipfile, no unzip binary needed).
    Each zip is extracted once (.extracted marker); incomplete/corrupt ones are skipped and extracted after download completes -> supports running while downloading."""
    import zipfile
    for z in sorted(glob.glob(f"{DATA}/*.zip")):
        marker = z + ".extracted"
        if os.path.exists(marker):
            continue
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(DATA)
            open(marker, "w").close()
            print(f"[prep] extracting {os.path.basename(z)} OK")
        except Exception as e:
            print(f"[prep] skipping {os.path.basename(z)} (incomplete/corrupt): {e}")
    print(f"[prep] mp4 available now: {len(glob.glob(f'{VIDEO_DIR}/*.mp4'))}")


# ---------------- decoding (CPU, parallelizable) ----------------
def decode_video(video, K):
    key = os.path.join(CACHE, f"{video}.K{K}.npz")
    if os.path.exists(key):
        d = np.load(key)
        return d["frames"], float(d["dur"]), d["times"]
    import cv2
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = nf / fps if nf > 0 else 0.0
    idxs = np.linspace(0, max(nf - 1, 0), K).astype(int) if nf > 0 else np.zeros(K, int)
    frames, times = [], []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if not ok:
            fr = np.zeros((224, 224, 3), np.uint8)
        else:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(fr, (224, 224)))
        times.append(float(idx) / fps)
    cap.release()
    frames, times = np.stack(frames), np.array(times)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(key, frames=frames, dur=dur, times=times)
    return frames, dur, times


def _predecode(args):
    v, K = args
    try:
        decode_video(v, K); return None
    except Exception as e:
        return f"{v}: {e}"


def predecode_all(videos, K, workers):
    from multiprocessing import Pool
    todo = sorted({v for v in videos if not os.path.exists(os.path.join(CACHE, f"{v}.K{K}.npz"))})
    if not todo:
        print("[decode] all cached"); return
    print(f"[decode] {len(todo)} videos, {workers} processes extracting frames in parallel")
    t0 = time.time()
    errs = []
    with Pool(workers) as p:
        for i, e in enumerate(p.imap_unordered(_predecode, [(v, K) for v in todo], chunksize=4), 1):
            if e: errs.append(e)
            if i % 200 == 0: print(f"  {i}/{len(todo)}  ({time.time()-t0:.0f}s)")
    print(f"[decode] done {len(todo)} items, {time.time()-t0:.0f}s, failed {len(errs)}")
    if errs: print("  sample errors:", errs[:3])


# ---------------- metrics ----------------
def iou_1d(a, b):
    s, e = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


_NUM = re.compile(r"[-+]?\d*\.?\d+")
def parse_window(text, dur):
    answer = re.search(r"<answer>(.*?)</answer>", text, re.I | re.S)
    target = answer.group(1) if answer else text
    m = re.search(r"start\s*[=:]?\s*([-+]?\d*\.?\d+).*?end\s*[=:]?\s*([-+]?\d*\.?\d+)", target, re.I | re.S)
    if m:
        s, e = float(m.group(1)), float(m.group(2))
    else:
        nums = _NUM.findall(target)
        if len(nums) < 2: return None
        s, e = float(nums[0]), float(nums[1])
    if e < s: s, e = e, s
    if dur > 0: s, e = max(0.0, min(s, dur)), max(0.0, min(e, dur))
    return (s, e)


# ---------------- GPU inference (one process per card) ----------------
def build_prompt(proc, frames, times, dur, caption, system_prompt=None, answer_instruction=None):
    from PIL import Image
    pil = [Image.fromarray(f) for f in frames]
    tline = ", ".join(f"{t:.1f}s" for t in times)
    instruction = answer_instruction or "Answer ONLY as: start=<sec> end=<sec>."
    q = (f"This is a {dur:.1f}-second video; the {len(pil)} frames above are sampled at: {tline}. "
         f"During which time interval does the following happen: \"{caption}\"? "
         f"{instruction}")
    content = [{"type": "image", "image": p} for p in pil] + [{"type": "text", "text": q}]
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    msgs.append({"role": "user", "content": content})
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info
        imgs, _ = process_vision_info(msgs)
    except Exception:
        imgs = pil
    return text, imgs


def gpu_worker(rank, shard, K, ret, adapter=None, blind=False, model_path=MODEL):
    import torch
    try:
        torch.cuda.set_device(rank)
        from transformers import AutoProcessor, AutoModelForImageTextToText
        proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        _fr = int(os.environ.get("INTERNVL_FRAME_RESIZE", "0") or "0")
        if _fr and getattr(proc, "image_processor", None) is not None:
            try:
                proc.image_processor.size = {"height": _fr, "width": _fr}
                if getattr(proc.image_processor, "crop_size", None) is not None:
                    proc.image_processor.crop_size = {"height": _fr, "width": _fr}
                if hasattr(proc, "image_seq_length"):
                    proc.image_seq_length = int(round((_fr / 28.0) ** 2))
                print("[mem-fit-eval] INTERNVL_FRAME_RESIZE=%d image_seq_length=%d" % (_fr, int(round((_fr/28.0)**2))), flush=True)
            except Exception as _e:
                print("[mem-fit-eval] resize skip: %s" % _e, flush=True)
        model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, device_map={"": rank}, trust_remote_code=True)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        res = []
        t0 = time.time()
        for j, it in enumerate(shard):
            frames, dur, times = decode_video(it["video"], K)   # cache hit
            if blind: frames = np.zeros_like(frames)            # blind floor: all frames blacked out, only caption + timestamp prior
            text, imgs = build_prompt(proc, frames, times, dur, it["caption"])
            inputs = proc(text=[text], images=imgs, videos=None, padding=True, return_tensors="pt").to(rank)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            ans = proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            pw = parse_window(ans, dur)
            gs = max(0.0, it["gs"]); ge = min(it["ge"], dur) if dur > 0 else it["ge"]  # clamp 966 out-of-range annotations
            iou = iou_1d(pw, (gs, ge)) if pw else 0.0
            res.append({**it, "dur": dur, "pred": pw, "iou": iou, "raw": ans[:80]})
            if rank == 0 and (j + 1) % 50 == 0:
                print(f"  [gpu0] {j+1}/{len(shard)}  ({time.time()-t0:.0f}s)", flush=True)
        ret[rank] = res
    except Exception as e:
        import traceback; ret[rank] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-500:]}


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--frames", "-K", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12, help="number of decoding processes (32 cores on this machine, leave headroom)")
    ap.add_argument("--available-only", action="store_true", help="only run queries whose videos are already extracted (run while downloading)")
    ap.add_argument("--model", default=MODEL, help="model path")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (to evaluate a DPO-trained model)")
    ap.add_argument("--blind", action="store_true", help="blind floor: black out all frames, measure caption + timestamp prior accuracy (faithfulness control)")
    ap.add_argument("--out", default=f"{ROOT}/probe3_selector_evidence/charades_base.json")
    args = ap.parse_args()

    import torch
    ndev = torch.cuda.device_count()
    items = load_annotations(None)
    ensure_videos()
    if args.available_only:
        items = [it for it in items if os.path.exists(os.path.join(VIDEO_DIR, it["video"]))]
    if args.n:
        items = items[:args.n]
    print(f"[eval] {len(items)} queries, K={args.frames}, GPUs={ndev}, available_only={args.available_only}")
    if not items:
        print("[eval] no queries to run (videos not extracted yet?)"); return
    predecode_all([it["video"] for it in items], args.frames, args.workers)

    # shard across cards
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    ret = ctx.Manager().dict()
    shards = [items[r::ndev] for r in range(ndev)]
    procs = [ctx.Process(target=gpu_worker, args=(r, shards[r], args.frames, ret, args.adapter, args.blind, args.model)) for r in range(ndev)]
    t0 = time.time()
    for p in procs: p.start()
    for p in procs: p.join()

    allres = []
    for r in range(ndev):
        v = ret.get(r)
        if isinstance(v, dict) and "error" in v:
            print(f"[GPU{r} ERROR] {v['error']}\n{v.get('tb','')}"); continue
        allres += (v or [])
    if not allres:
        print("[eval] no results, check the errors above"); return

    print(f"\n=== base Qwen2.5-VL Charades-STA R@1 (n={len(allres)}, {time.time()-t0:.0f}s) ===")
    for thr in IOU_THR:
        r = np.mean([x["iou"] >= thr for x in allres])
        print(f"  R@1 IoU>={thr}: {r:.4f}")
    print(f"  mIoU: {np.mean([x['iou'] for x in allres]):.4f}")
    parsed = np.mean([x["pred"] is not None for x in allres])
    print(f"  parse rate: {parsed:.3f}  (if low, tune prompt/parser)")
    json.dump(allres, open(args.out, "w"), ensure_ascii=False)
    print(f"  details -> {args.out}")


if __name__ == "__main__":
    main()
