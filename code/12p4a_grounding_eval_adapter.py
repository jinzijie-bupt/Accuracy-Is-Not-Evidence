#!/usr/bin/env python3
"""
12p4_sota_grounding_eval.py — P4 endpoint generation: Charades temporal grounding for six models, one generation pass, writes predictions with two GT sets to disk (sealed).

Spec (pred_sota_audit §2/§4; decoding parameters copied verbatim from 12_charades_grounding.py):
  - Decoding/prompt/parsing copied line by line from the original (K16 uniform frames/224x224/timestamp lines/max_new_tokens=32 do_sample=False greedy/
    same parse_window regex/<answer> tag first), no per-model tuning.
  - Generation is decoupled from scoring: the output of a model for a query does not depend on the annotation set, so each model generates once and writes per item
    the pred window + two GT sets (A=anno/charades_sta_test.txt original; B=TimeLens-Bench charades-timelens.json
    sha256^03fce0f6 re-annotation, matched by video + normalized caption; unmatched items are counted as-is, no relaxation).
    This script computes and prints no R@/IoU/drop -- that is the job of the analysis script (sealed protocol).
  - Deviation (loader compatibility fix): annotations read from canonical charades_sta_test.txt instead of parquet (no parquet on this machine,
    same field semantics, row order = canonical order = qid); GT clamping is left to the analysis script under the same rule (original set has 966 out-of-range annotations).
"""
import argparse, os, re, json, time
import numpy as np

# ROOT can be overridden with the env var P4_ROOT:
#   on some machines the default root is not writable, and this script puts five derived paths
#   (DATA/VIDEO_DIR/CACHE/TL_JSON/anno) under ROOT. Without P4_ROOT the value is identical to before,
#   so machines that do not set it behave exactly as before -- same approach as the --dtype parameter.
ROOT      = os.environ.get("P4_ROOT", "<remote>/vlm-probes")
DATA      = f"{ROOT}/data/charades_sta"
VIDEO_DIR = f"{DATA}/Charades_v1_480"
CACHE     = f"{ROOT}/framecache"
TL_JSON   = f"{ROOT}/data/TimeLens-Bench/charades-timelens.json"


# ---------------- data ----------------
def load_set_a(n=None):
    items = []
    with open(f"{DATA}/anno/charades_sta_test.txt") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            head, caption = line.split("##", 1)
            parts = head.split()
            items.append({"qid": i, "video": parts[0] + ".mp4", "vid": parts[0],
                          "caption": caption, "gsA": float(parts[1]), "geA": float(parts[2])})
    return items[:n] if n else items


def norm_cap(c):
    return re.sub(r"\s+", " ", c.strip().lower()).rstrip(".")


def load_set_b():
    d = json.load(open(TL_JSON))
    m = {}
    for vid, rec in d.items():
        for q, sp in zip(rec.get("queries", []), rec.get("spans", [])):
            m.setdefault((vid, norm_cap(q)), (float(sp[0]), float(sp[1])))
    return m


# ---------------- decoding (copied line by line from 12_charades_grounding.py) ----------------
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


# ---------------- parsing (copied line by line) ----------------
_NUM = re.compile(r"[-+]?\d*\.?\d+")
def parse_window(text, dur):
    answer = re.search(r"<answer>(.*?)</answer>", text, re.I | re.S)
    target = answer.group(1) if answer else text
    m = re.search(r"start\s*[=:]?\s*([-+]?\d*\.?\d+).*?end\s*[=:]?\s*([-+]?\d*\.?\d+)", target, re.I | re.S)
    if m:
        s, e = float(m.group(1)), float(m.group(2))
    else:
        nums = _NUM.findall(target)
        if len(nums) < 2:
            return None
        s, e = float(nums[0]), float(nums[1])
    if e < s:
        s, e = e, s
    if dur > 0:
        s, e = max(0.0, min(s, dur)), max(0.0, min(e, dur))
    return (s, e)


# ---------------- prompt (copied line by line) ----------------
def build_prompt(proc, frames, times, dur, caption):
    from PIL import Image
    pil = [Image.fromarray(f) for f in frames]
    tline = ", ".join(f"{t:.1f}s" for t in times)
    q = (f"This is a {dur:.1f}-second video; the {len(pil)} frames above are sampled at: {tline}. "
         f"During which time interval does the following happen: \"{caption}\"? "
         f"Answer ONLY as: start=<sec> end=<sec>.")
    content = [{"type": "image", "image": p} for p in pil] + [{"type": "text", "text": q}]
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info
        imgs, _ = process_vision_info(msgs)
    except Exception:
        imgs = pil
    return text, imgs


# ---------------- main ----------------
def load_set_timelens(n=None):
    """v2: the TimeLens-Bench re-annotation rewrites the query text (smoke test: original sentences match only 5/20),
    so set B is generated independently from its own (video, query, span) -- qid=vid#idx, GT=its spans."""
    d = json.load(open(TL_JSON))
    items = []
    for vid in sorted(d):
        rec = d[vid]
        for k, (q, sp) in enumerate(zip(rec.get("queries", []), rec.get("spans", []))):
            items.append({"qid": f"{vid}#{k}", "video": vid + ".mp4", "vid": vid,
                          "caption": q, "gsA": float(sp[0]), "geA": float(sp[1])})
    return items[:n] if n else items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--frames", "-K", type=int, default=16)
    ap.add_argument("--set", choices=["orig", "timelens"], default="orig")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)  # 12p4a: clean accuracy eval for LoRA arms
    args = ap.parse_args()

    if args.set == "timelens":
        items = load_set_timelens(args.n)
        for it in items:
            it["gsB"], it["geB"] = None, None
        nb = -1
    else:
        items = load_set_a(args.n)
        setb = load_set_b()
        nb = 0
        for it in items:
            gt = setb.get((it["vid"], norm_cap(it["caption"])))
            if gt is not None:
                it["gsB"], it["geB"] = gt
                nb += 1
            else:
                it["gsB"], it["geB"] = None, None
    print(f"[p4gen] set={args.set} n={len(items)} K={args.frames} model={os.path.basename(args.model)} "
          f"setB_matched={nb}/{len(items)}", flush=True)

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    # LABEL: AWQ/GPTQ quantized GEMM kernels only support fp16; bf16 activations crash Triton immediately
    #       (observed: 72B-AWQ raises c10::BFloat16 != c10::Half, kernel=AwqGEMMTritonLinear).
    #       The check is identical to 14_causal_provenance_probe.py:107-118 -- look for
    #       quantization_config in config.json; mechanical, reproducible, independent of the model name. Non-quantized models still use bfloat16,
    #       so the behavior of all non-quantized arms is byte-for-byte unchanged.
    load_dtype = torch.bfloat16
    try:
        with open(os.path.join(args.model, "config.json")) as _f:
            if "quantization_config" in json.load(_f):
                load_dtype = torch.float16
    except Exception:
        pass

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=load_dtype, device_map={"": 0}, trust_remote_code=True)
    if load_dtype == torch.float16:
        model = model.to(torch.float16)   # stray bf16 modules left by the unified quantizer (as in 14_:131-132)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    res, t0, nparse = [], time.time(), 0
    for j, it in enumerate(items):
        frames, dur, times = decode_video(it["video"], args.frames)
        text, imgs = build_prompt(proc, frames, times, dur, it["caption"])
        inputs = proc(text=[text], images=imgs, videos=None, padding=True, return_tensors="pt").to(0)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        ans = proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        pw = parse_window(ans, dur)
        if pw is not None:
            nparse += 1
        res.append({"set": args.set, "qid": it["qid"], "video": it["video"], "caption": it["caption"], "dur": dur,
                    "pred": pw, "gsA": it["gsA"], "geA": it["geA"],
                    "gsB": it["gsB"], "geB": it["geB"], "raw": ans[:80]})
        if (j + 1) % 100 == 0:
            print(f"  [gen] {j+1}/{len(items)}  parse={nparse/(j+1):.3f}  ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False)
    print(f"[p4gen] DONE n={len(res)} parse_rate={nparse/max(len(res),1):.3f} "
          f"elapsed={time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
