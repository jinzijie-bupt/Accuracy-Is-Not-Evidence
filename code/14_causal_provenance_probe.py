#!/usr/bin/env python3
"""
14_causal_provenance_probe.py — core causal measure of PROVE (key gates E1/E2).

Question: is a "correct answer" of the base VLM really causally dependent on the evidence frames?
Method: on the base-correct set, intervene on the evidence frames E* (mask/contradictory swap/shuffle),
      and compare with the same intervention on budget-matched random non-evidence frames N* (placebo, averaged over R runs).
  ΔFlip(op) = FlipRate(intervene E*) − FlipRate(intervene N*)
  the audit      = fraction still correct after intervening on E* (correct without relying on evidence = unfaithful).
  decoupling gate = per-query causal dependence vs clean-IoU Spearman should be ≈0 (shows ΔFlip is not an IoU proxy).
E* uses the full-precision DRN test.txt + framecache times[] (not the float16 parquet).
Stats: McNemar paired test (not an independent-samples χ²).

Usage: python 14_causal_provenance_probe.py --n 300 --ops swap,mask --R 3   # E1 gate
      python 14_causal_provenance_probe.py --ops swap,mask,shuffle          # E2 full set
"""
import argparse, os, re, csv, json, time, importlib.util, random
import numpy as np

_P = os.path.join(os.path.dirname(__file__), "12_charades_grounding.py")
_spec = importlib.util.spec_from_file_location("g12", _P); g12 = importlib.util.module_from_spec(_spec)
import sys as _sys; _sys.modules["g12"] = g12; _spec.loader.exec_module(g12)
decode_video, iou_1d, parse_window = g12.decode_video, g12.iou_1d, g12.parse_window
build_prompt, predecode_all, ensure_videos = g12.build_prompt, g12.predecode_all, g12.ensure_videos
MODEL, VIDEO_DIR, ROOT = g12.MODEL, g12.VIDEO_DIR, g12.ROOT
ANNO = f"{ROOT}/data/charades_sta/anno"


def load_durations():
    dur = {}
    for sp in ["train", "test"]:
        p = f"{ANNO}/Charades_v1_{sp}.csv"
        if os.path.exists(p):
            for row in csv.DictReader(open(p)):
                try: dur[row["id"]] = float(row["length"])
                except Exception: pass
    return dur


def load_test_drn(durs, n=None):
    items = []
    for line in open(f"{ANNO}/charades_sta_test.txt"):
        m = re.match(r"^(\S+)\s+([\d.]+)\s+([\d.]+)##(.*)$", line.strip())
        if not m: continue
        vid, s, e, sent = m.group(1), float(m.group(2)), float(m.group(3)), m.group(4).strip()
        video = f"{vid}.mp4"
        if not os.path.exists(os.path.join(VIDEO_DIR, video)): continue
        d = durs.get(vid, 0.0)
        items.append({"qid": len(items), "video": video, "caption": sent,
                      "gs": max(0.0, s), "ge": min(e, d) if d > 0 else e})
    return items[:n] if n else items


def predict_iou(model, proc, frames, times, dur, caption, gs, ge, device, max_new_tokens=32, system_prompt=None, answer_instruction=None):
    import torch
    text, imgs = build_prompt(proc, frames, times, dur, caption, system_prompt=system_prompt, answer_instruction=answer_instruction)
    inp = proc(text=[text], images=imgs, videos=None, padding=True, return_tensors="pt").to(device)
    pad_id = getattr(getattr(proc, "tokenizer", None), "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(pad_id, (list, tuple)):
        pad_id = pad_id[0]
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
    gen = out[:, inp.input_ids.shape[1]:]
    ans = proc.batch_decode(gen, skip_special_tokens=True)[0]
    pw = parse_window(ans, dur)
    return (iou_1d(pw, (gs, ge)) if pw else 0.0, pw is not None, ans, int(gen.shape[1]))   # parsed=False means parsing failed


def apply_op(frames, idx, op, distractor):
    """Apply the intervention at the idx positions of frames (frame count/positions unchanged, in-place on a copy)."""
    f = frames.copy()
    if op == "mask":
        f[idx] = 0
    elif op == "swap":
        f[idx] = distractor[:len(idx)]
    elif op == "shuffle":
        perm = idx.copy(); random.shuffle(perm)
        f[idx] = frames[perm]
    elif op == "blur":
        for i in np.asarray(idx, int):
            s = f[i].reshape(16, 14, 16, 14, 3).mean(axis=(1, 3)).astype(np.uint8)
            f[i] = np.repeat(np.repeat(s, 14, 0), 14, 1)
    elif op == "tshift":
        K = len(frames)
        for i in np.asarray(idx, int):
            f[i] = frames[(int(i) + K // 2) % K]
    return f


def load_model_and_processor(model_path, rank):
    import torch
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
    # AWQ/GPTQ checkpoints: quantized GEMM kernels are fp16-only; bf16 activations crash Triton.
    load_dtype = torch.bfloat16
    try:
        import json as _json, os as _os
        with open(_os.path.join(model_path, "config.json")) as _f:
            if "quantization_config" in _json.load(_f):
                load_dtype = torch.float16
    except Exception:
        pass

    def _from_pretrained(cls):
        try:
            return cls.from_pretrained(model_path, dtype=load_dtype, device_map={"": rank}, trust_remote_code=True)
        except TypeError as exc:
            if "unexpected keyword argument 'dtype'" not in str(exc):
                raise
            return cls.from_pretrained(model_path, torch_dtype=load_dtype, device_map={"": rank}, trust_remote_code=True)

    try:
        model = _from_pretrained(AutoModelForImageTextToText)
    except Exception as exc:
        from transformers import Qwen2_5_VLForConditionalGeneration
        print(f"[model-load] AutoModelForImageTextToText failed; fallback Qwen2_5_VLForConditionalGeneration: {type(exc).__name__}: {exc}", flush=True)
        model = _from_pretrained(Qwen2_5_VLForConditionalGeneration)
    if load_dtype == torch.float16:
        model = model.to(torch.float16)  # unify stray bf16 modules left by the quantizer path
    return proc, model


def gpu_worker(rank, shard, all_videos, K, ops, R, thr, ret, model_path=MODEL, adapter=None, placebo_map=None, max_new_tokens=32, system_prompt=None, answer_instruction=None, store_generations=False):
    import torch
    try:
        torch.cuda.set_device(rank)
        proc, model = load_model_and_processor(model_path, rank)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        model = model.eval()
        rng = random.Random(1000 + rank)
        res = []; t0 = time.time()
        for j, it in enumerate(shard):
            frames, dur, times = decode_video(it["video"], K)
            estar = np.where((times >= it["gs"]) & (times <= it["ge"]))[0]
            if len(estar) == 0 or len(estar) >= K:   # degenerate: no evidence frames, or all frames are evidence
                continue
            nonev = np.array([i for i in range(K) if i not in set(estar.tolist())])
            clean, clean_p, clean_text, clean_tokens = predict_iou(
                model, proc, frames, times, dur, it["caption"], it["gs"], it["ge"], rank, max_new_tokens, system_prompt, answer_instruction
            )
            rec = {"qid": it["qid"], "video": it["video"], "n_estar": int(len(estar)),
                   "clean_iou": clean, "hit": int(clean >= thr), "clean_parsed": int(clean_p)}
            if store_generations:
                rec["clean_generation_text"] = clean_text
                rec["clean_generation_tokens"] = clean_tokens
            if clean < thr:   # metrics/decoupling only on base-correct set S -> skip interventions on incorrect items, saves ~75% compute
                res.append(rec); continue
            for op in ops:
                # take frames from a contradictory distractor video (swap only)
                dvid = rng.choice([v for v in all_videos if v != it["video"]])
                dfr, _, _ = decode_video(dvid, K)
                didx = rng.sample(range(K), min(K, len(estar)))
                distractor = dfr[didx]
                # EVID: intervene on evidence frames
                fe = apply_op(frames, estar, op, distractor)
                evid_iou, evid_p, evid_text, evid_tokens = predict_iou(
                    model, proc, fe, times, dur, it["caption"], it["gs"], it["ge"], rank, max_new_tokens, system_prompt, answer_instruction
                )
                # RAND placebo: intervene on an equal number of random non-evidence frames, R times
                rand_ious = []; rand_ps = []; rand_texts = []; rand_tokens = []
                fixed = None
                if placebo_map:
                    raw = placebo_map.get(str(it["qid"]), placebo_map.get(it["qid"]))
                    if raw is not None:
                        es = set(estar.tolist())
                        cand = [int(x) for x in raw if 0 <= int(x) < K and int(x) not in es]
                        if len(cand) == len(estar):
                            fixed = np.asarray(cand, int)
                rand_sets = [fixed] if fixed is not None else [
                    np.array(rng.sample(list(nonev), min(len(nonev), len(estar)))) for _ in range(R)
                ]
                for ridx in rand_sets:
                    fr = apply_op(frames, ridx, op, distractor)
                    ri_iou, ri_p, ri_text, ri_tokens = predict_iou(
                        model, proc, fr, times, dur, it["caption"], it["gs"], it["ge"], rank, max_new_tokens, system_prompt, answer_instruction
                    )
                    rand_ious.append(ri_iou); rand_ps.append(ri_p)
                    if store_generations:
                        rand_texts.append(ri_text); rand_tokens.append(ri_tokens)
                rec[f"{op}_evid_iou"] = evid_iou
                rec[f"{op}_evid_parsed"] = int(evid_p)                        # measurement validity: does it still produce a parseable window after evidence is destroyed
                rec[f"{op}_rand_iou_mean"] = float(np.mean(rand_ious))
                rec[f"{op}_rand_parsed_mean"] = float(np.mean(rand_ps))
                rec[f"{op}_rand_strategy"] = "fixed_matched" if fixed is not None else "random"
                rec[f"{op}_rand_n"] = int(len(rand_sets))
                # flip (only meaningful for clean hits): hit -> miss
                rec[f"{op}_evid_flip"] = int(clean >= thr and evid_iou < thr)
                rec[f"{op}_rand_flip"] = float(np.mean([clean >= thr and ri < thr for ri in rand_ious]))
                # validity split: among flips, "still parseable but window changed" (real grounding change) vs "parse failure" (possibly garbage output)
                rec[f"{op}_evid_flip_parsed"] = int(clean >= thr and evid_iou < thr and evid_p)
                if store_generations:
                    rec[f"{op}_evid_generation_text"] = evid_text
                    rec[f"{op}_evid_generation_tokens"] = evid_tokens
                    rec[f"{op}_rand_generation_texts"] = rand_texts
                    rec[f"{op}_rand_generation_tokens"] = rand_tokens
            res.append(rec)
            if rank == 0 and (j + 1) % 25 == 0:
                print(f"  [gpu0] {j+1}/{len(shard)} ({time.time()-t0:.0f}s)", flush=True)
        ret[rank] = res
    except Exception as e:
        import traceback; ret[rank] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3: return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def mcnemar(evid_flips, rand_flips_bin):
    b = sum(1 for e, r in zip(evid_flips, rand_flips_bin) if e and not r)
    c = sum(1 for e, r in zip(evid_flips, rand_flips_bin) if r and not e)
    n = b + c
    return {"b_evid_only": b, "c_rand_only": c, "chi2_df1": ((abs(b-c)-1)**2/n) if n else 0.0}


def report(res, ops, thr):
    S = [r for r in res if r["hit"]]
    print(f"\n=== causal provenance probe (n={len(res)}, base-correct set |S|={len(S)}) ===")
    for op in ops:
        ef = np.mean([r[f"{op}_evid_flip"] for r in S]) if S else 0
        rf = np.mean([r[f"{op}_rand_flip"] for r in S]) if S else 0
        survive = 1 - ef  # still correct after evidence intervention = the audit candidate
        mc = mcnemar([r[f"{op}_evid_flip"] for r in S],
                     [r[f"{op}_rand_flip"] >= 0.5 for r in S])
        print(f"[{op}] flip_EVID={ef:.3f}  flip_RAND={rf:.3f}  ΔFlip={ef-rf:+.3f}  "
              f"the audit(survive EVID)={survive:.3f}  McNemar χ²(df1)={mc['chi2_df1']:.1f} (b={mc['b_evid_only']},c={mc['c_rand_only']})")
    # decoupling gate: per-query causal dependence vs clean IoU -- reported separately per operator, never averaged, swap not hard-coded.
    # (training uses only swap -> mask/shuffle are held-out and are the main faithfulness metrics; swap is only a training-operator control)
    if len(S) >= 3:
        iou = [r["clean_iou"] for r in S]
        print()
        for op in ops:
            causal = [r[f"{op}_evid_flip"] - r[f"{op}_rand_flip"] for r in S]
            rho = spearman(iou, causal)
            proj = np.mean([1 - r[f"{op}_evid_flip"] for r in S])
            tag = "training operator" if op == "swap" else "HELD-OUT"
            print(f"[decoupling gate:{op}|{tag}] Spearman(clean_IoU, causal dependence)={rho:+.3f}  "
                  f"the audit={proj:.1%}  (small |ρ| = not an IoU proxy)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0, help="index in the test list to start from; default 0 keeps the old behavior")
    ap.add_argument("--frames", "-K", type=int, default=16)
    ap.add_argument("--ops", default="swap,mask,shuffle")
    ap.add_argument("--R", type=int, default=3, help="number of RAND placebo draws")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model", default=MODEL, help="model path (E3 switches to LLaVA-OneVision)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter (measure faithfulness of the trained model)")
    ap.add_argument("--max-new-tokens", type=int, default=32, help="generation budget; default preserves existing probes")
    ap.add_argument("--system-prompt", default="", help="optional system prompt injected through the chat template")
    ap.add_argument("--system-prompt-file", default=None, help="optional file containing the system prompt")
    ap.add_argument("--answer-instruction", default="", help="optional replacement for the final answer instruction in the user prompt")
    ap.add_argument("--answer-instruction-file", default=None, help="optional file containing the replacement answer instruction")
    ap.add_argument("--store-generations", action="store_true", help="store raw generated text and generated-token counts in JSON")
    ap.add_argument("--placebo-index-file", default=None, help="qid -> fixed non-evidence frame indices for salience/continuity-matched placebo")
    ap.add_argument("--out", default=f"{ROOT}/probe3_selector_evidence/causal_probe.json")
    args = ap.parse_args()
    ops = args.ops.split(",")
    system_prompt = args.system_prompt
    if args.system_prompt_file:
        system_prompt = open(args.system_prompt_file).read().strip()
    answer_instruction = args.answer_instruction
    if args.answer_instruction_file:
        answer_instruction = open(args.answer_instruction_file).read().strip()

    import torch
    ndev = torch.cuda.device_count()
    durs = load_durations()
    ensure_videos()
    all_items = load_test_drn(durs, None)
    end = args.offset + args.n if args.n else None
    items = all_items[args.offset:end]
    all_videos = sorted({it["video"] for it in items})
    print(f"[probe] {len(items)} test queries, ops={ops} R={args.R} K={args.frames} GPUs={ndev} max_new_tokens={args.max_new_tokens}")
    placebo_map = None
    if args.placebo_index_file:
        placebo_map = json.load(open(args.placebo_index_file))
        print(f"[probe] fixed placebo index map loaded: {args.placebo_index_file} ({len(placebo_map)} qids)")
    predecode_all([it["video"] for it in items], args.frames, args.workers)

    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn"); ret = ctx.Manager().dict()
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
                placebo_map,
                args.max_new_tokens,
                system_prompt or None,
                answer_instruction or None,
                args.store_generations,
            ),
        )
        for r in range(ndev)
    ]
    t0 = time.time()
    for p in procs: p.start()
    for p in procs: p.join()
    res = []
    for r in range(ndev):
        v = ret.get(r)
        if isinstance(v, dict) and "error" in v: print(f"[GPU{r} ERR] {v['error']}\n{v.get('tb','')}"); continue
        res += (v or [])
    if not res: print("[probe] no results"); return
    print(f"[probe] {len(res)} items, {time.time()-t0:.0f}s")
    report(res, ops, args.thr)
    json.dump(res, open(args.out, "w"))
    print(f"details -> {args.out}")


if __name__ == "__main__":
    main()
