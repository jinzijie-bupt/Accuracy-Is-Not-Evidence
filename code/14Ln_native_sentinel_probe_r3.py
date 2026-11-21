# r3: r2 + three optional readouts for the review round; the r2 fields and their placebo draws are
#   unchanged (new readouts draw from a separate rng), so an r3 row reproduces the r2 row for the same model/qid.
#   --tok            per-token log-probs of the target under clean / evidence edit / placebo mean (+ token strings)
#   --contig         contiguous placebo: R blocks of |E*| consecutive frames outside E* ({op}_cdrop_* fields)
#   --refusal TEXT   log-prob of the refusal string under clean / evidence edit / placebo (where the lost mass goes)
#   --reloc          relocation: exchange the evidence block with a same-length block elsewhere; score the old and the
#                    moved interval on both videos and regenerate on the relocated video (does the answer follow?)
# r2: use_cache=True in the clean generation (VideoChat-R1 config has use_cache=false); scoring path unchanged.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""14Ln_native_sentinel_probe.py — fixed-answer audit under the NATIVE protocol.

Same estimands and record fields as the registered 14Ls_sentinel_probe.py (264bd380db19c756, --target-mode gt,
ops mask,swap, R placebo draws of |E*| random non-evidence frames, token-mean drops), but the video input is the
native protocol shared with 12p4n_native_acc_eval_tf5meta.py and 27_papo_train_qcdro_v3seln.py.
"""
import argparse, os, sys, json, time, random, hashlib, importlib.util, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_T = os.path.join(HERE, "27_papo_train_qcdro_v3seln.py")
_spec = importlib.util.spec_from_file_location("v3seln", _T)
V = importlib.util.module_from_spec(_spec); sys.modules["v3seln"] = V; _spec.loader.exec_module(V)
TRAINER_SHA16 = hashlib.sha256(open(_T, "rb").read()).hexdigest()[:16]

_NUM = re.compile(r"[-+]?\d*\.?\d+")


def parse_native(text, dur):
    m = re.search(r"<answer>(.*?)</answer>", text, re.I | re.S)
    tgt = m.group(1) if m else text
    m2 = re.search(r"([-+]?\d*\.?\d+)\s*(?:to|and|,)\s*([-+]?\d*\.?\d+)", tgt, re.I)
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


def apply_op(frames, idx, op, distractor):
    """Same semantics as 14L.apply_op for the two registered ops: mask=zero frames, swap=replace with distractor frames."""
    f = frames.copy(); e = np.asarray(idx, int)
    if op == "mask":
        f[e] = 0
    elif op == "swap":
        m = min(len(e), len(distractor)); f[e[:m]] = distractor[:m]
    else:
        raise ValueError(op)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--adapter", default=None)
    ap.add_argument("--anno", default=None, help="charades_sta_test.txt (default: <VIDEO_DIR>/../anno/charades_sta_test.txt)")
    ap.add_argument("--n", type=int, default=None); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--ops", default="mask,swap"); ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompt-style", choices=sorted(V.NATIVE_PROMPTS), default="timer1_default")
    ap.add_argument("--answer-style", choices=["timer1", "vchat"], default="timer1")
    ap.add_argument("--fps", type=float, default=2.0); ap.add_argument("--total-pixels", type=int, default=3584 * 28 * 28)
    ap.add_argument("--min-pixels", type=int, default=16 * 28 * 28); ap.add_argument("--max-frames", type=int, default=768)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--tok", action="store_true"); ap.add_argument("--contig", action="store_true")
    ap.add_argument("--refusal", default=None); ap.add_argument("--reloc", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for o in args.ops.split(","):
        if o and o not in ("mask", "swap"):
            raise SystemExit(f"unknown op {o}")
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    ops = [o for o in args.ops.split(",") if o]
    anno = args.anno or os.path.join(os.path.dirname(V.base.VIDEO_DIR), "anno", "charades_sta_test.txt")
    items = load_test(anno, args.n, args.offset)
    all_videos = sorted({it["video"] for it in load_test(anno)})
    torch.cuda.set_device(args.gpu)
    proc = AutoProcessor.from_pretrained(args.model)
    model = V._fp_dtype_compat(AutoModelForImageTextToText, args.model, device_map={"": args.gpu})
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    rng = random.Random(args.seed + args.gpu)            # r2 stream: donors and random placebo draws
    rng2 = random.Random(args.seed + 7919 + args.gpu)    # r3 stream: contiguous placebo, relocation
    print(f"[14Ln r3] n={len(items)} offset={args.offset} ops={ops} R={args.R} style={args.prompt_style}/{args.answer_style} "
          f"tok={args.tok} contig={args.contig} refusal={args.refusal!r} reloc={args.reloc} "
          f"trainer_sha16={TRAINER_SHA16} model={os.path.basename(args.model)} adapter={args.adapter}", flush=True)
    res = []; t0 = time.time(); nhit = 0; checked = False
    for j, it in enumerate(items):
        vpath = os.path.join(V.base.VIDEO_DIR, it["video"])
        try:
            frames, times, dur, sfps = V.native_decode(vpath, args)
        except Exception as exc:
            res.append({"qid": it["qid"], "video": it["video"], "decode_error": f"{type(exc).__name__}: {exc}"}); continue
        T = frames.shape[0]
        estar = np.asarray([i for i, t in enumerate(times) if it["gs"] - 1e-6 <= t <= it["ge"] + 1e-6], dtype=int)
        rec = {"qid": it["qid"], "video": it["video"], "n_estar": int(len(estar)), "nframes": int(T), "fps": float(sfps),
               "style": args.prompt_style, "target_mode": "gt", "sentinel": "none", "dilate_k": 0}
        if len(estar) == 0 or len(estar) >= T:
            rec["degenerate"] = 1; res.append(rec); continue
        prompt_text = V.NATIVE_PROMPTS[args.prompt_style].format(it["caption"])
        cache = {}
        vid = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
        msgs = [{"role": "user", "content": [{"type": "video", "video": "native_placeholder"}, {"type": "text", "text": prompt_text}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        cache["text"] = text
        inputs = V._native_inputs(proc, text, vid, sfps)
        cache["plen"] = int(inputs.input_ids.shape[1])
        inputs = inputs.to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
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

        def score_tok(fr, completion):
            """per-token log-probs of `completion` (same slicing as V.native_completion_logprob) and the token strings."""
            v = torch.from_numpy(np.ascontiguousarray(fr)).permute(0, 3, 1, 2)
            full = V._native_inputs(proc, cache["text"] + completion, v, sfps).to(model.device)
            plen = cache["plen"]
            with torch.no_grad():
                out = model(**full)
                lg = out.logits[:, plen - 1:-1, :].float()
                lab = full.input_ids[:, plen:]
                lp = torch.log_softmax(lg, dim=-1).gather(-1, lab.unsqueeze(-1)).squeeze(-1)[0]
            return lp.cpu().numpy().astype(np.float64), proc.tokenizer.convert_ids_to_tokens(lab[0].tolist())

        lp_c_sum, tgt_len = score(frames)
        rec["lp_clean_sum"] = lp_c_sum; rec["lp_clean_mean"] = lp_c_sum / max(1, tgt_len); rec["tgt_len"] = tgt_len
        if args.tok or args.contig or args.refusal or args.reloc:
            tok_c, toks = score_tok(frames, tgt_txt)
            if not checked:   # the per-token path must reproduce the registered scorer exactly
                assert abs(tok_c.sum() - lp_c_sum) < 1e-3 * max(1.0, abs(lp_c_sum)), (tok_c.sum(), lp_c_sum)
                checked = True
                print(f"  [r3] per-token path reproduces registered scorer: {tok_c.sum():.4f} vs {lp_c_sum:.4f}", flush=True)
            if args.tok:
                rec["tok_str"] = toks; rec["tok_clean"] = [round(x, 5) for x in tok_c.tolist()]
        if args.refusal:
            ref_c, ref_toks = score_tok(frames, args.refusal)
            rec["ref_clean_sum"] = float(ref_c.sum()); rec["ref_len"] = len(ref_toks)
        nonev = [i for i in range(T) if i not in set(estar.tolist())]
        L = len(estar); a0 = int(estar[0])
        for op in ops:
            dvid = rng.choice([v for v in all_videos if v != it["video"]])
            dfr, _, _, _ = V.native_decode(os.path.join(V.base.VIDEO_DIR, dvid), args)
            if dfr.shape[1:3] != frames.shape[1:3]:
                import cv2
                dfr = np.stack([cv2.resize(np.ascontiguousarray(f), (frames.shape[2], frames.shape[1])) for f in dfr])
            didx = rng.sample(range(dfr.shape[0]), min(dfr.shape[0], len(estar)))
            distractor = dfr[didx]
            fe = apply_op(frames, estar, op, distractor)
            lp_e_sum, _ = score(fe)
            rand_sums = []; rand_idx = []; rand_frames = []
            for _ in range(args.R):
                ridx = np.array(rng.sample(nonev, min(len(nonev), len(estar))))
                rand_idx.append([int(x) for x in ridx])
                fr_ = apply_op(frames, ridx, op, distractor)
                rs, _ = score(fr_)
                rand_sums.append(rs); rand_frames.append(fr_)
            rec[f"{op}_drop_evid"] = lp_c_sum - lp_e_sum
            rec[f"{op}_drop_rand"] = lp_c_sum - float(np.mean(rand_sums))
            rec[f"{op}_tokmean_drop_evid"] = (lp_c_sum - lp_e_sum) / max(1, tgt_len)
            rec[f"{op}_tokmean_drop_rand"] = (lp_c_sum - float(np.mean(rand_sums))) / max(1, tgt_len)
            rec[f"{op}_rand_sums"] = rand_sums; rec[f"{op}_rand_idx"] = rand_idx; rec[f"{op}_donor"] = dvid
            if args.tok:
                te, _ = score_tok(fe, tgt_txt)
                tr = np.mean([score_tok(f_, tgt_txt)[0] for f_ in rand_frames], axis=0)
                rec[f"{op}_tok_evid"] = [round(x, 5) for x in te.tolist()]
                rec[f"{op}_tok_rand"] = [round(x, 5) for x in tr.tolist()]
            if args.refusal:
                rec[f"{op}_ref_evid_sum"] = float(score_tok(fe, args.refusal)[0].sum())
                rec[f"{op}_ref_rand_sum"] = float(np.mean([score_tok(f_, args.refusal)[0].sum() for f_ in rand_frames]))
            del rand_frames
            if args.contig:
                starts = [s for s in range(0, T - L + 1) if s + L - 1 < a0 or s > a0 + L - 1]
                if not starts:
                    rec[f"{op}_contig_na"] = 1
                else:
                    csums = []; cst = []
                    for _ in range(args.R):
                        s0 = rng2.choice(starts); cst.append(int(s0))
                        csums.append(score(apply_op(frames, np.arange(s0, s0 + L), op, distractor))[0])
                    rec[f"{op}_cdrop_rand"] = lp_c_sum - float(np.mean(csums))
                    rec[f"{op}_tokmean_cdrop_rand"] = (lp_c_sum - float(np.mean(csums))) / max(1, tgt_len)
                    rec[f"{op}_contig_starts"] = cst
        if args.reloc:
            starts = [s for s in range(0, T - L + 1) if abs(s - a0) >= L]
            if not starts:
                rec["reloc_na"] = 1
            else:
                s0 = rng2.choice(starts)
                fr_rel = frames.copy()
                fr_rel[s0:s0 + L] = frames[a0:a0 + L]; fr_rel[a0:a0 + L] = frames[s0:s0 + L]
                shift = float(times[s0] - times[a0])
                ngs, nge = max(0.0, it["gs"] + shift), min(dur, it["ge"] + shift)
                new_txt = V.native_completion(ngs, nge, args.answer_style)
                cache_new = dict(cache)
                with torch.no_grad():
                    lp_old_rel = V.native_completion_logprob(model, proc, fr_rel, sfps, prompt_text, tgt_txt, model.device, cache)
                    lp_new_orig = V.native_completion_logprob(model, proc, frames, sfps, prompt_text, new_txt, model.device, cache_new)
                    lp_new_rel = V.native_completion_logprob(model, proc, fr_rel, sfps, prompt_text, new_txt, model.device, cache_new)
                    vr = torch.from_numpy(np.ascontiguousarray(fr_rel)).permute(0, 3, 1, 2)
                    inr = V._native_inputs(proc, text, vr, sfps).to(model.device)
                    g2 = model.generate(**inr, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
                rtxt = proc.batch_decode(g2[:, inr["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                pr = parse_native(rtxt, dur)
                n_new = int(lp_new_orig[1])
                rec.update({"reloc_start": int(s0), "reloc_shift": shift, "reloc_gt": [ngs, nge], "reloc_new_len": n_new,
                            "reloc_lp_old_rel": float(lp_old_rel[0]), "reloc_lp_new_orig": float(lp_new_orig[0]),
                            "reloc_lp_new_rel": float(lp_new_rel[0]), "reloc_pred": pr, "reloc_text": rtxt[-120:],
                            "reloc_iou_new": float(iou_1d(pr, (ngs, nge))) if pr else 0.0,
                            "reloc_iou_old": float(iou_1d(pr, (it["gs"], it["ge"]))) if pr else 0.0})
        res.append(rec)
        if (j + 1) % 25 == 0:
            print(f"  [14Ln r3] {j+1}/{len(items)} hits={nhit} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False)
    meta = {"instrument": "14Ln_native_sentinel_probe_v1r3", "trainer_sha16": TRAINER_SHA16, "args": vars(args),
            "n_rows": len(res), "n_hits": nhit, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=1)
    print(f"[14Ln] DONE n={len(res)} hits={nhit} elapsed={time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
