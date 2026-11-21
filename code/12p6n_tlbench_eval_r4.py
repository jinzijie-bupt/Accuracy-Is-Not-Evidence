# r4: r3 dcb6800736987902 + --video-op black (12p4n r5 93c38595d8d30bd7 l.100) + INSTRUMENT derived from --text-first (C2) + rows/meta carry text_first & video_op; defaults reproduce r3.
# r3: r2 2006fdc317f86f9d + trainer r4 (timelens prompt) + --image-patch-size + '-' separator; defaults = r2.
# text-first variant: source 5add95b72246a8b2 + --text-first (message order text->video; trainer r3 729232059d642960 scoring path follows the same flag). Flag off = source behaviour.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""12p6n_tlbench_eval.py — native-protocol direct-answer grounding eval on TimeLens-Bench splits
(TencentARC/TimeLens; qvhighlights-timelens.json = OOD videos/queries, charades-timelens.json = re-annotated Charades).
Same video pipeline / prompts / parsing / generation as 12p5n_native_perturb_eval.py (trainer native_decode, fps 2,
total_pixels 3584·28·28, greedy, use_cache=True). Item = (video key, query i, span i); spans[i] <-> queries[i] by index,
entries whose spans/queries lengths differ are paired up to the shorter length (count reported).
Rows use the 12p4n schema (qid, video, caption, dur, pred, gsA, geA, raw, style, fps, nframes) so analysis/read_p4nat_v1.py
reads them (pass --expect-n = number of items of the split). Per-row partial dump + resume by qid.
usage: 12p6n_tlbench_eval.py --model M [--adapter A] --anno qvhighlights-timelens.json --video-dir DIR [--ext .mp4]
         --prompt-style timer1_default --n N --offset 0 --gpu 0 --out OUT.json
"""
import argparse, os, sys, json, time, hashlib, importlib.util, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_T = None
for cand in (os.path.join(HERE, "27_papo_train_qcdro_v3seln_r4.py"), os.path.join(HERE, "..", "trainer_v3", "27_papo_train_qcdro_v3seln_r4.py")):
    if os.path.exists(cand):
        _T = cand; break
if _T is None:
    raise SystemExit("27_papo_train_qcdro_v3seln.py not found next to this file")
_spec = importlib.util.spec_from_file_location("v3seln", _T)
V = importlib.util.module_from_spec(_spec); sys.modules["v3seln"] = V
if "--selftest" in sys.argv[1:]:  # r4: offline bench on a box without torch; the real path below is the r3 line
    try:
        _spec.loader.exec_module(V)
    except Exception as _exc:  # noqa: BLE001 - selftest only
        print("[selftest] trainer module not importable here:", type(_exc).__name__, _exc)
else:
    _spec.loader.exec_module(V)
TRAINER_SHA16 = hashlib.sha256(open(_T, "rb").read()).hexdigest()[:16]
INSTRUMENT_BASE = "12p6n_tlbench_eval_v1r4_q3"  # r4: no hard-coded "_tf" (r2/r3 stamped it regardless of --text-first, C2)


def instrument_tag(text_first):
    """r4: message-order suffix derived from the flag that is actually executed, never hard-coded."""
    return INSTRUMENT_BASE + ("_tf" if text_first else "")


def apply_video_op(vid, op):
    """r4: 'black' = every decoded frame zeroed, same T/C/H/W so the token layout is unchanged.
    Verbatim semantics of 12p4n_native_acc_eval_tf5meta_r5.py (93c38595d8d30bd7) l.100
    `if args.video_op == 'black': vids = [torch.zeros_like(vids[0])]`."""
    if op == "black":
        import torch
        return torch.zeros_like(vid)
    if op != "none":
        raise ValueError(op)
    return vid
_NUM = re.compile(r"[-+]?\d*\.?\d+")


def parse_native(text, dur):
    m = re.search(r"<answer>(.*?)</answer>", text, re.I | re.S)
    tgt = m.group(1) if m else text
    m2 = re.search(r"([-+]?\d*\.?\d+)\s*(?:to|and|,|-)\s*([-+]?\d*\.?\d+)", tgt, re.I)  # r3: TimeLens "<s> - <e> seconds"
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


def load_tlbench(anno_path):
    """Flatten a TimeLens json {video_key: {duration, spans, queries}} into items ordered by (sorted key, index)."""
    d = json.load(open(anno_path, encoding="utf-8"))
    items = []; n_mismatch = 0
    for k in sorted(d):
        v = d[k]; sp, qs = v["spans"], v["queries"]
        if len(sp) != len(qs): n_mismatch += 1
        for i in range(min(len(sp), len(qs))):
            s, e = float(sp[i][0]), float(sp[i][1])
            items.append({"qid": len(items), "video": k, "gs": min(s, e), "ge": max(s, e), "caption": str(qs[i]).strip(), "dur_anno": float(v.get("duration") or 0.0)})
    return items, n_mismatch


def _partial_path(out): return out + ".partial.jsonl"


def load_partial(out):
    p = _partial_path(out); done = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: r = json.loads(line)
                except Exception: continue
                done[int(r["qid"])] = r
    return done


def selftest():
    import tempfile
    ok = True
    def chk(c, m):
        nonlocal ok; ok &= bool(c); print(("  PASS " if c else "  FAIL ") + m)
    d = tempfile.mkdtemp(); p = os.path.join(d, "a.json")
    json.dump({"vB": {"duration": 10.0, "spans": [[3, 1]], "queries": ["q1"]}, "vA": {"duration": 20.0, "spans": [[0, 5], [6, 9]], "queries": ["qa", "qb", "qc"]}}, open(p, "w"))
    items, nm = load_tlbench(p)
    chk([it["video"] for it in items] == ["vA", "vA", "vB"] and [it["qid"] for it in items] == [0, 1, 2], "items sorted by key, qids consecutive")
    chk(items[2]["gs"] == 1.0 and items[2]["ge"] == 3.0, "reversed span normalised")
    chk(nm == 1 and len(items) == 3, "length mismatch paired to the shorter length and counted")
    chk(parse_native("From 12.5 to 40 seconds.", 150.0) == (12.5, 40.0) and parse_native("no event", 10) is None, "parse_native")
    out = os.path.join(d, "x.json"); open(_partial_path(out), "w").write(json.dumps({"qid": 1}) + "\n" + "{bad")
    chk(sorted(load_partial(out)) == [1], "partial resume drops torn line")
    chk("_tf" not in instrument_tag(False) and instrument_tag(True).endswith("_tf"),
        "r4: INSTRUMENT derived from --text-first (off -> no _tf, on -> _tf)")
    chk(instrument_tag(False) + "_tf" == instrument_tag(True), "r4: the two tags differ only by that suffix")
    try:
        import torch as _torch
    except Exception:
        _torch = None
    if _torch is not None:
        v = (_torch.arange(2 * 3 * 2 * 2, dtype=_torch.uint8) + 1).reshape(2, 3, 2, 2)
        b = apply_video_op(v, "black")
        chk(int(b.sum()) == 0 and tuple(b.shape) == tuple(v.shape) and b.dtype == v.dtype,
            "r4: --video-op black -> every pixel zero, same T/C/H/W and dtype")
        chk(int(v.sum()) > 0, "r4 counter-example: the decoded tensor itself is not modified in place")
        chk(apply_video_op(v, "none") is v, "r4: --video-op none is the identity (r3 behaviour)")
        try:
            apply_video_op(v, "grey"); chk(False, "r4: an unknown video-op must raise")
        except ValueError:
            chk(True, "r4: an unknown video-op raises ValueError")
    else:
        print("  SKIP  --video-op black tensor checks (no torch here; this check runs on the GPU box)")
    print("[selftest]", "ALL PASS" if ok else "FAIL"); return ok


def main():
    if "--selftest" in sys.argv[1:]:  # r4: before parse_args, whose choices need the trainer module
        sys.exit(0 if selftest() else 1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--adapter", default=None)
    ap.add_argument("--anno"); ap.add_argument("--video-dir"); ap.add_argument("--ext", default=".mp4")
    ap.add_argument("--n", type=int, default=None); ap.add_argument("--offset", type=int, default=0); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompt-style", choices=sorted(V.NATIVE_PROMPTS), default="timer1_default")
    ap.add_argument("--fps", type=float, default=2.0); ap.add_argument("--total-pixels", type=int, default=3584 * 28 * 28)
    ap.add_argument("--min-pixels", type=int, default=16 * 28 * 28); ap.add_argument("--max-frames", type=int, default=768)
    ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--image-patch-size", type=int, default=14, help="14 Qwen2/2.5-VL (default), 16 Qwen3-VL"); ap.add_argument("--text-first", action="store_true", help="text part before the video part (TimeZero/VideoChat-R1 order)")
    ap.add_argument("--video-op", choices=["none", "black"], default="none", help="r4: black = zero every decoded frame (same token count, no visual content)")
    args = ap.parse_args()
    V._TEXT_FIRST["on"] = bool(args.text_first)
    INSTRUMENT = instrument_tag(args.text_first)  # r4: derived, not hard-coded
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not (args.model and args.anno and args.video_dir and args.out):
        ap.error("--model --anno --video-dir --out required")
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    items_all, n_mismatch = load_tlbench(args.anno)
    items = items_all[args.offset: (args.offset + args.n) if args.n else None]
    torch.cuda.set_device(args.gpu)
    proc = AutoProcessor.from_pretrained(args.model)
    model = V._fp_dtype_compat(AutoModelForImageTextToText, args.model, device_map={"": args.gpu})
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    done = load_partial(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if done: print(f"[p6n] resume: {len(done)} rows", flush=True)
    pf = open(_partial_path(args.out), "a", encoding="utf-8")
    print(f"[p6n] anno={os.path.basename(args.anno)} items_total={len(items_all)} (len-mismatch entries {n_mismatch}) n={len(items)} offset={args.offset} style={args.prompt_style} video_op={args.video_op} text_first={int(args.text_first)} instrument={INSTRUMENT} trainer_sha16={TRAINER_SHA16} model={os.path.basename(args.model)} adapter={args.adapter}", flush=True)
    res = []; t0 = time.time(); nparse = 0; nhit = 0
    for j, it in enumerate(items):
        if int(it["qid"]) in done:
            r = done[int(it["qid"])]; res.append(r); nparse += int(r.get("pred") is not None); continue
        vpath = os.path.join(args.video_dir, it["video"] + args.ext)
        rec = {"qid": it["qid"], "video": it["video"], "caption": it["caption"], "gsA": it["gs"], "geA": it["ge"], "gsB": None, "geB": None, "style": args.prompt_style, "instrument": INSTRUMENT, "text_first": int(args.text_first), "video_op": args.video_op}
        try:
            frames, times, dur, sfps = V.native_decode(vpath, args)
        except Exception as exc:
            rec.update({"decode_error": f"{type(exc).__name__}: {exc}", "pred": None, "dur": it["dur_anno"], "raw": "", "fps": args.fps, "nframes": -1})
            res.append(rec); pf.write(json.dumps(rec, ensure_ascii=False) + "\n"); pf.flush(); continue
        if dur <= 0: dur = it["dur_anno"]
        prompt_text = V.NATIVE_PROMPTS[args.prompt_style].format(it["caption"])
        vid = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
        vid = apply_video_op(vid, args.video_op)  # r4: 12p4n r5 l.100 equivalent
        _vp = {"type": "video", "video": "native_placeholder"}; _tp = {"type": "text", "text": prompt_text}; msgs = [{"role": "user", "content": ([_tp, _vp] if args.text_first else [_vp, _tp])}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = V._native_inputs(proc, text, vid, sfps).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        ans = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        pw = parse_native(ans, dur)
        if pw is not None: nparse += 1
        iou = 0.0
        if pw is not None:
            s1, e1 = pw; s2, e2 = it["gs"], it["ge"]; inter = max(0.0, min(e1, e2) - max(s1, s2)); uni = max(e1, e2) - min(s1, s2); iou = inter / uni if uni > 0 else 0.0
        nhit += int(iou >= 0.5)
        rec.update({"dur": float(dur), "pred": list(pw) if pw else None, "iou": float(iou), "raw": ans[-160:], "fps": float(sfps), "nframes": int(frames.shape[0])})
        res.append(rec); pf.write(json.dumps(rec, ensure_ascii=False) + "\n"); pf.flush()
        if (j + 1) % 50 == 0:
            print(f"  [p6n] {j+1}/{len(items)} parse={nparse/(j+1):.3f} R@0.5={nhit/(j+1):.3f} ({time.time()-t0:.0f}s)", flush=True)
    pf.close()
    assert len(res) == len(items) and [r["qid"] for r in res] == [it["qid"] for it in items], "row/qid order mismatch"
    json.dump(res, open(args.out, "w"), ensure_ascii=False)
    meta = {"instrument": INSTRUMENT, "trainer_sha16": TRAINER_SHA16, "args": vars(args),
            "text_first": bool(args.text_first), "video_op": args.video_op, "n_rows": len(res), "items_total": len(items_all), "len_mismatch_entries": n_mismatch, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=1)
    print(f"[p6n] DONE n={len(res)} parse_rate={nparse/max(1,len(res)):.3f} R@0.5={nhit/max(1,len(res)):.3f} elapsed={time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
