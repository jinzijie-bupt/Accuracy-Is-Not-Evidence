# r11 (rebuttal 2026-11-07, mk_14Ln_r11.py): r10 8a10ee350096c4e4 + --anno-override (self-audit on the own prediction; defaults = r10 rows).
# r10 (rebuttal 2026-11-06, mk_14Ln_r10.py): r9 4d4fee48d529b1c8 + --tok-lp/--tok-lp-xcheck (defaults = r9 rows).
# r9 (rebuttal 2026-11-06, mk_14Ln_r9.py): r8 0794417ae4d45d30 + --cands/--base-pred-file/--all-rows (defaults = r8).
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


# ---------------------------------------------------------------- r10 (--tok-lp): per-token log p + timestamp mask
import math

TSONLY_SHA16 = "7153d2cdae11bbd8"          # B3 trainer whose timestamp_char_spans/timestamp_token_mask are copied below
TRAINER_R4_SHA16_AT_GEN = "1fe0b70b91f6951a"   # trainer r4 whose native_completion_logprob body is copied below
TS_DEF = ("timestamp token = tokenizer offset span overlaps the char span of the formatter's start/end number "
          "(B3 trainer 27_papo_train_qcdro_v3seln_tsonly.py 7153d2cdae11bbd8, copied verbatim; timelens: *_1f copies)")

# ---- verbatim from 27_papo_train_qcdro_v3seln_tsonly.py (7153d2cdae11bbd8) ----
_TS_NUM = None


def timestamp_char_spans(completion: str, gs: float, ge: float) -> list[tuple[int, int]]:
    """Char spans of the start/end times inside the answer string built by native_completion (timer1 and vchat).
    Exactly two numbers must occur and they must be the formatter's f"{gs:.2f}" and f"{ge:.2f}"."""
    global _TS_NUM
    import re
    if _TS_NUM is None:
        _TS_NUM = re.compile(r"-?\d+\.\d+")
    spans = [(m.start(), m.end()) for m in _TS_NUM.finditer(completion)]
    got = [completion[a:b] for a, b in spans]
    want = [f"{gs:.2f}", f"{ge:.2f}"]
    if got != want:
        raise RuntimeError(f"[tsonly] timestamp spans {got} != formatter values {want} in {completion!r}")
    return spans


def timestamp_token_mask(tokenizer, completion: str, gs: float, ge: float):
    """(ids, mask, info): standalone tokenisation of the answer; mask[i]=1.0 iff token i's offset span overlaps a
    timestamp char span.  Every timestamp character must be covered; tokens that overlap but also reach outside a
    span are kept and counted in info['straddle'] (0 for the Qwen2 byte-level BPE, which splits digits singly)."""
    spans = timestamp_char_spans(completion, gs, ge)
    enc = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    offs = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    if len(offs) != len(ids):
        raise RuntimeError("[tsonly] offset_mapping length != input_ids length")
    mask, straddle, covered = [], 0, set()
    for a, b in offs:
        hit = b > a and any(a < e and s0 < b for s0, e in spans)
        inside = hit and any(s0 <= a and b <= e for s0, e in spans)
        straddle += int(hit and not inside)
        mask.append(1.0 if hit else 0.0)
        if hit:
            covered.update(range(a, b))
    need = {i for s0, e in spans for i in range(s0, e)}
    if not need <= covered:
        raise RuntimeError(f"[tsonly] timestamp chars not covered by selected tokens in {completion!r}")
    info = {
        "n_ts": int(sum(mask)), "n_tokens": len(ids), "straddle": int(straddle),
        "tokens": [completion[a:b] for a, b in offs],
        "ts_tokens": [completion[a:b] for (a, b), m in zip(offs, mask) if m],
    }
    return ids, mask, info


# ---- r10: timelens copies (answer_style timelens formats with .1f; B3 trains timer1/vchat only) ----
def timestamp_char_spans_1f(completion: str, gs: float, ge: float) -> list[tuple[int, int]]:
    """Char spans of the start/end times inside the answer string built by native_completion (timer1 and vchat).
    Exactly two numbers must occur and they must be the formatter's f"{gs:.2f}" and f"{ge:.2f}"."""
    global _TS_NUM
    import re
    if _TS_NUM is None:
        _TS_NUM = re.compile(r"-?\d+\.\d+")
    spans = [(m.start(), m.end()) for m in _TS_NUM.finditer(completion)]
    got = [completion[a:b] for a, b in spans]
    want = [f"{gs:.1f}", f"{ge:.1f}"]
    if got != want:
        raise RuntimeError(f"[tsonly] timestamp spans {got} != formatter values {want} in {completion!r}")
    return spans


def timestamp_token_mask_1f(tokenizer, completion: str, gs: float, ge: float):
    """(ids, mask, info): standalone tokenisation of the answer; mask[i]=1.0 iff token i's offset span overlaps a
    timestamp char span.  Every timestamp character must be covered; tokens that overlap but also reach outside a
    span are kept and counted in info['straddle'] (0 for the Qwen2 byte-level BPE, which splits digits singly)."""
    spans = timestamp_char_spans_1f(completion, gs, ge)
    enc = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    offs = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    if len(offs) != len(ids):
        raise RuntimeError("[tsonly] offset_mapping length != input_ids length")
    mask, straddle, covered = [], 0, set()
    for a, b in offs:
        hit = b > a and any(a < e and s0 < b for s0, e in spans)
        inside = hit and any(s0 <= a and b <= e for s0, e in spans)
        straddle += int(hit and not inside)
        mask.append(1.0 if hit else 0.0)
        if hit:
            covered.update(range(a, b))
    need = {i for s0, e in spans for i in range(s0, e)}
    if not need <= covered:
        raise RuntimeError(f"[tsonly] timestamp chars not covered by selected tokens in {completion!r}")
    info = {
        "n_ts": int(sum(mask)), "n_tokens": len(ids), "straddle": int(straddle),
        "tokens": [completion[a:b] for a, b in offs],
        "ts_tokens": [completion[a:b] for (a, b), m in zip(offs, mask) if m],
    }
    return ids, mask, info


def native_completion_token_logprobs_r10(model, proc, frames_np, sample_fps, prompt_text, completion, device, cache):
    """r10 (mk_14Ln_r10.py): trainer r4 native_completion_logprob copied verbatim, except that it returns
    (seg (1,n), scored answer token ids) instead of (seg.sum(), n) and reaches torch/_TEXT_FIRST/_native_inputs
    through the loaded trainer V.  r4 doc: (sum log p of completion tokens, n tokens) through the native videos= path. cache: per-example dict
    holding the chat-template text and prompt length (identical across clean/edited frames of one example)."""
    import torch
    vid = torch.from_numpy(np.ascontiguousarray(frames_np)).permute(0, 3, 1, 2)  # (T,3,H,W) uint8
    if "text" not in cache:
        _vp = {"type": "video", "video": "native_placeholder"}; _tp = {"type": "text", "text": prompt_text}
        msgs = [{"role": "user", "content": ([_tp, _vp] if V._TEXT_FIRST["on"] else [_vp, _tp])}]  # r3
        cache["text"] = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        cache["plen"] = int(V._native_inputs(proc, cache["text"], vid, sample_fps).input_ids.shape[1])
    full = V._native_inputs(proc, cache["text"] + completion, vid, sample_fps).to(device)
    plen = cache["plen"]
    out = model(**full)
    logits = out.logits[:, :-1, :].float()
    labels = full.input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    seg = logp[:, plen - 1:]
    return seg, [int(x) for x in full.input_ids[0, plen:].tolist()]


def ts_mask_for(tokenizer, completion, gs, ge, answer_style):
    """r10: (ids, mask, info) of the answer text under the B3 definition; timelens uses the .1f copies."""
    if answer_style == "timelens":
        return timestamp_token_mask_1f(tokenizer, completion, gs, ge)
    return timestamp_token_mask(tokenizer, completion, gs, ge)


def check_tok_row(rec, ops, tol=1e-3):
    """r10: every stored per-token array must reproduce its row sum field (|fsum - field| <= tol), have the answer
    length, and line up with tok_ids / ts_mask; candidates likewise against cand_lp.  Raises AssertionError."""
    def chk(c, m):
        if not c:
            raise AssertionError(f"[14Ln r10] qid={rec.get('qid')}: {m}")

    def side(name, ent, n, clean, evid, rand):
        ids, mask, tl = ent.get("tok_ids"), ent.get("ts_mask"), ent.get("tok_lp")
        chk(ids is not None and mask is not None and tl is not None, f"{name}: tok_ids/ts_mask/tok_lp missing")
        chk(len(ids) == n and len(mask) == n, f"{name}: len(tok_ids)={len(ids)} len(ts_mask)={len(mask)} != {n}")
        chk(set(mask) <= {0, 1} and sum(mask) > 0, f"{name}: ts_mask not 0/1 or empty")
        want = {"clean"} | {f"{op}_evid" for op in ops} | {f"{op}_rand" for op in ops}
        chk(set(tl) == want, f"{name}: tok_lp keys {sorted(tl)} != {sorted(want)}")

        def eq(arr, s, what):
            chk(len(arr) == n, f"{name} {what}: {len(arr)} tokens != {n}")
            chk(abs(math.fsum(arr) - float(s)) <= tol, f"{name} {what}: sum(tok)={math.fsum(arr):.6f} != field {float(s):.6f}")

        eq(tl["clean"], clean, "clean")
        for op in ops:
            eq(tl[f"{op}_evid"], evid[op], f"{op}_evid")
            chk(len(tl[f"{op}_rand"]) == len(rand[op]), f"{name} {op}_rand: {len(tl[f'{op}_rand'])} draws != {len(rand[op])}")
            for r_i, (arr, s) in enumerate(zip(tl[f"{op}_rand"], rand[op])):
                eq(arr, s, f"{op}_rand[{r_i}]")

    side("gt", rec, int(rec["tgt_len"]), rec["lp_clean_sum"],
         {op: rec["lp_clean_sum"] - rec[f"{op}_drop_evid"] for op in ops}, {op: rec[f"{op}_rand_sums"] for op in ops})
    ct, cl = rec.get("cand_tok") or {}, rec.get("cand_lp") or {}
    chk(set(ct) == set(cl), f"cand_tok {sorted(ct)} != cand_lp {sorted(cl)}")
    for k, ent in ct.items():
        side("cand:" + k, ent, int(cl[k]["len"]), cl[k]["clean"], {op: cl[k][f"{op}_evid"] for op in ops},
             {op: cl[k][f"{op}_rand"] for op in ops})
    return True


class _FakeTok:
    """r10 selftest stand-in for a fast tokenizer (offline, no transformers): Qwen2-like pre-tokenisation, digits
    split singly; merge_space_digit=True glues ' 1' into one token (a straddling token, as some BPEs would)."""

    def __init__(self, merge_space_digit=False):
        import re
        self.pat = re.compile((r" ?\d|" if merge_space_digit else r"\d|") + r" ?[A-Za-z]+| ?[^\sA-Za-z\d]+|\s+")

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        import zlib
        offs = [(m.start(), m.end()) for m in self.pat.finditer(text)]
        assert "".join(text[a:b] for a, b in offs) == text
        return {"input_ids": [zlib.crc32(text[a:b].encode()) % 100000 for a, b in offs], "offset_mapping": offs}


def selftest_r10():
    """r10 offline bench: the B3 timestamp mask on the three answer formats, its loud failures, and the row sum check
    with counter-examples (each check fails if the mechanism is removed)."""
    ok = True

    def chk(c, m):
        nonlocal ok
        ok &= bool(c); print(("  PASS " if c else "  FAIL ") + m)

    def fmt(gs, ge, style):                     # trainer r4 native_completion, restated for the offline bench
        if style == "timelens":
            return f"The event happens in {gs:.1f} - {ge:.1f} seconds."
        if style == "vchat":
            return f"<answer>[({gs:.2f}, {ge:.2f})]</answer>"
        return f"<answer> {gs:.2f} to {ge:.2f} </answer>"

    tk = _FakeTok()
    c = fmt(12.54, 17.83, "timer1")
    ids, mask, info = ts_mask_for(tk, c, 12.54, 17.83, "timer1")
    sel = info["ts_tokens"]; non = [t for t, m in zip(info["tokens"], mask) if not m]
    chk(c == "<answer> 12.54 to 17.83 </answer>" and "".join(sel) == "12.5417.83" and len(sel) == 10,
        f"timer1 {c!r}: mask selects exactly the 10 digit/point tokens {sel}")
    chk(all(set(t) <= set("0123456789.") for t in sel) and not any(ch.isdigit() or ch == "." for t in non for ch in t),
        "timer1: no tag / ' to' / space token selected, no digit or point left outside the mask")
    for style, (gs, ge), want in (("vchat", (3.5, 30.25), "3.5030.25"), ("timelens", (12.54, 17.83), "12.517.8"),
                                  ("timer1", (100.0, 123.45), "100.00123.45"), ("timer1", (0.0, 5.3), "0.005.30")):
        c = fmt(gs, ge, style); _, m_, inf = ts_mask_for(tk, c, gs, ge, style)
        chk("".join(inf["ts_tokens"]) == want and inf["straddle"] == 0 and len(m_) == inf["n_tokens"],
            f"{style} {c!r}: timestamp tokens {''.join(inf['ts_tokens'])!r}")
    try:
        ts_mask_for(tk, fmt(12.54, 17.83, "timer1"), 12.5, 17.83, "timer1"); raised = False
    except RuntimeError:
        raised = True
    chk(raised, "counter-example: formatter values != text numbers -> RuntimeError (never a silent empty mask)")
    try:
        ts_mask_for(tk, fmt(12.54, 17.83, "timelens"), 12.54, 17.83, "timer1"); raised = False
    except RuntimeError:
        raised = True
    chk(raised, "counter-example: timelens text under the .2f definition -> RuntimeError (style dispatch is load-bearing)")
    _, m2, inf2 = ts_mask_for(_FakeTok(merge_space_digit=True), fmt(12.54, 17.83, "timer1"), 12.54, 17.83, "timer1")
    chk(inf2["straddle"] == 2 and inf2["ts_tokens"][0] == " 1",
        "B3 rule on a space-merging BPE: ' 1' is kept as timestamp and counted in straddle (2)")

    import random as _r
    rr = _r.Random(5); n = 7; ops_ = ["mask", "swap"]; R_ = 3
    arr = lambda: [-rr.uniform(0.01, 3.0) for _ in range(n)]
    tl = {"clean": arr()}
    for op in ops_:
        tl[f"{op}_evid"] = arr(); tl[f"{op}_rand"] = [arr() for _ in range(R_)]
    rec = {"qid": 1, "tgt_len": n, "lp_clean_sum": math.fsum(tl["clean"]), "tok_ids": list(range(n)),
           "ts_mask": [0, 1, 1, 0, 1, 1, 0], "tok_lp": tl}
    for op in ops_:
        le = math.fsum(tl[f"{op}_evid"]); rs = [math.fsum(a) for a in tl[f"{op}_rand"]]
        rec[f"{op}_drop_evid"] = rec["lp_clean_sum"] - le; rec[f"{op}_rand_sums"] = rs
    chk(check_tok_row(rec, ops_), "check_tok_row passes on a consistent row")

    def bad(mut, m):
        import copy
        r2 = copy.deepcopy(rec); mut(r2)
        try:
            check_tok_row(r2, ops_); raised = False
        except AssertionError:
            raised = True
        chk(raised, "counter-example: " + m)

    bad(lambda r: r["tok_lp"]["swap_evid"].__setitem__(3, r["tok_lp"]["swap_evid"][3] - 0.01), "one E token off by 0.01 -> raises")
    bad(lambda r: r["tok_lp"]["mask_rand"][2].__setitem__(0, 0.0), "one RAND draw token changed -> raises")
    bad(lambda r: r["tok_lp"]["mask_rand"].pop(), "a missing RAND draw -> raises")
    bad(lambda r: r["tok_lp"].__setitem__("swap_evid", r["tok_lp"]["mask_evid"]), "E arrays of two ops swapped -> raises")
    bad(lambda r: r.__setitem__("ts_mask", r["ts_mask"][:-1]), "ts_mask shorter than the answer -> raises")
    bad(lambda r: r.__setitem__("cand_lp", {"own": {"clean": 0.0, "len": n}}), "cand_lp without cand_tok -> raises")
    print("[14Ln r10 selftest]", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- r11 (--anno-override): self-audit on the own prediction
OVR_GT_TOL = 1e-6                                         # r11: |12p4n gsA/geA - anno gs/ge| allowed (same float parse)


def load_anno_override(paths):
    """r11: merge one or more 12p4n row lists -> (by_qid {qid: row}, files [{path, sha256, n_rows}]).
    Each file must be a non-empty JSON list of rows carrying qid/video/pred/gsA/geA; a qid occurring twice (within or
    across files) aborts, even if the rows agree (12p4n shards are disjoint by construction)."""
    by, files = {}, []
    for p in paths:
        blob = open(p, "rb").read()
        rows = json.loads(blob.decode("utf-8"))
        if not isinstance(rows, list) or not rows:
            raise SystemExit(f"[14Ln r11] --anno-override {p}: not a non-empty 12p4n row list")
        for r in rows:
            miss = [k for k in ("qid", "video", "pred", "gsA", "geA") if k not in r]
            if miss:
                raise SystemExit(f"[14Ln r11] --anno-override {p}: row without {miss}: {str(r)[:120]}")
            q = int(r["qid"])
            if q in by:
                raise SystemExit(f"[14Ln r11] --anno-override: qid {q} occurs twice (second time in {p})")
            by[q] = r
        files.append({"path": os.path.abspath(p), "sha256": hashlib.sha256(blob).hexdigest(), "n_rows": len(rows)})
    return by, files


def override_item(it, orow, tol=OVR_GT_TOL):
    """r11: (item with gs/ge := the 12p4n prediction, None), or (None, "pred_none") when 12p4n parsed no window.
    The 12p4n row must be the same question as the anno item (video, caption, GT window); otherwise abort.
    The returned item keeps every other key and adds gt_gs/gt_ge (the true GT)."""
    q = it["qid"]
    if orow is None:
        raise SystemExit(f"[14Ln r11] qid {q}: not in --anno-override (the override shards must cover the run slice)")
    if str(orow["video"]) != str(it["video"]):
        raise SystemExit(f"[14Ln r11] qid {q}: override video {orow['video']!r} != anno video {it['video']!r} (qid misaligned)")
    if abs(float(orow["gsA"]) - float(it["gs"])) > tol or abs(float(orow["geA"]) - float(it["ge"])) > tol:
        raise SystemExit(f"[14Ln r11] qid {q}: override GT ({orow['gsA']}, {orow['geA']}) != anno GT ({it['gs']}, {it['ge']})")
    if orow.get("caption") is not None and str(orow["caption"]).strip() != str(it["caption"]).strip():
        raise SystemExit(f"[14Ln r11] qid {q}: override caption {orow['caption']!r} != anno caption {it['caption']!r}")
    pred = orow["pred"]
    if pred is None:
        return None, "pred_none"
    if not isinstance(pred, (list, tuple)) or len(pred) != 2:
        raise SystemExit(f"[14Ln r11] qid {q}: malformed override pred {pred!r}")
    s, e = float(pred[0]), float(pred[1])
    if not (math.isfinite(s) and math.isfinite(e)) or e < s or s < 0:   # 12p4n parse orders and clamps to [0, dur]
        raise SystemExit(f"[14Ln r11] qid {q}: malformed override pred {pred!r}")
    new = dict(it)
    new.update({"gs": s, "ge": e, "gt_gs": float(it["gs"]), "gt_ge": float(it["ge"])})
    return new, None


def selftest_r11():
    """r11 offline bench: shard merge, sha256, coverage/misalignment refusals, pred None skip, item replacement."""
    import tempfile
    ok = True

    def chk(c, m):
        nonlocal ok
        ok &= bool(c); print(("  PASS " if c else "  FAIL ") + m)

    def raises(fn):
        try:
            fn()
        except SystemExit:
            return True
        return False

    td = tempfile.mkdtemp(prefix="r11_selftest_")
    mk = lambda q, v, gs, ge, pred, cap=None: {"qid": q, "video": v, "caption": cap or f"cap{q}", "gsA": gs, "geA": ge, "pred": pred}
    s0 = [mk(0, "a.mp4", 1.0, 5.0, [2.0, 6.5]), mk(1, "a.mp4", 3.0, 9.0, None)]
    s1 = [mk(2, "b.mp4", 0.5, 4.25, [0.0, 3.0])]
    p0, p1 = os.path.join(td, "s0.json"), os.path.join(td, "s1.json")
    json.dump(s0, open(p0, "w")); json.dump(s1, open(p1, "w"))
    by, files = load_anno_override([p0, p1])
    chk(sorted(by) == [0, 1, 2] and [f["n_rows"] for f in files] == [2, 1], "two shards merged by qid")
    chk(files[0]["sha256"] == hashlib.sha256(open(p0, "rb").read()).hexdigest(), "sha256 is of the file bytes")
    chk(raises(lambda: load_anno_override([p0, p0])), "counter-example: the same shard twice -> qid twice -> abort")
    it0 = {"qid": 0, "video": "a.mp4", "gs": 1.0, "ge": 5.0, "caption": "cap0"}
    n0, why = override_item(it0, by[0])
    chk(why is None and (n0["gs"], n0["ge"], n0["gt_gs"], n0["gt_ge"]) == (2.0, 6.5, 1.0, 5.0) and it0["gs"] == 1.0,
        "pred replaces gs/ge, true GT kept as gt_gs/gt_ge, the anno item itself is not mutated")
    n1, why1 = override_item({"qid": 1, "video": "a.mp4", "gs": 3.0, "ge": 9.0, "caption": "cap1"}, by[1])
    chk(n1 is None and why1 == "pred_none", "pred None -> skipped with reason pred_none")
    chk(raises(lambda: override_item({"qid": 7, "video": "a.mp4", "gs": 1.0, "ge": 5.0, "caption": "c"}, None)),
        "counter-example: qid absent from the override -> abort (never a silent GT audit)")
    chk(raises(lambda: override_item(dict(it0, video="z.mp4"), by[0])), "counter-example: video differs -> abort")
    chk(raises(lambda: override_item(dict(it0, gs=1.5), by[0])), "counter-example: GT differs -> abort (qid misaligned)")
    chk(raises(lambda: override_item(dict(it0, caption="other"), by[0])), "counter-example: caption differs -> abort")
    chk(raises(lambda: override_item(it0, dict(by[0], pred=[6.5, 2.0]))), "counter-example: pred end < start -> abort")
    print("[14Ln r11 selftest]", "PASS" if ok else "FAIL")
    return ok


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
        sys.exit(0 if (selftest() & selftest_r8() & selftest_r10() & selftest_r11()) else 1)
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
    ap.add_argument("--cands", action="store_true", help="r9: also score own/base/shift competitor intervals")
    ap.add_argument("--base-pred-file", default=None, help="r9: r7/r8 product of the base model (field pred per qid)")
    ap.add_argument("--all-rows", action="store_true", help="r9: audit non-hit rows too")
    ap.add_argument("--tok-lp", action="store_true",
                    help="r10: also store per-token log p (clean/E/each RAND), answer token ids and the B3 timestamp mask")
    ap.add_argument("--tok-lp-xcheck", type=int, default=2,
                    help="r10: with --tok-lp, re-score the clean GT of the first N audited rows via the trainer's own "
                         "native_completion_logprob; |diff| > 1e-3 aborts")
    ap.add_argument("--anno-override", nargs="+", default=None, metavar="P4N_JSON",
                    help="r11: 12p4n row list(s); per qid the model's own pred replaces GT as the E window AND as the "
                         "scored target text (same formatter); pred None -> row skipped (override_skip). Needs --all-rows")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    V._TEXT_FIRST["on"] = bool(args.text_first)
    if args.anno_override and not args.all_rows:          # r11: the hit gate would compare regen vs the override window
        raise SystemExit("[14Ln r11] --anno-override requires --all-rows (the r7 hit gate must not drop own-pred rows)")
    if args.tok_lp and TRAINER_SHA16 != TRAINER_R4_SHA16_AT_GEN:   # r10: the per-token forward is a copy of r4's
        raise SystemExit(f"[14Ln r10] --tok-lp: trainer {_T} sha16 {TRAINER_SHA16} != {TRAINER_R4_SHA16_AT_GEN} "
                         f"(the body copied by mk_14Ln_r10.py); deploy the kit trainer r4")
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
    base_pred = {}
    if args.base_pred_file:                              # r9
        for _r in json.load(open(args.base_pred_file, encoding="utf-8")):
            if _r.get("pred"):
                base_pred[_r["qid"]] = _r["pred"]
        print(f"[14Ln r9] base preds loaded: {len(base_pred)} from {args.base_pred_file}", flush=True)
    tok_xcheck = []; _tokz = getattr(proc, "tokenizer", proc)   # r10
    if args.tok_lp:
        print(f"[14Ln r10] --tok-lp on: per-token log p + timestamp mask ({TS_DEF}); xcheck first {args.tok_lp_xcheck} rows", flush=True)
    ovr_by, ovr_files, ovr_skipped = None, [], []        # r11
    if args.anno_override:
        ovr_by, ovr_files = load_anno_override(args.anno_override)
        _miss = [it["qid"] for it in items if it["qid"] not in ovr_by]
        if _miss:
            raise SystemExit(f"[14Ln r11] {len(_miss)} qid(s) of the run slice missing from --anno-override, first {_miss[:5]}")
        print(f"[14Ln r11] --anno-override: {len(ovr_by)} rows from {len(ovr_files)} file(s) "
              f"{[f['sha256'][:16] for f in ovr_files]}; E window and target = own prediction", flush=True)
    res = []; t0 = time.time(); nhit = 0
    for j, it in enumerate(items):
        if ovr_by is not None:                            # r11: own prediction replaces GT (E window + scored target)
            it, _why = override_item(items[j], ovr_by.get(items[j]["qid"]))
            if it is None:
                ovr_skipped.append(items[j]["qid"])
                res.append({"qid": items[j]["qid"], "video": items[j]["video"], "anno_override": 1,
                            "target_mode": "own_pred", "override_skip": _why})
                continue
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
        if ovr_by is not None:                            # r11
            rec.update({"target_mode": "own_pred", "anno_override": 1, "ovr_gs": it["gs"], "ovr_ge": it["ge"],
                        "gt_gs": it["gt_gs"], "gt_ge": it["gt_ge"]})
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
        if ovr_by is not None:                            # r11: clean_iou/hit above are vs the override window (regen agreement)
            rec["regen_iou_gt"] = float(iou_1d(pw, (it["gt_gs"], it["gt_ge"]))) if pw else 0.0
        if clean < args.thr and not args.all_rows:          # r9: --all-rows audits non-hits too
            res.append(rec); continue
        nhit += int(clean >= args.thr)
        tgt_txt = V.native_completion(it["gs"], it["ge"], args.answer_style)
        rec["gt_target_text"] = tgt_txt

        tokrec = {}                                       # r10: "gt" / candidate name -> tok_ids, ts_mask, ts_info, tok_lp

        def score(fr, txt=None, key=None):
            if args.tok_lp:                               # r10
                return _score_tok(fr, txt, key)
            with torch.no_grad():
                s, n = V.native_completion_logprob(model, proc, fr, sfps, prompt_text, tgt_txt if txt is None else txt, model.device, cache)
            return float(s), int(n)

        def _score_tok(fr, txt, key):                     # r10: same forward, per-token vector kept
            _txt = tgt_txt if txt is None else txt
            with torch.no_grad():
                seg, ids = native_completion_token_logprobs_r10(model, proc, fr, sfps, prompt_text, _txt, model.device, cache)
            s, n = float(seg.sum()), int(seg.shape[1])
            toks = [float(x) for x in seg[0].tolist()]
            assert len(toks) == n == len(ids) and abs(math.fsum(toks) - s) <= 1e-3, (it["qid"], key, n, len(ids), math.fsum(toks), s)
            name, cond = key
            ent = tokrec.get(name)
            if ent is None:
                _ws, _we = (it["gs"], it["ge"]) if name == "gt" else cand_win[name]
                mids, mask, info = ts_mask_for(_tokz, _txt, _ws, _we, args.answer_style)
                if mids != ids:
                    raise RuntimeError(f"[14Ln r10] qid={it['qid']} {name}: standalone answer ids {mids} != scored ids {ids}")
                ent = tokrec[name] = {"tok_ids": ids, "ts_mask": [int(m) for m in mask], "ts_info": info, "tok_lp": {}}
            elif ent["tok_ids"] != ids:
                raise RuntimeError(f"[14Ln r10] qid={it['qid']} {name}/{cond}: scored ids changed between frame variants")
            if cond.endswith("_rand"):
                ent["tok_lp"].setdefault(cond, []).append(toks)
            else:
                ent["tok_lp"][cond] = toks
            return s, n

        cands = {}; cand_win = {}                         # r9 competitor intervals (same formatter as GT); r10: their (s, e)
        if args.cands:
            _gs, _ge = it["gs"], it["ge"]; _Lw = _ge - _gs; _D = float(dur) if dur and dur > 0 else _ge + _Lw
            def _add(name, w):
                if w is None:
                    return
                s_, e_ = max(0.0, float(w[0])), min(_D, float(w[1]))
                if e_ - s_ <= 0 or iou_1d((s_, e_), (_gs, _ge)) >= 0.999:
                    return
                t_ = V.native_completion(s_, e_, args.answer_style)
                if t_ != tgt_txt and t_ not in cands.values():
                    cands[name] = t_; cand_win[name] = (s_, e_)
            _add("own", pw)
            _add("base", base_pred.get(it["qid"]))
            _add("shift", (_gs + _Lw, _ge + _Lw) if _ge + _Lw <= _D else (_gs - _Lw, _ge - _Lw))
            rec["cand_texts"] = dict(cands)
            rec["cand_lp"] = {k: {} for k in cands}

        lp_c_sum, tgt_len = score(frames, key=("gt", "clean"))
        rec["lp_clean_sum"] = lp_c_sum; rec["lp_clean_mean"] = lp_c_sum / max(1, tgt_len); rec["tgt_len"] = tgt_len
        if args.tok_lp and len(tok_xcheck) < args.tok_lp_xcheck:   # r10: copied forward == trainer's own function
            with torch.no_grad():
                _s0, _n0 = V.native_completion_logprob(model, proc, frames, sfps, prompt_text, tgt_txt, model.device, cache)
            tok_xcheck.append({"qid": it["qid"], "tok_sum": lp_c_sum, "trainer_sum": float(_s0), "n": int(_n0)})
            if abs(float(_s0) - lp_c_sum) > 1e-3 or int(_n0) != tgt_len:
                raise RuntimeError(f"[14Ln r10] xcheck failed {tok_xcheck[-1]} (tgt_len {tgt_len})")
        for _k, _t in cands.items():                      # r9
            _s, _n = score(frames, _t, key=(_k, "clean")); rec["cand_lp"][_k].update({"clean": _s, "len": _n})
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
            _fe = apply_op(frames, estar, op, distractor, **okw)
            lp_e_sum, _ = score(_fe, key=("gt", f"{op}_evid"))
            for _k, _t in cands.items():                  # r9
                rec["cand_lp"][_k][f"{op}_evid"] = score(_fe, _t, key=(_k, f"{op}_evid"))[0]; rec["cand_lp"][_k][f"{op}_rand"] = []
            rand_sums = []; rand_idx = []
            for r_i in range(args.R):
                ridx = draw_rand(nonev, min(len(nonev), len(estar)), args.seed, it["qid"], op, r_i, rng)
                rand_idx.append([int(x) for x in ridx])
                _fr = apply_op(frames, ridx, op, distractor, **okw)
                rs, _ = score(_fr, key=("gt", f"{op}_rand"))
                rand_sums.append(rs)
                for _k, _t in cands.items():              # r9
                    rec["cand_lp"][_k][f"{op}_rand"].append(score(_fr, _t, key=(_k, f"{op}_rand"))[0])
            rec[f"{op}_drop_evid"] = lp_c_sum - lp_e_sum
            rec[f"{op}_drop_rand"] = lp_c_sum - float(np.mean(rand_sums))
            rec[f"{op}_tokmean_drop_evid"] = (lp_c_sum - lp_e_sum) / max(1, tgt_len)
            rec[f"{op}_tokmean_drop_rand"] = (lp_c_sum - float(np.mean(rand_sums))) / max(1, tgt_len)
            rec[f"{op}_rand_sums"] = rand_sums; rec[f"{op}_rand_idx"] = rand_idx; rec[f"{op}_donor"] = dvid
        if args.tok_lp:                                   # r10
            _g = tokrec["gt"]
            rec["tok_ids"] = _g["tok_ids"]; rec["ts_mask"] = _g["ts_mask"]; rec["ts_info"] = _g["ts_info"]; rec["tok_lp"] = _g["tok_lp"]
            if cands:
                rec["cand_tok"] = {_k: tokrec[_k] for _k in cands}
            check_tok_row(rec, ops)
        res.append(rec)
        if (j + 1) % 25 == 0:
            print(f"  [14Ln] {j+1}/{len(items)} hits={nhit} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False)
    meta = {"instrument": "14Ln_native_sentinel_probe_v1r2_tf_q3_r11", "trainer_sha16": TRAINER_SHA16, "args": vars(args),
            "ops": ops, "rng_mode": rng_mode, "probe_sha16": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16],
            "n_rows": len(res), "n_hits": nhit, "elapsed_s": round(time.time() - t0, 1)}
    if args.tok_lp:                                       # r10 (absent in default mode)
        meta["tok_lp"] = {"ts_def": TS_DEF, "tsonly_sha16": TSONLY_SHA16, "tokfwd_from_trainer_r4": TRAINER_R4_SHA16_AT_GEN,
                          "sum_tol": 1e-3, "xcheck": tok_xcheck}
    if ovr_by is not None:                                # r11 (absent in default mode)
        meta["anno_override"] = {"files": ovr_files, "n_override_rows": len(ovr_by), "n_items": len(items),
                                 "n_skipped_pred_none": len(ovr_skipped), "skipped_qids": ovr_skipped,
                                 "target_mode": "own_pred", "gt_tol": OVR_GT_TOL,
                                 "note": "E window and scored target = 12p4n pred; clean_iou/hit = regen vs override window"}
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=1)
    print(f"[14Ln] DONE n={len(res)} hits={nhit} elapsed={time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
