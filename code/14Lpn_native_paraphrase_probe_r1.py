# 14Lpn r1 — CLONE of 14Lqn_native_query_swap_probe_r3.py (124cfd516a39f427), one mechanism swapped:
#   the partner query is no longer "another event of the same video" (pick_partner) but the SAME-INTERVAL paraphrase of
#   14Lp_query_paraphrase_probe.py (1cad3b64487e5796): partner caption comes from --partner-map / --partner-mode map|self,
#   loaded with 14Lp's validation semantics (map caption == anno caption; partner_caption != caption; absent qid -> has_partner=0).
#   Because a paraphrase describes the SAME interval, E_j := E*_i (partner_gs/ge = the row's own gs/ge). The r3 placebo pool
#   T \ (E*_i u E_j) therefore reduces to T \ E*_i and its assertion is kept verbatim and still fires.
#   Read-out: the 2x2 cross terms (psi_own / psi_cross / spec) degenerate to 0 by construction here (E_j == E*_i); the live
#   read-out is the r2-style pair Psi(q_i) = tokmean_drop_evid - tokmean_drop_rand vs Psi(q_i') = its *_q2_* twin (14Lp §report).
#   Everything else (frames, native prompts/answer styles, fps sampling, likelihood scoring, ops, R draws, row schema) is r3 verbatim.
# r3: r2 86c33f31f3bdc772 + 2x2 cross design (query {q_i,q_j} x edit {none, E*_i, E_j}),
# placebo pool T \ (E*_i u E_j) with an assertion, per-(qid,op,r) RNG, --image-patch-size, prov on every row, --selftest.
# C9 §5: the r2 placebo pool always contained E_j, so psi_j / Gamma_agnostic measured the method working, not query-agnosticism.
# text-first variant: source f3042a76a02e178e + --text-first (message order text->video; trainer r3 729232059d642960 scoring path follows the same flag). Flag off = source behaviour.
# r2: use_cache=True in the clean generation (VideoChat-R1 config has use_cache=false); scoring path unchanged.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""14Lqn_native_query_swap_probe.py — 14Ln r2 + query-swap branch (port of 14Lq 27ba3362443e01a3):
same edited frames and the same scored target y_i, additionally scored under a partner query q_j from the SAME video whose
evidence frames do not intersect E*_i (deterministic partner: farthest evidence centre, then lowest qid). Fields *_q2_* as in 14Lq.

Base probe docstring follows.
14Ln_native_sentinel_probe.py — fixed-answer audit under the NATIVE protocol.

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
import argparse, os, sys, json, time, random, hashlib, importlib.util, inspect, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_CAND = ["27_papo_train_qcdro_v3seln_r4.py", "27_papo_train_qcdro_v3seln_r3.py"]  # r3: r4 adds timelens + image_patch_size; node4 ships only r3
_T = next((os.path.join(HERE, c) for c in _CAND if os.path.exists(os.path.join(HERE, c))), os.path.join(HERE, _CAND[-1]))
_IMPORT_ERR = None
try:  # r3: the offline bench must run on a box without torch; main() re-raises before it touches V
    _spec = importlib.util.spec_from_file_location("v3seln", _T)
    V = importlib.util.module_from_spec(_spec); sys.modules["v3seln"] = V; _spec.loader.exec_module(V)
except Exception as _exc:  # noqa: BLE001
    V = None; _IMPORT_ERR = _exc
TRAINER_SHA16 = hashlib.sha256(open(_T, "rb").read()).hexdigest()[:16]
TRAINER_FILE = os.path.basename(_T)
INSTRUMENT = "14Lpn_native_paraphrase_probe_v1_tf_r1"
PROBE_SHA16 = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]

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


def pick_partner(it, estar, times, same_video):
    """14Lq §2.1, deterministic: same video, different caption, non-empty E_j disjoint from E_i; farthest centre, then lowest qid."""
    ei = set(int(x) for x in estar); ci = float(np.mean(estar)); best = None
    for cand in same_video:
        if int(cand["qid"]) == int(it["qid"]) or cand["caption"] == it["caption"]:
            continue
        ej = np.asarray([i for i, t in enumerate(times) if cand["gs"] - 1e-6 <= t <= cand["ge"] + 1e-6], dtype=int)
        if len(ej) == 0 or (ei & set(int(x) for x in ej)):
            continue
        key = (-abs(float(ej.mean()) - ci), int(cand["qid"]))
        if best is None or key < best[0]:
            best = (key, cand, ej)
    return (None, None) if best is None else (best[1], best[2])


def load_partner_map(path):
    """14Lp file format: {"meta": {...}, "map": {"<qid>": {"caption", "partner_caption", "partner_kind", "partner_src_qid"}}}.
    Returns (pmap keyed by int qid, sha16). Empty map is fatal (an empty map would silently turn every row into has_partner=0)."""
    J = json.load(open(path, encoding="utf-8"))
    pmap = {int(k): v for k, v in J["map"].items()}
    if not pmap:
        raise SystemExit("partner-map is empty -- refusing to run (every row would silently lose its partner)")
    return pmap, hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def pick_partner_map(it, pmap, mode="map"):
    """14Lp §pick_partner, verbatim semantics: partner = another human caption of the SAME (video, gs, ge).
    Deterministic, touches no rng. Returns (partner_caption, kind, src_qid) or (None, None, None) when the qid
    is absent from the map. mode="self" (smoke gate only): the partner IS the row's own caption, so every
    *_q2_* field must come out bit-for-bit equal to its own-query twin."""
    if mode == "self":
        return it["caption"], "self", int(it["qid"])
    m = pmap.get(int(it["qid"]))
    if m is None:
        return None, None, None
    assert m["caption"] == it["caption"], "partner-map caption disagrees with the annotation qid=%s" % (it["qid"],)
    assert m["partner_caption"] != it["caption"], "partner caption identical to the original qid=%s" % (it["qid"],)
    return m["partner_caption"], m.get("partner_kind", "map"), int(m.get("partner_src_qid", -1))


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



def draw_placebo(pool, k, estar, ej, seed_key):
    """r3: k placebo frames from a pool that must exclude BOTH evidence sets.
    C9 §5: in r2 the pool was T \\ E*_i, so E_j was always inside it and every draw destroyed part of the partner
    evidence. The assertion below IS the mechanism; selftest() feeds it an r2-style pool to show that it fires."""
    rr = random.Random(str(seed_key))
    idx = sorted(int(x) for x in rr.sample(list(pool), min(len(pool), int(k))))
    bad = sorted(set(idx) & (set(int(x) for x in estar) | set(int(x) for x in ej)))
    assert not bad, "placebo draw overlaps evidence frames %s (pool not disjoint from E*_i u E_j)" % (bad,)
    return idx


def build_op_fields(rec, op, tgt_len, lp_c, lp_c2, lp_ei, lp_ej, rand_sums, lp_ei2, lp_ej2, rand2):
    """Write the ten scored log-probs of one operator and both read-outs.
    2x2 cross design:  psi_own = [lp(none) - lp(edit E*_i)]|q_i - [lp(none) - lp(edit E_j)]|q_i = (lp_ej - lp_ei)/L
                       psi_cross = same under q_j;  spec = psi_own - psi_cross  (the interaction term)
    Old read-out keeps the r2 field names; its placebo is now drawn from T \\ (E*_i u E_j)."""
    L = max(1, int(tgt_len))
    rec[op + "_lp_evid_i"] = lp_ei
    rec[op + "_lp_evid_j"] = lp_ej
    rec[op + "_lp_rand"] = list(rand_sums)
    rec[op + "_q2_lp_evid_i"] = lp_ei2
    rec[op + "_q2_lp_evid_j"] = lp_ej2
    rec[op + "_q2_lp_rand"] = list(rand2)
    rec[op + "_rand_sums"] = list(rand_sums)
    rec[op + "_drop_evid"] = lp_c - lp_ei
    rec[op + "_tokmean_drop_evid"] = (lp_c - lp_ei) / L
    if rand_sums:
        mr = float(np.mean(rand_sums))
        rec[op + "_drop_rand"] = lp_c - mr
        rec[op + "_tokmean_drop_rand"] = (lp_c - mr) / L
    if lp_ej is not None:
        rec[op + "_drop_evid_j"] = lp_c - lp_ej
        rec[op + "_tokmean_drop_evid_j"] = (lp_c - lp_ej) / L
        rec[op + "_psi_own"] = (lp_ej - lp_ei) / L
    if lp_c2 is not None and lp_ei2 is not None:
        rec[op + "_q2_drop_evid"] = lp_c2 - lp_ei2
        rec[op + "_q2_tokmean_drop_evid"] = (lp_c2 - lp_ei2) / L
        if rand2:
            m2 = float(np.mean(rand2))
            rec[op + "_q2_drop_rand"] = lp_c2 - m2
            rec[op + "_q2_tokmean_drop_rand"] = (lp_c2 - m2) / L
        if lp_ej2 is not None:
            rec[op + "_q2_drop_evid_j"] = lp_c2 - lp_ej2
            rec[op + "_q2_tokmean_drop_evid_j"] = (lp_c2 - lp_ej2) / L
            rec[op + "_q2_psi_cross"] = (lp_ej2 - lp_ei2) / L
            rec[op + "_psi_cross"] = rec[op + "_q2_psi_cross"]
            rec[op + "_spec"] = rec[op + "_psi_own"] - rec[op + "_psi_cross"]
    return rec


def selftest():
    """Offline (no torch): positives = the two operators, the estimand identities, partner selection;
    negatives = r2-style placebo pool must raise, and E_j := E*_i must collapse Specificity to 0."""
    ok = True

    def chk(c, m):
        nonlocal ok
        ok &= bool(c)
        print(("  PASS " if c else "  FAIL ") + m)

    frames = np.stack([np.full((2, 2, 3), v, dtype=np.uint8) for v in (10, 20, 30)])
    dist = np.stack([np.full((2, 2, 3), v, dtype=np.uint8) for v in (200, 201, 202)])
    m = apply_op(frames, [0, 2], "mask", dist)
    chk(int(m[0].max()) == 0 and int(m[2].max()) == 0 and int(m[1].max()) == 20 and int(frames[0].max()) == 10,
        "mask zeroes exactly the listed frames and leaves the input array untouched")
    w = apply_op(frames, [1], "swap", dist)
    chk(int(w[1].max()) == 200 and int(w[0].max()) == 10 and int(w[2].max()) == 30,
        "swap writes the donor prefix into exactly the listed frames")

    rec = {}
    build_op_fields(rec, "mask", 10, -20.0, -25.0, -30.0, -24.0, [-22.0, -23.0], -34.0, -29.0, [-26.0, -27.0])
    chk(len([k for k in rec if k.startswith("mask_lp_") or k.startswith("mask_q2_lp_")]) == 6
        and len(rec["mask_lp_rand"]) == 2 and len(rec["mask_q2_lp_rand"]) == 2,
        "all ten scored log-probs of the operator are written to the row")
    chk(abs(rec["mask_psi_own"] - (-24.0 - -30.0) / 10) < 1e-12 and abs(rec["mask_psi_cross"] - (-29.0 - -34.0) / 10) < 1e-12,
        "psi_own / psi_cross = (lp(edit E_j) - lp(edit E*_i)) / tgt_len")
    chk(abs(rec["mask_spec"] - (rec["mask_psi_own"] - rec["mask_psi_cross"])) < 1e-12,
        "Specificity = psi_own - psi_cross (2x2 interaction)")
    d_own = (rec["mask_drop_evid"] - rec["mask_drop_evid_j"]) / 10
    chk(abs(d_own - rec["mask_psi_own"]) < 1e-12,
        "psi_own equals the difference of the two drops from the SAME clean anchor (anchor cancels)")

    base = {}
    build_op_fields(base, "mask", 10, -20.0, -25.0, -28.0, -26.0, [-22.5, -23.5], -33.0, -31.0, [-26.5, -27.5])
    g_spec = rec["mask_spec"] - base["mask_spec"]
    chk(abs(g_spec - ((rec["mask_psi_own"] - base["mask_psi_own"]) - (rec["mask_psi_cross"] - base["mask_psi_cross"]))) < 1e-12,
        "Gamma_spec = Spec(arm) - Spec(base) = dGamma(psi_own) - dGamma(psi_cross)")
    psi_i = lambda r: r["mask_tokmean_drop_evid"] - r["mask_tokmean_drop_rand"]
    psi_j = lambda r: r["mask_q2_tokmean_drop_evid"] - r["mask_q2_tokmean_drop_rand"]
    G = psi_i(rec) - psi_i(base); Ga = psi_j(rec) - psi_j(base)
    chk(abs(G - (Ga + (G - Ga))) < 1e-12, "old read-out identity Gamma = Gamma_agnostic + Gamma_sem still holds")

    items = [{"qid": 5, "caption": "b", "gs": 0.0, "ge": 0.4}, {"qid": 7, "caption": "c", "gs": 0.8, "ge": 1.2}]
    times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    p, e = pick_partner({"qid": 1, "caption": "a"}, np.asarray([0, 1]), times, items)
    chk(p is not None and int(p["qid"]) == 7 and [int(x) for x in e] == [4, 5, 6],
        "pick_partner: farthest non-overlapping same-video event, E_j disjoint from E*_i")

    chk(draw_placebo([3], 1, [0], [1, 2], "k") == [3], "clean pool T \\ (E*_i u E_j) draws without tripping the guard")
    chk(draw_placebo([3, 4], 5, [0], [1, 2], "k") == [3, 4], "pool shorter than |E*| -> draw what exists (row keeps pool_short=1)")
    raised = False
    try:
        draw_placebo([0, 1, 2, 3], 3, [0], [1, 2], "k")  # r2-style pool: only E*_i removed, so E_j is inside it
    except AssertionError:
        raised = True
    chk(raised, "NEGATIVE: drop the E_j exclusion (r2 pool) -> draw_placebo raises")
    r0 = {}
    build_op_fields(r0, "mask", 10, -20.0, -25.0, -30.0, -30.0, [-22.0], -34.0, -34.0, [-26.0])
    chk(r0["mask_spec"] == 0.0 and r0["mask_psi_own"] == 0.0,
        "NEGATIVE: E_j := E*_i (control does not vary on the tested dimension) -> Specificity degenerates to 0")

    # ---- 14Lpn r1: the swapped mechanism (paraphrase partner map) ----
    it0 = {"qid": 11, "caption": "person turn a light on.", "gs": 0.0, "ge": 2.0}
    good = {11: {"caption": "person turn a light on.", "partner_caption": "person flips the light switch.",
                 "partner_kind": "sibling", "partner_src_qid": 42}}
    pc, pk, ps = pick_partner_map(it0, good, "map")
    chk(pc == "person flips the light switch." and pk == "sibling" and ps == 42,
        "map mode: partner caption/kind/src_qid come from the map entry of this qid")
    chk(pick_partner_map({"qid": 99, "caption": "x"}, good, "map") == (None, None, None),
        "map mode: a qid absent from the map yields no partner (row will carry has_partner=0)")
    pcs, pks, pss = pick_partner_map(it0, {}, "self")
    chk(pcs == it0["caption"] and pks == "self" and pss == 11,
        "self mode: partner IS the row's own caption (smoke gate expects bit-identical *_q2_* fields)")
    raised = False
    try:
        pick_partner_map(it0, {11: dict(good[11], caption="person turned A LIGHT on.")}, "map")
    except AssertionError:
        raised = True
    chk(raised, "NEGATIVE: doctored map caption (map entry not the annotation's caption) -> pick_partner_map raises")
    raised = False
    try:
        pick_partner_map(it0, {11: dict(good[11], partner_caption=it0["caption"])}, "map")
    except AssertionError:
        raised = True
    chk(raised, "NEGATIVE: partner caption equal to the original (no paraphrase at all) -> pick_partner_map raises")
    import tempfile
    _fp = os.path.join(tempfile.mkdtemp(), "pm.json")
    json.dump({"meta": {}, "map": {"11": good[11]}}, open(_fp, "w"))
    _pm, _sha = load_partner_map(_fp)
    chk(list(_pm) == [11] and isinstance(list(_pm)[0], int) and len(_sha) == 16,
        "load_partner_map: json keys become int qids and the file sha16 is reported")
    json.dump({"meta": {}, "map": {}}, open(_fp, "w"))
    raised = False
    try:
        load_partner_map(_fp)
    except SystemExit:
        raised = True
    chk(raised, "NEGATIVE: empty partner map -> refuse to run (would silently zero every partner)")
    # E_j := E*_i is this probe's DEFINITION (same interval), so the r3 placebo guard must still accept the reduced pool
    chk(draw_placebo([2, 3], 2, [0, 1], [0, 1], "k") == [2, 3],
        "paraphrase geometry (E_j == E*_i): placebo pool T \\ E*_i passes the unweakened r3 assertion")
    raised = False
    try:
        draw_placebo([0, 2, 3], 3, [0, 1], [0, 1], "k")
    except AssertionError:
        raised = True
    chk(raised, "NEGATIVE: an E*_i frame left in the pool still trips the r3 assertion under E_j == E*_i")
    print("[selftest]", "ALL PASS" if ok else "FAIL")
    return ok


def main():
    if "--selftest" in sys.argv[1:]:  # r3: before parse_args and before V is needed
        sys.exit(0 if selftest() else 1)
    if V is None:
        raise _IMPORT_ERR
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--adapter", default=None)
    ap.add_argument("--anno", default=None, help="charades_sta_test.txt (default: <VIDEO_DIR>/../anno/charades_sta_test.txt)")
    ap.add_argument("--n", type=int, default=None); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--ops", default="mask,swap"); ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompt-style", choices=sorted(V.NATIVE_PROMPTS), default="timer1_default")
    ap.add_argument("--answer-style", choices=["timer1", "vchat", "timelens"], default="timer1")
    ap.add_argument("--image-patch-size", type=int, default=14, help="14 Qwen2/2.5-VL (default), 16 Qwen3-VL/TimeLens-8B; read by V.native_decode via getattr")
    ap.add_argument("--fps", type=float, default=2.0); ap.add_argument("--total-pixels", type=int, default=3584 * 28 * 28)
    ap.add_argument("--min-pixels", type=int, default=16 * 28 * 28); ap.add_argument("--max-frames", type=int, default=768)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--text-first", action="store_true", help="text part before the video part (TimeZero/VideoChat-R1 order)")
    ap.add_argument("--partner-mode", choices=["map", "self"], default="map",
                    help="map = same-interval paraphrase from --partner-map (canonical); self = partner is the row's own caption (smoke gate only)")
    ap.add_argument("--partner-map", default=None, help="14Lp paraphrase map JSON; required in map mode")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true", help="offline bench (handled before parse_args, so --out stays required for real runs)")
    args = ap.parse_args()
    V._TEXT_FIRST["on"] = bool(args.text_first)
    _dec_src = inspect.getsource(V.native_decode)  # r3: the gate reads the module that was imported, not a file name
    if int(args.image_patch_size) != 14:
        assert "image_patch_size" in _dec_src, "--image-patch-size %d but %s ignores it (needs trainer r4)" % (args.image_patch_size, TRAINER_FILE)
    if args.answer_style == "timelens":
        assert V.native_completion(0.0, 1.0, "timelens") == "The event happens in 0.0 - 1.0 seconds.", "%s has no timelens answer format" % TRAINER_FILE
    pmap = {}; pmap_sha16 = None
    if args.partner_mode == "map":
        if not args.partner_map:
            raise SystemExit("--partner-mode map requires --partner-map")
        pmap, pmap_sha16 = load_partner_map(args.partner_map)
        print("[14Lpn] partner-map %s entries=%d sha16=%s" % (args.partner_map, len(pmap), pmap_sha16), flush=True)
    PROV = {"instrument": INSTRUMENT, "probe_sha16": PROBE_SHA16, "trainer_file": TRAINER_FILE, "trainer_sha16": TRAINER_SHA16,
            "text_first": bool(args.text_first), "prompt_style": args.prompt_style, "answer_style": args.answer_style,
            "adapter": args.adapter, "seed": int(args.seed), "ops": args.ops, "image_patch_size": int(args.image_patch_size),
            "partner_mode": args.partner_mode, "partner_map": args.partner_map, "partner_map_sha16": pmap_sha16}
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    ops = [o for o in args.ops.split(",") if o]
    anno = args.anno or os.path.join(os.path.dirname(V.base.VIDEO_DIR), "anno", "charades_sta_test.txt")
    items = load_test(anno, args.n, args.offset)
    all_items = load_test(anno); all_videos = sorted({it["video"] for it in all_items})
    byvid = {}
    for x in all_items:
        byvid.setdefault(x["video"], []).append(x)
    if pmap:  # 14Lp: verify the map against the WHOLE population up front, not row by row after an hour of GPU
        byq = {int(it["qid"]): it for it in items}
        bad = [q for q, m in pmap.items() if q in byq and m["caption"] != byq[q]["caption"]]
        if bad:
            raise SystemExit("partner-map disagrees with the annotation on %d rows, first qid=%s" % (len(bad), bad[0]))
        n_part = sum(1 for q in byq if q in pmap)
        print("[14Lpn] rows with a paraphrase partner: %d/%d" % (n_part, len(items)), flush=True)
        if n_part == 0:
            raise SystemExit("no row of this population has a paraphrase partner -- refusing to run")
    torch.cuda.set_device(args.gpu)
    proc = AutoProcessor.from_pretrained(args.model)
    model = V._fp_dtype_compat(AutoModelForImageTextToText, args.model, device_map={"": args.gpu})
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    rng = random.Random(args.seed + args.gpu)
    print(f"[14Ln] n={len(items)} offset={args.offset} ops={ops} R={args.R} style={args.prompt_style}/{args.answer_style} "
          f"fps={args.fps} trainer_sha16={TRAINER_SHA16} model={os.path.basename(args.model)} adapter={args.adapter}", flush=True)
    res = []; t0 = time.time(); nhit = 0
    for j, it in enumerate(items):
        vpath = os.path.join(V.base.VIDEO_DIR, it["video"])
        try:
            frames, times, dur, sfps = V.native_decode(vpath, args)
        except Exception as exc:
            res.append({"qid": it["qid"], "video": it["video"], "decode_error": f"{type(exc).__name__}: {exc}"}); continue
        T = frames.shape[0]
        estar = np.asarray([i for i, t in enumerate(times) if it["gs"] - 1e-6 <= t <= it["ge"] + 1e-6], dtype=int)
        rec = {"qid": it["qid"], "video": it["video"], "n_estar": int(len(estar)), "nframes": int(T), "fps": float(sfps),
               "style": args.prompt_style, "target_mode": "gt", "sentinel": "none", "dilate_k": 0, "prov": PROV}
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
        pcap, pkind, psrc = pick_partner_map(it, pmap, args.partner_mode)  # 14Lpn: paraphrase of the SAME interval; touches no rng
        has_p = pcap is not None; rec["has_partner"] = int(has_p)
        rec["estar_idx"] = [int(x) for x in estar]
        ej = np.asarray([], dtype=int)
        lp_c2_sum = None
        if has_p:
            ej = np.asarray(estar, dtype=int)  # same interval -> E_j IS E*_i (see file header)
            rec["partner_qid"] = it["qid"]; rec["partner_caption"] = pcap; rec["partner_gs"] = it["gs"]; rec["partner_ge"] = it["ge"]
            rec["partner_kind"] = pkind; rec["partner_src_qid"] = psrc
            rec["partner_n_estar"] = int(len(ej)); rec["ej_idx"] = [int(x) for x in ej]
            prompt_q2 = V.NATIVE_PROMPTS[args.prompt_style].format(pcap); cache2 = {}

            def score2(fr):
                with torch.no_grad():
                    s2, n2 = V.native_completion_logprob(model, proc, fr, sfps, prompt_q2, tgt_txt, model.device, cache2)
                return float(s2), int(n2)

            lp_c2_sum, tgt_len2 = score2(frames)
            assert tgt_len2 == tgt_len, "q2 target length changed; target not fixed"
            rec["lp_clean_q2_sum"] = lp_c2_sum; rec["lp_clean_q2_mean"] = lp_c2_sum / max(1, tgt_len)
        excl = set(int(x) for x in estar) | set(int(x) for x in ej)  # r3: the pool excludes BOTH evidence sets (C9 §5)
        pool = [i for i in range(T) if i not in excl]
        k_rand = min(len(pool), int(len(estar)))
        rec["pool_size"] = int(len(pool)); rec["pool_short"] = int(len(pool) < int(len(estar)))
        for op in ops:
            drng = random.Random(f"{args.seed}:{it['qid']}:{op}:donor")  # r3: per-(qid,op) stream -> row-wise reproducible across arms
            dvid = drng.choice([v for v in all_videos if v != it["video"]])
            dfr, _, _, _ = V.native_decode(os.path.join(V.base.VIDEO_DIR, dvid), args)
            if dfr.shape[1:3] != frames.shape[1:3]:
                import cv2
                dfr = np.stack([cv2.resize(np.ascontiguousarray(f), (frames.shape[2], frames.shape[1])) for f in dfr])
            n_need = max(int(len(estar)), int(len(ej)), int(k_rand))
            didx = sorted(drng.sample(range(dfr.shape[0]), min(dfr.shape[0], n_need)))
            distractor = dfr[didx]  # same donor frames for E*_i, E_j and the placebo -> the swap axis stays isomorphic
            fe = apply_op(frames, estar, op, distractor)
            lp_ei = score(fe)[0]
            fj = None; lp_ej = None; lp_ei2 = None; lp_ej2 = None
            if has_p:
                fj = apply_op(frames, ej, op, distractor)
                lp_ej = score(fj)[0]
            rand_sums = []; rand_idx = []; fr_list = []
            for rdraw in range(args.R):
                if k_rand <= 0:
                    break
                ridx = draw_placebo(pool, k_rand, estar, ej, f"{args.seed}:{it['qid']}:{op}:{rdraw}")
                rand_idx.append(ridx)
                f2 = apply_op(frames, np.asarray(ridx, int), op, distractor); fr_list.append(f2)
                rand_sums.append(score(f2)[0])
            rand2 = []
            if has_p:  # same edited frames, partner query, same target y_i
                lp_ei2 = score2(fe)[0]; lp_ej2 = score2(fj)[0]; rand2 = [score2(f2)[0] for f2 in fr_list]
            build_op_fields(rec, op, tgt_len, lp_c_sum, lp_c2_sum, lp_ei, lp_ej, rand_sums, lp_ei2, lp_ej2, rand2)
            rec[f"{op}_rand_idx"] = rand_idx; rec[f"{op}_donor"] = dvid; rec[f"{op}_donor_idx"] = didx
        res.append(rec)
        if (j + 1) % 25 == 0:
            print(f"  [14Ln] {j+1}/{len(items)} hits={nhit} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False)
    meta = {"instrument": INSTRUMENT, "trainer_sha16": TRAINER_SHA16, "trainer_file": TRAINER_FILE, "probe_sha16": PROBE_SHA16,
            "prov": PROV, "args": vars(args), "n_rows": len(res), "n_hits": nhit,
            "n_pool_short": sum(1 for r in res if r.get("pool_short")), "elapsed_s": round(time.time() - t0, 1),
            "partner_mode": args.partner_mode, "partner_map_sha16": pmap_sha16,
            "n_partner": sum(1 for r in res if r.get("has_partner"))}
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=1)
    print(f"[14Lpn] DONE n={len(res)} hits={nhit} partners={sum(1 for r in res if r.get(chr(104)+'as_partner'))} elapsed={time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
