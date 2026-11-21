# r3: r2 e325d1d4f20ad887 + --swap-frac dose-graded swap (nested prefix subsets, zero extra rng)
#   + --gate-against self-verifying gate (rho=1.0 must reproduce an r2 six-op product, hit+pred exactly) + --ops validation
#   + row/meta prov + --image-patch-size + resume completeness check.  Defaults = r2 behaviour.
# text-first variant: source 5d0de96818c8231a + --text-first (message order text->video; trainer r3 729232059d642960 scoring path follows the same flag). Flag off = source behaviour.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""12p5n_native_perturb_eval.py — BEHAVIORAL evidence test under the native protocol.

The fixed-answer audit (14Ln) measures the log-likelihood of the GT answer; this instrument measures what the model
actually PREDICTS when evidence frames are removed at test time. Same decode / E* / ops as 14Ln r2 (b33f38c0ec599a3a):
  frames, times = trainer.native_decode (fps 2, total_pixels 3584·28·28, smart-resized), E* = frames with t in [gs, ge].
Conditions (--ops, one greedy generation each):
  none       clean video (= 12p4n accuracy)
  mask_evid  zero all E* frames                       (evidence removed)
  mask_rand  zero |E*| random non-evidence frames      (placebo, same number of frames)
  swap_evid  replace E* frames by frames of another video
  swap_rand  replace |E*| random non-evidence frames the same way
  black      zero every frame (= 12p4n r4 --video-op black)
Randomness is seeded PER QUESTION (seed + qid), so rows are identical across shardings and machines given the same decode.
Rows (one per question): qid, video, caption, dur, gsA, geA, n_estar, nframes, fps, style, degenerate,
  {op}_pred, {op}_parsed, {op}_iou, {op}_hit, {op}_raw (tail), {op}_idx (edited frame indices, rand ops), swap_donor.
Degenerate questions (E* empty or E* = all frames) get degenerate=1 and only the 'none'/'black' conditions.
Per-row partial dump to <out>.partial.jsonl with resume by qid (torn last line dropped). Reader: analysis/read_p5n_v1.py.
usage: 12p5n_native_perturb_eval.py --model M [--adapter A] --prompt-style timer1_default --ops none,mask_evid,mask_rand
         --n 3720 --offset 0 --gpu 0 --out OUT.json
"""
import argparse, os, sys, json, time, random, hashlib, importlib.util, re, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_T = None
for cand in (os.path.join(HERE, "27_papo_train_qcdro_v3seln_r3.py"), os.path.join(HERE, "..", "trainer_v3", "27_papo_train_qcdro_v3seln_r3.py")):
    if os.path.exists(cand):
        _T = cand; break
if _T is None:
    raise SystemExit("27_papo_train_qcdro_v3seln.py not found next to this file")
_spec = importlib.util.spec_from_file_location("v3seln", _T)
V = importlib.util.module_from_spec(_spec); sys.modules["v3seln"] = V; _spec.loader.exec_module(V)
TRAINER_SHA16 = hashlib.sha256(open(_T, "rb").read()).hexdigest()[:16]
SELF_SHA16 = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
TRAINER_TAKES_PATCH = "image_patch_size" in open(_T, encoding="utf-8").read()
INSTRUMENT = "12p5n_native_perturb_eval_v1_tf_dose"
ALL_OPS = ("none", "mask_evid", "mask_rand", "swap_evid", "swap_rand", "black")

_NUM = re.compile(r"[-+]?\d*\.?\d+")


def parse_native(text, dur):
    """Identical to 14Ln r2 / 12p4n parse_native."""
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


def edit_frames(frames, idx, kind, distractor=None):
    """kind mask -> zero the frames at idx; kind swap -> replace them by distractor frames (14Ln apply_op semantics)."""
    f = frames.copy(); e = np.asarray(idx, int)
    if kind == "mask":
        f[e] = 0
    elif kind == "swap":
        m = min(len(e), len(distractor)); f[e[:m]] = distractor[:m]
    else:
        raise ValueError(kind)
    return f


def pick_indices(op, estar, nonev, rng):
    """Frame indices edited by op: *_evid -> E*, *_rand -> |E*| frames drawn from non-evidence (per-question rng)."""
    if op.endswith("_evid"):
        return [int(x) for x in estar]
    if op.endswith("_rand"):
        return sorted(int(x) for x in rng.sample(nonev, min(len(nonev), len(estar))))
    raise ValueError(op)


def rho_tag(rho):
    """0.25 -> 'r25', 1.0 -> 'r100' (round-half-up on the percent)."""
    return "r%d" % int(math.floor(float(rho) * 100 + 0.5))


def parse_swap_frac(spec):
    """'0.25,0.5,0.75,1.0' -> [0.25, 0.5, 0.75, 1.0].  Empty -> [] (= r2 behaviour)."""
    out = []
    for tok in [t.strip() for t in str(spec or "").split(",") if t.strip()]:
        try:
            v = float(tok)
        except ValueError:
            raise ValueError("bad --swap-frac token %r" % tok)
        if not (0.0 < v <= 1.0):
            raise ValueError("--swap-frac must be in (0, 1], got %r" % tok)
        out.append(v)
    tags = [rho_tag(v) for v in out]
    if len(set(tags)) != len(tags):
        raise ValueError("duplicate --swap-frac condition tags %s" % tags)
    return out


def validate_ops(ops):
    """Unknown / duplicated op names are an error (r2 only checked after the model import)."""
    for o in ops:
        if o not in ALL_OPS:
            raise ValueError("unknown op %r; choose from %s" % (o, list(ALL_OPS)))
    if len(set(ops)) != len(ops):
        raise ValueError("duplicate op in %s" % list(ops))
    return list(ops)


def dose_indices(full_idx, rho):
    """NESTED prefix subset of the rho=1.0 index set: the first ceil(rho*|full|) entries of the r2 index
    order (ascending frame index).  Nesting holds by construction: rho1 <= rho2 implies dose_indices(.,rho1)
    is a prefix of dose_indices(.,rho2); rho=1.0 returns the whole set unchanged."""
    n = len(full_idx)
    k = max(0, min(n, int(math.ceil(float(rho) * n - 1e-9))))
    return [int(x) for x in full_idx[:k]]


def dose_conditions(fracs):
    """[(condition name, base r2 op, rho)] - evid family first, then rand family."""
    return ([("swap_evid_" + rho_tag(v), "swap_evid", v) for v in fracs] +
            [("swap_rand_" + rho_tag(v), "swap_rand", v) for v in fracs])


def _gate_core(ref, rows, fracs, label, rsha):
    """Self-verifying gate: the rho=1.0 dose conditions actually produced by THIS run must reproduce, question
    by question, swap_evid / swap_rand of the reference.  hit and pred must be EXACTLY equal.
    Raw generated text is counted and reported but NOT enforced: it is machine dependent (verify_journal C3
    section 7: node5 vs node4 agree on raw for only 287/1000 rows) while hit/pred is what the reader uses."""
    if not any(abs(float(v) - 1.0) < 1e-9 for v in fracs):
        print("[gate] FAIL: the gate needs rho=1.0 in --swap-frac (nothing to compare)", flush=True)
        return False
    pairs = [("swap_evid_r100", "swap_evid"), ("swap_rand_r100", "swap_rand")]
    n_cmp = 0
    n_raw = 0
    first = None
    for r in rows:
        q = int(r["qid"])
        if q not in ref:
            continue
        g = ref[q]
        for new, old in pairs:
            has_new = (new + "_hit") in r
            has_old = (old + "_hit") in g
            if not has_new and not has_old:
                continue                      # both absent (degenerate / decode error): nothing to compare
            n_cmp += 1
            if has_new != has_old:            # absence is a value
                first = first or ("qid %d %s: present=%s but reference %s present=%s" % (q, new, has_new, old, has_old))
                continue
            a_hit = int(r[new + "_hit"])
            b_hit = int(g[old + "_hit"])
            a_pred = list(r[new + "_pred"]) if r.get(new + "_pred") is not None else None
            b_pred = list(g[old + "_pred"]) if g.get(old + "_pred") is not None else None
            if a_hit != b_hit or a_pred != b_pred:
                first = first or ("qid %d %s: hit %s vs %s, pred %s vs %s" % (q, new, a_hit, b_hit, a_pred, b_pred))
            if r.get(new + "_raw") != g.get(old + "_raw"):
                n_raw += 1
    if n_cmp == 0:
        print("[gate] FAIL: 0 comparable (qid, condition) pairs against %s sha16=%s -- 'nothing read' is not 'zero'"
              % (label, rsha), flush=True)
        return False
    if first:
        print("[gate] FAIL after %d comparisons vs %s sha16=%s; first difference: %s"
              % (n_cmp, label, rsha, first), flush=True)
        return False
    print("[gate] PASS: %d (qid, condition) pairs exactly equal in hit+pred vs %s sha16=%s; raw text differs on %d (not enforced)"
          % (n_cmp, label, rsha, n_raw), flush=True)
    return True


def gate_against(rep_path, rows, fracs):
    """rho=1.0 must reproduce an INDEPENDENT r2 six-op product of the same arm on the same machine."""
    ref = {int(r["qid"]): r for r in json.load(open(rep_path, encoding="utf-8"))}
    return _gate_core(ref, rows, fracs, os.path.basename(rep_path), hashlib.sha256(open(rep_path, "rb").read()).hexdigest()[:16])


def gate_self(rows, fracs):
    """Same-run gate for machines with no same-arm r2 six-op product: swap_*_r100 must equal swap_* of the SAME
    row, i.e. the dose block consumed no rng and used the r2 index set at rho=1.0.  Weaker than --gate-against
    (one process, one model load) but it is a real landing gate: it fails if the dose wiring drifts."""
    return _gate_core({int(r["qid"]): r for r in rows}, rows, fracs, "SELF(same run swap_evid/swap_rand)", "n/a")


def _partial_path(out): return out + ".partial.jsonl"


def load_partial(out):
    p = _partial_path(out); done = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: r = json.loads(line)
                except Exception: continue  # torn last line
                done[int(r["qid"])] = r
    return done


def selftest():
    ok = True
    def chk(c, m):
        nonlocal ok; ok &= bool(c); print(("  PASS " if c else "  FAIL ") + m)
    fr = np.arange(5 * 2 * 2 * 3, dtype=np.uint8).reshape(5, 2, 2, 3) + 1
    e = np.array([1, 2]); nonev = [0, 3, 4]
    m = edit_frames(fr, e, "mask"); chk((m[[1, 2]] == 0).all() and (m[[0, 3, 4]] == fr[[0, 3, 4]]).all() and (fr[[1, 2]] != 0).all(), "mask zeroes exactly the given frames, input untouched")
    d = np.full((3, 2, 2, 3), 9, np.uint8); s = edit_frames(fr, e, "swap", d); chk((s[[1, 2]] == 9).all() and (s[[0, 3, 4]] == fr[[0, 3, 4]]).all(), "swap replaces exactly the given frames")
    r1 = pick_indices("mask_rand", e, nonev, random.Random(1000 + 7)); r2 = pick_indices("mask_rand", e, nonev, random.Random(1000 + 7)); r3 = pick_indices("mask_rand", e, nonev, random.Random(1000 + 8))
    chk(r1 == r2 and len(r1) == 2 and set(r1) <= set(nonev), "rand draw is per-question deterministic, |E*| frames, outside E*")
    chk(r1 != r3 or True, "different qid -> independent draw (not asserted unequal)")
    chk(pick_indices("mask_evid", e, nonev, random.Random(0)) == [1, 2], "evid op edits E*")
    chk(parse_native("The event happens from 3.5 to 8.0 seconds.", 10.0) == (3.5, 8.0) and parse_native("There are none.", 10.0) is None, "parse_native: window / refusal")
    import tempfile
    dd = tempfile.mkdtemp(); out = os.path.join(dd, "x.json")
    open(_partial_path(out), "w").write(json.dumps({"qid": 3}) + "\n" + '{"qid": 5, "no')
    chk(sorted(load_partial(out)) == [3], "partial resume drops torn line")
    # --- r3: dose-graded swap ---------------------------------------------------------------
    chk(parse_swap_frac("") == [] and parse_swap_frac("0.25,0.5,0.75,1.0") == [0.25, 0.5, 0.75, 1.0], "parse_swap_frac: empty = r2 behaviour, list parsed")
    for bad in ("0", "0.0", "1.5", "-0.5", "0.5,0.5", "abc"):
        try:
            parse_swap_frac(bad); chk(False, "parse_swap_frac rejects %r" % bad)
        except ValueError:
            chk(True, "parse_swap_frac rejects %r" % bad)
    chk([rho_tag(v) for v in (0.25, 0.5, 0.75, 1.0)] == ["r25", "r50", "r75", "r100"], "rho tags r25/r50/r75/r100")
    full = list(range(10))
    dsets = [dose_indices(full, v) for v in (0.25, 0.5, 0.75, 1.0)]
    chk([len(x) for x in dsets] == [3, 5, 8, 10], "ceil(rho*|E*|) sizes 3/5/8/10 for |E*|=10")
    chk(all(dsets[i] == dsets[i + 1][:len(dsets[i])] for i in range(len(dsets) - 1)), "dose sets are NESTED prefixes (rho1<=rho2 -> prefix)")
    chk(dsets[-1] == full, "rho=1.0 returns the r2 index set unchanged")
    chk(all(len(dose_indices(list(range(n)), 1.0)) == n and len(dose_indices(list(range(n)), 0.25)) >= 1 for n in range(1, 40)), "rho=1.0 total, rho>0 never empty, |E*|=1..39")
    # rho=1.0 index sets are bit-identical to what r2's own code path produces, with the same rng stream
    est = np.array([2, 3, 4, 7]); nev = [0, 1, 5, 6, 8, 9]
    g1 = random.Random(1000 + 7); g2 = random.Random(1000 + 7)
    r2_evid = pick_indices("swap_evid", est, nev, g1); r2_rand = pick_indices("swap_rand", est, nev, g1)
    chk(dose_indices(pick_indices("swap_evid", est, nev, g2), 1.0) == r2_evid, "swap_evid_r100 index set == r2 swap_evid index set")
    chk(dose_indices(pick_indices("swap_rand", est, nev, g2), 1.0) == r2_rand, "swap_rand_r100 index set == r2 swap_rand index set (same rng stream)")
    chk([c[0] for c in dose_conditions([0.25, 1.0])] == ["swap_evid_r25", "swap_evid_r100", "swap_rand_r25", "swap_rand_r100"], "condition names")
    # gate: positive (identical run) then the two counterexamples
    import tempfile as _tf
    gd = _tf.mkdtemp(); refp = os.path.join(gd, "ref_r2.json")
    ref_rows = [{"qid": q, "video": "v%d" % (q // 2), "swap_evid_hit": q % 2, "swap_evid_pred": [1.0 * q, 2.0 + q], "swap_evid_raw": "t%d" % q,
                 "swap_rand_hit": 1 - q % 2, "swap_rand_pred": [3.0, 4.0], "swap_rand_raw": "u%d" % q} for q in range(6)]
    json.dump(ref_rows, open(refp, "w"))
    same = [{"qid": r["qid"], "swap_evid_r100_hit": r["swap_evid_hit"], "swap_evid_r100_pred": r["swap_evid_pred"], "swap_evid_r100_raw": r["swap_evid_raw"],
             "swap_rand_r100_hit": r["swap_rand_hit"], "swap_rand_r100_pred": r["swap_rand_pred"], "swap_rand_r100_raw": r["swap_rand_raw"]} for r in ref_rows]
    chk(gate_against(refp, same, [0.25, 1.0]) is True, "gate PASSES when rho=1.0 reproduces the r2 product")
    chk(gate_against(refp, [], [1.0]) is False, "gate FAILS on 0 comparable pairs ('nothing read' is not 'zero')")
    chk(gate_against(refp, same, [0.25, 0.5]) is False, "gate FAILS when rho=1.0 was not run")
    # counterexample A: nested prefix replaced by an INDEPENDENT redraw -> non-nested, and the gate catches the outcome change
    ind_rng = random.Random(0)
    ind = [sorted(ind_rng.sample(full, k)) for k in (3, 5, 8, 10)]
    chk(not all(ind[i] == ind[i + 1][:len(ind[i])] for i in range(len(ind) - 1)), "counterexample: independent redraw is NOT nested (prefix property has teeth)")
    bad_rows = [dict(r) for r in same]; bad_rows[2]["swap_evid_r100_hit"] = 1 - bad_rows[2]["swap_evid_r100_hit"]
    chk(gate_against(refp, bad_rows, [1.0]) is False, "counterexample: one flipped hit at rho=1.0 -> gate FAILS")
    bad2 = [dict(r) for r in same]; bad2[1]["swap_rand_r100_pred"] = [9.0, 9.5]
    chk(gate_against(refp, bad2, [1.0]) is False, "counterexample: one changed pred at rho=1.0 -> gate FAILS")
    bad3 = [dict(r) for r in same]; del bad3[4]["swap_evid_r100_hit"]
    chk(gate_against(refp, bad3, [1.0]) is False, "counterexample: a missing rho=1.0 condition -> gate FAILS (absence is a value)")
    # same-run self gate
    selfrows = [dict(r, **{"swap_evid_hit": r["swap_evid_r100_hit"], "swap_evid_pred": r["swap_evid_r100_pred"], "swap_evid_raw": r["swap_evid_r100_raw"],
                           "swap_rand_hit": r["swap_rand_r100_hit"], "swap_rand_pred": r["swap_rand_r100_pred"], "swap_rand_raw": r["swap_rand_r100_raw"]}) for r in same]
    chk(gate_self(selfrows, [0.5, 1.0]) is True, "gate_self PASSES when rho=1.0 equals the same run's swap_evid/swap_rand")
    sbad = [dict(r) for r in selfrows]; sbad[3]["swap_rand_r100_hit"] = 1 - sbad[3]["swap_rand_r100_hit"]
    chk(gate_self(sbad, [1.0]) is False, "counterexample: rho=1.0 diverging from the same run's swap_* -> gate_self FAILS")
    chk(gate_self([], [1.0]) is False, "gate_self FAILS on 0 comparable pairs")
    # counterexample B: unknown / duplicated op name must raise
    chk(validate_ops(["none", "swap_evid"]) == ["none", "swap_evid"], "validate_ops accepts known ops")
    for bad in (["none", "mask_evd"], ["swap_ev"], ["none", "none"]):
        try:
            validate_ops(bad); chk(False, "counterexample: validate_ops rejects %s" % bad)
        except ValueError:
            chk(True, "counterexample: validate_ops rejects %s" % bad)
    print("[selftest]", "ALL PASS" if ok else "FAIL"); return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--adapter", default=None)
    ap.add_argument("--anno", default=None); ap.add_argument("--n", type=int, default=None); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--ops", default="none,mask_evid,mask_rand"); ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompt-style", choices=sorted(V.NATIVE_PROMPTS), default="timer1_default")
    ap.add_argument("--fps", type=float, default=2.0); ap.add_argument("--total-pixels", type=int, default=3584 * 28 * 28)
    ap.add_argument("--min-pixels", type=int, default=16 * 28 * 28); ap.add_argument("--max-frames", type=int, default=768)
    ap.add_argument("--seed", type=int, default=1000); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true");
    ap.add_argument("--swap-frac", default="", help="comma list of swap doses in (0,1], e.g. 0.25,0.5,0.75,1.0; adds swap_evid_rNN / swap_rand_rNN (nested prefix subsets of the rho=1.0 set). Empty = r2 behaviour")
    ap.add_argument("--gate-against", default=None, help="r2 six-op product: after the run, swap_*_r100 must equal its swap_* question by question (hit+pred) or exit 3")
    ap.add_argument("--gate-self", action="store_true", help="after the run, swap_*_r100 must equal the swap_evid/swap_rand of THIS same run, question by question, or exit 3")
    ap.add_argument("--image-patch-size", type=int, default=14, help="14 Qwen2/2.5-VL (default), 16 Qwen3-VL; prov + pass-through to native_decode")
    ap.add_argument("--text-first", action="store_true", help="text part before the video part (TimeZero/VideoChat-R1 order)")
    args = ap.parse_args()
    V._TEXT_FIRST["on"] = bool(args.text_first)
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.model or not args.out:
        ap.error("--model and --out required")
    try:
        ops = validate_ops([o for o in args.ops.split(",") if o])
        fracs = parse_swap_frac(args.swap_frac)
    except ValueError as exc:
        ap.error(str(exc))
    if fracs and not ("swap_evid" in ops and "swap_rand" in ops):
        ap.error("--swap-frac needs both swap_evid and swap_rand in --ops: the donor cache, the rng stream and the rho=1.0 self-gate all come from them")
    if (args.gate_against or args.gate_self) and not any(abs(float(v) - 1.0) < 1e-9 for v in fracs):
        ap.error("--gate-against / --gate-self need --swap-frac containing 1.0")
    if args.gate_against and not os.path.exists(args.gate_against):
        ap.error("--gate-against file not found: " + args.gate_against)
    if int(args.image_patch_size) != 14 and not TRAINER_TAKES_PATCH:
        ap.error("--image-patch-size %d requested but the imported trainer (%s) does not take it; it would be silently ignored" % (int(args.image_patch_size), os.path.basename(_T)))
    dose = dose_conditions(fracs)
    conds = list(ops) + [c[0] for c in dose]
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
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
    done = load_partial(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if done: print(f"[p5n] resume: {len(done)} rows from {_partial_path(args.out)}", flush=True)
    pf = open(_partial_path(args.out), "a", encoding="utf-8")
    print(f"[p5n] n={len(items)} offset={args.offset} ops={ops} style={args.prompt_style} fps={args.fps} trainer_sha16={TRAINER_SHA16} model={os.path.basename(args.model)} adapter={args.adapter}", flush=True)

    def generate(frames, sfps, prompt_text):
        vid = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
        _vp = {"type": "video", "video": "native_placeholder"}; _tp = {"type": "text", "text": prompt_text}; msgs = [{"role": "user", "content": ([_tp, _vp] if args.text_first else [_vp, _tp])}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = V._native_inputs(proc, text, vid, sfps).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        return proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    res = []; t0 = time.time(); nh = {c: 0 for c in conds}
    for j, it in enumerate(items):
        if int(it["qid"]) in done:
            r = done[int(it["qid"])]
            want = [] if "decode_error" in r else [c for c in conds if not (r.get("degenerate") and c not in ("none", "black"))]
            miss = [c for c in want if f"{c}_hit" not in r]
            if not miss:
                res.append(r)
                for c in conds: nh[c] += int(r.get(f"{c}_hit", 0) or 0)
                continue
            print(f"[p5n] qid {r['qid']}: partial row missing {miss} -> recompute", flush=True)
        vpath = os.path.join(V.base.VIDEO_DIR, it["video"])
        rec = {"qid": it["qid"], "video": it["video"], "caption": it["caption"], "gsA": it["gs"], "geA": it["ge"], "style": args.prompt_style, "instrument": INSTRUMENT,
               "instrument_sha16": SELF_SHA16, "trainer_sha16": TRAINER_SHA16, "prompt_style": args.prompt_style, "text_first": bool(args.text_first),
               "adapter": args.adapter, "seed": int(args.seed), "ops": ",".join(ops), "swap_frac": [float(v) for v in fracs], "image_patch_size": int(args.image_patch_size)}
        try:
            frames, times, dur, sfps = V.native_decode(vpath, args)
        except Exception as exc:
            rec["decode_error"] = f"{type(exc).__name__}: {exc}"; res.append(rec); pf.write(json.dumps(rec, ensure_ascii=False) + "\n"); pf.flush(); continue
        T = int(frames.shape[0])
        estar = np.asarray([i for i, t in enumerate(times) if it["gs"] - 1e-6 <= t <= it["ge"] + 1e-6], dtype=int)
        nonev = [i for i in range(T) if i not in set(estar.tolist())]
        degenerate = int(len(estar) == 0 or len(estar) >= T)
        rec.update({"dur": float(dur), "n_estar": int(len(estar)), "nframes": T, "fps": float(sfps), "degenerate": degenerate})
        prompt_text = V.NATIVE_PROMPTS[args.prompt_style].format(it["caption"])
        rng = random.Random(args.seed + int(it["qid"]))
        distractor = None
        for op in ops:
            if degenerate and op not in ("none", "black"):
                continue
            if op == "none":
                fr = frames
            elif op == "black":
                fr = np.zeros_like(frames)
            else:
                kind = op.split("_")[0]
                idx = pick_indices(op, estar, nonev, rng)
                if kind == "swap":
                    if distractor is None:
                        dvid = rng.choice([v for v in all_videos if v != it["video"]])
                        dfr, _, _, _ = V.native_decode(os.path.join(V.base.VIDEO_DIR, dvid), args)
                        if dfr.shape[1:3] != frames.shape[1:3]:
                            import cv2
                            dfr = np.stack([cv2.resize(np.ascontiguousarray(f), (frames.shape[2], frames.shape[1])) for f in dfr])
                        didx = rng.sample(range(dfr.shape[0]), min(dfr.shape[0], len(estar)))
                        distractor = dfr[didx]; rec["swap_donor"] = dvid
                    fr = edit_frames(frames, idx, "swap", distractor)
                else:
                    fr = edit_frames(frames, idx, "mask")
                rec[f"{op}_idx"] = idx
            txt = generate(fr, sfps, prompt_text)
            pw = parse_native(txt, dur)
            iou = iou_1d(pw, (it["gs"], it["ge"])) if pw else 0.0
            rec[f"{op}_pred"] = pw; rec[f"{op}_parsed"] = int(pw is not None); rec[f"{op}_iou"] = float(iou); rec[f"{op}_hit"] = int(iou >= args.thr); rec[f"{op}_raw"] = txt[-120:]
            nh[op] += rec[f"{op}_hit"]
        if dose and not degenerate:
            for cname, bop, rho in dose:
                bidx = rec.get(bop + "_idx")
                if bidx is None:
                    raise SystemExit("[p5n-r3] qid %d: %s_idx missing - the r2 swap ops must run before the dose conditions" % (int(it["qid"]), bop))
                idx = dose_indices(bidx, rho)
                rec[cname + "_idx"] = idx
                txt = generate(edit_frames(frames, idx, "swap", distractor), sfps, prompt_text)
                pw = parse_native(txt, dur)
                iou = iou_1d(pw, (it["gs"], it["ge"])) if pw else 0.0
                rec[cname + "_pred"] = pw; rec[cname + "_parsed"] = int(pw is not None); rec[cname + "_iou"] = float(iou); rec[cname + "_hit"] = int(iou >= args.thr); rec[cname + "_raw"] = txt[-120:]
                nh[cname] += rec[cname + "_hit"]
        res.append(rec); pf.write(json.dumps(rec, ensure_ascii=False) + "\n"); pf.flush()
        if (j + 1) % 25 == 0:
            print(f"  [p5n] {j+1}/{len(items)} hits={{{', '.join(f'{o}:{nh[o]}' for o in ops)}}} ({time.time()-t0:.0f}s)", flush=True)
    pf.close()
    assert len(res) == len(items) and [r["qid"] for r in res] == [it["qid"] for it in items], "row/qid order mismatch"
    json.dump(res, open(args.out, "w"), ensure_ascii=False)
    meta = {"instrument": INSTRUMENT, "trainer_sha16": TRAINER_SHA16, "args": vars(args), "n_rows": len(res), "hits": nh, "elapsed_s": round(time.time() - t0, 1),
            "prov": {"instrument": INSTRUMENT, "instrument_sha16": SELF_SHA16, "trainer_sha16": TRAINER_SHA16, "model": args.model, "adapter": args.adapter,
                     "prompt_style": args.prompt_style, "text_first": bool(args.text_first), "seed": int(args.seed), "ops": ops, "conditions": conds,
                     "swap_frac": [float(v) for v in fracs], "image_patch_size": int(args.image_patch_size), "n": args.n, "offset": args.offset,
                     "thr": args.thr, "fps": args.fps, "max_new_tokens": args.max_new_tokens}}
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=1)
    print(f"[p5n] DONE n={len(res)} hits={nh} elapsed={time.time()-t0:.0f}s -> {args.out}", flush=True)
    if args.gate_self or args.gate_against:
        g_ok = True; meta["gate"] = {}
        if args.gate_self:
            s_ok = gate_self(res, fracs); g_ok = g_ok and s_ok; meta["gate"]["self"] = bool(s_ok)
        if args.gate_against:
            a_ok = gate_against(args.gate_against, res, fracs); g_ok = g_ok and a_ok
            meta["gate"]["against"] = os.path.basename(args.gate_against)
            meta["gate"]["against_sha16"] = hashlib.sha256(open(args.gate_against, "rb").read()).hexdigest()[:16]
            meta["gate"]["against_pass"] = bool(a_ok)
        meta["gate"]["pass"] = bool(g_ok)
        json.dump(meta, open(args.out + ".meta.json", "w"), indent=1)
        if not g_ok:
            sys.exit(3)


if __name__ == "__main__":
    main()
