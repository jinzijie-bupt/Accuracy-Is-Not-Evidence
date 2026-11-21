#!/usr/bin/env python3
"""
18_papo_train.py — PAPO: counterfactual contrastive LoRA training (the actual method contribution).

Difference from 16 (IoU-DPO) is the contrast axis:
  16: same context (frames), chosen=high-IoU window vs rejected=low-IoU window   -> learns "which window is accurate" (contains IoU signal, i.e. a proxy)
  18: same target window w, chosen=w|clean frames vs rejected=w|evidence-destroyed frames   -> learns "the answer must causally depend on evidence" (zero IoU signal)

loss = -logσ( β·[ (logp(w|clean)-logp_ref(w|clean)) - (logp(w|corrupt)-logp_ref(w|corrupt)) ] )
  ref = same model with adapter disabled (peft disable_adapter, no second copy in memory); 4 forward passes/sample, same cost as DPO.
  margin>0 means "prefers w with clean frames, not when evidence is destroyed" = faithful. Directly aligned with the headline metrics (the audit/ΔFlip).
  --alpha>0 adds a clean-side SFT (NLL of w) term = PAPO+accuracy arm; --alpha 0 = pure faithfulness arm.

corrupt frames are rebuilt on the fly from the pair estar + distractor (times unchanged, only E* pixels replaced, same as 14_causal_probe).
Usage: python 18_papo_train.py --pairs papo_pairs.jsonl --alpha 0 --out adapter_PAPO
"""
import argparse, os, json, csv, re, importlib.util, random, time
import numpy as np
import torch, torch.nn.functional as F

_P = os.path.join(os.path.dirname(__file__), "12_charades_grounding.py")
_spec = importlib.util.spec_from_file_location("g12", _P); g12 = importlib.util.module_from_spec(_spec)
import sys as _sys; _sys.modules["g12"] = g12; _spec.loader.exec_module(g12)
decode_video, build_prompt, MODEL, ROOT = g12.decode_video, g12.build_prompt, g12.MODEL, g12.ROOT
VIDEO_DIR = g12.VIDEO_DIR
ANNO = f"{ROOT}/data/charades_sta/anno"

_PN = os.path.join(os.path.dirname(__file__), "native_video_causal_probe.py")
_specn = importlib.util.spec_from_file_location("native_train_helpers", _PN)
nvid = importlib.util.module_from_spec(_specn)
_sys.modules["native_train_helpers"] = nvid
_specn.loader.exec_module(nvid)


def completion_logprob_n(model, proc, frames, times, dur, caption, completion, device):
    """Return (sum of completion token log-likelihoods, token count). Only the completion span is scored."""
    user_text, imgs = build_prompt(proc, frames, times, dur, caption)
    full = proc(text=[user_text + completion], images=imgs, videos=None, return_tensors="pt").to(device)
    plen = proc(text=[user_text], images=imgs, videos=None, return_tensors="pt").input_ids.shape[1]
    out = model(**full)
    logits = out.logits[:, :-1, :].float()
    labels = full.input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    seg = logp[:, plen - 1:]
    return seg.sum(), seg.shape[1]


def completion_logprob_native(
    model,
    proc,
    frames,
    times,
    dur,
    caption,
    completion,
    device,
    resized=336,
    prompt_kind="official_hms",
    sample_fps=2.0,
):
    """Completion logprob through the native `videos=` processor path."""
    text, image_inputs, video_inputs, video_kwargs = nvid.build_native_prompt(
        proc, frames, times, dur, caption, resized, prompt_kind, sample_fps
    )
    full = proc(
        text=[text + completion],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(device)
    prompt = proc(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    plen = prompt.input_ids.shape[1]
    out = model(**full)
    logits = out.logits[:, :-1, :].float()
    labels = full.input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    seg = logp[:, plen - 1:]
    return seg.sum(), seg.shape[1]


def corrupt_op(frames, idx, distractor_frames, op="swap"):
    """Corrupt frames at positions idx with op (positions/times unchanged). The cross-corruption generalization matrix uses 4 ops:
    swap=frames from another video (cross-video content) / mask=black out / blur=block pooling destroys detail (same frame) / tshift=content from another time in the same video (misaligned)."""
    f = frames.copy(); e = np.asarray(idx, int); K = len(frames)
    if op == "swap":
        m = min(len(e), len(distractor_frames)); f[e[:m]] = distractor_frames[:m]
    elif op == "mask":
        f[e] = 0
    elif op == "blur":
        for i in e:
            h, w = f[i].shape[:2]
            gh = gw = 16
            bh, bw = max(1, h // gh), max(1, w // gw)
            hh, ww = gh * bh, gw * bw
            crop = f[i][:hh, :ww]
            s = crop.reshape(gh, bh, gw, bw, 3).mean(axis=(1, 3)).astype(np.uint8)
            blurred = np.repeat(np.repeat(s, bh, 0), bw, 1)
            f[i][:hh, :ww] = blurred
    elif op == "tshift":
        for i in e:
            f[i] = frames[(int(i) + K // 2) % K]
    return f


def parse_actions(s):
    out = []
    for tok in (s or "").split(";"):
        parts = tok.strip().split()
        if len(parts) != 3:
            continue
        try:
            out.append((parts[0], float(parts[1]), float(parts[2])))
        except ValueError:
            pass
    return out


def interval_overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def load_action_index():
    rows, by_code = {}, {}
    for split in ["train", "test"]:
        path = f"{ANNO}/Charades_v1_{split}.csv"
        if not os.path.exists(path):
            continue
        for row in csv.DictReader(open(path)):
            vid = row["id"]
            acts = parse_actions(row.get("actions", ""))
            rows[vid] = acts
            for code, s, e in acts:
                video = f"{vid}.mp4"
                if os.path.exists(os.path.join(VIDEO_DIR, video)):
                    by_code.setdefault(code, []).append((video, s, e))
    return rows, by_code


def best_action_for_ex(ex, action_rows):
    vid = ex["video"].rsplit(".", 1)[0]
    gs, ge = float(ex["gs"]), float(ex["ge"])
    best = None
    for code, s, e in action_rows.get(vid, []):
        ov = interval_overlap((gs, ge), (s, e))
        if ov <= 0:
            continue
        score = ov / max(1e-6, min(ge - gs, e - s))
        cand = (score, ov, code)
        if best is None or cand > best:
            best = cand
    return best[2] if best else None


def same_action_corrupt(frames, idx, ex, nct_ctx, rng):
    if not nct_ctx:
        return None
    code = best_action_for_ex(ex, nct_ctx["action_rows"])
    if not code:
        return None
    pool = [x for x in nct_ctx["by_code"].get(code, []) if x[0] != ex["video"]]
    if not pool:
        return None
    donor_video, ds, de = rng.choice(pool)
    dframes, _, dtimes = decode_video(donor_video, ex["K"])
    e = np.asarray(idx, int)
    didx = np.where((dtimes >= ds) & (dtimes <= de))[0]
    if len(didx) == 0:
        didx = np.arange(len(dframes))
    if len(didx) == 1:
        pick = np.repeat(didx, len(e))
    else:
        pos = np.linspace(0, len(didx) - 1, len(e))
        pick = didx[np.round(pos).astype(int)]
    f = frames.copy()
    f[e] = dframes[pick]
    return f


def build_corrupt(frames, idx, distractor_frames, op="swap"):  # Backward-compatible alias
    return corrupt_op(frames, idx, distractor_frames, op)


def _rand_nonev(estar, K, rng):
    """Equal-count random **non-evidence** frame indices (for the N* pole)."""
    s = set(int(x) for x in estar)
    nonev = [i for i in range(K) if i not in s]
    return rng.sample(nonev, min(len(nonev), len(estar)))


def train_step(
    model,
    proc,
    ex,
    device,
    mode,
    beta,
    alpha,
    npole,
    rng,
    cop="swap",
    npole_sym=False,
    nct_ctx=None,
    nct_same_alpha=0.0,
    input_encoding="frame",
    native_frames=64,
    native_decode_size=336,
    native_resized=336,
    native_prompt="official_hms",
    native_sample_fps=2.0,
):
    """mode: papo (counterfactual contrast) / gtsft (pure GT-window SFT floor) / corruptaug (SFT on both clean+corrupt, no contrast)
    / mdpo (clean vs random-corruption contrast) / defacto (DeFacto-style three independent SFT terms, no paired margin; concurrent-work ablation)
    / nct (natural counterfactual training: swap+tshift destroy evidence, same-action as invariance term).
    npole (papo only): add the N* pole — destroying non-evidence frames should not lower w; blocks the collapse solution 'lower confidence under any perturbation' (blocker#4).
    npole_sym: symmetric N*-pole ablation — clean>N* same as clean>E* (removes E*/N* directionality), isolates 'whether directionality is load-bearing'.
    cop: corruption operator (swap/mask/blur/tshift), for the cross-corruption generalization matrix."""
    w, cap = ex["w_text"], ex["caption"]
    if input_encoding == "native":
        frames, dur, times = nvid.decode_video_size(ex["video"], native_frames, native_decode_size)
        estar = np.where((times >= float(ex["gs"])) & (times <= float(ex["ge"])))[0].astype(int)
        K = len(frames)
        if len(estar) == 0 or len(estar) >= K:
            raise ValueError(f"degenerate native evidence set: |E*|={len(estar)} K={K}")
        clp = lambda fr: completion_logprob_native(
            model,
            proc,
            fr,
            times,
            dur,
            cap,
            w,
            device,
            resized=native_resized,
            prompt_kind=native_prompt,
            sample_fps=native_sample_fps,
        )
    else:
        frames, dur, times = decode_video(ex["video"], ex["K"])
        estar = np.asarray(ex["estar"], int)
        K = len(frames)
        clp = lambda fr: completion_logprob_n(model, proc, fr, times, dur, cap, w, device)

    if mode == "gtsft":                       # Accuracy / window-prior floor (PAPO must clearly exceed it)
        pc, nc = clp(frames)
        l = -(pc / max(1, nc))
        return l, 0.0, float(-l)

    if input_encoding == "native":
        dfr, _, _ = nvid.decode_video_size(ex["distractor"], native_frames, native_decode_size)
    else:
        dfr, _, _ = decode_video(ex["distractor"], ex["K"])
    # Family-level intervention training: sample one operator per example and
    # reuse it for E* and N*, so evidence location remains the only contrast.
    cop_eff = rng.choice(("swap", "mask", "blur", "tshift")) if cop == "mixed" else cop
    corruptE = corrupt_op(frames, estar, dfr, cop_eff)

    if mode == "corruptaug":                  # Only 'look more at corrupted frames', no logσ contrast (isolates robustness ≠ contrast)
        pc, nc = clp(frames); pe, ne = clp(corruptE)
        l = -(pc / max(1, nc)) - (pe / max(1, ne))
        return l, 0.0, float(-l)

    if mode == "mdpo":                        # mDPO-video: clean vs random corruption (evidence-agnostic) contrastive DPO. Only difference from papo = corruption target E* -> random
        ne_ = len(estar)   # Corruption budget = |E*| matched, only 'position' changes
        ridx = rng.sample(range(K), min(ne_, K))
        corruptR = corrupt_op(frames, ridx, dfr, cop_eff)
        pc, nc = clp(frames); pr, _ = clp(corruptR)
        with torch.no_grad(), model.disable_adapter():
            rc, _ = clp(frames); rr, _ = clp(corruptR)
        margin = beta * ((pc - rc) - (pr - rr))        # clean should give w more than random corruption (generic mDPO objective, not evidence-specific)
        loss = -F.logsigmoid(margin)
        if alpha > 0:
            loss = loss + alpha * (-(pc / max(1, nc)))
        return loss, float((margin > 0).float()), float(margin)

    if mode == "defacto":                     # DeFacto-style (adapted to video TSG): three **independent** SFT scalars, no paired margin -> isolates "paired contrast vs independent rewards"
        ABSTAIN = "start=0.0 end=0.0"          # Counterfactual (E* destroyed) target = null window (abstain; DeFacto "no evidence, do not answer")
        ne_ = len(estar)
        ridx = rng.sample(range(K), min(ne_, K)); corruptR = corrupt_op(frames, ridx, dfr, cop_eff)
        pc, nc = clp(frames)                                          # Factual: clean -> w
        pr, nr = clp(corruptR)                                        # Random mask: still -> w (robustness term)
        pa, na = completion_logprob_n(model, proc, corruptE, times, dur, cap, ABSTAIN, device)  # Counterfactual: E* destroyed -> abstain
        l = -(pc / max(1, nc)) - (pr / max(1, nr)) - (pa / max(1, na))   # Three independent NLL terms, no logσ paired contrast
        return l, 0.0, float(-l)

    if mode == "nct":
        # NCT branch: keep one synthetic E* corruption (swap/mask/blur),
        # add natural temporal shift as E* corruption, and use same-action replacement
        # as the "should preserve w" invariant term. This targets the same-action
        # sensitivity exposed by SIGNATURE_MATRIX_PILOT while keeping the original loss shape.
        if input_encoding == "native":
            raise ValueError("nct mode is not yet supported for native input encoding")
        corruptT = corrupt_op(frames, estar, dfr, "tshift")
        corruptSA = same_action_corrupt(frames, estar, ex, nct_ctx, rng)
        pc, nc = clp(frames)
        pe, _ = clp(corruptE)
        pt, _ = clp(corruptT)
        with torch.no_grad(), model.disable_adapter():
            rc, _ = clp(frames)
            re_, _ = clp(corruptE)
            rt, _ = clp(corruptT)
        margin_e = beta * ((pc - rc) - (pe - re_))   # clean > synthetic E*
        margin_t = beta * ((pc - rc) - (pt - rt))    # clean > natural tshift E*
        loss = -F.logsigmoid(margin_e) - F.logsigmoid(margin_t)
        margins = [margin_e, margin_t]
        if corruptSA is not None:
            ps, ns = clp(corruptSA)
            with torch.no_grad(), model.disable_adapter():
                rs, _ = clp(corruptSA)
            margin_sa_e = beta * ((ps - rs) - (pe - re_))  # same-action should beat E*-destroyed
            margin_sa_t = beta * ((ps - rs) - (pt - rt))
            loss = loss - F.logsigmoid(margin_sa_e) - F.logsigmoid(margin_sa_t)
            margins += [margin_sa_e, margin_sa_t]
            if nct_same_alpha > 0:
                loss = loss + nct_same_alpha * (-(ps / max(1, ns)))  # parse guard: same-action must still emit w
        if npole:
            corruptN = corrupt_op(frames, _rand_nonev(estar, K, rng), dfr, cop_eff)
            pn, _ = clp(corruptN)
            with torch.no_grad(), model.disable_adapter():
                rn, _ = clp(corruptN)
            margin_n = beta * ((pn - rn) - (pe - re_))
            loss = loss - F.logsigmoid(margin_n)
            margins.append(margin_n)
        if alpha > 0:
            loss = loss + alpha * (-(pc / max(1, nc)))
        mstack = torch.stack(margins)
        return loss, float((mstack.detach() > 0).float().mean()), float(mstack.detach().mean())

    # mode == "papo": counterfactual contrast
    pc, nc = clp(frames); pe, _ = clp(corruptE)
    with torch.no_grad(), model.disable_adapter():
        rc, _ = clp(frames); re_, _ = clp(corruptE)
    margin = beta * ((pc - rc) - (pe - re_))
    loss = -F.logsigmoid(margin)
    macc = float((margin > 0).float())
    if npole:
        corruptN = corrupt_op(frames, _rand_nonev(estar, K, rng), dfr, cop_eff)
        pn, _ = clp(corruptN)
        with torch.no_grad(), model.disable_adapter():
            rn, _ = clp(corruptN)
        if npole_sym:
            margin2 = beta * ((pc - rc) - (pn - rn))   # Symmetric ablation: clean>N* same as clean>E*, removes E*/N* directionality
        else:
            margin2 = beta * ((pn - rn) - (pe - re_))   # True N* pole (asymmetric): N*-destroyed should keep w more than E*-destroyed => evidence specificity
        loss = loss - F.logsigmoid(margin2)
    if alpha > 0:
        loss = loss + alpha * (-(pc / max(1, nc)))   # Clean-side SFT, preserves accuracy
    return loss, macc, float(margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=f"{ROOT}/probe3_selector_evidence/papo_pairs.jsonl")
    ap.add_argument("--out", default=f"{ROOT}/probe3_selector_evidence/adapter_PAPO")
    ap.add_argument("--model", default=MODEL, help="base model path; defaults to local Qwen2.5-VL-7B")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.0, help=">0 adds clean-side SFT = PAPO+accuracy arm")
    ap.add_argument("--mode", default="papo", choices=["papo", "gtsft", "corruptaug", "mdpo", "defacto", "nct"],
                    help="papo=counterfactual contrast; gtsft=GT-window SFT floor; corruptaug=SFT on both clean+corrupt; mdpo=clean vs random-corruption contrast; defacto=DeFacto-style three independent SFT terms, no paired margin (concurrent-work baseline); nct=natural counterfactual training")
    ap.add_argument("--npole", action="store_true", help="papo plus the N* pole (blocks the collapse solution, blocker#4)")
    ap.add_argument("--npole-symmetric", action="store_true", help="symmetric N*-pole ablation: clean>N* same as clean>E* (no directionality), isolates whether directionality is load-bearing")
    ap.add_argument("--corrupt-op", default="swap", choices=["swap", "mask", "blur", "tshift", "mixed"],
                    help="training corruption; mixed samples swap/mask/blur/tshift per example")
    ap.add_argument("--input-encoding", default="frame", choices=["frame", "native"],
                    help="frame=K independent images; native=single videos= input path for encoding-coupling 2x2 training")
    ap.add_argument("--native-frames", type=int, default=64)
    ap.add_argument("--native-decode-size", type=int, default=336)
    ap.add_argument("--native-resized", type=int, default=336)
    ap.add_argument("--native-prompt", default="official_hms", choices=["timestamps", "official_sec", "official_hms"])
    ap.add_argument("--native-sample-fps", type=float, default=2.0)
    ap.add_argument("--nct-same-alpha", type=float, default=0.0,
                    help="NCT only: same-action invariant side SFT weight, forcing parseable original window instead of parse-fail escape")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=150,
                    help="save a checkpoint every N optimizer steps -> allows early stopping after convergence, no compute wasted on the near-zero-gradient tail")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0, help="multi-seed: LoRA init + data shuffle, for error bars")
    args = ap.parse_args()

    device = args.gpu
    torch.cuda.set_device(device)
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model

    rows = [json.loads(l) for l in open(args.pairs) if l.strip()]
    if args.limit: rows = rows[:args.limit]
    armname = {"gtsft": "GT-SFT floor", "corruptaug": "CorruptAug-SFT", "mdpo": "mDPO-video concurrent-work control", "defacto": "DeFacto-style three independent SFT concurrent-work control", "nct": "NCT natural counterfactual branch"}.get(
        args.mode, f"PAPO{'+npole' if args.npole else ''}{'-sym' if args.npole_symmetric else ''}{'+acc' if args.alpha>0 else '-only'}")
    print(f"[papo] {len(rows)} pairs, mode={args.mode} input={args.input_encoding} npole={args.npole} corrupt_op={args.corrupt_op} "
          f"epochs={args.epochs} lr={args.lr} beta={args.beta} alpha={args.alpha} accum={args.accum} "
          f"gpu={device} model={args.model} -> {armname}")
    if args.input_encoding == "native":
        print(
            f"[native-train] K={args.native_frames} decode={args.native_decode_size} "
            f"resized={args.native_resized} prompt={args.native_prompt} sample_fps={args.native_sample_fps}"
        )

    torch.manual_seed(args.seed); random.seed(args.seed)
    proc = AutoProcessor.from_pretrained(args.model)
    try:
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": device})
    except Exception as exc:
        from transformers import Qwen2_5_VLForConditionalGeneration
        print(f"[model-load] AutoModelForImageTextToText failed; fallback Qwen2_5_VLForConditionalGeneration: {type(exc).__name__}: {exc}", flush=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": device})
    lcfg = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = random.Random(4321 + device + 1000 * args.seed)
    nct_ctx = None
    if args.mode == "nct":
        action_rows, by_code = load_action_index()
        nct_ctx = {"action_rows": action_rows, "by_code": by_code}
        covered = sum(1 for ex in rows if best_action_for_ex(ex, action_rows))
        print(f"[nct] same-action action coverage: {covered}/{len(rows)} ({covered/max(1,len(rows)):.1%})")
    step = 0; run_loss = run_acc = 0.0; t0 = time.time()
    for ep in range(args.epochs):
        random.Random(1234 + ep + 1000 * args.seed).shuffle(rows)
        for i, ex in enumerate(rows):
            try:
                loss, acc, margin = train_step(
                    model,
                    proc,
                    ex,
                    device,
                    args.mode,
                    args.beta,
                    args.alpha,
                    args.npole,
                    rng,
                    args.corrupt_op,
                    args.npole_symmetric,
                    nct_ctx,
                    args.nct_same_alpha,
                    args.input_encoding,
                    args.native_frames,
                    args.native_decode_size,
                    args.native_resized,
                    args.native_prompt,
                    args.native_sample_fps,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); print(f"  [oom] skip {ex['video']}"); continue
            except ValueError as exc:
                print(f"  [skip] {ex.get('video')} {exc}", flush=True); continue
            (loss / args.accum).backward()
            run_loss += float(loss.detach()); run_acc += acc
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 5 == 0:
                    n = args.accum * 5
                    print(f"  ep{ep} step{step} loss={run_loss/n:.4f} margin_acc={run_acc/n:.3f} "
                          f"({i+1}/{len(rows)}, {time.time()-t0:.0f}s)", flush=True)
                    run_loss = run_acc = 0.0
                if args.save_every and step % args.save_every == 0:
                    os.makedirs(args.out, exist_ok=True); model.save_pretrained(args.out)
                    print(f"  [ckpt] step{step} -> {args.out}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    print(f"[papo] adapter -> {args.out}  ({time.time()-t0:.0f}s)")
    print(f"     eval accuracy: python probe3_selector_evidence/12_charades_grounding.py --adapter {args.out} --n 1000")
    if args.input_encoding == "native":
        print(
            "     eval native faithfulness: python probe3_selector_evidence/native_video_causal_probe.py "
            f"--model {args.model} --adapter {args.out} --n 500 --frames {args.native_frames} "
            f"--decode-size {args.native_decode_size} --resized {args.native_resized} "
            f"--native-prompt {args.native_prompt} --sample-fps {args.native_sample_fps} --ops mask,swap --R 3"
        )
        print(
            f"     eval frame provenance: python probe3_selector_evidence/14_causal_provenance_probe.py "
            f"--model {args.model} --adapter {args.out} --n 500 --ops swap,mask"
        )
    else:
        print(f"     eval faithfulness: python probe3_selector_evidence/14_causal_provenance_probe.py --model {args.model} --adapter {args.out} --n 1000 --ops swap,mask")


if __name__ == "__main__":
    main()
