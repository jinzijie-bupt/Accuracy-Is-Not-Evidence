#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""29_grpo_native_iou.py -- IoU-GRPO baseline re-trained under the NATIVE protocol of our method.

Rebuttal request: "IoU-GRPO was trained on 16 frames but audited at 2 fps; train it under matched conditions."
Everything below except the objective is taken from our trainer 27_papo_train_qcdro_v3seln.py (sha16
98be87f7e92e6ed4, loaded as a module, sha pinned):

  model / processor / LoRA  : same load path (V._fp_dtype_compat, bf16, device_map {"": gpu}), same LoRA
                              (r=16, alpha=2r=32, dropout .05, targets q/k/v/o/gate/up/down, CAUSAL_LM), same
                              torch.manual_seed(seed)+random.seed(seed) before loading (=> same LoRA init as the
                              method's seed-0 arm), gradient_checkpointing_enable + enable_input_require_grads.
  video input               : V.native_decode (qwen_vl_utils.fetch_video, fps 2, total_pixels 3584*28*28,
                              min_pixels 16*28*28, max_frames 768) -> uint8 frames -> V._native_inputs
                              (tf5: video_metadata + do_sample_frames=False; tf4.51: fps list).
  prompt                    : V.NATIVE_PROMPTS[timer1_default], message = [video placeholder, text], chat template
                              with add_generation_prompt=True (identical to V.native_completion_logprob).
  questions / order         : rows = first --limit (2400) lines of --pairs; order = V.build_schedule_v2(epochs=1,
                              ops=swap,mask,blur,tshift,shuffle, k=2, seed) -- the SAME schedule our method used
                              (fingerprint printed; pass --expected-schedule-fingerprint to pin it). Only the row
                              order/membership is used; the operators are ignored. One pass.
  answer parser             : parse_native() of the native evaluator 12p4n_native_acc_eval_tf5meta_r8.py (sha pinned;
                              identical to tf451_r5's) -- the parser the 2 fps audit uses for timer1 answers.

GRPO (semantics of the old IoU-GRPO baseline: 04_train_grpo_v7.py --arm iou --no-spec 1 + grpo_decoupled_v7 +
03_rewards_v7/r5 03_rewards on TRL 1.7, --agg sum_then_normalize --scale-rewards none):
  reward     r_i = fmt_i + IoU_i ; fmt_i = 1[parse_native(text_i, dur) is not None], IoU_i = IoU(pred_i, [gs,ge])
             (0 if unparsable). Old: r5 reward_format + reward_iou with g12.parse_window; here OUR timer1 parser.
  advantage  --adv-norm group_z (default) = (r - mean_G r) / (std_G(r, ddof=0) + 1e-6).  This is what the old
             baseline ACTUALLY optimised: with --no-spec 1 the decoupled v7 trainer replaces the reward column by
             A_task = zscore_g(fmt+iou) (grpo_decoupled_v7.py L483, ddof 0, eps 1e-6) and TRL's
             sum_then_normalize + scale_rewards=none then only subtracts the (already zero) group mean.
             --adv-norm center gives the plain r - mean_G r reading of "scale_rewards=none".
  sampling   per question ONE generate() call over a G=8 batch (prompt + video tiled 8x; HF prefills the tiled
             batch), temperature 1.0 (TRL GRPOConfig default; the old command never set it), top_p 1, no top_k,
             repetition_penalty 1, max_new_tokens 64, use_cache=True, model.eval() (no LoRA dropout, KV cache on).
             Sampling = exact Gumbel-max over the full vocabulary inside a LogitsProcessor driven by a
             torch.Generator seeded from sha256(seed|row_index) -> reruns/resumes reproduce the same draws
             (up to GPU kernel nondeterminism). The model's own generation_config (Qwen: top_k 1, top_p .001,
             temperature .1, repetition_penalty 1.05) is explicitly overridden.
  loss       on-policy, one update per batch => ratio exp(logp - logp.detach()) == 1 (TRL num_iterations=1):
             per-token  l_t = -ratio_t * A_i + beta * k3_t,  k3_t = exp(ref_t - logp_t) - (ref_t - logp_t) - 1,
             ref = same model under peft disable_adapter(), beta = 0.04, logits / temperature before log_softmax
             (TRL). Aggregation --loss-type dapo (default; TRL >= 0.22 default, assumed for TRL 1.7 -- NOT verified
             here): step loss = sum over all completion tokens of the optimizer step / #completion tokens of the
             step; bnpo / grpo also available. Each micro-loss is further divided by --ga-loss-divisor (default =
             questions per step = grad-accum 4), reproducing HF Trainer's 1/GA scaling (GRPOTrainer sets
             model_accepts_loss_kwargs=False) -- a constant factor that only matters through grad clipping / Adam eps.
             NOTE: "sum_then_normalize" in the old command is the multi-REWARD aggregation (weighted sum of reward
             columns, then group normalisation), not the token aggregation of the loss.
  optimiser  AdamW(lr 1e-5, betas .9/.999, eps 1e-8, weight_decay 0.0 = HF TrainingArguments defaults), LambdaLR
             constant_with_warmup(20) (HF: lr = step/20 for step < 20, so optimizer step 1 runs at lr 0), grad clip 1.0
             (HF max_grad_norm default), no autocast (same as our trainer: bf16 weights, fp32 LoRA params via peft).

STEP DEFINITION (mirrors the old baseline: TRL "--launch single --grad-accum 4 --questions-per-step 4
--num-generations 8" => generation_batch_size 32 = 4 questions x 8 rollouts, steps_per_generation 4, ONE optimizer
step per 32 rollouts):  1 optimizer step = 4 consecutive scheduled questions x G=8 completions = 32 rollouts,
accumulated as 4 micro-batches (one question each; each split into --score-batch chunks for memory).
2,400 questions, one pass  =>  600 optimizer steps  (old baseline: --max-steps 700 = 2,800 question draws).
Phase 1 of a step samples+rewards all 4 questions with the step-start policy (as TRL does), phase 2 computes the
loss and backprops; then clip, AdamW step, scheduler step.

Throughput: the policy and reference log-prob passes are batched over the G completions (chunks of --score-batch)
and reuse the prompt tensors built once per question; the vision tower is still re-run per sequence in a batch
(LoRA also wraps the vision MLPs' gate/up/down_proj, so the policy pass needs gradients through it). Video decoding
of upcoming questions is prefetched in a background thread (--prefetch).

Usage (full):  see the bottom of this docstring or the report.
  python 29_grpo_native_iou.py --pairs qcdro_v2_pairs_train.jsonl --limit 2400 --seed 0 --out <dir> [--resume]
  python 29_grpo_native_iou.py ... --smoke          # 3 questions, 1 optimizer step (runs at lr 0 because of
                                                    #  the HF warmup; add --warmup-steps 0 to see the weights move)
  python 29_grpo_native_iou.py ... --dry-shapes     # processor + video only, no weights
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER_FILE = "27_papo_train_qcdro_v3seln.py"
TRAINER_SHA16 = "2010cd89375b909f"
PARSER_FILE = "12p4n_native_acc_eval_tf5meta_r8.py"
PARSER_SHA16 = "a892cad30a491021"
METHOD_OPS = "swap,mask,blur,tshift,shuffle"      # the trainer's --ops default (only used to rebuild its schedule)
ADV_EPS = 1e-6                                     # grpo_decoupled_v7.SPEC_EPS
ADV_DDOF = 0                                       # grpo_decoupled_v7.ZSCORE_DDOF
LOSS_TYPES = ("dapo", "bnpo", "grpo")
ADV_MODES = ("group_z", "center")
TRAINER_NAME = "grpo_native_iou_v1"


# ============================================================================ files / modules
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha16_file(path: str) -> str:
    return sha256_file(path)[:16]


def load_module(name: str, path: str):
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_trainer(path: str, expect_sha16: str | None):
    sh = sha16_file(path)
    if expect_sha16 and sh != expect_sha16:
        raise SystemExit(f"STOP trainer {path} sha16={sh} != expected {expect_sha16} "
                         "(this script reuses its functions by name; re-audit before overriding --trainer-sha16)")
    return load_module("v3seln", path), sh


def load_parser(path: str, expect_sha16: str | None):
    sh = sha16_file(path)
    if expect_sha16 and sh != expect_sha16:
        raise SystemExit(f"STOP parser module {path} sha16={sh} != expected {expect_sha16}")
    mod = load_module("p4n_parser", path)
    if not callable(getattr(mod, "parse_native", None)):
        raise SystemExit(f"STOP {path} has no parse_native()")
    return mod.parse_native, mod, sh


# ============================================================================ reward / advantage (pure python)
def iou_1d(a, b) -> float:
    """== 12_charades_grounding.iou_1d (the r5 reward_iou's IoU)."""
    s, e = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def reward_one(text: str, gs: float, ge: float, dur: float, parse_fn):
    """Old iou arm = reward_format + reward_iou (r5 03_rewards L259-266), with OUR parser."""
    pred = parse_fn(text, dur)
    fmt = 1.0 if pred is not None else 0.0
    iou = iou_1d(pred, (float(gs), float(ge))) if pred is not None else 0.0
    return fmt, float(iou), pred


def compute_rewards(texts, gs, ge, dur, parse_fn) -> dict:
    fmt, iou, preds = [], [], []
    for t in texts:
        f, i, p = reward_one(t, gs, ge, dur, parse_fn)
        fmt.append(f); iou.append(i); preds.append(None if p is None else [float(p[0]), float(p[1])])
    fmt = np.asarray(fmt, dtype=np.float64)
    iou = np.asarray(iou, dtype=np.float64)
    return {"fmt": fmt, "iou": iou, "reward": fmt + iou, "pred": preds}


def group_advantages(rewards, mode: str = "group_z", eps: float = ADV_EPS, ddof: int = ADV_DDOF) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 1 or r.size == 0:
        raise ValueError(f"rewards must be a non-empty 1-D group, got shape {r.shape}")
    if not np.all(np.isfinite(r)):
        raise ValueError("non-finite reward in group")
    c = r - r.mean()
    if mode == "center":
        return c
    if mode == "group_z":
        return c / (r.std(ddof=ddof) + float(eps))
    raise ValueError(f"unknown adv mode {mode!r}; choose from {ADV_MODES}")


def question_seed(seed: int, row_index: int, tag: str = "grpo_sample") -> int:
    d = hashlib.sha256(f"{int(seed)}|{int(row_index)}|{tag}".encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big") & ((1 << 63) - 1)


def group_steps(order, questions_per_step: int):
    q = int(questions_per_step)
    if q <= 0:
        raise ValueError("questions_per_step must be positive")
    order = list(order)
    return [order[i:i + q] for i in range(0, len(order), q)]


def constant_with_warmup_lambda(num_warmup_steps: int):
    """== transformers.get_constant_schedule_with_warmup's lr_lambda."""
    w = int(num_warmup_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < w:
            return float(current_step) / float(max(1.0, w))
        return 1.0
    return lr_lambda


# ============================================================================ torch pieces (lazy import)
def k3_kl(logp, ref_logp):
    """TRL per-token KL estimator: exp(ref - logp) - (ref - logp) - 1  (>= 0, unbiased for KL(pi || ref))."""
    import torch
    d = ref_logp - logp
    return torch.exp(d) - d - 1.0


def grpo_per_token_loss(logp, ref_logp, adv, beta: float):
    """On-policy GRPO per-token loss. ratio = exp(logp - logp.detach()) (== 1, gradient = grad logp)."""
    import torch
    ratio = torch.exp(logp - logp.detach())
    per = -ratio * adv.to(logp.dtype).unsqueeze(1)
    kl = None
    if ref_logp is not None:
        kl = k3_kl(logp, ref_logp)
        if beta != 0.0:
            per = per + float(beta) * kl
    return per, kl


def chunk_loss(per, mask, loss_type: str, *, step_tokens: int, q_tokens: int, n_seq_question: int,
               ga_div: float):
    """Loss contribution of one score chunk (a subset of one question's G completions).

    dapo : sum(per*mask) / #completion tokens of the whole optimizer step   (chunking-invariant)
    bnpo : sum(per*mask) / #completion tokens of this question (micro-batch = one question)
    grpo : sum_seq( mean_tokens(per) ) / G  (sequence-mean, then mean over the micro-batch)
    all  : / ga_div  (HF Trainer's 1/gradient_accumulation_steps)
    """
    m = mask.to(per.dtype)
    if loss_type == "dapo":
        val = (per * m).sum() / float(max(1, int(step_tokens)))
    elif loss_type == "bnpo":
        val = (per * m).sum() / float(max(1, int(q_tokens)))
    elif loss_type == "grpo":
        val = ((per * m).sum(1) / m.sum(1).clamp(min=1.0)).sum() / float(max(1, int(n_seq_question)))
    else:
        raise ValueError(f"unknown loss_type {loss_type!r}")
    return val / float(ga_div)


def gumbel_perturb(scores, generator, temperature: float):
    """argmax(scores/T + Gumbel) is an exact sample from softmax(scores/T) over the full vocabulary."""
    import torch
    u = torch.rand(scores.shape, generator=generator, device=scores.device, dtype=torch.float32)
    u = u.clamp_(min=torch.finfo(torch.float32).tiny, max=1.0 - 2.0 ** -24)
    g = -torch.log(-torch.log(u))
    return scores.float() / float(temperature) + g


def make_gumbel_processor(seed_int: int, temperature: float):
    import torch
    from transformers import LogitsProcessor

    class GumbelMaxSampler(LogitsProcessor):
        def __init__(self):
            self.seed = int(seed_int)
            self.temperature = float(temperature)
            self.gen = None
            self.n_calls = 0

        def __call__(self, input_ids, scores):
            if self.gen is None:
                self.gen = torch.Generator(device=scores.device)
                self.gen.manual_seed(self.seed)
            self.n_calls += 1
            return gumbel_perturb(scores, self.gen, self.temperature)

    return GumbelMaxSampler()


def completion_mask_from_ids(ids, eos_ids):
    """mask[i, j] = 1 for j <= first EOS (EOS included, TRL convention); no EOS -> all ones. Returns (mask, lens)."""
    import torch
    is_eos = torch.zeros_like(ids, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= ids == int(e)
    G, L = ids.shape
    first = torch.full((G,), L, dtype=torch.long, device=ids.device)
    has = is_eos.any(dim=1)
    if bool(has.any()):
        first[has] = is_eos.int().argmax(dim=1)[has]
    lens = torch.clamp(first + 1, max=L)
    mask = (torch.arange(L, device=ids.device).unsqueeze(0) < lens.unsqueeze(1)).long()
    return mask, lens


def expand_inputs(inputs: dict, n: int) -> dict:
    """Tile one processed (prompt, video) sample n times. Per-sample tensors (input_ids, attention_mask,
    mm_token_type_ids, video_grid_thw, second_per_grid_ts) get n rows; pixel_values_videos (patches of the one
    video) is concatenated n times, which is exactly the processor layout for n copies of the same video."""
    import torch
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            out[k] = v.repeat(n, *([1] * (v.dim() - 1))) if v.dim() >= 1 else v
        elif isinstance(v, list):
            out[k] = list(v) * n
        else:
            out[k] = v
    return out


def build_score_inputs(prompt_inputs: dict, comp_ids, comp_mask, pad_id: int):
    """prompt (1, P) tiled to B rows + right-padded completions (B, Lc). Other (1, P)-shaped per-token keys
    (e.g. tf5 mm_token_type_ids) are extended with zeros (= text tokens). Returns (inputs, P)."""
    import torch
    B, Lc = comp_ids.shape
    P = int(prompt_inputs["input_ids"].shape[1])
    base = expand_inputs(prompt_inputs, B)
    ids_c = torch.where(comp_mask.bool(), comp_ids, torch.full_like(comp_ids, int(pad_id)))
    out = dict(base)
    out["input_ids"] = torch.cat([base["input_ids"], ids_c.to(base["input_ids"].dtype)], dim=1)
    am = base.get("attention_mask")
    if am is None:
        am = torch.ones_like(base["input_ids"])
    out["attention_mask"] = torch.cat([am, comp_mask.to(am.dtype)], dim=1)
    for k, v in base.items():
        if k in ("input_ids", "attention_mask") or not torch.is_tensor(v):
            continue
        if v.dim() == 2 and tuple(v.shape) == (B, P):
            out[k] = torch.cat([v, torch.zeros((B, Lc), dtype=v.dtype, device=v.device)], dim=1)
    return out, P


def completion_logps(model, inp: dict, P: int, Lc: int, temperature: float, keep_name: str | None):
    """Per-token log pi(token) of the Lc completion tokens (B, Lc), logits / temperature as in TRL."""
    import torch
    kw = {keep_name: Lc + 1} if keep_name else {}
    out = model(**inp, use_cache=False, **kw)
    logits = out.logits
    S = int(inp["input_ids"].shape[1])
    if int(logits.shape[1]) == Lc + 1:
        sl = logits[:, :-1, :]
    else:
        sl = logits[:, P - 1:S - 1, :]
    if int(sl.shape[1]) != Lc:
        raise RuntimeError(f"logit slice length {sl.shape[1]} != completion length {Lc} (P={P}, S={S})")
    lsm = torch.log_softmax(sl.float() / float(temperature), dim=-1)
    tgt = inp["input_ids"][:, P:]
    return lsm.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)


def logits_keep_kwarg(model) -> str | None:
    import inspect
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    try:
        params = inspect.signature(base.forward).parameters
    except (TypeError, ValueError):
        return None
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return name
    return None


# ============================================================================ prompt construction (trainer reuse)
def chat_text(proc, prompt_text: str) -> str:
    """Identical to V.native_completion_logprob L393-395."""
    msgs = [{"role": "user", "content": [{"type": "video", "video": "native_placeholder"},
                                          {"type": "text", "text": prompt_text}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def prompt_inputs_for(V, proc, frames_np, sample_fps: float, caption: str, prompt_style: str):
    import torch
    vid = torch.from_numpy(np.ascontiguousarray(frames_np)).permute(0, 3, 1, 2)  # (T,3,H,W) uint8, as V L391
    text = chat_text(proc, V.NATIVE_PROMPTS[prompt_style].format(caption))
    return V._native_inputs(proc, text, vid, sample_fps), text


def gen_eos_pad(proc, model):
    tok = getattr(proc, "tokenizer", proc)
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = tok.eos_token_id
    eos_ids = [int(x) for x in (eos if isinstance(eos, (list, tuple)) else [eos])]
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos_ids[0]
    return eos_ids, int(pad)


def sample_group(model, pin: dict, G: int, max_new_tokens: int, temperature: float, seed_int: int,
                 eos_ids, pad_id: int):
    """One generate() over the G-tiled prompt; Gumbel-max sampling with a per-question generator."""
    import torch
    from transformers import LogitsProcessorList
    sampler = make_gumbel_processor(seed_int, temperature)
    gin = expand_inputs(pin, G)
    P = int(pin["input_ids"].shape[1])
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model.generate(**gin, max_new_tokens=int(max_new_tokens), do_sample=False, num_beams=1,
                                 temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                                 use_cache=True, logits_processor=LogitsProcessorList([sampler]),
                                 eos_token_id=list(eos_ids), pad_token_id=int(pad_id))
    finally:
        if was_training:
            model.train()
    seq = out if torch.is_tensor(out) else out.sequences
    comp = seq[:, P:]
    if sampler.n_calls != int(comp.shape[1]) or int(comp.shape[1]) == 0:
        raise RuntimeError(f"STOP sampler was called {sampler.n_calls}x for {comp.shape[1]} generated positions "
                           "-- the Gumbel processor is not wired into generate(); sampling would be greedy")
    mask, lens = completion_mask_from_ids(comp, eos_ids)
    return comp, mask, lens


# ============================================================================ main
def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", required=True, help="qcdro_v2_pairs_train.jsonl (same file as the method run)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="default: trainer MODEL ($DATA_ROOT/models/Qwen2.5-VL-7B-Instruct)")
    ap.add_argument("--limit", type=int, default=2400, help="first N pairs rows (method run: --limit 2400)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--r", type=int, default=16, help="LoRA rank (trainer default 16; alpha = 2r)")
    # native protocol knobs (trainer defaults)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--total-pixels", type=int, default=3584 * 28 * 28)
    ap.add_argument("--min-pixels", type=int, default=16 * 28 * 28)
    ap.add_argument("--max-frames", type=int, default=768)
    ap.add_argument("--prompt-style", default="timer1_default")
    ap.add_argument("--answer-style", choices=["timer1", "vchat"], default="timer1",
                    help="recorded; only used for the GT-answer token count in --dry-shapes")
    # GRPO (old IoU-GRPO baseline values)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--questions-per-step", type=int, default=4, help="= old --questions-per-step 4 / --grad-accum 4")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0, help="TRL GRPOConfig default (old run did not set it)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04, help="beta_KL")
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=0.0, help="HF TrainingArguments default")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--adv-norm", choices=ADV_MODES, default="group_z")
    ap.add_argument("--loss-type", choices=LOSS_TYPES, default="dapo")
    ap.add_argument("--ga-loss-divisor", type=float, default=None,
                    help="divide every micro-loss by this (HF 1/GA); default = --questions-per-step")
    ap.add_argument("--score-batch", type=int, default=4, help="completions per policy/ref log-prob forward")
    # run control
    ap.add_argument("--save-every", type=int, default=25, help="optimizer steps between resume checkpoints")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (0 = full pass)")
    ap.add_argument("--smoke", action="store_true", help="3 questions, 1 optimizer step")
    ap.add_argument("--dry-shapes", action="store_true", help="build inputs + print token counts, no weights")
    ap.add_argument("--dry-n", type=int, default=5)
    ap.add_argument("--prefetch", type=int, default=1, choices=[0, 1], help="background video decoding")
    ap.add_argument("--max-skip-frac", type=float, default=0.02,
                    help="abort if more than this fraction of questions fail to decode (after 50)")
    ap.add_argument("--expected-schedule-fingerprint", default=None)
    ap.add_argument("--trainer", default=os.path.join(HERE, TRAINER_FILE))
    ap.add_argument("--trainer-sha16", default=TRAINER_SHA16, help="'' disables the pin")
    ap.add_argument("--parser-module", default=os.path.join(HERE, PARSER_FILE))
    ap.add_argument("--parser-sha16", default=PARSER_SHA16, help="'' disables the pin")
    ap.add_argument("--cpu-debug", action="store_true",
                    help="TESTS ONLY: run on CPU (tiny random model); never for real runs")
    return ap


def _ident(args, pairs_sha, schedule_fp):
    keys = ("seed", "limit", "r", "fps", "total_pixels", "min_pixels", "max_frames", "prompt_style",
            "num_generations", "questions_per_step", "max_new_tokens", "temperature", "lr", "beta",
            "warmup_steps", "weight_decay", "max_grad_norm", "adv_norm", "loss_type", "ga_loss_divisor",
            "smoke", "model")
    d = {k: getattr(args, k) for k in keys}
    d.update({"pairs_sha256": pairs_sha, "schedule_fp": schedule_fp})
    return d


def _versions():
    out = {"python": platform.python_version(), "platform": platform.platform(), "host": socket.gethostname()}
    try:
        from importlib.metadata import version
    except Exception:  # pragma: no cover
        version = None
    for pkg in ("torch", "transformers", "peft", "qwen-vl-utils", "accelerate", "decord", "torchvision"):
        try:
            out[pkg] = version(pkg) if version else None
        except Exception:
            out[pkg] = None
    return out


class Prefetcher:
    def __init__(self, fn, enabled: bool):
        self.fn = fn
        self.ex = None
        self.fut = {}
        if enabled:
            from concurrent.futures import ThreadPoolExecutor
            self.ex = ThreadPoolExecutor(max_workers=2)

    def want(self, keys):
        if self.ex is None:
            return
        for k in keys:
            if k not in self.fut:
                self.fut[k] = self.ex.submit(self.fn, k)

    def get(self, k):
        if self.ex is None:
            return self.fn(k)
        f = self.fut.pop(k, None)
        if f is None:
            f = self.ex.submit(self.fn, k)
        return f.result()

    def close(self):
        if self.ex is not None:
            self.ex.shutdown(wait=False, cancel_futures=True)


def _append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _truncate_trace(path, last_step):
    if not os.path.isfile(path):
        return 0
    keep = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue          # torn last line
            if int(rec.get("step", 0)) <= last_step:
                keep.append(line)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("".join(x + "\n" for x in keep))
    os.replace(tmp, path)
    return len(keep)


def run_dry_shapes(V, args, rows, order, decode):
    from transformers import AutoProcessor
    import torch
    proc = AutoProcessor.from_pretrained(args.model)
    tok = proc.tokenizer
    video_pad_id = tok.convert_tokens_to_ids("<|video_pad|>")
    for ri in order[: args.dry_n]:
        ex = rows[ri]
        frames, times, dur, sfps = decode(ri)
        pin, text = prompt_inputs_for(V, proc, frames, sfps, str(ex["caption"]), args.prompt_style)
        ids = pin["input_ids"]
        vtok = int((ids == int(video_pad_id)).sum())
        gt = V.native_completion(float(ex["gs"]), float(ex["ge"]), args.answer_style)
        n_gt = len(tok(gt, add_special_tokens=False).input_ids)
        info = {k: (list(v.shape) if torch.is_tensor(v) else v) for k, v in pin.items()}
        print(f"[dry] row={ri} video={ex['video']} T={len(frames)} HxW={frames.shape[1]}x{frames.shape[2]} "
              f"dur={dur:.2f}s sample_fps={sfps:.3f} prompt_tokens={ids.shape[1]} video_tokens={vtok} "
              f"gt_answer_tokens={n_gt} score_seq<= {ids.shape[1] + args.max_new_tokens} keys={info}", flush=True)
    print("[dry] done (no weights loaded)", flush=True)


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    if args.smoke:
        args.questions_per_step = 3
        args.max_steps = 1
    if args.ga_loss_divisor is None:
        args.ga_loss_divisor = float(args.questions_per_step)
    if args.num_generations < 2:
        ap.error("--num-generations must be >= 2 (group advantages)")
    if args.score_batch <= 0 or args.temperature <= 0:
        ap.error("--score-batch and --temperature must be positive")

    V, trainer_sha = load_trainer(args.trainer, args.trainer_sha16 or None)
    parse_fn, P4, parser_sha = load_parser(args.parser_module, args.parser_sha16 or None)
    if args.prompt_style not in V.NATIVE_PROMPTS:
        ap.error(f"--prompt-style must be one of {sorted(V.NATIVE_PROMPTS)}")
    if getattr(P4, "PROMPTS", {}).get(args.prompt_style) != V.NATIVE_PROMPTS[args.prompt_style]:
        raise SystemExit("STOP trainer and evaluator disagree on the prompt text for " + args.prompt_style)
    if args.model is None:
        args.model = V.MODEL
    self_sha = sha16_file(os.path.abspath(__file__))

    # ---------------- questions: identical rows + order as the method run
    rows = [json.loads(line) for line in open(args.pairs, encoding="utf-8") if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    pairs_sha = sha256_file(args.pairs)
    schedule = V.build_schedule_v2(rows, epochs=1, requested_ops=V.parse_ops_v2(METHOD_OPS), ops_per_example=2,
                                   seed=args.seed, query_group_key="query_group")
    if args.expected_schedule_fingerprint and args.expected_schedule_fingerprint != schedule.fingerprint:
        raise SystemExit(f"STOP schedule fingerprint {schedule.fingerprint} != expected "
                         f"{args.expected_schedule_fingerprint}")
    order = [int(it.row_index) for it in schedule.items]
    if args.smoke:
        order = order[:3]
    steps = group_steps(order, args.questions_per_step)
    if args.max_steps and args.max_steps > 0:
        steps = steps[: args.max_steps]
    n_q_planned = sum(len(s) for s in steps)
    print(f"[data] rows={len(rows)} scheduled={len(schedule.items)} fingerprint={schedule.fingerprint} "
          f"skipped_by_schedule={schedule.skipped_too_few_valid} -> {n_q_planned} questions in {len(steps)} "
          f"optimizer steps of {args.questions_per_step} x G={args.num_generations}", flush=True)
    print(f"[prov] trainer={os.path.basename(args.trainer)} sha16={trainer_sha} parser={os.path.basename(args.parser_module)} "
          f"sha16={parser_sha} self sha16={self_sha} pairs sha256={pairs_sha[:16]}", flush=True)

    nsargs = argparse.Namespace(fps=args.fps, total_pixels=args.total_pixels, min_pixels=args.min_pixels,
                                max_frames=args.max_frames)

    def decode(ri):
        ex = rows[ri]
        return V.native_decode(os.path.join(V.base.VIDEO_DIR, str(ex["video"])), nsargs)

    if args.dry_shapes:
        run_dry_shapes(V, args, rows, order, decode)
        return

    os.makedirs(args.out, exist_ok=True)
    trace_path = os.path.join(args.out, "grpo_trace.jsonl")
    manifest_path = os.path.join(args.out, "grpo_native_manifest.json")
    rs_path = os.path.join(args.out, "resume_state.pt")
    with open(os.path.join(args.out, "grpo_schedule.jsonl"), "w", encoding="utf-8") as f:
        for si, st in enumerate(steps):
            f.write(json.dumps({"step": si + 1, "row_index": st, "qid": [rows[r].get("qid") for r in st]}) + "\n")
    train_videos = sorted({str(rows[r]["video"]) for r in order})
    with open(os.path.join(args.out, "train_videos.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_videos) + "\n")

    # ---------------- model (same construction as trainer main() L713-741)
    import torch
    cuda = not args.cpu_debug
    if cuda:
        device = args.gpu                      # trainer: torch.cuda.set_device(gpu); device_map {"": gpu}
        torch.cuda.set_device(device)
    else:
        device = "cpu"
        print("[cpu-debug] CPU run -- for tests with a tiny model only", flush=True)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    proc = AutoProcessor.from_pretrained(args.model)
    try:
        model = V._fp_dtype_compat(AutoModelForImageTextToText, args.model, device_map={"": device}, max_memory=None)
    except Exception as exc:
        from transformers import Qwen2_5_VLForConditionalGeneration
        print(f"[model-load] fallback Qwen2_5_VL: {type(exc).__name__}: {exc}", flush=True)
        model = V._fp_dtype_compat(Qwen2_5_VLForConditionalGeneration, args.model, device_map={"": device},
                                   max_memory=None)
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=lora_targets)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    lora_mods = [n for n, m in model.named_modules() if hasattr(m, "lora_A") and len(getattr(m, "lora_A", {})) > 0]
    n_vis = sum(1 for n in lora_mods if ".visual." in n or n.startswith("visual") or "vision" in n)
    print(f"[lora] wrapped modules={len(lora_mods)} (vision tower: {n_vis}, language: {len(lora_mods) - n_vis})",
          flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = int(sum(p.numel() for p in params))
    keep_name = logits_keep_kwarg(model)
    print(f"[score] logits-trimming kwarg = {keep_name} (None -> full logits are sliced)", flush=True)
    eos_ids, pad_id = gen_eos_pad(proc, model)
    tok = proc.tokenizer
    gcfg = getattr(model, "generation_config", None)
    model_gen_cfg = {k: getattr(gcfg, k, None) for k in ("do_sample", "temperature", "top_p", "top_k",
                                                          "repetition_penalty", "eos_token_id", "pad_token_id")}
    print(f"[gen] model generation_config {model_gen_cfg} -> OVERRIDDEN by: greedy argmax over logits/T + Gumbel "
          f"(T={args.temperature}, full vocab), repetition_penalty=1.0, max_new_tokens={args.max_new_tokens}, "
          f"eos={eos_ids}, pad={pad_id}, use_cache=True, eval mode", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, constant_with_warmup_lambda(args.warmup_steps))

    ident = _ident(args, pairs_sha, schedule.fingerprint)
    step_done = 0
    t_prev = 0.0
    n_skipped = 0
    n_q_done = 0
    n_method_degenerate = 0
    if args.resume and os.path.isfile(rs_path):
        from peft import set_peft_model_state_dict
        rs = torch.load(rs_path, map_location="cpu", weights_only=False)
        if rs["ident"] != ident:
            diff = {k: (rs["ident"].get(k), ident.get(k)) for k in ident if rs["ident"].get(k) != ident.get(k)}
            raise SystemExit(f"[resume-abort] ident mismatch (ckpt, now): {diff}")
        if rs.get("self_sha16") != self_sha:
            print(f"[resume][WARN] this file changed since the checkpoint ({rs.get('self_sha16')} -> {self_sha})",
                  flush=True)
        res = set_peft_model_state_dict(model, rs["adapter"])
        unexpected = list(getattr(res, "unexpected_keys", []) or [])
        if unexpected:
            raise SystemExit(f"[resume-abort] unexpected adapter keys: {unexpected[:5]}")
        opt.load_state_dict(rs["opt"])
        sched.load_state_dict(rs["sched"])
        torch.set_rng_state(rs["rng"]["torch"])
        if cuda and rs["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(rs["rng"]["cuda"])
        random.setstate(rs["rng"]["python"])
        np.random.set_state(rs["rng"]["numpy"])
        step_done = int(rs["step"])
        t_prev = float(rs["elapsed"])
        n_skipped = int(rs.get("n_skipped", 0))
        n_q_done = int(rs.get("n_q_done", 0))
        n_method_degenerate = int(rs.get("n_method_degenerate", 0))
        kept = _truncate_trace(trace_path, step_done)
        print(f"[resume] step={step_done}/{len(steps)} questions_done={n_q_done} trace_rows_kept={kept} "
              f"elapsed_prev={t_prev:.0f}s", flush=True)
    elif args.resume:
        print("[resume] no resume_state.pt -> fresh start", flush=True)
        _truncate_trace(trace_path, 0)
    else:
        _truncate_trace(trace_path, 0)

    t0 = time.time() - t_prev
    manifest = {
        "trainer": TRAINER_NAME, "status": "running", "args": vars(args),
        "self_sha16": self_sha, "trainer_module": os.path.basename(args.trainer), "trainer_sha16": trainer_sha,
        "parser_module": os.path.basename(args.parser_module), "parser_sha16": parser_sha,
        "helper_sha16": {fn: sha16_file(os.path.join(HERE, fn)) for fn in
                         ("18_papo_train_operator_mix.py", "12_charades_grounding.py", "native_video_causal_probe.py",
                          "12p4a_grounding_eval_adapter.py") if os.path.isfile(os.path.join(HERE, fn))},
        "pairs": os.path.abspath(args.pairs), "pairs_sha256": pairs_sha, "limit": args.limit,
        "schedule": {"fingerprint": schedule.fingerprint, "n_scheduled": len(schedule.items),
                     "skipped_by_schedule": schedule.skipped_too_few_valid, "method_ops": METHOD_OPS,
                     "n_questions_planned": n_q_planned, "n_optimizer_steps_planned": len(steps)},
        "train_video_count": len(train_videos),
        "train_video_sha256": hashlib.sha256("\n".join(train_videos).encode()).hexdigest(),
        "lora": {"r": args.r, "alpha": 2 * args.r, "dropout": 0.05, "targets": lora_targets,
                 "task_type": "CAUSAL_LM", "n_wrapped": len(lora_mods), "n_wrapped_vision": n_vis,
                 "n_trainable_params": n_trainable},
        "generation": {"model_generation_config": model_gen_cfg, "temperature": args.temperature, "top_p": 1.0,
                       "top_k": None, "repetition_penalty": 1.0, "sampler": "gumbel-max, torch.Generator per "
                       "question seeded sha256(seed|row_index|grpo_sample)", "max_new_tokens": args.max_new_tokens,
                       "eos_ids": eos_ids, "pad_id": pad_id, "mode": "eval (no LoRA dropout), use_cache=True"},
        "objective": {"reward": "fmt + IoU (parse_native of the native evaluator)", "adv_norm": args.adv_norm,
                      "adv_eps": ADV_EPS, "adv_ddof": ADV_DDOF, "loss_type": args.loss_type,
                      "ga_loss_divisor": args.ga_loss_divisor, "beta_kl": args.beta, "kl_estimator": "k3",
                      "ratio": "on-policy, exp(logp - logp.detach())", "ref": "peft disable_adapter()"},
        "optimizer": {"type": "torch.optim.AdamW", "lr": args.lr, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": args.weight_decay, "scheduler": f"constant_with_warmup({args.warmup_steps})",
                      "max_grad_norm": args.max_grad_norm},
        "step_definition": f"1 optimizer step = {args.questions_per_step} scheduled questions x "
                           f"G={args.num_generations} completions; micro-batch = 1 question",
        "logits_keep_kwarg": keep_name, "versions": _versions(),
        "gpu": (torch.cuda.get_device_name(device) if cuda else "cpu-debug"), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out": os.path.abspath(args.out), "trace": trace_path,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False, default=str)

    def save_state(step: int) -> None:
        from peft import get_peft_model_state_dict
        rs = {"step": step, "ident": ident, "self_sha16": self_sha,
              "adapter": {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
              "opt": opt.state_dict(), "sched": sched.state_dict(),
              "rng": {"torch": torch.get_rng_state(), "cuda": (torch.cuda.get_rng_state_all() if cuda else None),
                      "python": random.getstate(), "numpy": np.random.get_state()},
              "elapsed": time.time() - t0, "n_skipped": n_skipped, "n_q_done": n_q_done,
              "n_method_degenerate": n_method_degenerate}
        tmp = rs_path + ".tmp"
        torch.save(rs, tmp)
        os.replace(tmp, rs_path)
        model.save_pretrained(args.out)       # convenience copy of the adapter at the same step

    pref = Prefetcher(decode, bool(args.prefetch))
    G = int(args.num_generations)
    sb = int(args.score_batch)
    kl_first = None
    try:
        for si in range(step_done, len(steps)):
          try:                                   # (2-space level: body below kept at its original indentation)
            step_rows = steps[si]
            pref.want(step_rows)
            if si + 1 < len(steps):
                pref.want(steps[si + 1])
            t_step = time.time()
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            # ---------------- phase 1: sample + reward (policy at step start)
            qs = []
            t_dec = t_gen = 0.0
            for ri in step_rows:
                ex = rows[ri]
                ta = time.time()
                try:
                    frames, times, dur, sfps = pref.get(ri)
                except Exception as exc:          # decode failure -> skip question (logged + counted)
                    n_skipped += 1
                    print(f"  [skip] row={ri} video={ex.get('video')} decode {type(exc).__name__}: {exc}", flush=True)
                    _append_jsonl(trace_path, {"step": si + 1, "skip": True, "row_index": ri,
                                               "qid": ex.get("qid"), "error": f"{type(exc).__name__}: {exc}"})
                    continue
                t_dec += time.time() - ta
                tb = time.time()
                gs, ge = float(ex["gs"]), float(ex["ge"])
                n_estar = int(sum(1 for t in times if gs - 1e-6 <= t <= ge + 1e-6))
                method_degenerate = bool(n_estar >= len(frames) - 1)
                try:
                    pin_cpu, _ = prompt_inputs_for(V, proc, frames, sfps, str(ex["caption"]), args.prompt_style)
                    pin = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in pin_cpu.items()}
                    comp, cmask, lens = sample_group(model, pin, G, args.max_new_tokens, args.temperature,
                                                     question_seed(args.seed, ri), eos_ids, pad_id)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    n_skipped += 1
                    print(f"  [oom-gen] skip row={ri}", flush=True)
                    _append_jsonl(trace_path, {"step": si + 1, "skip": True, "row_index": ri,
                                               "qid": ex.get("qid"), "error": "oom in generation"})
                    continue
                t_gen += time.time() - tb
                texts = [tok.decode(comp[i, : int(lens[i])].tolist(), skip_special_tokens=True) for i in range(G)]
                rw = compute_rewards(texts, gs, ge, float(dur), parse_fn)
                adv = group_advantages(rw["reward"], args.adv_norm)
                truncated = [bool(int(lens[i]) == comp.shape[1] and int(comp[i, -1]) not in eos_ids) for i in range(G)]
                qs.append({"ri": ri, "ex": ex, "pin": pin, "comp": comp, "mask": cmask, "lens": lens,
                           "adv": torch.tensor(adv, dtype=torch.float32, device=comp.device), "rw": rw,
                           "texts": texts, "dur": float(dur), "T": len(frames), "n_estar": n_estar,
                           "method_degenerate": method_degenerate, "truncated": truncated,
                           "P": int(pin["input_ids"].shape[1])})
                n_method_degenerate += int(method_degenerate)
            n_q_done += len(step_rows)
            if n_q_done >= 50 and n_skipped / max(1, n_q_done) > args.max_skip_frac:
                raise SystemExit(f"STOP {n_skipped}/{n_q_done} questions skipped (> --max-skip-frac "
                                 f"{args.max_skip_frac}); check videos/decoder")
            if not qs:
                print(f"  [step {si + 1}] all questions skipped -> no update", flush=True)
                continue

            # ---------------- phase 2: loss + backward
            step_tokens = int(sum(int(q["mask"].sum()) for q in qs))
            tc = time.time()
            stats = {"kl_sum": 0.0, "pg_sum": 0.0, "loss_opt": 0.0, "tok": 0}
            oom = False
            try:
                opt.zero_grad(set_to_none=True)
                for q in qs:
                    q_tokens = int(q["mask"].sum())
                    for c0 in range(0, G, sb):
                        sl = slice(c0, min(G, c0 + sb))
                        cm = q["mask"][sl]
                        Lc = int(cm.sum(1).max())
                        cids, cm = q["comp"][sl, :Lc], cm[:, :Lc]
                        inp, P = build_score_inputs(q["pin"], cids, cm, pad_id)
                        ref = None
                        if args.beta != 0.0:
                            model.eval()
                            with torch.no_grad(), model.disable_adapter():
                                ref = completion_logps(model, inp, P, Lc, args.temperature, keep_name)
                            model.train()
                        logp = completion_logps(model, inp, P, Lc, args.temperature, keep_name)
                        per, kl = grpo_per_token_loss(logp, ref, q["adv"][sl], args.beta)
                        loss = chunk_loss(per, cm, args.loss_type, step_tokens=step_tokens, q_tokens=q_tokens,
                                          n_seq_question=G, ga_div=args.ga_loss_divisor)
                        loss.backward()
                        m = cm.to(per.dtype)
                        stats["loss_opt"] += float(loss.detach())
                        stats["tok"] += int(cm.sum())
                        stats["pg_sum"] += float((-(q["adv"][sl].unsqueeze(1)) * m).sum())
                        if kl is not None:
                            stats["kl_sum"] += float((kl.detach() * m).sum())
                        del inp, logp, per, kl, loss, ref
            except torch.cuda.OutOfMemoryError:
                oom = True
                model.train()                     # the OOM may have hit inside the eval-mode reference pass
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"  [oom-train] step {si + 1}: update skipped", flush=True)
            if oom:
                _append_jsonl(trace_path, {"step": si + 1, "skip": True, "error": "oom in loss/backward",
                                           "row_index": [q["ri"] for q in qs]})
                continue
            lr_used = float(sched.get_last_lr()[0])
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            t_train = time.time() - tc
            kl_mean = stats["kl_sum"] / max(1, stats["tok"])
            if kl_first is None:
                kl_first = kl_mean
                flag = "OK" if (si > 0 or kl_mean < 1e-3) else "WARN (LoRA B=0 at step 1 => expected ~0)"
                print(f"[check] first-step token-mean KL(policy||ref) = {kl_mean:.3e} {flag}", flush=True)

            all_r = np.concatenate([q["rw"]["reward"] for q in qs])
            all_f = np.concatenate([q["rw"]["fmt"] for q in qs])
            all_i = np.concatenate([q["rw"]["iou"] for q in qs])
            rec = {
                "step": si + 1, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "row_index": [q["ri"] for q in qs], "qid": [q["ex"].get("qid") for q in qs],
                "pair_id": [q["ex"].get("pair_id") for q in qs], "video": [q["ex"].get("video") for q in qs],
                "gt": [[float(q["ex"]["gs"]), float(q["ex"]["ge"])] for q in qs],
                "rewards": [[round(float(x), 6) for x in q["rw"]["reward"]] for q in qs],
                "fmt": [[int(x) for x in q["rw"]["fmt"]] for q in qs],
                "iou": [[round(float(x), 6) for x in q["rw"]["iou"]] for q in qs],
                "pred": [q["rw"]["pred"] for q in qs],
                "adv": [[round(float(x), 6) for x in q["adv"].tolist()] for q in qs],
                "completion_len": [[int(x) for x in q["lens"].tolist()] for q in qs],
                "truncated": [q["truncated"] for q in qs],
                "n_unique": [len(set(q["texts"])) for q in qs],
                "texts": [[t[:120] for t in q["texts"]] for q in qs],
                "prompt_tokens": [q["P"] for q in qs], "frames": [q["T"] for q in qs],
                "n_estar": [q["n_estar"] for q in qs], "method_degenerate": [q["method_degenerate"] for q in qs],
                "mean_reward": float(all_r.mean()), "mean_iou": float(all_i.mean()), "format_rate": float(all_f.mean()),
                "frac_zero_std_groups": float(np.mean([float(np.std(q["rw"]["reward"]) == 0.0) for q in qs])),
                "kl": kl_mean, "pg_term_token_mean": stats["pg_sum"] / max(1, stats["tok"]),
                "loss": stats["loss_opt"], "loss_unscaled": stats["loss_opt"] * float(args.ga_loss_divisor),
                "step_tokens": step_tokens, "lr": lr_used, "grad_norm": gnorm,
                "sec_step": round(time.time() - t_step, 3), "sec_decode_wait": round(t_dec, 3),
                "sec_generate": round(t_gen, 3), "sec_train": round(t_train, 3),
                "gpu_mem_max_gb": (round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2) if cuda else 0.0),
                "elapsed": round(time.time() - t0, 1), "n_skipped_total": n_skipped,
            }
            _append_jsonl(trace_path, rec)
            print(f"  step={si + 1}/{len(steps)} R={rec['mean_reward']:.3f} IoU={rec['mean_iou']:.3f} "
                  f"fmt={rec['format_rate']:.3f} KL={kl_mean:.4f} loss={rec['loss']:.4f} gnorm={gnorm:.3f} "
                  f"lr={lr_used:.2e} uniq={rec['n_unique']} {rec['sec_step']:.1f}s mem={rec['gpu_mem_max_gb']}G",
                  flush=True)
            del qs
            if args.save_every and (si + 1) % args.save_every == 0:
                save_state(si + 1)
                print(f"  [ckpt] step={si + 1} -> {args.out}", flush=True)
          finally:
            step_done = si + 1     # also for skipped steps (all-decode-fail / OOM): they are consumed, not retried
    finally:
        pref.close()

    save_state(step_done)
    ad = os.path.join(args.out, "adapter_model.safetensors")
    ad_sha = sha16_file(ad) if os.path.isfile(ad) else None
    with open(os.path.join(args.out, "DONE.txt"), "w", encoding="utf-8") as f:
        f.write(f"step={step_done}\nutc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\nadapter_sha16={ad_sha}\n")
    manifest.update({"status": "done", "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "optimizer_steps_done": step_done, "questions_done": n_q_done, "questions_skipped": n_skipped,
                     "questions_method_degenerate_estar": n_method_degenerate, "first_step_kl": kl_first,
                     "elapsed_seconds": time.time() - t0, "adapter_sha16": ad_sha})
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    print(f"[done] adapter={args.out} sha16={ad_sha} steps={step_done}/{len(steps)} questions={n_q_done} "
          f"skipped={n_skipped} elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
