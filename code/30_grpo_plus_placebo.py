#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""30_grpo_plus_placebo.py -- rebuttal B5: native IoU-GRPO (29_) + our placebo-controlled evidence term, same question.

Objective per optimizer step (4 scheduled questions):
    J  =  J_GRPO  +  lam * J_ev
  J_GRPO : EXACTLY 29_grpo_native_iou.py (sha16 7a762e2306146777, imported by path, sha pinned): G=8 Gumbel-max rollouts,
           reward fmt+IoU, group-z advantage, k3 KL (beta .04) to the frozen base, dapo aggregation, 1/GA (= /4).
  J_ev   : EXACTLY the per-example loss of our method's trainer 27_papo_train_qcdro_v3seln.py (sha16 98be87f7e92e6ed4)
           with the main-arm settings (--objective groupdro --npole --dro-group-standardize --alpha .5 --beta .1
           --rand-hinge 1 --rand-ratio 1.25 --rand-slack .10), evaluated on the SAME scheduled question and with the SAME
           scheduled operator pair (schedule item ops of V.build_schedule_v2, seed, epoch 0, row_index):
             clean / ref_clean,  per op: E*-edit (corrupt_op_v2, keyed rng "shufperm") / ref,
                                          RAND-edit = same op on |E*| random NON-evidence frames (base._rand_nonev with
                                          keyed rng "npole", shuffle perm rng "shufperm_n") / ref
             L_op = softplus(-b[(c-c0)-(e-e0)]) + softplus(-b[(n-n0)-(e-e0)]) + h*relu((c-n)/tok - (ratio*(c0-n0)/tok + slack))
             L_ev(q) = sum_op w_g(op) L_op + alpha * NLL_tokenmean(GT | clean)
           (c,e,n = adapter log-prob sums of the GT answer under clean / E*-edit / RAND-edit; c0,e0,n0 = the same under
           peft disable_adapter() = frozen base, which is also the hinge reference.)  The forwards go through the trainer's
           own V.native_completion_logprob (batch 1, full logits, same processor path) and the backward through its
           V.backward_one_with_clean_proxy; only the closure that combines the scores is restated (it is a nested function
           in the trainer) and is checked bit-for-bit against V._train_scheduled_example by test_30_grpo_plus_placebo_cpu.py.
  Scaling: question q's evidence loss is backpropagated with weight lam / ga_div (ga_div = --ga-loss-divisor = 4), the
           trainer's own 1/accum convention => J_ev = mean over the step's questions of L_ev, i.e. each term enters at the
           absolute gradient scale its own trainer optimised. lam = 1 is "both objectives at native scale"; the per-step
           trace logs ||g_GRPO||, ||g_ev|| and their cosine so lam can be judged from data.

Design decisions (B5):
  * GT answer: yes. J_ev scores the GT window (V.native_completion(gs, ge, timer1)), not the sampled rollouts -- as in the
    method, and because a margin on (mostly wrong) early rollouts would reward evidence dependence of wrong answers;
    it also keeps the two gradients separable (GRPO: which answer; J_ev: why that answer).
  * GroupDRO: same groups and pre-registered gate as the trainer (product query_group/op groups; any product group with
    schedule exposure < 20 -> validity scheme; with --limit 2400 seed 0 the gate fires exactly as in the method run),
    same GroupDRO(eta .05, ema .9, clip 5), same within-group standardisation, weights normalised over the question's two
    sampled groups, weights lag one update. The method updates q once per 8 examples (accum 8); a B5 step has 4
    questions, so q is updated every --ev-dro-every 8 evidence records (= every 2 steps), keeping the method's number of
    q updates per example. Separation stats are logged at each update; the method's step-40 separation abort is NOT
    applied (q is not a B5 read-out).
  * Anchors: the trainer's --beta (.1) is the margin temperature, not an anchor; it stays (--ev-beta). The trainer's
    anchor is --alpha (clean SFT on the GT answer). GRPO's KL anchors the policy's distribution over its OWN samples to the
    base; it does not hold up the GT answer's likelihood on clean video, which is what alpha protects (without it the
    margins can be won by lowering the edited-video scores). So the default keeps alpha .5 (= the method's term
    exactly); --ev-alpha 0 drops the SFT anchor (accuracy then comes from GRPO only). The reference subtraction inside the
    margins and the hinge's base reference are part of the method and stay.
  * Memory: the GRPO chunks (--score-batch, logits trimmed) are backpropagated and freed before the evidence phase starts;
    the evidence phase holds one clean graph plus one branch graph (batch 1, as the method). Peak = max of the two
    phases, not the sum; both are logged (gpu_mem_grpo_gb / gpu_mem_ev_gb).
  * Evidence-term failures (degenerate |E*| >= T-1, shuffle with native |E*| < 2, donor decode error) skip J_ev for that
    question only (GRPO still updates on it), are logged, and abort the run if > --ev-max-skip-frac after 50 questions.
    All edits are built on CPU before any evidence forward, so a failure never leaves a half-accumulated gradient.
  * --lam 0 runs no evidence code at all (no donor decode, no extra forward) and is bit-identical to 29_ (tested).

Everything else (model/LoRA init, native video path, prompt, schedule order, sampler, optimiser, warmup, clipping,
resume/manifest/trace files) is 29_'s. Output files keep 29_'s names (grpo_trace.jsonl, grpo_native_manifest.json,
resume_state.pt, ...); the manifest's "trainer" field is "grpo_plus_placebo_v1" and carries both sha16 pins.

Usage:  python 30_grpo_plus_placebo.py --pairs qcdro_v2_pairs_train.jsonl --limit 2400 --seed 0 --lam 1.0 --out <dir> [--resume]
        ... --smoke (3 questions, 1 step)   ... --dry-shapes
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRPO_FILE = "29_grpo_native_iou.py"
GRPO_SHA16 = "e94329d0b9318489"
TRAINER_NAME = "grpo_plus_placebo_v1"
QUERY_GROUP_KEY = "query_group"


def _load_grpo_module(path: str):
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    spec = importlib.util.spec_from_file_location("grpo29_b5", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["grpo29_b5"] = mod
    spec.loader.exec_module(mod)
    return mod


G29_PATH = os.path.join(HERE, GRPO_FILE)
G29 = _load_grpo_module(G29_PATH)
sha16_file = G29.sha16_file
sha256_file = G29.sha256_file


def load_trainer(path, expect_sha16):          # module-level indirection (tests patch these two)
    return G29.load_trainer(path, expect_sha16)


def load_parser(path, expect_sha16):
    return G29.load_parser(path, expect_sha16)


# ============================================================================ evidence term (trainer reuse)
def ev_cfg_from_args(args) -> SimpleNamespace:
    return SimpleNamespace(beta=float(args.ev_beta), alpha=float(args.ev_alpha), npole=bool(args.ev_npole),
                           npole_symmetric=bool(args.ev_npole_symmetric), rand_hinge=float(args.ev_rand_hinge),
                           rand_ratio=float(args.ev_rand_ratio), rand_slack=float(args.ev_rand_slack))


def trainer_equivalent_args(args) -> dict:
    """The 27_ command-line values this evidence term corresponds to (recorded in the manifest)."""
    return {"objective": args.ev_objective, "npole": bool(args.ev_npole), "npole_symmetric": bool(args.ev_npole_symmetric),
            "dro_group_standardize": bool(args.ev_dro_group_standardize), "alpha": args.ev_alpha, "beta": args.ev_beta,
            "rand_hinge": args.ev_rand_hinge, "rand_ratio": args.ev_rand_ratio, "rand_slack": args.ev_rand_slack,
            "dro_eta": args.ev_dro_eta, "dro_ema_decay": args.ev_dro_ema_decay, "dro_update_clip": args.ev_dro_update_clip,
            "group_scheme": args.ev_group_scheme, "min_group_exposure": args.ev_min_group_exposure,
            "ops": G29.METHOD_OPS, "ops_per_example": 2, "seed": args.seed, "limit": args.limit,
            "prompt_style": args.prompt_style, "answer_style": args.answer_style,
            "accum_equivalent": f"grad weight lam/ga_div = {args.lam}/{args.ga_loss_divisor} per question"}


def evidence_plan(V, ex, frames, times, sample_fps, donor_frames, ops, *, seed, epoch, row_index, npole,
                  answer_style, prompt_style) -> dict:
    """CPU half of the method's per-example evidence term, built from already-decoded frames.
    == V._prepare_example (E* from timestamps, nearest-frame fallback, degenerate guard, donor resize, GT completion,
    prompt) + the edit construction of V._train_scheduled_example (same keyed rngs, same operators). Raises ValueError
    exactly where the trainer would (degenerate E*, shuffle with |E*| < 2)."""
    caption = str(ex["caption"])
    gs, ge = float(ex["gs"]), float(ex["ge"])
    estar = np.asarray([i for i, t in enumerate(times) if gs - 1e-6 <= t <= ge + 1e-6], dtype=int)
    if len(estar) == 0:
        j = int(np.argmin([abs(t - 0.5 * (gs + ge)) for t in times]))
        estar = np.asarray([j], dtype=int)
    if len(estar) >= len(frames) - 1:
        raise ValueError(f"degenerate evidence set |E*|={len(estar)} T={len(frames)}")
    if donor_frames.shape[1:3] != frames.shape[1:3]:
        H, W = int(frames.shape[1]), int(frames.shape[2])
        donor_frames = np.stack([V.cv2.resize(np.ascontiguousarray(f), (W, H)) for f in donor_frames])
    edits = []
    for op in ops:
        op_rng = V._stable_rng(seed, epoch, row_index, op, "shufperm")
        corrupt_e = V.corrupt_op_v2(frames, estar, donor_frames, op, rng=op_rng)
        corrupt_n, nonev_idx = None, None
        if npole:
            local_rng = V._stable_rng(seed, epoch, row_index, op, "npole")
            nonev_idx = V.base._rand_nonev(estar, len(frames), local_rng)
            n_rng = V._stable_rng(seed, epoch, row_index, op, "shufperm_n")
            corrupt_n = V.corrupt_op_v2(frames, nonev_idx, donor_frames, op, rng=n_rng)
        edits.append({"op": op, "corrupt_e": corrupt_e, "corrupt_n": corrupt_n,
                      "nonev_idx": None if nonev_idx is None else [int(x) for x in nonev_idx]})
    return {"frames": frames, "sample_fps": float(sample_fps), "estar": [int(x) for x in estar],
            "completion": V.native_completion(gs, ge, answer_style),
            "prompt_text": V.NATIVE_PROMPTS[prompt_style].format(caption), "edits": edits, "ops": list(ops)}


def evidence_backward(V, model, proc, plan, device, *, group_names, aggregation, cfg, grad_scale: float) -> dict:
    """GPU half: forwards in the trainer's order (clean, ref_clean, per op: E*, ref_E*, RAND, ref_RAND), the trainer's
    loss closure, V.backward_one_with_clean_proxy per op, alpha SFT on the clean proxy, one clean backward.
    Gradients are accumulated into the LoRA params with weight grad_scale (trainer: 1/accum)."""
    import torch
    import torch.nn.functional as F
    cache: dict = {}

    def completion_logp(fr):
        return V.native_completion_logprob(model, proc, fr, plan["sample_fps"], plan["prompt_text"],
                                           plan["completion"], device, cache)

    clean_score, clean_tokens = completion_logp(plan["frames"])
    with torch.no_grad(), model.disable_adapter():
        ref_clean, _ = completion_logp(plan["frames"])

    clean_gradient = torch.zeros_like(clean_score)
    group_losses: dict = {}
    group_margins: dict = {}
    branches = []
    reported_total = 0.0
    tok = float(max(1, clean_tokens))
    for e in plan["edits"]:
        op = e["op"]
        g = group_names[op]
        corrupt_score, _ = completion_logp(e["corrupt_e"])
        with torch.no_grad(), model.disable_adapter():
            ref_corrupt, _ = completion_logp(e["corrupt_e"])
        nonevidence_score = None
        ref_nonevidence = None
        if cfg.npole:
            nonevidence_score, _ = completion_logp(e["corrupt_n"])
            with torch.no_grad(), model.disable_adapter():
                ref_nonevidence, _ = completion_logp(e["corrupt_n"])

        def build_loss(clean_proxy):           # == the closure in V._train_scheduled_example (27_ L471-492)
            margin = cfg.beta * ((clean_proxy - ref_clean) - (corrupt_score - ref_corrupt))
            value = F.softplus(-margin)
            if cfg.npole:
                assert nonevidence_score is not None and ref_nonevidence is not None
                if cfg.npole_symmetric:
                    margin_n = cfg.beta * (
                        (clean_proxy - ref_clean) - (nonevidence_score - ref_nonevidence)
                    )
                else:
                    margin_n = cfg.beta * (
                        (nonevidence_score - ref_nonevidence) - (corrupt_score - ref_corrupt)
                    )
                value = value + F.softplus(-margin_n)
                if cfg.rand_hinge > 0:
                    tok_ = float(max(1, clean_tokens))
                    drop_rand_arm = (clean_proxy - nonevidence_score) / tok_
                    drop_rand_base = (ref_clean - ref_nonevidence) / tok_
                    excess = drop_rand_arm - (cfg.rand_ratio * drop_rand_base + cfg.rand_slack)
                    value = value + cfg.rand_hinge * F.relu(excess)
            return value

        weight = aggregation[g]
        detached_loss, proxy_gradient = V.backward_one_with_clean_proxy(
            clean_score, build_loss, weight=weight * grad_scale
        )
        clean_gradient = clean_gradient + proxy_gradient
        group_losses[g] = float(detached_loss.float().cpu())
        primary_margin = cfg.beta * (
            (clean_score.detach() - ref_clean) - (corrupt_score.detach() - ref_corrupt)
        )
        group_margins[g] = float(primary_margin.float().cpu())
        reported_total += weight * group_losses[g]

        # ---- read-outs (detached; not part of the loss)
        c, c0 = float(clean_score.detach().float()), float(ref_clean.float())
        s_e, s_e0 = float(corrupt_score.detach().float()), float(ref_corrupt.float())
        br = {"op": op, "group": g, "weight": float(weight), "loss": group_losses[g], "margin": group_margins[g],
              "drop_e_tok": (c - s_e) / tok, "drop_e_base_tok": (c0 - s_e0) / tok}
        if cfg.npole:
            s_n, s_n0 = float(nonevidence_score.detach().float()), float(ref_nonevidence.float())
            if cfg.npole_symmetric:
                m_n = cfg.beta * ((c - c0) - (s_n - s_n0))
            else:
                m_n = cfg.beta * ((s_n - s_n0) - (s_e - s_e0))
            d_ra, d_rb = (c - s_n) / tok, (c0 - s_n0) / tok
            exc = d_ra - (cfg.rand_ratio * d_rb + cfg.rand_slack)
            br.update({"margin_n": m_n, "drop_rand_tok": d_ra, "drop_rand_base_tok": d_rb,
                       "selectivity_tok": (c - s_e) / tok - d_ra, "hinge_excess": exc,
                       "hinge_active": bool(cfg.rand_hinge > 0 and exc > 0)})
        branches.append(br)

    clean_nll = float((-clean_score.detach() / max(1, clean_tokens)).float().cpu())
    if cfg.alpha > 0:
        clean_gradient = clean_gradient + torch.full_like(
            clean_score, -cfg.alpha * grad_scale / max(1, clean_tokens)
        )
        reported_total += cfg.alpha * clean_nll
    clean_score.backward(clean_gradient)
    return {"loss": reported_total, "clean_nll": clean_nll, "group_losses": group_losses,
            "group_margins": group_margins, "aggregation_weights": dict(aggregation), "branches": branches,
            "clean_tokens": int(clean_tokens), "grad_scale": float(grad_scale)}


def build_groups(V, rows, schedule, *, scheme: str, min_exposure: int, ops):
    """== trainer main() L623-648: product groups, starvation gate on the schedule only, fallback to validity."""
    exposure_product = V.group_exposure(schedule, rows, scheme="product", query_group_key=QUERY_GROUP_KEY)
    exposure_validity = V.group_exposure(schedule, rows, scheme="validity", query_group_key=QUERY_GROUP_KEY)
    fallback = None
    if scheme == "product" and exposure_product and min(exposure_product.values()) < min_exposure:
        starved = {g: n for g, n in exposure_product.items() if n < min_exposure}
        fallback = {"from": "product", "to": "validity",
                    "rule": f"min product-group exposure < --min-group-exposure({min_exposure})",
                    "starved_groups": starved, "evaluated_on": "schedule only (no training signal observed)"}
        scheme = "validity"
    exposure_groups = exposure_product if scheme == "product" else exposure_validity
    names_by_row = [{op: V.group_of(r, op, scheme=scheme, query_group_key=QUERY_GROUP_KEY) for op in ops}
                    for r in rows]
    return {"scheme": scheme, "fallback": fallback, "exposure": exposure_groups,
            "exposure_product": exposure_product, "exposure_validity": exposure_validity,
            "groups": sorted(exposure_groups.keys()), "names_by_row": names_by_row}


def dro_update_from_records(dro, records, run_mean, run_n, standardize: bool) -> dict:
    """== the GroupDRO block of the trainer's optimizer_step() (27_ L828-847) on one window of records."""
    import torch
    grouped = defaultdict(list)
    for record in records:
        for g, value in record["group_losses"].items():
            grouped[g].append(float(value))
    signal = {}
    for g, values in grouped.items():
        raw = sum(values) / len(values)
        if standardize:
            n = run_n[g]
            mu = run_mean[g]
            centered = raw - mu if n > 0 else 0.0
            run_mean[g] = raw if n == 0 else mu + (raw - mu) / (n + 1)
            run_n[g] = n + 1
            signal[g] = torch.tensor(centered, dtype=torch.float32)
        else:
            signal[g] = torch.tensor(raw, dtype=torch.float32)
    dro.update(signal)
    return {g: float(v) for g, v in signal.items()}


def separation_stats(records, softplus0_pin: float):
    """== the trainer's group-loss separation read-out (27_ L870-884)."""
    window_groups = defaultdict(list)
    for record in records:
        for g, value in record["group_losses"].items():
            window_groups[g].append(float(value))
    if len(window_groups) < 2:
        return None
    means = {g: sum(v) / len(v) for g, v in window_groups.items()}
    spread = max(means.values()) - min(means.values())
    within = float(np.mean([np.std(v) for v in window_groups.values() if len(v) > 1] or [0.0]))
    return {"group_loss_spread": round(spread, 6), "within_group_sd": round(within, 6),
            "separation_ratio": round(spread / within, 4) if within > 1e-12 else None,
            "softplus0_pin": round(abs(float(np.mean(list(means.values()))) - softplus0_pin), 6)}


def grad_snapshot(params):
    return [None if p.grad is None else p.grad.detach().clone() for p in params]


def term_grad_norms(params, g_first):
    """||g_first||, ||g_now - g_first||, cos(g_first, g_now - g_first) over all trainable params (float64)."""
    import torch
    n0 = n1 = dot = None
    for p, a in zip(params, g_first):
        b = p.grad
        if a is None and b is None:
            continue
        a64 = a.double() if a is not None else torch.zeros_like(b, dtype=torch.float64)
        b64 = b.double() if b is not None else torch.zeros_like(a, dtype=torch.float64)
        d = b64 - a64
        x0, x1, xd = (a64 * a64).sum(), (d * d).sum(), (a64 * d).sum()
        n0 = x0 if n0 is None else n0 + x0
        n1 = x1 if n1 is None else n1 + x1
        dot = xd if dot is None else dot + xd
    if n0 is None:
        return 0.0, 0.0, None
    g0, g1 = float(n0.sqrt()), float(n1.sqrt())
    cos = float(dot) / (g0 * g1) if g0 > 0 and g1 > 0 else None
    return g0, g1, cos


def ev_skip_bucket(reason: str) -> str:
    if reason.startswith("donor decode"):
        return "donor_decode"
    if "degenerate" in reason:
        return "degenerate_estar"
    if "shuffle needs" in reason:
        return "shuffle_estar_lt2"
    return "other"


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def ev_step_summary(qs) -> dict:
    recs = [q["ev_rec"] for q in qs if q.get("ev_rec") is not None]
    br = [b for r in recs for b in r["branches"]]
    out = {"ev_n": len(recs), "ev_loss_mean": _mean(r["loss"] for r in recs),
           "ev_clean_nll_mean": _mean(r["clean_nll"] for r in recs),
           "ev_margin_mean": _mean(b["margin"] for b in br), "ev_margin_n_mean": _mean(b.get("margin_n") for b in br),
           "ev_drop_e_tok_mean": _mean(b["drop_e_tok"] for b in br),
           "ev_drop_e_base_tok_mean": _mean(b["drop_e_base_tok"] for b in br),
           "ev_drop_rand_tok_mean": _mean(b.get("drop_rand_tok") for b in br),
           "ev_drop_rand_base_tok_mean": _mean(b.get("drop_rand_base_tok") for b in br),
           "ev_selectivity_tok_mean": _mean(b.get("selectivity_tok") for b in br),
           "ev_hinge_excess_mean": _mean(b.get("hinge_excess") for b in br),
           "ev_hinge_active_frac": _mean(float(b.get("hinge_active", False)) for b in br),
           "ev_loss_opt": float(sum(r["loss"] * r["grad_scale"] for r in recs))}
    return out


# ============================================================================ main
def build_argparser():
    ap = G29.build_argparser()                 # every 29_ flag, same defaults
    ap.description = __doc__.split("\n")[0]
    ev = ap.add_argument_group("B5 evidence term (defaults = method main arm)")
    ev.add_argument("--lam", type=float, default=1.0, help="weight of J_ev; 0 = exactly 29_ (no evidence code runs)")
    ev.add_argument("--ev-objective", choices=["groupdro", "mean"], default="groupdro")
    ev.add_argument("--ev-beta", type=float, default=0.1, help="margin temperature (27_ --beta)")
    ev.add_argument("--ev-alpha", type=float, default=0.5, help="clean SFT anchor on the GT answer (27_ --alpha); 0 drops it")
    ev.add_argument("--ev-npole", type=int, choices=[0, 1], default=1, help="RAND (non-evidence) placebo branch (27_ --npole)")
    ev.add_argument("--ev-npole-symmetric", action="store_true")
    ev.add_argument("--ev-rand-hinge", type=float, default=1.0)
    ev.add_argument("--ev-rand-ratio", type=float, default=1.25)
    ev.add_argument("--ev-rand-slack", type=float, default=0.10)
    ev.add_argument("--ev-group-scheme", choices=["product", "validity"], default="product")
    ev.add_argument("--ev-min-group-exposure", type=int, default=20)
    ev.add_argument("--ev-dro-eta", type=float, default=0.05)
    ev.add_argument("--ev-dro-ema-decay", type=float, default=0.9)
    ev.add_argument("--ev-dro-update-clip", type=float, default=5.0)
    ev.add_argument("--ev-dro-group-standardize", type=int, choices=[0, 1], default=1)
    ev.add_argument("--ev-dro-every", type=int, default=8,
                    help="evidence records per GroupDRO update (method: --accum 8 examples per update)")
    ev.add_argument("--ev-max-skip-frac", type=float, default=0.10,
                    help="abort if more than this fraction of questions get no evidence term (after 50)")
    ev.add_argument("--log-term-gnorm", type=int, choices=[0, 1], default=1,
                    help="log ||g_GRPO||, ||g_ev||, cos (one clone of the LoRA grads per step)")
    ev.add_argument("--grpo-sha16", default=GRPO_SHA16, help="pin of 29_grpo_native_iou.py; '' disables")
    return ap


def _ev_ident(args) -> dict:
    keys = ("lam", "ev_objective", "ev_beta", "ev_alpha", "ev_npole", "ev_npole_symmetric", "ev_rand_hinge",
            "ev_rand_ratio", "ev_rand_slack", "ev_group_scheme", "ev_min_group_exposure", "ev_dro_eta",
            "ev_dro_ema_decay", "ev_dro_update_clip", "ev_dro_group_standardize", "ev_dro_every", "answer_style")
    return {k: getattr(args, k) for k in keys}


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
    if args.lam < 0:
        ap.error("--lam must be >= 0")
    if args.ev_beta <= 0 or args.ev_alpha < 0:
        ap.error("require ev_beta>0, ev_alpha>=0 (27_ constraint)")
    if args.ev_rand_hinge < 0 or args.ev_rand_ratio <= 0 or args.ev_rand_slack < 0:
        ap.error("require ev_rand_hinge>=0, ev_rand_ratio>0, ev_rand_slack>=0 (27_ constraint)")
    if args.ev_rand_hinge > 0 and not args.ev_npole:
        ap.error("--ev-rand-hinge needs --ev-npole 1 (reuses the non-evidence corruption pass)")
    if args.ev_dro_every <= 0:
        ap.error("--ev-dro-every must be positive")
    use_ev = args.lam > 0

    g29_sha = sha16_file(G29_PATH)
    if args.grpo_sha16 and g29_sha != args.grpo_sha16:
        raise SystemExit(f"STOP {GRPO_FILE} sha16={g29_sha} != expected {args.grpo_sha16} (this file reuses its "
                         "GRPO step by name; re-audit before overriding --grpo-sha16)")
    V, trainer_sha = load_trainer(args.trainer, args.trainer_sha16 or None)
    parse_fn, P4, parser_sha = load_parser(args.parser_module, args.parser_sha16 or None)
    if args.prompt_style not in V.NATIVE_PROMPTS:
        ap.error(f"--prompt-style must be one of {sorted(V.NATIVE_PROMPTS)}")
    if getattr(P4, "PROMPTS", {}).get(args.prompt_style) != V.NATIVE_PROMPTS[args.prompt_style]:
        raise SystemExit("STOP trainer and evaluator disagree on the prompt text for " + args.prompt_style)
    if args.model is None:
        args.model = V.MODEL
    self_sha = sha16_file(os.path.abspath(__file__))

    # ---------------- questions: identical rows + order as the method run (29_)
    rows = [json.loads(line) for line in open(args.pairs, encoding="utf-8") if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    pairs_sha = sha256_file(args.pairs)
    requested_ops = V.parse_ops_v2(G29.METHOD_OPS)
    schedule = V.build_schedule_v2(rows, epochs=1, requested_ops=requested_ops, ops_per_example=2,
                                   seed=args.seed, query_group_key=QUERY_GROUP_KEY)
    if args.expected_schedule_fingerprint and args.expected_schedule_fingerprint != schedule.fingerprint:
        raise SystemExit(f"STOP schedule fingerprint {schedule.fingerprint} != expected "
                         f"{args.expected_schedule_fingerprint}")
    order = [int(it.row_index) for it in schedule.items]
    item_by_row = {int(it.row_index): it for it in schedule.items}
    if len(item_by_row) != len(order):
        raise SystemExit("STOP schedule repeats a row (epochs must be 1): evidence ops would be ambiguous")
    if args.smoke:
        order = order[:3]
    steps = G29.group_steps(order, args.questions_per_step)
    if args.max_steps and args.max_steps > 0:
        steps = steps[: args.max_steps]
    n_q_planned = sum(len(s) for s in steps)
    grp = build_groups(V, rows, schedule, scheme=args.ev_group_scheme, min_exposure=args.ev_min_group_exposure,
                       ops=requested_ops)
    group_names_by_row = grp["names_by_row"]
    print(f"[data] rows={len(rows)} scheduled={len(schedule.items)} fingerprint={schedule.fingerprint} "
          f"skipped_by_schedule={schedule.skipped_too_few_valid} -> {n_q_planned} questions in {len(steps)} "
          f"optimizer steps of {args.questions_per_step} x G={args.num_generations}", flush=True)
    print(f"[prov] grpo={GRPO_FILE} sha16={g29_sha} trainer={os.path.basename(args.trainer)} sha16={trainer_sha} "
          f"parser={os.path.basename(args.parser_module)} sha16={parser_sha} self sha16={self_sha} "
          f"pairs sha256={pairs_sha[:16]}", flush=True)
    print(f"[b5] lam={args.lam} evidence term {'ON' if use_ev else 'OFF (= 29_ exactly)'}; groups scheme="
          f"{grp['scheme']} (requested {args.ev_group_scheme}; fallback={'yes' if grp['fallback'] else 'no'}) "
          f"n={len(grp['groups'])} exposure={grp['exposure']}", flush=True)

    nsargs = argparse.Namespace(fps=args.fps, total_pixels=args.total_pixels, min_pixels=args.min_pixels,
                                max_frames=args.max_frames)

    def decode(ri):
        ex = rows[ri]
        return V.native_decode(os.path.join(V.base.VIDEO_DIR, str(ex["video"])), nsargs)

    def decode_bundle(ri):
        main_dec = decode(ri)                   # raises -> question skipped (29_ path)
        donor, derr = None, None
        try:
            donor = V.native_decode(os.path.join(V.base.VIDEO_DIR, str(rows[ri]["distractor"])), nsargs)[0]
        except Exception as exc:                # donor failure -> no evidence term for this question only
            derr = f"donor decode {type(exc).__name__}: {exc}"
        return main_dec, donor, derr

    if args.dry_shapes:
        G29.run_dry_shapes(V, args, rows, order, decode)
        return

    os.makedirs(args.out, exist_ok=True)
    trace_path = os.path.join(args.out, "grpo_trace.jsonl")
    manifest_path = os.path.join(args.out, "grpo_native_manifest.json")
    rs_path = os.path.join(args.out, "resume_state.pt")
    with open(os.path.join(args.out, "grpo_schedule.jsonl"), "w", encoding="utf-8") as f:
        for si, st in enumerate(steps):
            f.write(json.dumps({"step": si + 1, "row_index": st, "qid": [rows[r].get("qid") for r in st],
                                "ops": [list(item_by_row[r].ops) for r in st]}) + "\n")
    train_videos = sorted({str(rows[r]["video"]) for r in order})
    with open(os.path.join(args.out, "train_videos.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_videos) + "\n")

    # ---------------- model (same construction as 29_ / trainer main() L713-741)
    import torch
    cuda = not args.cpu_debug
    if cuda:
        device = args.gpu
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
    keep_name = G29.logits_keep_kwarg(model)
    print(f"[score] logits-trimming kwarg = {keep_name} (None -> full logits are sliced)", flush=True)
    eos_ids, pad_id = G29.gen_eos_pad(proc, model)
    tok = proc.tokenizer
    gcfg = getattr(model, "generation_config", None)
    model_gen_cfg = {k: getattr(gcfg, k, None) for k in ("do_sample", "temperature", "top_p", "top_k",
                                                          "repetition_penalty", "eos_token_id", "pad_token_id")}
    print(f"[gen] model generation_config {model_gen_cfg} -> OVERRIDDEN by: greedy argmax over logits/T + Gumbel "
          f"(T={args.temperature}, full vocab), repetition_penalty=1.0, max_new_tokens={args.max_new_tokens}, "
          f"eos={eos_ids}, pad={pad_id}, use_cache=True, eval mode", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, G29.constant_with_warmup_lambda(args.warmup_steps))

    # ---------------- evidence-term state (GroupDRO as the trainer)
    evcfg = ev_cfg_from_args(args)
    dro = None
    if use_ev and args.ev_objective == "groupdro":
        dro = V.GroupDRO(grp["groups"], eta=args.ev_dro_eta, ema_decay=args.ev_dro_ema_decay,
                         update_clip=args.ev_dro_update_clip)
    grp_run_mean = defaultdict(float)
    grp_run_n = defaultdict(int)
    dro_buf: list = []
    n_dro_updates = 0
    n_ev_done = 0
    n_ev_skipped = 0
    ev_skip_reasons: dict = defaultdict(int)
    ev_wired_checked = False

    ident = G29._ident(args, pairs_sha, schedule.fingerprint)
    ident.update(_ev_ident(args))
    step_done = 0
    t_prev = 0.0
    n_skipped = 0
    n_q_done = 0
    n_method_degenerate = 0
    if args.resume and os.path.isfile(rs_path):
        from peft import set_peft_model_state_dict
        rs = torch.load(rs_path, map_location="cpu", weights_only=False)
        if rs["ident"] != ident:
            diff = {k: (rs["ident"].get(k), ident.get(k)) for k in set(ident) | set(rs["ident"])
                    if rs["ident"].get(k) != ident.get(k)}
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
        evs = rs.get("ev") or {}
        if use_ev:
            if dro is not None and evs.get("groupdro") is not None:
                dro.load_state_dict(evs["groupdro"])
            grp_run_mean.update(evs.get("grp_run_mean", {}))
            grp_run_n.update(evs.get("grp_run_n", {}))
            dro_buf = list(evs.get("dro_buf", []))
            n_dro_updates = int(evs.get("n_dro_updates", 0))
            n_ev_done = int(evs.get("n_ev_done", 0))
            n_ev_skipped = int(evs.get("n_ev_skipped", 0))
            ev_skip_reasons.update(evs.get("ev_skip_reasons", {}))
            ev_wired_checked = bool(evs.get("ev_wired_checked", False))
        kept = G29._truncate_trace(trace_path, step_done)
        print(f"[resume] step={step_done}/{len(steps)} questions_done={n_q_done} trace_rows_kept={kept} "
              f"elapsed_prev={t_prev:.0f}s ev_done={n_ev_done} dro_updates={n_dro_updates}", flush=True)
    elif args.resume:
        print("[resume] no resume_state.pt -> fresh start", flush=True)
        G29._truncate_trace(trace_path, 0)
    else:
        G29._truncate_trace(trace_path, 0)

    t0 = time.time() - t_prev
    fwd_ev = (2 + 2 * 2 + (2 * 2 if args.ev_npole else 0))
    manifest = {
        "trainer": TRAINER_NAME, "status": "running", "args": vars(args),
        "self_sha16": self_sha, "grpo_module": GRPO_FILE, "grpo_sha16": g29_sha,
        "trainer_module": os.path.basename(args.trainer), "trainer_sha16": trainer_sha,
        "parser_module": os.path.basename(args.parser_module), "parser_sha16": parser_sha,
        "helper_sha16": {fn: sha16_file(os.path.join(HERE, fn)) for fn in
                         ("18_papo_train_operator_mix.py", "12_charades_grounding.py", "native_video_causal_probe.py",
                          "12p4a_grounding_eval_adapter.py", "operator_groupdro_pilot_helpers.py",
                          "qcdro_objective_draft.py") if os.path.isfile(os.path.join(HERE, fn))},
        "pairs": os.path.abspath(args.pairs), "pairs_sha256": pairs_sha, "limit": args.limit,
        "schedule": {"fingerprint": schedule.fingerprint, "n_scheduled": len(schedule.items),
                     "skipped_by_schedule": schedule.skipped_too_few_valid, "method_ops": G29.METHOD_OPS,
                     "operator_exposure": schedule.exposure,
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
        "objective": {"form": "J = J_GRPO(29_) + lam * J_ev(27_ per-example loss, mean over the step's questions)",
                      "lam": args.lam,
                      "grpo": {"reward": "fmt + IoU (parse_native of the native evaluator)", "adv_norm": args.adv_norm,
                               "adv_eps": G29.ADV_EPS, "adv_ddof": G29.ADV_DDOF, "loss_type": args.loss_type,
                               "ga_loss_divisor": args.ga_loss_divisor, "beta_kl": args.beta, "kl_estimator": "k3",
                               "ratio": "on-policy, exp(logp - logp.detach())", "ref": "peft disable_adapter()"},
                      "evidence": {"enabled": use_ev, "trainer_equivalent_args": trainer_equivalent_args(args),
                                   "answer": "GT window via V.native_completion (not the rollouts)",
                                   "edits": "schedule item ops; V.corrupt_op_v2 with V._stable_rng(seed, epoch, "
                                            "row_index, op, 'shufperm'|'npole'|'shufperm_n'); RAND = base._rand_nonev",
                                   "scores": "V.native_completion_logprob (batch 1, full logits), ref = disable_adapter()",
                                   "backward": "V.backward_one_with_clean_proxy per op + alpha SFT on the clean proxy",
                                   "grad_weight_per_question": (args.lam / args.ga_loss_divisor),
                                   "forwards_per_question": fwd_ev,
                                   "groups": grp["groups"], "group_scheme": grp["scheme"],
                                   "group_scheme_requested": args.ev_group_scheme,
                                   "group_scheme_fallback": grp["fallback"], "group_exposure_selected": grp["exposure"],
                                   "groupdro_update": f"every {args.ev_dro_every} evidence records, at optimizer-step end"
                                                      "; weights lag (method: every --accum 8 examples)",
                                   "separation_abort": "not applied (logged only)",
                                   "skip_policy": "J_ev skipped for the question only (GRPO still applied); "
                                                  f"abort if > {args.ev_max_skip_frac} after 50 questions"}},
        "optimizer": {"type": "torch.optim.AdamW", "lr": args.lr, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": args.weight_decay, "scheduler": f"constant_with_warmup({args.warmup_steps})",
                      "max_grad_norm": args.max_grad_norm},
        "step_definition": f"1 optimizer step = {args.questions_per_step} scheduled questions x "
                           f"G={args.num_generations} completions (+ the evidence term on each question); "
                           "micro-batch = 1 question",
        "logits_keep_kwarg": keep_name, "versions": G29._versions(),
        "gpu": (torch.cuda.get_device_name(device) if cuda else "cpu-debug"),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
              "n_method_degenerate": n_method_degenerate,
              "ev": {"groupdro": None if dro is None else dro.state_dict(), "grp_run_mean": dict(grp_run_mean),
                     "grp_run_n": dict(grp_run_n), "dro_buf": list(dro_buf), "n_dro_updates": n_dro_updates,
                     "n_ev_done": n_ev_done, "n_ev_skipped": n_ev_skipped, "ev_skip_reasons": dict(ev_skip_reasons),
                     "ev_wired_checked": ev_wired_checked}}
        tmp = rs_path + ".tmp"
        torch.save(rs, tmp)
        os.replace(tmp, rs_path)
        model.save_pretrained(args.out)       # convenience copy of the adapter at the same step

    pref = G29.Prefetcher(decode_bundle if use_ev else decode, bool(args.prefetch))
    G = int(args.num_generations)
    sb = int(args.score_batch)
    kl_first = None
    try:
        for si in range(step_done, len(steps)):
          try:                                   # (2-space level: body below kept at 29_'s indentation)
            step_rows = steps[si]
            pref.want(step_rows)
            if si + 1 < len(steps):
                pref.want(steps[si + 1])
            t_step = time.time()
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            # ---------------- phase 1: sample + reward (policy at step start)  [29_ verbatim + evidence plan]
            qs = []
            t_dec = t_gen = 0.0
            for ri in step_rows:
                ex = rows[ri]
                ta = time.time()
                try:
                    got = pref.get(ri)
                except Exception as exc:          # decode failure -> skip question (logged + counted)
                    n_skipped += 1
                    print(f"  [skip] row={ri} video={ex.get('video')} decode {type(exc).__name__}: {exc}", flush=True)
                    G29._append_jsonl(trace_path, {"step": si + 1, "skip": True, "row_index": ri,
                                                   "qid": ex.get("qid"), "error": f"{type(exc).__name__}: {exc}"})
                    continue
                if use_ev:
                    (frames, times, dur, sfps), donor, derr = got
                else:
                    frames, times, dur, sfps = got
                t_dec += time.time() - ta
                tb = time.time()
                gs, ge = float(ex["gs"]), float(ex["ge"])
                n_estar = int(sum(1 for t in times if gs - 1e-6 <= t <= ge + 1e-6))
                method_degenerate = bool(n_estar >= len(frames) - 1)
                try:
                    pin_cpu, _ = G29.prompt_inputs_for(V, proc, frames, sfps, str(ex["caption"]), args.prompt_style)
                    pin = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in pin_cpu.items()}
                    comp, cmask, lens = G29.sample_group(model, pin, G, args.max_new_tokens, args.temperature,
                                                         G29.question_seed(args.seed, ri), eos_ids, pad_id)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    n_skipped += 1
                    print(f"  [oom-gen] skip row={ri}", flush=True)
                    G29._append_jsonl(trace_path, {"step": si + 1, "skip": True, "row_index": ri,
                                                   "qid": ex.get("qid"), "error": "oom in generation"})
                    continue
                t_gen += time.time() - tb
                texts = [tok.decode(comp[i, : int(lens[i])].tolist(), skip_special_tokens=True) for i in range(G)]
                rw = G29.compute_rewards(texts, gs, ge, float(dur), parse_fn)
                adv = G29.group_advantages(rw["reward"], args.adv_norm)
                truncated = [bool(int(lens[i]) == comp.shape[1] and int(comp[i, -1]) not in eos_ids) for i in range(G)]
                qs.append({"ri": ri, "ex": ex, "pin": pin, "comp": comp, "mask": cmask, "lens": lens,
                           "adv": torch.tensor(adv, dtype=torch.float32, device=comp.device), "rw": rw,
                           "texts": texts, "dur": float(dur), "T": len(frames), "n_estar": n_estar,
                           "method_degenerate": method_degenerate, "truncated": truncated,
                           "P": int(pin["input_ids"].shape[1])})
                n_method_degenerate += int(method_degenerate)
                if use_ev:                         # CPU only: E*, edits (no torch RNG consumed)
                    item = item_by_row[ri]
                    q = qs[-1]
                    q["ops"] = list(item.ops)
                    q["ev"], q["ev_skip"], q["ev_rec"] = None, None, None
                    if derr is not None:
                        q["ev_skip"] = derr
                    else:
                        try:
                            q["ev"] = evidence_plan(V, ex, frames, times, sfps, donor, item.ops, seed=args.seed,
                                                    epoch=item.epoch, row_index=ri, npole=bool(args.ev_npole),
                                                    answer_style=args.answer_style, prompt_style=args.prompt_style)
                        except ValueError as exc:
                            q["ev_skip"] = f"ValueError: {exc}"
                    if q["ev_skip"] is not None:
                        n_ev_skipped += 1
                        ev_skip_reasons[ev_skip_bucket(q["ev_skip"])] += 1
                        print(f"  [ev-skip] row={ri} ops={q['ops']} {q['ev_skip']}", flush=True)
            n_q_done += len(step_rows)
            if n_q_done >= 50 and n_skipped / max(1, n_q_done) > args.max_skip_frac:
                raise SystemExit(f"STOP {n_skipped}/{n_q_done} questions skipped (> --max-skip-frac "
                                 f"{args.max_skip_frac}); check videos/decoder")
            if use_ev and n_q_done >= 50 and (n_ev_skipped + n_skipped) / max(1, n_q_done) > args.ev_max_skip_frac:
                raise SystemExit(f"STOP evidence term missing on {n_ev_skipped + n_skipped}/{n_q_done} questions "
                                 f"(> --ev-max-skip-frac {args.ev_max_skip_frac}); reasons={dict(ev_skip_reasons)}")
            if not qs:
                print(f"  [step {si + 1}] all questions skipped -> no update", flush=True)
                continue

            # ---------------- phase 2a: GRPO loss + backward  [29_ verbatim]
            step_tokens = int(sum(int(q["mask"].sum()) for q in qs))
            tc = time.time()
            stats = {"kl_sum": 0.0, "pg_sum": 0.0, "loss_opt": 0.0, "tok": 0}
            oom = False
            gn_grpo = gn_ev = cos_ge = None
            mem_grpo = mem_ev = None
            t_ev = 0.0
            try:
                opt.zero_grad(set_to_none=True)
                for q in qs:
                    q_tokens = int(q["mask"].sum())
                    for c0 in range(0, G, sb):
                        sl = slice(c0, min(G, c0 + sb))
                        cm = q["mask"][sl]
                        Lc = int(cm.sum(1).max())
                        cids, cm = q["comp"][sl, :Lc], cm[:, :Lc]
                        inp, P = G29.build_score_inputs(q["pin"], cids, cm, pad_id)
                        ref = None
                        if args.beta != 0.0:
                            model.eval()
                            with torch.no_grad(), model.disable_adapter():
                                ref = G29.completion_logps(model, inp, P, Lc, args.temperature, keep_name)
                            model.train()
                        logp = G29.completion_logps(model, inp, P, Lc, args.temperature, keep_name)
                        per, kl = G29.grpo_per_token_loss(logp, ref, q["adv"][sl], args.beta)
                        loss = G29.chunk_loss(per, cm, args.loss_type, step_tokens=step_tokens, q_tokens=q_tokens,
                                              n_seq_question=G, ga_div=args.ga_loss_divisor)
                        loss.backward()
                        m = cm.to(per.dtype)
                        stats["loss_opt"] += float(loss.detach())
                        stats["tok"] += int(cm.sum())
                        stats["pg_sum"] += float((-(q["adv"][sl].unsqueeze(1)) * m).sum())
                        if kl is not None:
                            stats["kl_sum"] += float((kl.detach() * m).sum())
                        del inp, logp, per, kl, loss, ref

                # ---------------- phase 2b: evidence term (same questions, same policy, model.train())
                if use_ev and any(q.get("ev") is not None for q in qs):
                    te = time.time()
                    if cuda:
                        mem_grpo = torch.cuda.max_memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    g_first = grad_snapshot(params) if args.log_term_gnorm else None
                    model.train()
                    for q in qs:
                        if q.get("ev") is None:
                            continue
                        names = group_names_by_row[q["ri"]]
                        groups = [names[op] for op in q["ops"]]
                        agg = V.conditional_aggregation_weights(groups, objective=args.ev_objective, dro=dro)
                        q["ev_rec"] = evidence_backward(V, model, proc, q["ev"], device, group_names=names,
                                                        aggregation=agg, cfg=evcfg,
                                                        grad_scale=float(args.lam) / float(args.ga_loss_divisor))
                    if g_first is not None:
                        gn_grpo, gn_ev, cos_ge = term_grad_norms(params, g_first)
                        del g_first
                    if cuda:
                        mem_ev = torch.cuda.max_memory_allocated(device)
                    t_ev = time.time() - te
            except torch.cuda.OutOfMemoryError:
                oom = True
                model.train()                     # the OOM may have hit inside the eval-mode reference pass
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"  [oom-train] step {si + 1}: update skipped", flush=True)
            if oom:
                G29._append_jsonl(trace_path, {"step": si + 1, "skip": True, "error": "oom in loss/backward",
                                               "row_index": [q["ri"] for q in qs]})
                continue
            if use_ev and args.log_term_gnorm and not ev_wired_checked and gn_ev is not None:
                ev_wired_checked = True
                print(f"[check] first evidence step: ||g_GRPO||={gn_grpo:.3e} ||g_ev||={gn_ev:.3e} cos={cos_ge}",
                      flush=True)
                if not gn_ev > 0.0:
                    raise SystemExit("STOP the evidence term produced a zero gradient (declared but not wired)")
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

            # ---------------- GroupDRO update (trainer semantics, every --ev-dro-every evidence records)
            dro_events = []
            if use_ev:
                for q in qs:
                    if q.get("ev_rec") is not None:
                        n_ev_done += 1
                        dro_buf.append({"group_losses": dict(q["ev_rec"]["group_losses"])})
                while len(dro_buf) >= args.ev_dro_every:
                    window, dro_buf = dro_buf[: args.ev_dro_every], dro_buf[args.ev_dro_every:]
                    sig = (dro_update_from_records(dro, window, grp_run_mean, grp_run_n,
                                                   bool(args.ev_dro_group_standardize)) if dro is not None else None)
                    n_dro_updates += 1
                    dro_events.append({"update": n_dro_updates, "signal": sig,
                                       "separation": separation_stats(window, V.SOFTPLUS0_PIN)})

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
                "gpu_mem_max_gb": (round(max(torch.cuda.max_memory_allocated(device), mem_grpo or 0) / 2 ** 30, 2)
                                   if cuda else 0.0),
                "elapsed": round(time.time() - t0, 1), "n_skipped_total": n_skipped,
            }
            if use_ev:
                rec.update(ev_step_summary(qs))
                rec.update({
                    "lam": args.lam, "ev_ops": [q.get("ops") for q in qs], "ev_skip": [q.get("ev_skip") for q in qs],
                    "ev_estar": [None if q.get("ev") is None else len(q["ev"]["estar"]) for q in qs],
                    "ev_q": [None if q.get("ev_rec") is None else round(q["ev_rec"]["loss"], 6) for q in qs],
                    "ev_branches": [None if q.get("ev_rec") is None else
                                    [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in b.items()}
                                     for b in q["ev_rec"]["branches"]] for q in qs],
                    "loss_total_opt": stats["loss_opt"] + rec.get("ev_loss_opt", 0.0),
                    "gnorm_grpo": gn_grpo, "gnorm_ev": gn_ev, "cos_grpo_ev": cos_ge,
                    "dro_q": (None if dro is None else {g: round(v, 6) for g, v in dro.weights().items()}),
                    "dro_events": dro_events, "n_dro_updates": n_dro_updates,
                    "n_ev_done_total": n_ev_done, "n_ev_skipped_total": n_ev_skipped,
                    "sec_ev": round(t_ev, 3),
                    "gpu_mem_grpo_gb": (round(mem_grpo / 2 ** 30, 2) if mem_grpo is not None else None),
                    "gpu_mem_ev_gb": (round(mem_ev / 2 ** 30, 2) if mem_ev is not None else None),
                })
            G29._append_jsonl(trace_path, rec)
            msg = (f"  step={si + 1}/{len(steps)} R={rec['mean_reward']:.3f} IoU={rec['mean_iou']:.3f} "
                   f"fmt={rec['format_rate']:.3f} KL={kl_mean:.4f} loss={rec['loss']:.4f} gnorm={gnorm:.3f} "
                   f"lr={lr_used:.2e} uniq={rec['n_unique']} {rec['sec_step']:.1f}s mem={rec['gpu_mem_max_gb']}G")
            if use_ev and rec.get("ev_n"):
                f6 = lambda x: "na" if x is None else f"{x:.4f}"   # noqa: E731
                msg += (f" | ev n={rec['ev_n']} L={f6(rec['ev_loss_mean'])} m={f6(rec['ev_margin_mean'])} "
                        f"dE={f6(rec['ev_drop_e_tok_mean'])} dRAND={f6(rec['ev_drop_rand_tok_mean'])}"
                        f"/base {f6(rec['ev_drop_rand_base_tok_mean'])} hinge={f6(rec['ev_hinge_active_frac'])} "
                        f"g={f6(gn_grpo)}/{f6(gn_ev)} cos={f6(cos_ge)} {t_ev:.1f}s mem_ev={rec['gpu_mem_ev_gb']}G")
            print(msg, flush=True)
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
                     "evidence_done": n_ev_done, "evidence_skipped": n_ev_skipped,
                     "evidence_skip_reasons": dict(ev_skip_reasons), "groupdro_updates": n_dro_updates,
                     "groupdro_final_q": (None if dro is None else dro.weights()),
                     "elapsed_seconds": time.time() - t0, "adapter_sha16": ad_sha})
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    print(f"[done] adapter={args.out} sha16={ad_sha} steps={step_done}/{len(steps)} questions={n_q_done} "
          f"skipped={n_skipped} evidence={n_ev_done} (skipped {n_ev_skipped}) dro_updates={n_dro_updates} "
          f"elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
