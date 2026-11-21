#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU tests for 30_grpo_plus_placebo.py (no GPU, no real weights).

  python test_30_grpo_plus_placebo_cpu.py
  B5_TEST_PROC=<dir with a Qwen2.5-VL processor> python test_30_grpo_plus_placebo_cpu.py   # + tiny-model tests

Pure tests: argparser = 29_'s flags + main-arm evidence defaults, GroupDRO update/standardisation, separation read-out,
per-term gradient norms, group gate on the real 2400-row schedule, evidence edit construction (E*, RAND placebo,
guards). Tiny-model tests (random 2-layer Qwen2.5-VL built in a temp dir; the processor is taken from $B5_TEST_PROC or
the trainer's MODEL dir; skipped if neither exists):
  * the evidence term == the trainer's own V._train_scheduled_example (loss, group losses/margins, every LoRA gradient)
    bit-for-bit, for the main-arm config, a forced-hinge config, the mean/alpha-0 config and the symmetric-npole config;
  * gradients flow from both terms (GRPO-only, evidence-only, both = sum; vision + language LoRA);
  * --lam 0 reproduces 29_ exactly (same adapter bytes, same trace) after 2 optimizer steps with the same seed,
    and --lam 1 does not.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if "cv2" not in sys.modules:
    try:
        import cv2  # noqa: F401
    except Exception:
        sys.modules["cv2"] = types.ModuleType("cv2")

try:
    import torch
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    HAVE_TORCH = False

_spec = importlib.util.spec_from_file_location("b5_30", os.path.join(HERE, "30_grpo_plus_placebo.py"))
M = importlib.util.module_from_spec(_spec)
sys.modules["b5_30"] = M
_spec.loader.exec_module(M)
G29 = M.G29

V = None
if HAVE_TORCH:
    V, _ = M.load_trainer(os.path.join(HERE, G29.TRAINER_FILE), G29.TRAINER_SHA16)


def _rows(n=2400):
    return [json.loads(l) for l in open(os.path.join(HERE, "qcdro_v2_pairs_train.jsonl"), encoding="utf-8")
            if l.strip()][:n]


class TestArgs(unittest.TestCase):
    def test_superset_of_29_with_same_defaults(self):
        base = ["--pairs", "p.jsonl", "--out", "o"]
        a29 = vars(G29.build_argparser().parse_args(base))
        a30 = vars(M.build_argparser().parse_args(base))
        for k, v in a29.items():
            self.assertIn(k, a30)
            self.assertEqual(a30[k], v, k)

    def test_main_arm_defaults(self):
        a = M.build_argparser().parse_args(["--pairs", "p", "--out", "o"])
        self.assertEqual((a.lam, a.ev_objective, a.ev_alpha, a.ev_beta, a.ev_npole, a.ev_npole_symmetric),
                         (1.0, "groupdro", 0.5, 0.1, 1, False))
        self.assertEqual((a.ev_rand_hinge, a.ev_rand_ratio, a.ev_rand_slack), (1.0, 1.25, 0.10))
        self.assertEqual((a.ev_dro_eta, a.ev_dro_ema_decay, a.ev_dro_update_clip, a.ev_dro_group_standardize),
                         (0.05, 0.9, 5.0, 1))
        self.assertEqual((a.ev_group_scheme, a.ev_min_group_exposure, a.ev_dro_every), ("product", 20, 8))
        self.assertEqual(M.GRPO_SHA16, M.sha16_file(M.G29_PATH))


class TestPure(unittest.TestCase):
    def test_skip_bucket(self):
        self.assertEqual(M.ev_skip_bucket("donor decode RuntimeError: x"), "donor_decode")
        self.assertEqual(M.ev_skip_bucket("ValueError: degenerate evidence set |E*|=9 T=10"), "degenerate_estar")
        self.assertEqual(M.ev_skip_bucket("ValueError: shuffle needs |E*|>=2, got 1"), "shuffle_estar_lt2")

    def test_separation(self):
        recs = [{"group_losses": {"a": 1.0, "b": 2.0}}, {"group_losses": {"a": 3.0, "b": 2.0}}]
        s = M.separation_stats(recs, 2 * np.log(2))
        self.assertAlmostEqual(s["group_loss_spread"], 0.0)
        self.assertAlmostEqual(s["within_group_sd"], 0.5)
        self.assertIsNone(M.separation_stats([{"group_losses": {"a": 1.0}}], 1.0))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestDRO(unittest.TestCase):
    def test_standardized_update_matches_trainer_rule(self):
        dro = V.GroupDRO(["g1", "g2", "g3"], eta=0.05, ema_decay=0.9, update_clip=5.0)
        rm, rn = defaultdict(float), defaultdict(int)
        w1 = [{"group_losses": {"g1": 1.0, "g2": 2.0}}, {"group_losses": {"g1": 3.0, "g3": 0.5}}]
        sig = M.dro_update_from_records(dro, w1, rm, rn, True)
        self.assertEqual(sig, {"g1": 0.0, "g2": 0.0, "g3": 0.0})          # first sighting -> centered 0
        self.assertEqual(dict(rn), {"g1": 1, "g2": 1, "g3": 1})
        self.assertEqual(rm["g1"], 2.0)
        w2 = [{"group_losses": {"g1": 4.0, "g2": 1.0}}]
        sig = M.dro_update_from_records(dro, w2, rm, rn, True)
        self.assertAlmostEqual(sig["g1"], 2.0)
        self.assertAlmostEqual(sig["g2"], -1.0)
        self.assertAlmostEqual(rm["g1"], 3.0)                              # 2 + (4-2)/2
        q = dro.weights()
        self.assertGreater(q["g1"], q["g3"])
        self.assertGreater(q["g3"], q["g2"])
        # reference: same updates issued directly on a fresh GroupDRO
        ref = V.GroupDRO(["g1", "g2", "g3"], eta=0.05, ema_decay=0.9, update_clip=5.0)
        ref.update({g: torch.tensor(0.0) for g in ("g1", "g2", "g3")})
        ref.update({"g1": torch.tensor(2.0), "g2": torch.tensor(-1.0)})
        for g in q:
            self.assertAlmostEqual(q[g], ref.weights()[g], places=12)

    def test_raw_update(self):
        dro = V.GroupDRO(["a", "b"], eta=0.5, ema_decay=0.0)
        sig = M.dro_update_from_records(dro, [{"group_losses": {"a": 3.0, "b": 1.0}}], defaultdict(float),
                                        defaultdict(int), False)
        self.assertEqual(sig, {"a": 3.0, "b": 1.0})
        self.assertGreater(dro.weights()["a"], 0.5)

    def test_term_grad_norms(self):
        p1 = torch.zeros(3, requires_grad=True)
        p2 = torch.zeros(2, requires_grad=True)
        p1.grad = torch.tensor([1.0, 0.0, 0.0])
        p2.grad = None
        snap = M.grad_snapshot([p1, p2])
        p1.grad = p1.grad + torch.tensor([0.0, 2.0, 0.0])
        p2.grad = torch.tensor([0.0, 0.0])
        g0, g1, cos = M.term_grad_norms([p1, p2], snap)
        self.assertAlmostEqual(g0, 1.0)
        self.assertAlmostEqual(g1, 2.0)
        self.assertAlmostEqual(cos, 0.0)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestGroupsAndPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = _rows()
        cls.sched = V.build_schedule_v2(cls.rows, epochs=1, requested_ops=V.parse_ops_v2(G29.METHOD_OPS),
                                        ops_per_example=2, seed=0, query_group_key="query_group")

    def test_group_gate_as_method(self):
        g = M.build_groups(V, self.rows, self.sched, scheme="product", min_exposure=20,
                           ops=V.parse_ops_v2(G29.METHOD_OPS))
        # with --limit 2400 seed 0 temporal_order/* is starved -> the trainer's pre-registered fallback fires
        self.assertEqual(g["scheme"], "validity")
        self.assertIsNotNone(g["fallback"])
        self.assertIn("temporal_order/swap", g["fallback"]["starved_groups"])
        for it in self.sched.items:                                         # every scheduled op is registered
            for op in it.ops:
                self.assertIn(g["names_by_row"][it.row_index][op], g["groups"])
        self.assertEqual(sum(g["exposure"].values()), 2 * len(self.sched.items))
        print(f"\n  [groups] {g['exposure']}")

    def _frames(self, T=12, seed=0):
        rng = np.random.default_rng(seed)
        return rng.integers(1, 255, size=(T, 28, 28, 3), dtype=np.uint8)

    def test_plan_edits(self):
        T = 12
        fr, donor = self._frames(T, 0), self._frames(T, 1)
        times = [i * 1.0 for i in range(T)]
        ex = {"caption": "x", "gs": 3.0, "ge": 6.0}
        p = M.evidence_plan(V, ex, fr, times, 1.0, donor, ("mask", "shuffle"), seed=0, epoch=0, row_index=5,
                            npole=True, answer_style="timer1", prompt_style="timer1_default")
        self.assertEqual(p["estar"], [3, 4, 5, 6])
        self.assertEqual(p["completion"], "<answer> 3.00 to 6.00 </answer>")
        mask = p["edits"][0]
        self.assertTrue((mask["corrupt_e"][3:7] == 0).all())
        self.assertTrue(np.array_equal(np.delete(mask["corrupt_e"], [3, 4, 5, 6], 0), np.delete(fr, [3, 4, 5, 6], 0)))
        n_idx = mask["nonev_idx"]
        self.assertEqual(len(n_idx), 4)                                  # RAND: |E*| frames ...
        self.assertFalse(set(n_idx) & {3, 4, 5, 6})                       # ... outside E*
        self.assertTrue((mask["corrupt_n"][n_idx] == 0).all())
        sh = p["edits"][1]                                               # shuffle: permutation within E*, non-identity
        self.assertFalse(np.array_equal(sh["corrupt_e"], fr))
        self.assertEqual(sorted(map(bytes, sh["corrupt_e"][3:7])), sorted(map(bytes, fr[3:7])))
        # keyed: identical to the trainer's own construction for the same (seed, epoch, row, op)
        ref = V.corrupt_op_v2(fr, np.asarray([3, 4, 5, 6]), donor, "shuffle",
                              rng=V._stable_rng(0, 0, 5, "shuffle", "shufperm"))
        self.assertTrue(np.array_equal(ref, sh["corrupt_e"]))
        p2 = M.evidence_plan(V, ex, fr, times, 1.0, donor, ("mask", "shuffle"), seed=0, epoch=0, row_index=5,
                             npole=True, answer_style="timer1", prompt_style="timer1_default")
        self.assertTrue(all(np.array_equal(a["corrupt_n"], b["corrupt_n"]) for a, b in zip(p["edits"], p2["edits"])))

    def test_plan_guards(self):
        fr = self._frames(6)
        times = [float(i) for i in range(6)]
        with self.assertRaises(ValueError):                              # |E*| >= T-1
            M.evidence_plan(V, {"caption": "x", "gs": 0.0, "ge": 4.5}, fr, times, 1.0, fr, ("mask", "blur"),
                            seed=0, epoch=0, row_index=0, npole=True, answer_style="timer1",
                            prompt_style="timer1_default")
        with self.assertRaises(ValueError):                              # shuffle with native |E*| = 1
            M.evidence_plan(V, {"caption": "x", "gs": 2.0, "ge": 2.4}, fr, times, 1.0, fr, ("mask", "shuffle"),
                            seed=0, epoch=0, row_index=0, npole=True, answer_style="timer1",
                            prompt_style="timer1_default")
        p = M.evidence_plan(V, {"caption": "x", "gs": 50.0, "ge": 60.0}, fr, times, 1.0, fr, ("mask", "blur"),
                            seed=0, epoch=0, row_index=0, npole=True, answer_style="timer1",
                            prompt_style="timer1_default")
        self.assertEqual(p["estar"], [5])                                # nearest-frame fallback (trainer rule)


# ============================================================================ tiny-model tests
def _proc_dir():
    cands = [os.environ.get("B5_TEST_PROC")]
    if V is not None:
        cands.append(V.MODEL)
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "preprocessor_config.json")):
            return c
    return None


PROC_DIR = _proc_dir() if HAVE_TORCH else None


def fake_native_decode(video_path, args):
    """Deterministic random 56x56 clip per file name; 10/12/14 frames spread over 32 s so GT windows of the real
    rows give non-trivial E*."""
    h = int(hashlib.sha256(os.path.basename(video_path).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)
    T = 10 + 2 * (h % 3)
    fr = rng.integers(0, 255, size=(T, 56, 56, 3), dtype=np.uint8)
    step = 32.0 / T
    return fr, [i * step for i in range(T)], 32.0, 1.0 / step


def fake_parse(text, dur):            # random model never emits numbers -> text-hash rewards (as harness_29)
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:6], 16)
    if h % 4 == 0:
        return None
    s = float(h % 20)
    return (s, s + 1.0 + (h // 20) % 10)


def _tf_major():
    import transformers
    return int(str(transformers.__version__).split(".")[0])


def fp32_from_pretrained(cls, path, **kw):
    """tf5: dtype=, tf4.51: torch_dtype= (CPU oneDNN has no bf16 backward -> tiny model in fp32)."""
    if _tf_major() >= 5:
        return cls.from_pretrained(path, dtype=torch.float32, **kw)
    return cls.from_pretrained(path, torch_dtype=torch.float32, **kw)


def patch_trainer(Vm):
    Vm.native_decode = fake_native_decode
    Vm._fp_dtype_compat = lambda cls, path, **kw: fp32_from_pretrained(cls, path, **kw)
    return Vm


def build_tiny(dst):
    from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, GenerationConfig
    text = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                intermediate_size=128, vocab_size=152064, max_position_embeddings=4096,
                rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]}, rms_norm_eps=1e-6,
                tie_word_embeddings=False)
    vision = dict(depth=2, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=64,
                  patch_size=14, spatial_merge_size=2, temporal_patch_size=2, window_size=112,
                  fullatt_block_indexes=[1], tokens_per_second=2, in_channels=3)
    ids = dict(image_token_id=151655, video_token_id=151656, vision_start_token_id=151652,
               vision_end_token_id=151653, bos_token_id=151643, eos_token_id=151645)
    if _tf_major() >= 5:
        cfg = Qwen2_5_VLConfig(text_config=text, vision_config=vision, **ids)
    else:                                    # tf4.51: flat text config (branch not exercised on the dev box)
        cfg = Qwen2_5_VLConfig(vision_config=vision, **text, **ids)
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(cfg).to(torch.bfloat16)
    model.generation_config = GenerationConfig(do_sample=True, temperature=0.1, top_p=0.001, top_k=1,
                                               repetition_penalty=1.05, eos_token_id=[151645, 151643],
                                               pad_token_id=151643, bos_token_id=151643)
    model.save_pretrained(dst)
    AutoProcessor.from_pretrained(PROC_DIR).save_pretrained(dst)


def lora_model(tiny, dropout):
    from transformers import AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model
    model = fp32_from_pretrained(AutoModelForImageTextToText, tiny, device_map={"": "cpu"})
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=dropout, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():                    # move off B=0 so adapter != base (margins, hinge, lora_A grads live)
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model


def grads(model):
    return {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in model.named_parameters()
            if p.requires_grad}


TOY_EX = {"video": "toyA.mp4", "distractor": "toyB.mp4", "caption": "person opens a door.", "gs": 9.0, "ge": 17.0,
          "query_group": "temporal_order", "shuffle_valid": True, "n_estar": 4}


@unittest.skipUnless(HAVE_TORCH and PROC_DIR, "needs torch + a Qwen2.5-VL processor dir ($B5_TEST_PROC)")
class TestTinyModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="b5test_")
        cls.tiny = os.path.join(cls.tmp, "tiny")
        build_tiny(cls.tiny)
        from transformers import AutoProcessor
        cls.proc = AutoProcessor.from_pretrained(cls.tiny)
        cls.V = patch_trainer(M.load_trainer(os.path.join(HERE, G29.TRAINER_FILE), G29.TRAINER_SHA16)[0])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _trainer_args(self, **kw):
        a = dict(objective="groupdro", npole=True, npole_symmetric=False, beta=0.1, alpha=0.5, rand_hinge=1.0,
                 rand_ratio=1.25, rand_slack=0.10, accum=4, seed=0, answer_style="timer1",
                 prompt_style="timer1_default", fps=2.0, total_pixels=1, min_pixels=1, max_frames=768)
        a.update(kw)
        return argparse.Namespace(**a)

    def _dro(self, groups):
        dro = self.V.GroupDRO(groups, eta=0.5, ema_decay=0.0)
        dro.update({groups[0]: torch.tensor(1.0), groups[-1]: torch.tensor(-0.5)})   # non-uniform q
        return dro

    def _compare(self, ops, row_index, **kw):
        Vt = self.V
        targs = self._trainer_args(**kw)
        names = {op: Vt.group_of(TOY_EX, op, scheme="validity", query_group_key="query_group")
                 for op in Vt.SUPPORTED_V2}
        groups = sorted(set(names.values()))
        model = lora_model(self.tiny, dropout=0.05)                      # dropout ON: RNG order must match too
        # --- the trainer's own function
        dro = self._dro(groups) if targs.objective == "groupdro" else None
        model.zero_grad(set_to_none=True)
        torch.manual_seed(1234)
        rt = Vt._train_scheduled_example(model, self.proc, TOY_EX, epoch=0, row_index=row_index, ops=ops, dro=dro,
                                         device="cpu", args=targs, group_names=names)
        gt = grads(model)
        # --- 30_: plan from decoded frames + evidence_backward with lam/ga_div = 1/accum
        dro = self._dro(groups) if targs.objective == "groupdro" else None
        model.zero_grad(set_to_none=True)
        torch.manual_seed(1234)
        fr, times, dur, sfps = Vt.native_decode(TOY_EX["video"], None)
        donor = Vt.native_decode(TOY_EX["distractor"], None)[0]
        plan = M.evidence_plan(Vt, TOY_EX, fr, times, sfps, donor, ops, seed=targs.seed, epoch=0,
                               row_index=row_index, npole=targs.npole, answer_style="timer1",
                               prompt_style="timer1_default")
        agg = Vt.conditional_aggregation_weights([names[o] for o in ops], objective=targs.objective, dro=dro)
        cfg = types.SimpleNamespace(beta=targs.beta, alpha=targs.alpha, npole=targs.npole,
                                    npole_symmetric=targs.npole_symmetric, rand_hinge=targs.rand_hinge,
                                    rand_ratio=targs.rand_ratio, rand_slack=targs.rand_slack)
        lam, ga = 1.0, float(targs.accum)
        rm = M.evidence_backward(Vt, model, self.proc, plan, "cpu", group_names=names, aggregation=agg, cfg=cfg,
                                 grad_scale=lam / ga)
        gm = grads(model)
        self.assertEqual(rm["loss"], rt["loss"])
        self.assertEqual(rm["clean_nll"], rt["clean_nll"])
        self.assertEqual(rm["group_losses"], rt["group_losses"])
        self.assertEqual(rm["group_margins"], rt["group_margins"])
        self.assertEqual(rm["aggregation_weights"], rt["aggregation_weights"])
        worst, n_nonzero = 0.0, 0
        for n in gt:
            a, b = gt[n], gm[n]
            self.assertEqual(a is None, b is None, n)
            if a is None:
                continue
            worst = max(worst, float((a - b).abs().max()))
            n_nonzero += int(bool(a.abs().max() > 0))
        self.assertEqual(worst, 0.0, f"max |grad diff| = {worst}")
        self.assertGreater(n_nonzero, 0)
        return rm

    def test_evidence_term_equals_trainer_main_arm(self):
        rm = self._compare(("swap", "shuffle"), 11)
        rm2 = self._compare(("mask", "blur"), 12)
        rm3 = self._compare(("tshift", "mask"), 13)
        print(f"\n  [ev==trainer] main arm: loss {rm['loss']:.6f} / {rm2['loss']:.6f} / {rm3['loss']:.6f}; "
              f"hinge active {[b['hinge_active'] for r in (rm, rm2, rm3) for b in r['branches']]}")

    def test_evidence_term_equals_trainer_forced_hinge(self):
        rm = self._compare(("swap", "blur"), 21, rand_slack=-1.0)          # excess > 0 -> hinge gradient live
        self.assertTrue(all(b["hinge_active"] for b in rm["branches"]))

    def test_evidence_term_equals_trainer_mean_alpha0_symmetric(self):
        self._compare(("mask", "shuffle"), 31, objective="mean", alpha=0.0)
        self._compare(("blur", "tshift"), 32, npole_symmetric=True, rand_hinge=0.0)

    def test_gradients_flow_from_both_terms(self):
        Vt = self.V
        model = lora_model(self.tiny, dropout=0.0)                       # deterministic -> additivity is exact-ish
        params = [p for p in model.parameters() if p.requires_grad]
        fr, times, dur, sfps = Vt.native_decode(TOY_EX["video"], None)
        donor = Vt.native_decode(TOY_EX["distractor"], None)[0]
        pin, _ = G29.prompt_inputs_for(Vt, self.proc, fr, sfps, TOY_EX["caption"], "timer1_default")
        pin = dict(pin)
        eos_ids, pad_id = G29.gen_eos_pad(self.proc, model)
        comp, mask, lens = G29.sample_group(model, pin, 4, 6, 1.0, G29.question_seed(0, 3), eos_ids, pad_id)
        adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
        keep = G29.logits_keep_kwarg(model)
        names = {op: Vt.group_of(TOY_EX, op, scheme="validity", query_group_key="query_group")
                 for op in Vt.SUPPORTED_V2}
        ops = ("swap", "shuffle")
        plan = M.evidence_plan(Vt, TOY_EX, fr, times, sfps, donor, ops, seed=0, epoch=0, row_index=3, npole=True,
                               answer_style="timer1", prompt_style="timer1_default")
        agg = Vt.conditional_aggregation_weights([names[o] for o in ops], objective="mean")
        cfg = M.ev_cfg_from_args(M.build_argparser().parse_args(["--pairs", "p", "--out", "o"]))

        def grpo():
            N = int(mask.sum())
            for c0 in range(0, 4, 3):                                    # 29_'s chunk loop, score-batch 3
                sl = slice(c0, min(4, c0 + 3))
                cm = mask[sl]
                Lc = int(cm.sum(1).max())
                inp, P = G29.build_score_inputs(pin, comp[sl, :Lc], cm[:, :Lc], pad_id)
                model.eval()
                with torch.no_grad(), model.disable_adapter():
                    ref = G29.completion_logps(model, inp, P, Lc, 1.0, keep)
                model.train()
                logp = G29.completion_logps(model, inp, P, Lc, 1.0, keep)
                per, _ = G29.grpo_per_token_loss(logp, ref, adv[sl], 0.04)
                G29.chunk_loss(per, cm[:, :Lc], "dapo", step_tokens=N, q_tokens=N, n_seq_question=4,
                               ga_div=4.0).backward()

        def ev():
            model.train()
            M.evidence_backward(Vt, model, self.proc, plan, "cpu", group_names=names, aggregation=agg, cfg=cfg,
                                grad_scale=1.0 / 4.0)

        model.zero_grad(set_to_none=True); grpo(); g1 = grads(model)
        model.zero_grad(set_to_none=True); ev(); g2 = grads(model)
        model.zero_grad(set_to_none=True); grpo(); snap = M.grad_snapshot(params); ev(); g12 = grads(model)
        n_grpo, n_ev, cos = M.term_grad_norms(params, snap)
        norm = lambda g: float(torch.sqrt(sum((v.double() ** 2).sum() for v in g.values() if v is not None)))  # noqa
        self.assertGreater(norm(g1), 0.0)
        self.assertGreater(norm(g2), 0.0)
        self.assertAlmostEqual(n_grpo, norm(g1), delta=1e-6 * norm(g1))
        self.assertAlmostEqual(n_ev, norm(g2), delta=1e-5 * norm(g2))
        worst = max(float((g12[n] - g1[n] - g2[n]).abs().max()) for n in g12 if g12[n] is not None)
        self.assertLess(worst, 1e-6 * max(norm(g1), norm(g2)) + 1e-9)
        ev_vis = sum(float(v.abs().sum()) for n, v in g2.items() if v is not None and "visual" in n)
        ev_lm = sum(float(v.abs().sum()) for n, v in g2.items() if v is not None and "visual" not in n)
        self.assertGreater(ev_vis, 0.0)
        self.assertGreater(ev_lm, 0.0)
        print(f"\n  [both terms] ||g_GRPO||={norm(g1):.4e} ||g_ev||={norm(g2):.4e} cos={cos:.4f} "
              f"additivity err={worst:.2e}; ev grad vision={ev_vis:.3e} language={ev_lm:.3e}")

    # ---------------------------------------------------------------- main(): --lam 0 == 29_
    def _run(self, mod, argv):
        old = sys.argv
        olt, olp = mod.load_trainer, mod.load_parser

        def lt(path, sha):
            Vm, sh = olt(path, sha)
            return patch_trainer(Vm), sh

        def lp(path, sha):
            _, pmod, sh = olp(path, sha)
            return fake_parse, pmod, sh
        sys.argv = ["x"] + argv
        mod.load_trainer, mod.load_parser = lt, lp
        try:
            mod.main()
        finally:
            sys.argv = old
            mod.load_trainer, mod.load_parser = olt, olp

    def test_lam0_reproduces_29_and_lam1_differs(self):
        from safetensors.torch import load_file
        common = ["--pairs", os.path.join(HERE, "qcdro_v2_pairs_train.jsonl"), "--limit", "12", "--model", self.tiny,
                  "--cpu-debug", "--r", "4", "--num-generations", "4", "--questions-per-step", "2",
                  "--max-new-tokens", "6", "--score-batch", "3", "--save-every", "2", "--warmup-steps", "1",
                  "--lr", "1e-3", "--max-steps", "2"]
        d29, d30, d31 = (os.path.join(self.tmp, x) for x in ("r29", "r30_lam0", "r30_lam1"))
        self._run(G29, common + ["--out", d29])
        self._run(M, common + ["--out", d30, "--lam", "0"])
        self._run(M, common + ["--out", d31, "--lam", "1"])
        a29 = load_file(os.path.join(d29, "adapter_model.safetensors"))
        a30 = load_file(os.path.join(d30, "adapter_model.safetensors"))
        a31 = load_file(os.path.join(d31, "adapter_model.safetensors"))
        self.assertEqual(sorted(a29), sorted(a30))
        diff0 = max(float((a29[k] - a30[k]).abs().max()) for k in a29)
        diff1 = max(float((a29[k] - a31[k]).abs().max()) for k in a29)
        self.assertEqual(diff0, 0.0)
        self.assertGreater(diff1, 0.0)
        t29 = [json.loads(l) for l in open(os.path.join(d29, "grpo_trace.jsonl"), encoding="utf-8")]
        t30 = [json.loads(l) for l in open(os.path.join(d30, "grpo_trace.jsonl"), encoding="utf-8")]
        t31 = [json.loads(l) for l in open(os.path.join(d31, "grpo_trace.jsonl"), encoding="utf-8")]
        self.assertEqual(len(t29), 2)
        for x, y in zip(t29, t30):
            for k in ("rewards", "texts", "adv", "loss", "grad_norm", "kl", "lr", "row_index"):
                self.assertEqual(x[k], y[k], k)
            self.assertNotIn("ev_n", y)
        self.assertEqual([r["texts"] for r in t31][0], t29[0]["texts"])  # step 1 samples before any update
        self.assertTrue(all(r["ev_n"] >= 1 for r in t31))
        self.assertTrue(all(r["gnorm_ev"] > 0 and r["gnorm_grpo"] > 0 for r in t31))
        man = json.load(open(os.path.join(d31, "grpo_native_manifest.json"), encoding="utf-8"))
        self.assertEqual(man["trainer"], M.TRAINER_NAME)
        self.assertEqual(man["grpo_sha16"], M.GRPO_SHA16)
        print(f"\n  [lam0==29] max|adapter diff| = {diff0}; lam1 vs 29 = {diff1:.3e}; "
              f"lam1 step2 ev_loss={t31[-1]['ev_loss_mean']:.4f} gnorm grpo/ev={t31[-1]['gnorm_grpo']:.3e}/"
              f"{t31[-1]['gnorm_ev']:.3e} hinge_frac={t31[-1]['ev_hinge_active_frac']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
