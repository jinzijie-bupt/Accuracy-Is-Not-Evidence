# File guide

Folder-by-folder description of the code, identifiers and result files. The `results/` folder is distributed through the Hugging Face dataset [`jinzijie-bupt/Accuracy-Is-Not-Evidence`](https://huggingface.co/datasets/jinzijie-bupt/Accuracy-Is-Not-Evidence); download it into `results/` at the repository root. The folder `code/iou_grpo/` is not included (the IoU-GRPO baseline is released as weights only), and the scripts of the experiments in Appendix F are described in `README.md`.

## `code/` — training and evaluation

| File | Purpose |
|---|---|
| `27_papo_train_qcdro_v3seln_r5.py` | Trainer for the placebo-controlled objective (LoRA). SFT is the same trainer with the evidence and placebo terms off. |
| `27_papo_train_qcdro_v3seln_mdpo1.py` | Trainer variant for the mDPO-style baseline. |
| `27_papo_train_qcdro_v3seln.py`, `..._r3.py`, `..._r4.py` | Earlier revisions of the trainer, imported by the audit and evaluation scripts for video decoding and scoring. Each file header lists what the revision adds; defaults are unchanged. |
| `18_papo_train_operator_mix.py`, `operator_groupdro_pilot_helpers.py`, `qcdro_objective_draft.py` | Trainer helpers: edit operators, operator schedule and Group-DRO weights, objective terms and query groups. |
| `14Ln_native_sentinel_probe_r7.py` | Likelihood audit: evidence drop, placebo drop (three draws) and ψ under mask and swap, plus the unseen operators (noise, patch scrambling, Gaussian blur). |
| `14Ln_native_sentinel_probe_r3.py` | Likelihood audit with per-token, contiguous-placebo, refusal and relocation readouts. |
| `14Lqn_native_query_swap_probe_r3.py` | Query-crossed audit with the partner event of the same video. |
| `12p4n_native_acc_eval_tf5meta_r8.py`, `12p4n_native_acc_eval_tf451_r5.py` | Generation at 2 fps with greedy decoding: accuracy, black video (`--video-op black`) and mismatched query (`--query-op foreign --query-seed 0`). Two variants for the transformers 5.x and 4.51 APIs. |
| `14Lpn_native_paraphrase_probe_r1.py`, `chain_node2_14Lpn.sh` | Likelihood audit with the reworded query (second human-written description of the same interval; map in `data_ids/reworded_queries_map.json`), and the command lines used to run it. Imports the trainer revision `27_papo_train_qcdro_v3seln_r4.py` from this folder. |
| `12p5n_native_perturb_eval_r3.py` | Behavioural audit: regenerated answers under evidence and placebo edits. |
| `12p6n_tlbench_eval_r4.py` | Evaluation on the TimeLens-Bench splits (QVHighlights, re-annotated Charades-STA). |
| `12_charades_grounding.py`, `native_video_causal_probe.py`, `14_causal_provenance_probe.py`, `12p4a_grounding_eval_adapter.py` | Shared helpers (data loading, frame operators, parsing, IoU). |
| `configs/adapter_config.json` | LoRA configuration of the trained adapters (rank 16, alpha 32, dropout 0.05). Adapter weights are not included. |

Data and model paths: the scripts take `--model`, `--adapter`, `--anno` and `--video-dir` arguments. Default paths are built from a root directory (the environment variable `DATA_ROOT` in `12_charades_grounding.py`, the parent of `code/` in `native_video_causal_probe.py`) that is expected to hold `data/charades_sta/` (Charades-STA annotations and the `Charades_v1_480` videos) and `models/`.

## `readers/` — scoring

Readers turn the per-question outputs of the scripts in `code/` into the quantities reported in the paper, with 95% bootstrap intervals over videos: `read_14Ln_v1.py` (Γ, ψ, d_E, d_O), `read_14Lqn_v2.py` (query-crossed contrasts), `read_14Lpn_v1.py` (reworded queries), `read_p4nat_v1.py` and `read_p4nat_selective_v1.py` (R@0.5, parse rate, abstention, coverage, paired differences), `read_blackdecline_v1.py` (black video), `read_p4nat_tneg_v1.py` (mismatched query), `read_p5n_v1.py`/`read_p5n_v2.py` (behavioural audit), `behav_split.py`, `read_r3.py`, `read_r3_tokclass.py` (where the dependence sits), `read_tl_labelswap_v1.py` (re-annotation), `read_window_oracle_v1.py` (window-length substitution), `read_newly_correct_v1.py`, `seed_resolution.py` (seed-to-seed resolution), `make_paraphrase_map_v1.py` (builds `data_ids/reworded_queries_map.json`).

`readers/derived/` holds analyses that combine audit outputs (clean-likelihood scale, newly correct questions, public checkpoints against the Qwen base) together with their JSON outputs, which the table and figure scripts read.

## `results/`

Reader products (JSON) that the table and figure scripts read, plus the per-question outputs read directly by the message-order and Charades-CD tables. `results/run_order/` records the message order of every run. Run names end in an anonymised compute-node tag (`node1` … `node5`); `tables_and_figures/scripts/canonical_v2.py` maps each tag to its software stack and pairs every trained model with a base run from the same stack. Paths and node names inside the files were rewritten for anonymity, so the `sha16` fields recorded in the files refer to the files before rewriting.

## Per-question outputs

This variant also contains the per-question outputs of every run the readers and the derived analyses consume (485 JSON files under `results/`, at the paths the readers expect), anonymised like the rest. With them, `readers/*.py` and `readers/derived/*.py` rerun from this folder; e.g. `python readers/read_14Ln_v1.py --base <base run> --arm <trained run> --out out.json`, or `python a2_newly_correct.py` inside `readers/derived/` (it rewrites its JSON next to itself).

## `tables_and_figures/`

```
cd tables_and_figures
python make_all.py            # tables and data figures
python make_all.py --tables   # tables only
```

`make_all.py` runs, in order: `build_numbers_v2.py` (registry of all reader products), `canonical_v3.py` (one evaluation stack per model configuration), the table builders (every table of the paper: `tables/T*.tex`, `tables/S*.tex`), `rename_display_v3.py`, and `plot_figs_v6.py` (data figures). Outputs go to `tables/` and `figures/`; the copies shipped there were produced by this command. The qualitative-case figure additionally needs the Charades-STA videos `AK2KG.mp4` and `BL7OF.mp4` in `assets/source_videos/` and is skipped without them. `concept_figures/` holds the builders of the schematic figures; the video frames they show (Charades-STA `K7NRW`, `P6JGZ`) are not included.

## `data_ids/`

| File | Content |
|---|---|
| `training_pairs_pool_8434.jsonl` | The 8,434-pair pool (Charades-STA training split); the 2,400 training pairs are its first 2,400 lines. |
| `audit_set_200.jsonl` | The 200 audit questions (first 200 Charades-STA test questions, 79 videos). |
| `generation_set_1000.jsonl` | The 1,000 questions regenerated in the behavioural audit. |
| `behavioural_placebo_draws.jsonl` | Per question: evidence frames, placebo frames for mask and swap, swap donor video. The same draws are used for every model. |
| `partner_event_pairs.jsonl` | Question → partner event of the same video used by the query-crossed audit (recorded for questions localised by at least one audited model). |
| `reworded_queries_map.json` | Question → second human-written description of the same interval. |
| `mismatched_query_pairing.jsonl` | Question → query taken from a different video (seed 0). |
| `charades_cd/` | Charades-CD test-iid and test-ood items. |

`qid` is the 0-based line index in the Charades-STA test annotation (`charades_sta_test.txt`).

## Requirements

Tables and figures: Python 3 with numpy, scipy, matplotlib and SciencePlots (opencv-python for the qualitative-case figure; PyMuPDF and fontTools for the schematic figures). Regenerated here with Python 3.14, numpy 2.5, scipy 1.18, matplotlib 3.11, SciencePlots 2.2.

Training and evaluation: a CUDA GPU, PyTorch, transformers 5.12/5.13 or 4.51 (the `tf451` evaluator), peft 0.20, qwen-vl-utils, decord or OpenCV, safetensors. Base models: Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct, and the public TimeZero, Time-R1, VideoChat-R1 and TimeLens-8B checkpoints.

## Known gaps

- The per-question outputs of the reworded-query runs are in `results/native_eval_20260918_node2/lpn/`.
- The IoU-GRPO baseline trainer is not included (weights only).
- Table 4 (notation) and Table 5 (audit controls) are written by hand in `sections/appendix.tex` of the paper and have no script.
- The concept-figure scripts need the original Charades-STA videos and extracted frames, which we do not redistribute.
