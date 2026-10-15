# AutoVLA Phase II — `model/`

Qwen2.5-VL-3B + LoRA backbone, SFT, open-loop evaluation, the Bayesian uncertainty
head, uncertainty-gated reasoning, and GRPO. Phase I (`data_pipeline/`) is imported,
never modified.

## Read this first: what has and has not been run

This code was written and tested on a MacBook with **no CUDA GPU**, and the Phase-I corpus is
**synthetic** (no camera images, no nuScenes download). So:

| Piece | Status |
|---|---|
| Every module's logic (tokenizer extension, LoRA-only invariant, Eq. 2 loss, 3:1 mixing, constrained decoding, metrics, uncertainty, gate, GRPO, reward) | Unit-tested on CPU in `model/tests/` against a tiny random-weight **real `Qwen2_5_VLForConditionalGeneration`** (`model/standin.py`) |
| Loading `Qwen/Qwen2.5-VL-3B-Instruct`, NF4 4-bit via bitsandbytes, gradient checkpointing, SFT, open-loop evaluation | **Run on a Kaggle T4** (fp16 + loss scaling): smoke test, 1200-step SFT, evaluation. Results below. GRPO and the uncertainty/gating stages have still only run on the stand-in |
| Anything with images (vision features for the uncertainty head, `PromptEncoder` with a processor) | Vision *forward* path tested on the stand-in with random pixel tensors; real image loading untested |
| DriveLM-based CoT | Reformatter implemented and tested on a hand-written fixture in DriveLM's layout; **no DriveLM data was available**, so all CoT in `outputs/phase2/cot_annotations.jsonl` is trajectory-derived text (no perception content) |
| BDD-X benchmark (BLEU-4/METEOR/CIDEr) | Metrics verified against sacreBLEU / pycocoevalcap (BLEU, CIDEr to 1e-9); **BDD-X was not run** (CSV not downloaded: `data_pipeline/scripts/download_bddx.sh`). METEOR is exact+stem only |
| Numbers in `results/` | Except `openloop_sft_t4_1200.csv` (real model, below), produced by the **stand-in model on synthetic data with synthetic agent boxes**. They show the pipeline works end to end. They are *not* results for Qwen2.5-VL-3B or nuScenes, and the CSV says so in its `provenance` column |

## Layout

| File | Milestone | What |
|---|---|---|
| `config.py` | all | constants (paper values cited inline); re-exports Phase-I config |
| `prompting.py` | 4–6 | ChatML rendering, fast/slow response templates, response parser |
| `standin.py`, `runtime.py` | all | tiny CPU stand-in; build a backbone + restore a checkpoint |
| `backbone.py`, `inference.py`, `forward_test.py` | M4 | load/extend/freeze/LoRA; constrained plan generation; smoke test |
| `cot_annotator.py`, `explanation_metrics.py` | M5 | DriveLM→4-field CoT, trajectory-template fallback; BLEU/METEOR/CIDEr |
| `dataset.py`, `trainer.py`, `../configs/sft.yaml` | M6 | SFT with Eq. 2 |
| `metrics.py`, `evaluate.py`, `splits.py` | M7 | L2/collision per horizon, latency, fast/slow/random/auto |
| `uncertainty.py`, `calibration.py` | M8 | aleatoric head, MC-dropout epistemic, fusion, temperature scaling, ECE, AUROC |
| `gating.py` | M9 | `ConfidenceGate`, minimal-risk manoeuvre, threshold calibration |
| `reward.py`, `grpo.py` | M10 | rewards (incl. calibration term), GRPO |
| `benchmark.py` | M11 | four-row ablation → `results/ablation_table.csv` |

## Run

```bash
.venv/bin/python -m pytest model/tests -q            # all CPU smoke tests
.venv/bin/python -m model.forward_test               # M4 on the stand-in
.venv/bin/python -m model.forward_test --real        # M4 on Qwen2.5-VL-3B (needs weights + ideally CUDA)
.venv/bin/python -m model.cot_annotator              # M5 -> outputs/phase2/cot_annotations.jsonl
.venv/bin/python -m model.trainer --config configs/sft.yaml              # M6 on a GPU
.venv/bin/python -m model.evaluate --checkpoint checkpoints/sft/adapter.pt
.venv/bin/python -m model.uncertainty --checkpoint checkpoints/sft/adapter.pt --tau 0.5
.venv/bin/python -m model.gating --checkpoint checkpoints/sft/adapter.pt
.venv/bin/python -m model.grpo --sft-checkpoint checkpoints/sft/adapter.pt --out checkpoints/grpo
.venv/bin/python -m model.benchmark --sft ... --grpo-nocal ... --grpo-cal ...
```

Add `--standin` to any of them to use the tiny CPU model (`trainer`, `evaluate`, `uncertainty`,
`gating`, `grpo`, `benchmark` all accept it).

## Design decisions worth knowing

* **Constrained decoding.** After the first `<action_i>` (or the CoT budget) the logits are masked to
  the 2048 action ids until exactly 10 are emitted, then EOS is forced, so every plan decodes to a
  drivable trajectory. `evaluate_holdout` additionally reports *free-running* validity with the mask
  off, which is the honest measure of what SFT learned.
* **LoRA-only invariant.** `assert_lora_only` runs at every trainer start. Trainable = LoRA on the
  language model's projections + the *rows* of the embedding that belong to the 2048 new tokens
  (PEFT `trainable_token_indices`; tied head handled, untied head gets its own rows). The vision tower
  is frozen and excluded from LoRA by regex.
* **L_LM normalisation.** Eq. 2 does not say how L_LM is averaged. A per-sample mean starves the
  fast/slow opener token in long CoT records (1/500 vs 1/8 weight) and biases the learned switch to
  "fast"; the default is therefore a token-level mean (`lm_norm: token`).
* **Uncertainty features.** The head reads pooled pre-decoder representations: vision-encoder tokens
  (zero when there are no images) ‖ mean prompt-token embedding ‖ ego state. On the image-free
  synthetic corpus only the last two carry signal.
* **Epistemic = policy-level MC-dropout.** Every `nn.Dropout` (the LoRA dropout, p=0.1) is switched on
  and the plan is re-decoded T=10 times; σ²_epi is the mean per-step position variance (m²). An
  ensemble sampler is provided but would need 3 trained adapters.
* **Calibration confidence.** `confidence = 1/(1+σ²_ale+σ²_epi)` is read as P(plan succeeds), success
  being ADE ≤ τ. Temperature scaling fits `sigmoid(logit(conf)/T)` by NLL on a split disjoint from
  both head-training and test.
* **r_calibration confidence source.** In GRPO the sample-level confidence is the policy's own
  geometric-mean action-token probability (it varies across the G samples, so it carries gradient);
  the head's scene-level confidence is constant within a group and would only re-weight scenes.
  Pass `confidence_fn` to `GRPOTrainer` to use another source.
* **Stand-in rescaling.** The stand-in's tokenizer yields far more tokens per sentence than Qwen's, so
  stand-in GRPO runs rescale `L_tol`/γ/`cot_max_tokens` and `τ_fail` (the stand-in's ADE is metres, not
  decimetres). Real runs use the paper constants in `config.py`.

## Real-model result: Qwen2.5-VL-3B + LoRA, Kaggle T4, synthetic corpus

`model/run_kaggle.sh` (SFT 1200 optimizer steps at ~9.5 s/step, fp16, `configs/sft_t4.yaml`), then
`model.evaluate` on 40 of the 200 held-out scenes. Synthetic corpus, no images, trajectory-templated CoT,
synthetic agent boxes. These are not nuScenes numbers. Full table: `results/openloop_sft_t4_1200.csv`.

| mode (all, n=40) | L2@1s | L2@3s | ADE | Col@3s | latency |
|---|---|---|---|---|---|
| fast | 0.46 | 4.01 | 4.13 | 0.050 | 2.0 s |
| slow | 0.47 | 4.13 | 4.23 | 0.050 | 27.9 s |
| random | 0.46 | 3.97 | 4.07 | 0.050 | 17.4 s |
| auto | 0.46 | 4.01 | 4.13 | 0.050 | 2.5 s |
| const_vel (baseline) | 0.69 | 6.10 | 6.11 | 0.100 | 0 |

* The model beats constant velocity on every split (ADE easy 1.00 vs 2.48, hard 7.94 vs 10.54). An earlier
  300-step adapter did not (ADE 8.29): at that point it was worse than a straight line.
* `slow` is no more accurate than `fast` and ~14x slower, as expected without images or informative CoT.
  `auto` never chose slow (identical to `fast`). With 40 scenes (18 hard), differences of ~0.1 m ADE between
  modes are noise.

## Results so far (tiny stand-in, synthetic corpus, synthetic agents; reproduce with `model/run_standin_pipeline.sh`)

Every number below validates the pipeline. None is a Qwen2.5-VL-3B or nuScenes result. Stand-in-only
rescalings: LoRA r=32 (3B uses 8), CoT cap 120 tokens, `L_tol=2`, `gamma=0.5`, `tau_fail=8 m`,
failure threshold `tau` = median ADE, gate ADE scale 25 m. Latencies are CPU wall-clock of a ~250k-parameter model.

**M6 SFT** (1500 optimizer steps, grad-accum 8, fast:slow 3:1): mean loss over the first quarter of
steps 14.2 → last quarter 8.6. On 20 held-out records: 100 % emit exactly 10 valid action tokens with
constrained decoding, **85 % free-running** (decoder mask off), token accuracy 20 %.

**M7 open-loop** (200 held-out scenes; per horizon, synthetic agent boxes):

| mode | L2@1s | L2@2s | L2@3s | Col@1s | Col@2s | Col@3s | latency |
|---|---|---|---|---|---|---|---|
| fast | 2.66 | 5.69 | 8.91 | 0.010 | 0.060 | 0.115 | 16.0 ms |
| slow | 4.10 | 8.54 | 13.21 | 0.015 | 0.050 | 0.085 | 127.1 ms |
| random | 3.49 | 7.50 | 11.83 | 0.010 | 0.055 | 0.115 | 78.3 ms |
| auto (learned switch) | 2.66 | 5.69 | 8.91 | 0.010 | 0.060 | 0.115 | 19.8 ms |

Fast mode cuts latency by 87.4 % (paper: 66.8 %, different hardware and CoT length). Slow mode is *worse*
here because the stand-in cannot write a useful CoT; `auto` picks fast every time (75 % prior, greedy).

**M8 uncertainty** (800 test scenes, `tau` = 7.1 m, failure rate 48 %): failure-prediction **AUROC 0.906**
(aleatoric alone 0.931, epistemic alone 0.760); mean σ² on the hard split 789 m² vs 266 m² easy
(AUROC of hard-vs-easy from uncertainty 0.715). **ECE 0.337 → 0.049** after calibration, but only with the
NLL-selected *temperature + bias* calibrator: pure logit-temperature scaling reaches just 0.218 and
variance-temperature scaling 0.068 (candidates are all reported in `outputs/phase2/unc_report_*.json`).
0.049 is under the 0.05 target by a hair; across seeds on similar data it wobbles between 0.04 and 0.06.
High AUROC is partly because the synthetic error is almost a function of speed/command, which the head sees.

**M9 gate** (250 test scenes): the optimal thresholds have **no slow band** (high = low) because slow never
beats fast on this stand-in. Gate on the hard split: 18 % minimal-risk vs 6 % on easy; epistemic-top-quartile
scenes 21 % MRM vs 5 %. With a forced ≥15 % slow band: slow 14.8 % easy vs 17.6 % hard, 24 % on epistemic-high
vs 13 % otherwise. Mean latency on the easy split: gated 60 ms vs always-slow 128 ms (−53 %) and random 74 ms
(−19 %), but **above always-fast (16 ms)** because confidence estimation (features + 10 batched MC passes) isn't free.
Aleatoric-only gating costs 16 ms.

**M10 GRPO** (300 steps, 4 scenes x G=8 per step, 200 fixed held-out scenes, one sample each):
reward −1.075 → −0.901 after 75 steps without r_calibration, then flat (−0.923 at step 300). The per-sample reward std is 0.87 (SE of a 200-sample mean 0.06), so the early gain is about 2σ: suggestive, not decisive;
with r_calibration −1.078 → −0.928 → −0.867. Sampled slow fraction 7.5 % → ~0, mean CoT length on easy scenes
0.29 → 0.0 tokens. The CoT-length part is **weak evidence**: this policy writes ~0 CoT tokens to begin with.
A fixed-4-scene sanity run shows the optimiser itself works (training reward −1.40 → −0.85).

**M11 ablation** (500 test scenes; `results/ablation_table.csv`, 95 % CIs in the CSV):

| Model variant | L2@3s ↓ | Col@3s ↓ | ECE ↓ | Latency ↓ | AUROC ↑ | route compl. |
|---|---|---|---|---|---|---|
| SFT only | 9.68 | 0.148 | 0.259 | 19.8 ms | 0.703 | 0.68 |
| + GRPO (no head) | 9.45 | 0.216 | 0.104 | 19.7 ms | 0.763 | 0.74 |
| + uncertainty head (no calib reward) | 7.35 | 0.086 | 0.049 | 55.5 ms | 0.926 | 0.60 |
| + calibration reward (FULL) | 7.61 | 0.102 | 0.039 | 56.7 ms | 0.930 | 0.58 |

How *not* to read it: rows 1→2 differ by less than their CIs on L2, and GRPO raised collisions. Rows 3-4 improve
L2/collision mostly because 30-35 % of scenes get the stop manoeuvre (route completion falls 0.7 → 0.6) and they
are slower, not faster, than rows 1-2. Rows 3 and 4 differ by less than their CIs, so the table does **not** show a
benefit of the calibration reward; the ECE gap (0.049 vs 0.039) is within seed noise. Rows 1-2 report ECE for the
policy's own action-token probability (uncalibrated), rows 3-4 for the calibrated fused confidence.

## Bugs/design problems found while validating (and what was done)

* Per-sample-mean L_LM starves the fast/slow opener token → switch collapses to "fast". Default is now a token-level mean.
* Stop token inside L_LM was learned too slowly (free-running validity 0 %) → it now sits in the action term (85 %).
* Pure temperature scaling cannot calibrate saturated confidences (σ² in the hundreds of m²) → three calibrators, NLL-selected.
* 10 sequential MC-dropout passes cost more than the slow path they avoid → one batched `generate` (`generate_action_batch`).
* A driving score without route completion made "always brake to a stop" the optimal gate (100 % MRM in the ablation rows)
  → score is now route_completion x (1-collided) x accuracy. Its ADE scale must match the policy's error level.
* At lr 2e-3 on a larger stand-in, GRPO collapsed to "slow mode with an empty CoT" (length 0, so no length penalty).
  `L` counts only tokens between the mode prefix and the first action; a format reward would be the real fix and is not implemented.
