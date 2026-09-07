# AutoVLA Reimplementation — Milestone Workflow

**Team 5** · Raghul D (23BAI0034), Sahithya D (23BAI0035), Medha S (23BAI0039)
**Course** BCSE332L Deep Learning · Phase I Course Based Design Project
**Base paper** Zhou et al., *AutoVLA: A Vision-Language-Action Model for End-to-End
Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning*,
arXiv:2506.13757 (2025) · <https://autovla.github.io/>

---

## How the work splits

AutoVLA is the spine; the project's six objectives extend it with an
uncertainty channel that the paper does not have. Milestones 1–3 rebuild the
paper's data layer, 4–7 rebuild its model and training, and 8–11 add the
project's own contribution.

| Objective | Owner | Lands in |
|---|---|---|
| 1. Multi-camera → BEV perception backbone | Raghul | M4 (see risk R2) |
| 2. Bayesian uncertainty head (aleatoric + epistemic) | Raghul | M8 |
| 3. VLM + LoRA, CoT justification generation | Sahithya | M5, M6 |
| 4. Physical action tokenizer + explanation metrics | Sahithya | **M2 (done)**, M7 |
| 5. Uncertainty-gated adaptive reasoning | Medha | M9 |
| 6. GRPO reinforcement fine-tuning + benchmarking | Medha | M10, M11 |

---

## Phase I — data (this review)

### M0 · Environment and repo scaffold — **done**
Python venv, numpy/scipy/matplotlib, package layout, config with every constant
traced to the paper.
**Done when:** `python scripts/run_pipeline.py` runs start to finish.

### M1 · Data acquisition and unified schema — **done (synthetic), nuScenes pending**
Owner: Raghul
The paper standardises nuPlan/nuScenes/Waymo/CARLA into one record (Appendix
E.1): 2 Hz ego-frame future trajectory, 4-frame × 3-camera image paths, CoT
annotation, vehicle state, navigation instruction. `autovla/schema.py` is that
record; `autovla/adapters/` converts each source into it.
- `adapters/synthetic.py` — kinematic-bicycle corpus, runs today, no download.
- `adapters/nuscenes.py` — reads the nuScenes JSON tables directly (no devkit),
  works on v1.0-mini and v1.0-trainval.
**Done when:** nuScenes mini extracted and `--source nuscenes` reproduces the
same report. **Blocked only on the 4 GB download** (needs a signed-in account).

### M2 · Physical action tokenizer — **done**
Owner: Sahithya
K-disk clustering of 0.5 s motion segments into K = 2048 tokens, each a
body-frame `(Δx, Δy, Δθ)`; encode by nearest contour distance, decode by SE(2)
composition. Every token inverted through the bicycle model to confirm
feasibility (Objective 4's dynamics check).
**Done when:** round-trip ADE within an order of magnitude of the paper's
Table 4. **Achieved: ADE 2.40 cm / FDE 2.54 cm vs the paper's 1.82 / 2.03.**

### M3 · Statistics and visualisation — **done ← next week's deliverable**
Owner: all three
Eight figures in `outputs/figures/`, plus `outputs/artifacts/report.txt`.
See [README.md](README.md) for what each figure shows and says.

---

## Phase II — model

### M4 · VLM backbone and action-token vocabulary
Owner: Sahithya (backbone), Raghul (vision path)
Load Qwen2.5-VL-3B, extend the tokenizer with the 2048 `<action_i>` tokens
(already emitted to `outputs/artifacts/action_vocab.json`), resize embeddings,
attach LoRA (r=8, α=8, dropout 0.1 — the paper's RFT setting), freeze the
vision encoder. Verify a forward pass over one built record.
**Done when:** the model consumes a record from `sft_train.jsonl` and emits 10
action tokens that decode to a trajectory (untrained, so it will be wrong —
the point is that the plumbing is correct).
**Depends on:** M2 (vocabulary), M1 (records). **Needs a GPU** — see R1.

### M5 · CoT reasoning annotation
Owner: Sahithya
The paper distils reasoning from Qwen2.5-VL-72B with ground-truth actions given
as a hint, producing four fields: scene description, critical objects, agent
intentions, driving decision (`Reasoning` in `schema.py` is already this shape,
currently stubbed). For a student budget, substitute:
- DriveLM-nuScenes QA reformatted into the four fields (the paper does this too
  for nuScenes), plus
- BDD-X action/justification pairs for the explanation-quality benchmark.
**Done when:** ≥15% of samples carry real CoT text and BLEU-4/METEOR/CIDEr run
against BDD-X references.

### M6 · Supervised fine-tuning, dual thinking modes
Owner: Sahithya + Medha
Loss from Eq. 2: `L = w·(L_LM + λ_a·L_action)` with `w = λ_cot` when CoT is
present. Fast-thinking records get the short no-reasoning template; slow ones
get the four-step CoT. Mix both in one batch stream.
**Done when:** training loss decreases and the model emits exactly 10 valid
action tokens on held-out samples.

### M7 · Open-loop evaluation
Owner: Sahithya
nuScenes protocol (UniAD convention): L2 at 1 s / 2 s / 3 s and collision rate,
reported per horizon, not averaged. Compare fast vs slow mode for both accuracy
and latency — the paper's Table 2 motivation for adaptive reasoning.
**Done when:** a results table exists with a fast/slow latency gap reproduced.

---

## Phase II — the project's own contribution

### M8 · Bayesian uncertainty head
Owner: Raghul
Two heads over the visual features: a predictive log-variance branch
(aleatoric) and MC-dropout / small-ensemble sampling (epistemic), fused into a
confidence score per predicted trajectory. Calibrate with temperature scaling.
**Done when:** ECE and AUROC-for-failure-prediction are reported, and
uncertainty is measurably higher on a held-out hard split than an easy one —
this is the SUPER-AD result the proposal builds on.
**This is the part that is genuinely new relative to AutoVLA.**

### M9 · Uncertainty-gated adaptive reasoning
Owner: Medha
Replace the paper's implicit scenario-difficulty gate with a confidence gate:
low uncertainty → fast thinking; moderate → slow thinking with CoT; above a
calibrated threshold → conservative minimal-risk manoeuvre.
**Done when:** an ablation shows the gate changing mode on a hard split and
cutting mean latency on an easy one.

### M10 · GRPO reinforcement fine-tuning
Owner: Medha
Group-relative advantage `A_i = (r_i − r̄)/σ_r`, KL to the SFT reference policy,
LoRA-only updates. Reward `r = r_Driving − λ_r·r_CoT` with `λ_r = 0.3` and the
sigmoid CoT-length penalty `r_CoT = 1/(1+e^{−(L−L_tol)γ})`, `L_tol = 400`,
`γ = 2e−3`. For nuScenes (no PDMS scorer available) use the paper's Waymo
variant `r_Driving = (δ − ADE)/κ` with `δ = 2`, `κ = 10`.
**Extension:** add the calibration penalty term from the proposal, so
confident-but-wrong actions are punished — the novel element, since existing
RFT optimises task reward only.
**Done when:** the reward curve rises and mean CoT length falls on easy scenes.

### M11 · Benchmarking and report
Owner: all three
Collision rate, route completion, L2, ECE, AUROC; ablations for codebook size
(already have the data pipeline half), fast/slow/adaptive, and with/without the
calibration reward term.

---

## Risk register

**R1 · Compute.** Qwen2.5-VL-3B LoRA fine-tuning will not train usefully on a
24 GB M4 Mac. Plan on Colab/Kaggle T4 or A100 sessions and keep checkpoints
small. Mitigation: 4-bit loading, batch size 1 with gradient accumulation, and
a subset of nuScenes rather than nuPlan's 166k samples. *Decide this before M4
starts, because it sets the dataset size for M1.*

**R2 · Objective 1 is not in the base paper.** AutoVLA has no BEV transformer —
it feeds camera frames straight into the Qwen2.5-VL vision encoder. Objective 1
(BEVFormer-style deformable cross-attention into a BEV grid) is an *addition*,
and training a BEV encoder is a large job on its own. Two honest options: (a)
scope Objective 1 down to using a pretrained/frozen BEV encoder as an auxiliary
feature source, or (b) reinterpret it as the visual-token representation the VLM
already produces and put the effort into M8 instead. Worth settling with the
faculty guide early — it changes Raghul's workload substantially.

**R3 · CARLA closed-loop is the highest-risk deliverable.** CARLA 0.9.x needs
Linux/Windows with an NVIDIA GPU; it will not run on the Macs in this team.
Mitigation: do the RL fine-tuning against the open-loop ADE reward (the paper's
own Waymo setting) and use the CARLA-Garage *dataset* for offline evaluation
rather than live closed-loop rollout. Only attempt live CARLA if a lab machine
is available.

**R4 · Dataset size on disk.** ~29 GB free on the dev machine. nuScenes mini
(4 GB) fits; full trainval (300+ GB) does not. Every milestone above is written
to work on mini.

**R5 · Explanation faithfulness.** BLEU/METEOR/CIDEr measure fluency against a
reference, not whether the explanation reflects the actual policy. If M5 lands
early, add a simple counterfactual check (perturb the critical object, see
whether the justification changes) — the proposal calls out hallucinated
rationales as a known failure and this is cheap evidence against it.

---

## Suggested schedule

| Week | Milestone | Owner |
|---|---|---|
| 1 (now) | M0–M3 complete, nuScenes mini downloaded | all |
| 2–3 | M4 backbone plumbing, M5 CoT annotation started | Sahithya, Raghul |
| 4–5 | M6 SFT run, M7 open-loop numbers | Sahithya, Medha |
| 6–7 | M8 uncertainty head + calibration | Raghul |
| 8–9 | M9 gating, M10 GRPO | Medha |
| 10 | M11 benchmarks, ablations, report | all |
