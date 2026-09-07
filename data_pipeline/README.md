# AutoVLA — reimplementation (Phase I: data processing and visualisation)

Reimplementation of **AutoVLA: A Vision-Language-Action Model for End-to-End
Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning**
(Zhou et al., 2025, [arXiv:2506.13757](https://arxiv.org/abs/2506.13757)),
for BCSE332L Deep Learning, Team 5.

Phase I covers the paper's **data layer**: the unified multi-dataset schema, the
physical action codebook, the tokenizer, and the analysis figures.
[MILESTONES.md](MILESTONES.md) has the full workflow through GRPO fine-tuning.

## Quick start

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

```bash
./.venv/bin/python scripts/run_pipeline.py --n-traj 4000 --ablation
```

Runs in ~90 s on a laptop, no GPU and no dataset download. Writes figures to
`outputs/figures/` and artifacts to `outputs/artifacts/`.

To run on real data, get nuScenes mini (4 GB, free account) from
<https://www.nuscenes.org/nuscenes#download>, then:

```bash
bash scripts/download_nuscenes.sh ~/Downloads/v1.0-mini.tgz && ./.venv/bin/python scripts/run_pipeline.py --source nuscenes
```

## What the pipeline does

```
raw driving logs
  → unified sample schema      (2 Hz ego-frame trajectory, 3 cameras × 4 frames,
                                velocity/acceleration, navigation instruction)
  → 0.5 s motion segments      (body-frame Δx, Δy, Δθ)
  → K-disk action codebook     (K = 2048 physically feasible motion primitives)
  → action tokenizer           (5 s trajectory ⇄ 10 discrete <action_i> tokens)
  → SFT chat records           (system prompt + observation + CoT + action tokens)
```

The point of the tokenizer is that it turns trajectory planning into
**next-token prediction**, so one autoregressive model emits its reasoning and
its trajectory in a single stream — the paper's central idea.

## Results against the paper

Tokenizer round-trip accuracy over 5 s trajectories, versus AutoVLA's Table 4
(K-disk column):

| K | our ADE | paper ADE | our FDE | paper FDE |
|---:|---:|---:|---:|---:|
| 256 | 8.42 cm | 6.87 cm | 9.48 cm | 10.34 cm |
| 1024 | 3.67 cm | 2.53 cm | 3.96 cm | 2.82 cm |
| **2048** | **2.40 cm** | **1.82 cm** | **2.54 cm** | **2.03 cm** |
| 4096 | 1.58 cm | 1.41 cm | 1.70 cm | 1.55 cm |

Same order of magnitude and the same shape of curve. Two other numbers match
the paper's construction directly: the achieved **covering radius is 0.063 m**
against the paper's stated δ = 0.05 m, and **99.9% of codebook tokens are
physically feasible** when inverted through the kinematic bicycle model.

The residual gap is expected — these numbers come from the procedural corpus,
not the Waymo Open Motion Dataset the paper clusters. Real logs are far denser
and more stereotyped, which is exactly what a nearest-neighbour codebook
rewards.

### One finding worth reporting

The paper says only that each 0.5 s segment is mapped "to its nearest action
token". *Which pose the segment is measured from* turns out to decide the
result. Measuring each segment from the **ground-truth** pose lets heading
quantization error rotate every later waypoint, and error compounds over the
horizon — 14.67 cm ADE at K = 2048, with FDE ≫ ADE. Measuring it from the pose
the decoder has **actually reached** lets each token correct its predecessor's
drift: 2.40 cm ADE, with FDE ≈ ADE.

The paper reports FDE ≈ ADE (2.03 vs 1.82), which is the signature of the
second scheme — so this is almost certainly what AutoVLA does, and it is worth
knowing because the naive reading of that sentence costs a 6× accuracy loss.
Both are implemented: `encode(traj, mode="anchored" | "open_loop")`.

## Figures

| File | What it shows |
|---|---|
| `00_pipeline_overview.png` | The pipeline with live counts at each stage |
| `01_action_codebook.png` | The 2048 learned motion primitives, footprints, speed coverage |
| `02_kinematic_feasibility.png` | Every token inverted through the bicycle model vs vehicle limits |
| `03_trajectory_reconstruction.png` | Round-trip on eight manoeuvre types |
| `04_codebook_size_ablation.png` | Error vs K, ours vs the paper, anchored vs open-loop |
| `05_token_usage.png` | Heavy-tailed token usage; 90% of usage from ~520 tokens |
| `06_dataset_statistics.png` | Speed, curvature, acceleration, command mix, endpoint density |
| `07_sample_card.png` | One processed sample end to end, as the VLM sees it |
| `08_camera_overlay.png` | Trajectory projected onto CAM_FRONT (nuScenes only) |

## Layout

```
autovla/
  config.py       constants, each traced to the paper
  geometry.py     SE(2) composition, vehicle footprint contours
  kinematics.py   bicycle model rollout, segment inversion, feasibility
  codebook.py     K-disk clustering
  tokenizer.py    encode / decode / reconstruction metrics
  schema.py       unified DrivingSample record
  prompts.py      system prompt, fast/slow response format
  build_dataset.py  → SFT chat records + action vocabulary
  stats.py        corpus statistics
  viz.py          all figures
  adapters/
    synthetic.py  procedural bicycle-model corpus (no download needed)
    nuscenes.py   direct nuScenes JSON reader + camera projection
scripts/
  run_pipeline.py       end-to-end driver
  download_nuscenes.sh  extract nuScenes mini
  download_bddx.sh      BDD-X explanation annotations
```

## A note on the synthetic corpus

`adapters/synthetic.py` is a **stand-in, not a contribution**. It integrates the
kinematic bicycle model over sampled manoeuvre profiles so that the codebook,
tokenizer, statistics and figures can be built and tested before nuScenes is
downloaded. It is not a model of driving behaviour and no claim in this report
depends on its realism — swap `--source nuscenes` and every stage downstream is
unchanged. Statistics reported from it are labelled as such.

## Key parameters (all from the paper)

| Parameter | Value | Source |
|---|---|---|
| Action token duration | 0.5 s | Sec. 3.1 |
| Planning horizon | 5 s → 10 tokens | Appendix A |
| Codebook size K | 2048 | Sec. 3.1, Table 4 |
| K-disk threshold δ | 0.05 m contour distance | Appendix A |
| Camera views | front-left, front, front-right | Sec. 3.1 |
| Frames per view | 4 at 2 Hz (2 s history) | Sec. 3.1 |
| Backbone | Qwen2.5-VL-3B | Sec. 3.1 |
| LoRA rank / alpha / dropout | 8 / 8 / 0.1 | Appendix D.3 |
| GRPO reward | `r_Driving − 0.3·r_CoT` | Appendix D.2 |
| CoT penalty | `1/(1+e^{−(L−400)·2e−3})` | Eq. S5 |
