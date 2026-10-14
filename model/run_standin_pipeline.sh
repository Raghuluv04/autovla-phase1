#!/usr/bin/env bash
# Reproduces every stand-in number in results/ (tiny CPU model, synthetic corpus, synthetic agents).
# These runs validate the pipeline; they are NOT Qwen2.5-VL-3B / nuScenes results.
# On a GPU, drop --standin, point the commands at configs/sft.yaml and the real checkpoints, and
# remove the stand-in rescalings (--cot-max-tokens/--l-tol/--gamma/--tau-fail/--lora-rank).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CK=checkpoints
SFT=$CK/sft_standin/adapter.pt   # hidden 64, 2 layers, LoRA r=32
COT=120          # stand-in CoT cap (its tokenizer is ~4x less efficient than Qwen's)
ADE_SCALE=25     # stand-in ADE is ~8-10 m; the 5 m default assumes real-driving ADE of ~1 m

$PY -m model.cot_annotator
$PY -m model.trainer --standin --steps 1500 --lr 3e-3 --lora-rank 32 --holdout 3000                                                       # M6
$PY -m model.evaluate --standin --checkpoint $SFT --n 200 \
    --cot-max-tokens $COT --tag standin_sft                                                          # M7
$PY -m model.uncertainty --standin --checkpoint $SFT --tag standin_sft          # M8
$PY -m model.gating --standin --checkpoint $SFT --tag standin_sft \
    --cot-max-tokens $COT --ade-scale $ADE_SCALE                                                    # M9

COMMON="--standin --sft-checkpoint $SFT --steps 300 --scenes-per-step 4 --lr 1e-3 \
    --cot-max-tokens $COT --l-tol 2 --gamma 0.5 --tau-fail 8.0 --n-scenes 800 --n-eval 200 --eval-every 50"
OMP_NUM_THREADS=4 $PY -m model.grpo $COMMON --no-calibration --out $CK/grpo_standin_nocal &          # M10
OMP_NUM_THREADS=4 $PY -m model.grpo $COMMON --out $CK/grpo_standin_cal &
wait

$PY -m model.benchmark --standin --sft $SFT \
    --grpo-nocal $CK/grpo_standin_nocal/adapter.pt --grpo-cal $CK/grpo_standin_cal/adapter.pt \
    --cot-max-tokens $COT --ade-scale $ADE_SCALE                                                    # M11
