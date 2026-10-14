#!/usr/bin/env bash
# Whole Phase-II real-model run on a Kaggle T4, start to finish, in ONE cell:
#     !bash /kaggle/working/autovla-phase1/model/run_kaggle.sh      (after the clone below)
# or paste the clone + this script as the only cell and use "Save Version -> Save & Run All (Commit)".
# A committed run executes headless and keeps /kaggle/working as notebook output, so the adapter survives.
# Interactive sessions are wiped when they end; that is how the first adapters were lost.
#
#   STEPS=1200  optimizer steps (about 9.5 s/step on a T4 -> 1200 steps is about 3.2 h)
#   NEVAL=40    held-out scenes scored per mode
set -euo pipefail
STEPS=${STEPS:-1200}
NEVAL=${NEVAL:-40}
cd /kaggle/working
[ -d autovla-phase1 ] || git clone -q https://github.com/Raghuluv04/autovla-phase1.git
cd autovla-phase1
git pull -q

pip install -q -r data_pipeline/requirements.txt transformers peft bitsandbytes accelerate

echo "== corpus + CoT annotations"
python data_pipeline/scripts/run_pipeline.py --n-traj 4000 --outdir data_pipeline/outputs > pipeline.log 2>&1
python -m model.cot_annotator

echo "== smoke test (real model)"
python -m model.forward_test --real

echo "== SFT: $STEPS steps"
python -u -m model.trainer --config configs/sft_t4.yaml --steps "$STEPS" --n-eval 20

echo "== open-loop evaluation: fast / slow / random / auto + const_vel baseline"
python -u -m model.evaluate --checkpoint checkpoints/sft_t4/adapter.pt --compute-dtype float16 \
    --n "$NEVAL" --tag "sft_t4_${STEPS}"
cat "results/openloop_sft_t4_${STEPS}.csv"
echo "== done; outputs: checkpoints/sft_t4/{adapter.pt,history.json} results/openloop_sft_t4_${STEPS}.csv"
