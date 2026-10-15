#!/usr/bin/env bash
# Milestone 8 (uncertainty head) on the real SFT adapter, Kaggle T4. Headless-friendly: use
# "Save Version -> Save & Run All (Commit)" with the single cell shown in model/README.md.
#
# Needs the SFT adapter attached as a Kaggle Dataset (Add Input -> your dataset containing adapter.pt).
# SFT held out only 200 scenes, but the head needs ~1600 the policy never fit, so scenes are drawn from the
# records the 1200-step SFT run never saw (replayed from its seeded sampler; valid for an un-resumed run).
#
#   N_TRAIN/N_CALIB/N_TEST  scenes per split (default 800/300/500; each scene costs a plan, 10 MC-dropout
#                            passes and a feature pass, so read the printed s/scene before scaling up)
#   SFT_STEPS               optimizer steps of the SFT run that produced the adapter (default 1200)
set -euo pipefail
N_TRAIN=${N_TRAIN:-800}; N_CALIB=${N_CALIB:-300}; N_TEST=${N_TEST:-500}; SFT_STEPS=${SFT_STEPS:-1200}
ADAPTER=${ADAPTER:-$(find /kaggle/input -name adapter.pt 2>/dev/null | head -n1)}
[ -n "$ADAPTER" ] || { echo "no adapter.pt under /kaggle/input: attach the dataset holding it"; exit 1; }
echo "adapter: $ADAPTER"
cd /kaggle/working
[ -d autovla-phase1 ] || git clone -q https://github.com/Raghuluv04/autovla-phase1.git
cd autovla-phase1
git pull -q

pip install -q -r data_pipeline/requirements.txt transformers peft bitsandbytes accelerate
echo "== corpus + CoT annotations (same seed as the SFT run, so the unseen-record replay lines up)"
python data_pipeline/scripts/run_pipeline.py --n-traj 4000 --outdir data_pipeline/outputs > pipeline.log 2>&1
python -m model.cot_annotator

echo "== uncertainty: $N_TRAIN/$N_CALIB/$N_TEST scenes"
python -u -m model.uncertainty --checkpoint "$ADAPTER" --compute-dtype float16 \
    --unseen-sft-config configs/sft_t4.yaml --sft-steps "$SFT_STEPS" \
    --n-train "$N_TRAIN" --n-calib "$N_CALIB" --n-test "$N_TEST" --tag sft_t4_1200
echo "== done; outputs/phase2/{unc_head,unc_report,unc_scenes}_sft_t4_1200.*"
