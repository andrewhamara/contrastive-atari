#!/bin/bash
# A/B test: EfficientZero baseline vs EfficientZero + SupCon value-bin loss
#
# Runs 5 Atari games with 3 seeds each, comparing:
#   A) Baseline (supcon_coeff=0)
#   B) SupCon   (supcon_coeff=0.5)
#
# Results go to results/baseline_* and results/supcon_*
#
# Usage: bash scripts/run_ab_test.sh [--gpu_actor 2] [--cpu_actor 8]

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Config ────────────────────────────────────────────────────────────
GAMES=("BreakoutNoFrameskip-v4" "PongNoFrameskip-v4" "QbertNoFrameskip-v4" "SeaquestNoFrameskip-v4" "MsPacmanNoFrameskip-v4")
SEEDS=(0 1 2)
SUPCON_COEFF=0.5
NUM_GPUS=${NUM_GPUS:-1}
NUM_CPUS=${NUM_CPUS:-16}
GPU_ACTOR=${GPU_ACTOR:-2}
CPU_ACTOR=${CPU_ACTOR:-8}
AMP_TYPE=${AMP_TYPE:-torch_amp}

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu_actor) GPU_ACTOR=$2; shift 2;;
        --cpu_actor) CPU_ACTOR=$2; shift 2;;
        --num_gpus) NUM_GPUS=$2; shift 2;;
        --num_cpus) NUM_CPUS=$2; shift 2;;
        --supcon_coeff) SUPCON_COEFF=$2; shift 2;;
        --amp_type) AMP_TYPE=$2; shift 2;;
        --games) IFS=',' read -ra GAMES <<< "$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

COMMON_ARGS="--case atari --opr train --amp_type $AMP_TYPE --num_gpus $NUM_GPUS --num_cpus $NUM_CPUS --gpu_actor $GPU_ACTOR --cpu_actor $CPU_ACTOR --use_priority --use_max_priority --use_augmentation"

echo "════════════════════════════════════════════════════════════════"
echo "  EfficientZero A/B Test: Baseline vs SupCon (coeff=$SUPCON_COEFF)"
echo "  Games: ${GAMES[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "════════════════════════════════════════════════════════════════"

for game in "${GAMES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "── Baseline: $game (seed=$seed) ──────────────────────────────"
        python main.py --env "$game" --seed "$seed" \
            --result_dir results/baseline \
            --supcon_coeff 0.0 \
            --info "baseline_s${seed}" \
            $COMMON_ARGS || echo "FAILED: baseline $game seed=$seed"

        echo ""
        echo "── SupCon: $game (seed=$seed) ────────────────────────────────"
        python main.py --env "$game" --seed "$seed" \
            --result_dir results/supcon \
            --supcon_coeff "$SUPCON_COEFF" \
            --info "supcon_s${seed}" \
            $COMMON_ARGS || echo "FAILED: supcon $game seed=$seed"
    done
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All runs complete. Compare with tensorboard:"
echo "  tensorboard --logdir results/"
echo "════════════════════════════════════════════════════════════════"
