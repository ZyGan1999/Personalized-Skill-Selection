#!/bin/bash
# Main paper experiments: Phase 1 (one-hot + soft preference) for all API models.
# Each experiment uses 50 users x 500 rounds x 3 seeds.
#
# Usage:
#   export OPENAI_API_KEY=your_packy_key
#   export OPENAI_API_BASE=https://www.packyapi.com/v1
#   bash run_main_experiments.sh

set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

# Default to packy if no base URL set
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://www.packyapi.com/v1}"
# Unset OPENAI_CHAT_URL — it takes priority over OPENAI_API_BASE in agents._llm_call
unset OPENAI_CHAT_URL
echo "API base: $OPENAI_API_BASE"

# Models to test (gpt-5.2 already done via right.codes — skip if you don't want to re-run)
MODELS=(
    "qwen3-30b-a3b-instruct-2507"
    "qwen3-235b-a22b-thinking-2507"
    "deepseek-v4-flash"
    # "gpt-5.2"   # uncomment to re-run via packy
)

OUTPUT_BASE="paper-exp"
USERS=50
ROUNDS=500
SEEDS="0 1 2"
TEMP=3.0
WANDB_PROJECT="${WANDB_PROJECT:-tool-call-bandit}"  # set WANDB_PROJECT="" to disable

mkdir -p "$OUTPUT_BASE"

run_experiment() {
    local model=$1
    local mode=$2          # "onehot" or "soft0.3"
    local extra_args=$3
    # Sanitize model name for filesystem (replace / with _)
    local model_safe="${model//\//_}"
    local out_dir="$OUTPUT_BASE/main-${mode}-${model_safe}"

    echo
    echo "========================================================"
    echo "[$(date '+%H:%M:%S')] Running: model=$model, mode=$mode"
    echo "Output: $out_dir"
    echo "========================================================"

    local wandb_args=""
    if [[ -n "$WANDB_PROJECT" ]]; then
        wandb_args="--wandb-project $WANDB_PROJECT --wandb-run-name ${mode}-${model_safe}"
    fi

    conda run --no-capture-output -n tool-call python -u main.py \
        --benchmark benchmark_data/toolbench_60.json \
        --model "$model" \
        --users $USERS --rounds $ROUNDS --seeds $SEEDS \
        --temperature $TEMP \
        --resume --verbose --export-csv \
        --output-dir "$out_dir" \
        $wandb_args \
        $extra_args
}

for model in "${MODELS[@]}"; do
    # One-hot main experiment
    run_experiment "$model" "onehot" ""

    # Soft preference (concentration=0.3) main experiment
    run_experiment "$model" "soft0.3" "--soft-preferences --concentration 0.3"
done

echo
echo "========================================================"
echo "[$(date '+%H:%M:%S')] All experiments completed."
echo "Results in: $OUTPUT_BASE/"
echo "========================================================"
