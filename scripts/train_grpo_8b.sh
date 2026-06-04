#!/bin/bash
set -e

EXPERIMENT_NAME=${EXPERIMENT_NAME:-floorplan_8b_grpo_$(date +%Y%m%d_%H%M%S)}
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== RL Vectorizer EasyR1 Workflow ==="
echo "Model:      Qwen3-VL-8B-Instruct"
echo "Algorithm:  GRPO"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Root:       ${PROJECT_ROOT}"
echo ""

if [ ! -d "data/easyr1_resplan" ]; then
    echo "[1/3] Preparing ResPlan dataset for EasyR1 ..."
    python scripts/prepare_easyr1_data.py \
        --metadata data/resplan/metadata.jsonl \
        --base-dir data/resplan \
        --output-dir data/easyr1_resplan \
        --train-ratio 0.9
    echo ""
fi

if [ ! -d ".venv_easyr1" ]; then
    echo "[2/3] Setting up EasyR1 virtual environment ..."
    python -m venv .venv_easyr1
    source .venv_easyr1/bin/activate

    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

    pip install easyr1
    pip install -r requirements.txt

    pip install WandB
    wandb login || echo "WandB login skipped — set WANDB_API_KEY env var instead"
    echo ""
else
    echo "[2/3] EasyR1 venv already exists, activating ..."
    source .venv_easyr1/bin/activate
fi

echo "[3/3] Launching GRPO training (8B, single-node, 8 GPUs) ..."
python -m verl.trainer.main \
    config=config/easyr1/qwen3_vl_8b_grpo.yaml \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS:-8}" \
    "$@"
