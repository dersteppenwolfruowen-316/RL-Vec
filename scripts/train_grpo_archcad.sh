#!/bin/bash
set -e

EXPERIMENT_NAME=${EXPERIMENT_NAME:-archcad_8b_grpo_$(date +%Y%m%d_%H%M%S)}
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== RL Vectorizer EasyR1 Workflow (ArchCAD) ==="
echo "Dataset:    ArchCAD (40K CAD drawings)"
echo "Model:      Qwen3-VL-8B-Instruct"
echo "Algorithm:  GRPO"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Root:       ${PROJECT_ROOT}"
echo ""

if [ ! -d "data/archcad" ]; then
    echo "[1/4] Downloading ArchCAD dataset ..."
    python download_archcad.py --output-dir data/archcad
    echo ""
fi

if [ ! -d "data/easyr1_archcad" ]; then
    echo "[2/4] Preparing ArchCAD dataset for EasyR1 ..."
    python scripts/prepare_archcad_data.py \
        --data-dir data/archcad/data \
        --output-dir data/easyr1_archcad \
        --train-ratio 0.95
    echo ""
fi

if [ ! -d ".venv_easyr1" ]; then
    echo "[3/4] Setting up EasyR1 virtual environment ..."
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
    echo "[3/4] EasyR1 venv already exists, activating ..."
    source .venv_easyr1/bin/activate
fi

echo "[4/4] Launching GRPO training (ArchCAD, 8B, 8 GPUs) ..."
python -m verl.trainer.main \
    config=config/easyr1/qwen3_vl_8b_archcad_grpo.yaml \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS:-8}" \
    "$@"
