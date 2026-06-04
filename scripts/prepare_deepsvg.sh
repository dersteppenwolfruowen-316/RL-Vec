#!/bin/bash

echo "Downloading DeepSVG Icons8 dataset..."

cd "$(dirname "$0")/.."

DATA_DIR="./data/deepsvg"

if [ -d "$DATA_DIR" ]; then
    echo "Dataset already exists at $DATA_DIR"
    exit 0
fi

mkdir -p "$DATA_DIR"

echo "DeepSVG requires manual download from Google Drive (3GB)"
echo "Please visit: https://google.github.io/deepsvg/"
echo "Alternative: Use starvector/svg-stack from HuggingFace instead:"
echo "  python -c \"from datasets import load_dataset; ds = load_dataset('starvector/svg-stack', split='train')\""
