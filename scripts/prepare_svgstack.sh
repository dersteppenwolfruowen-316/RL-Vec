#!/bin/bash

set -e

echo "=========================================="
echo "Preparing SVG-Stack Dataset"
echo "=========================================="

cd "$(dirname "$0")/.."
DATA_DIR="./data/svgstack"
mkdir -p "$DATA_DIR"

if [ -f "$DATA_DIR/train_metadata.jsonl" ]; then
    echo "SVG-Stack dataset already exists at $DATA_DIR"
    echo "To re-download, remove the directory first:"
    echo "  rm -rf $DATA_DIR"
    exit 0
fi

echo "Downloading SVG-Stack from HuggingFace..."
echo "This may take a while depending on your internet connection."
echo ""

python scripts/prepare_svgstack.py \
    --output-dir "$DATA_DIR" \
    --max-train 100000 \
    --max-val 5000 \
    --max-test 2000 \
    --validate

echo ""
echo "SVG-Stack dataset ready at $DATA_DIR"
echo ""
echo "Files:"
ls -lh "$DATA_DIR"/*.jsonl 2>/dev/null || echo "No metadata files found"
