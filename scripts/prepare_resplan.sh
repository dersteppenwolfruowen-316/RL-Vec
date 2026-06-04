#!/bin/bash

set -e

echo "=========================================="
echo "Downloading ResPlan dataset"
echo "=========================================="

cd "$(dirname "$0")/.."
DATA_DIR="./data/resplan"
mkdir -p "$DATA_DIR"

if [ -d "$DATA_DIR/train" ] && [ -d "$DATA_DIR/val" ]; then
    echo "ResPlan dataset already exists at $DATA_DIR"
    ls -la "$DATA_DIR"
    exit 0
fi

echo "Downloading ResPlan v1.0..."
wget -O "$DATA_DIR/resplan_v1.tar.gz" \
    https://github.com/m-agour/ResPlan/releases/download/v1.0/resplan_v1.tar.gz \
    2>/dev/null || echo "Download failed, please download manually from:"
    echo "https://github.com/m-agour/ResPlan/releases/download/v1.0/resplan_v1.tar.gz"

if [ -f "$DATA_DIR/resplan_v1.tar.gz" ]; then
    echo "Extracting..."
    tar -xzf "$DATA_DIR/resplan_v1.tar.gz" -C "$DATA_DIR"
    rm "$DATA_DIR/resplan_v1.tar.gz"
    echo "ResPlan dataset ready at $DATA_DIR"
else
    echo "Please download manually and place in $DATA_DIR"
fi
