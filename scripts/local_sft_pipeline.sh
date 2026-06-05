#!/bin/bash
# ═══════════════════════════════════════════════════════
# RL Vectorizer — 本地小规模 SFT Pipeline
# 用途: 在 Mac (CPU/MPS) 上跑通完整的 SFT 流程
# 用法: bash scripts/local_sft_pipeline.sh
# ═══════════════════════════════════════════════════════

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "═══════════════════════════════════════════════════"
echo "  RL Vectorizer — Local SFT Pipeline"
echo "═══════════════════════════════════════════════════"
echo ""

echo "[0/4] Checking data..."

if [ ! -f "data/resplan/ResPlan.pkl" ]; then
    echo "  X ResPlan.pkl not found at data/resplan/"
    echo "  Run these first (or use Colab):"
    echo "    mkdir -p data/resplan"
    echo "    # Download ResPlan.zip from https://github.com/m-agour/ResPlan/releases"
    echo "    unzip ResPlan.zip -d data/resplan/"
    exit 1
fi
echo "  OK ResPlan.pkl exists"

echo ""
echo "[1/4] Converting ResPlan → SVG + PNG..."
if [ ! -d "data/resplan/svgs" ] || [ ! -d "data/resplan/bitmaps" ]; then
    python convert_resplan.py
    echo "  OK Done"
else
    echo "  OK Already exists, skipping"
fi

echo ""
echo "[2/4] Preparing mini SFT dataset (25 samples)..."
if [ ! -f "data/resplan/sft_train.jsonl" ]; then
    python scripts/prepare_sft_data.py
fi

# 取前 20 条做训练，后 5 条做测试
mkdir -p data/resplan/mini
head -20 data/resplan/sft_train.jsonl > data/resplan/mini/train.jsonl
tail -5 data/resplan/sft_train.jsonl > data/resplan/mini/test.jsonl
echo "  OK Train: $(wc -l < data/resplan/mini/train.jsonl) samples"
echo "  OK Test:  $(wc -l < data/resplan/mini/test.jsonl) samples"

echo ""
echo "[3/4] SFT training..."
echo "  Model: Qwen2.5-VL-3B-Instruct (4bit)"
echo "  Data:  data/resplan/mini/train.jsonl"
echo "  Epochs: 5, LR: 5e-5, LoRA rank: 32"
echo ""

python scripts/train_sft.py \
    --data-path data/resplan/mini/train.jsonl \
    --max-samples 20 \
    --batch-size 1 \
    --epochs 5 \
    --lr 5e-5 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --quantization 4bit \
    --save-dir checkpoints/sft \
    --log-interval 2

echo ""
echo "  OK SFT training complete"

echo ""
echo "[4/4] Evaluation..."
echo "  Note: 本地 CPU 评估较慢，仅在 GPU 可用时运行"
echo ""

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "  Running evaluation on GPU..."
    python -c "
import sys, json
sys.path.insert(0, 'src')
from pathlib import Path
import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel
from skimage.metrics import structural_similarity as ssim
import numpy as np
from lxml import etree

TEST_SAMPLES = []
with open('data/resplan/mini/test.jsonl') as f:
    for line in f:
        TEST_SAMPLES.append(json.loads(line.strip()))

MODEL_NAME = 'Qwen/Qwen2.5-VL-3B-Instruct'
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)

def validate_svg(svg_code):
    try:
        root = etree.fromstring(svg_code.encode('utf-8'))
        return root.tag.endswith('svg')
    except: return False

prompt = 'Convert this architectural floor plan to SVG format.'

print('  Zero-shot:')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, quantization_config=quant_config, device_map='auto', torch_dtype=torch.float16,
)
processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=False)
for s in TEST_SAMPLES:
    img_path = s['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join('data/resplan/bitmaps', os.path.basename(img_path))
    img = Image.open(img_path).convert('RGB')
    msgs = [{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':prompt}]}]
    txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = processor(text=[txt], images=[img], return_tensors='pt')
    inp = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=1024, do_sample=False)
    resp = processor.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
    valid = 'OK' if validate_svg(resp) else 'X'
    print(f'    [{valid}] {s[\"id\"]}')
del model; torch.cuda.empty_cache()

print('  SFT:')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, quantization_config=quant_config, device_map='auto', torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(model, 'checkpoints/sft/final')
for s in TEST_SAMPLES:
    img_path = s['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join('data/resplan/bitmaps', os.path.basename(img_path))
    img = Image.open(img_path).convert('RGB')
    msgs = [{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':prompt}]}]
    txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = processor(text=[txt], images=[img], return_tensors='pt')
    inp = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=1024, do_sample=False)
    resp = processor.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
    valid = 'OK' if validate_svg(resp) else 'X'
    print(f'    [{valid}] {s[\"id\"]}')
" 2>&1
else
    echo "  WARN No GPU available, skipping evaluation."
    echo "  Evaluation available in Colab notebook after SFT training."
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Pipeline complete!"
echo "  SFT checkpoint: checkpoints/sft/final"
echo "  Train data:     data/resplan/mini/train.jsonl"
echo "  Test data:      data/resplan/mini/test.jsonl"
echo "═══════════════════════════════════════════════════"
