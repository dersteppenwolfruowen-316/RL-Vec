"""SFT 评估 Cell — 粘贴到 Colab notebook 作为新 cell 运行。
比较 zero-shot 和 SFT 在 5 张测试图上的效果。
"""
# ═══════════════════════════════════════════════════
# 把这个 cell 加到 notebook 的训练 cell 之后运行
# ═══════════════════════════════════════════════════

import sys, os, json, re, time
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from lxml import etree

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ── 配置 ──────────────────────────────────────────
TEST_IDS = [
    "resplan_00013", "resplan_00017", "resplan_00021",
    "resplan_00025", "resplan_00033",
]
DATA_DIR = "data/resplan"
SFT_CKPT = "checkpoints/sft/final"
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_NEW_TOKENS = 1536
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 工具函数 ──────────────────────────────────────

def load_image(sid):
    path = os.path.join(DATA_DIR, "bitmaps", f"{sid}.png")
    return Image.open(path).convert("RGB")

def load_gt_svg(sid):
    path = os.path.join(DATA_DIR, "svgs", f"{sid}.svg")
    return open(path).read()

def extract_svg_from_response(text):
    """从模型输出中提取 SVG 代码（兼容带中间指令的格式）。"""
    # 先尝试 <svg_output> 标签
    m = re.search(r"<svg_output>\s*(.*?)\s*</svg_output>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 再尝试 ```svg ... ``` 代码块
    m = re.search(r"```(?:svg)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 最后直接找 <svg ...>...</svg>
    m = re.search(r"(<svg[\s\S]*?</svg>)", text)
    if m:
        return m.group(1).strip()
    return text.strip()

def validate_svg(svg_code):
    try:
        root = etree.fromstring(svg_code.encode("utf-8"))
        return root.tag.endswith("svg")
    except Exception:
        return False

def count_svg_elements(svg_code):
    try:
        root = etree.fromstring(svg_code.encode("utf-8"))
        ns = {"svg": "http://www.w3.org/2000/svg"}
        paths = root.xpath("//svg:path", namespaces=ns)
        lines = root.xpath("//svg:line", namespaces=ns)
        rects = root.xpath("//svg:rect", namespaces=ns)
        return len(paths) + len(lines) + len(rects)
    except Exception:
        return 0

def check_instruction_format(text):
    """检查输出是否包含中间指令格式。"""
    tags = ["<analysis>", "<outer_wall>", "<svg_output>"]
    return sum(1 for t in tags if t in text)

def render_svg(svg_code, output_size=(512, 512)):
    """SVG → numpy array (RGB)。"""
    try:
        import cairosvg
        import io
        png = cairosvg.svg2png(
            bytestring=svg_code.encode(),
            output_width=output_size[0],
            output_height=output_size[1],
        )
        img = Image.open(io.BytesIO(png))
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        return np.array(img.convert("RGB"))
    except Exception:
        # fallback: 空白图
        return np.ones((*output_size[::-1], 3), dtype=np.uint8) * 255

def compute_ssim_score(gen_svg, target_img):
    try:
        gen_arr = render_svg(gen_svg, (target_img.width, target_img.height))
        tgt_arr = np.array(target_img.convert("RGB"))
        # 统一尺寸
        h = min(gen_arr.shape[0], tgt_arr.shape[0])
        w = min(gen_arr.shape[1], tgt_arr.shape[1])
        gen_arr = gen_arr[:h, :w]
        tgt_arr = tgt_arr[:h, :w]
        score = ssim(gen_arr, tgt_arr, channel_axis=2)
        return float(score)
    except Exception:
        return 0.0

def infer(model, processor, image, prompt):
    """单次推理。"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image], return_tensors="pt"
    )
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.6,
            top_p=0.95,
        )

    output_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(output_ids, skip_special_tokens=True)
    return response

# ── 加载模型 ──────────────────────────────────────

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

results = []

# ---- 1) Zero-shot baseline ----
print("=" * 60)
print("Evaluating: Zero-shot (Qwen2.5-VL-3B)")
print("=" * 60)

model_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=False)

prompt = "Convert this architectural floor plan to SVG format. First analyze its structure, then generate the SVG step by step."

for sid in TEST_IDS:
    img = load_image(sid)
    gt_svg = load_gt_svg(sid)
    t0 = time.time()
    resp = infer(model_base, processor, img, prompt)
    elapsed = time.time() - t0
    svg = extract_svg_from_response(resp)
    valid = validate_svg(svg)
    ss = compute_ssim_score(svg, img) if valid else 0.0
    elem_cnt = count_svg_elements(svg) if valid else 0
    fmt_score = check_instruction_format(resp)
    results.append({
        "id": sid, "model": "zero-shot",
        "valid": valid, "ssim": ss,
        "elements": elem_cnt, "format": fmt_score,
        "response_len": len(resp), "time": elapsed,
        "has_svg": bool(re.search(r"<svg", resp)),
    })
    status = "✓" if valid else "✗"
    print(f"  [{status}] {sid}: SSIM={ss:.3f}, elems={elem_cnt}, fmt={fmt_score}/3, {elapsed:.1f}s")

del model_base
torch.cuda.empty_cache()

# ---- 2) SFT model ----
print()
print("=" * 60)
print(f"Evaluating: SFT (from {SFT_CKPT})")
print("=" * 60)

if os.path.exists(SFT_CKPT):
    model_sft = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model_sft = PeftModel.from_pretrained(model_sft, SFT_CKPT)

    for sid in TEST_IDS:
        img = load_image(sid)
        gt_svg = load_gt_svg(sid)
        t0 = time.time()
        resp = infer(model_sft, processor, img, prompt)
        elapsed = time.time() - t0
        svg = extract_svg_from_response(resp)
        valid = validate_svg(svg)
        ss = compute_ssim_score(svg, img) if valid else 0.0
        elem_cnt = count_svg_elements(svg) if valid else 0
        fmt_score = check_instruction_format(resp)
        results.append({
            "id": sid, "model": "sft",
            "valid": valid, "ssim": ss,
            "elements": elem_cnt, "format": fmt_score,
            "response_len": len(resp), "time": elapsed,
            "has_svg": bool(re.search(r"<svg", resp)),
        })
        status = "✓" if valid else "✗"
        print(f"  [{status}] {sid}: SSIM={ss:.3f}, elems={elem_cnt}, fmt={fmt_score}/3, {elapsed:.1f}s")

    del model_sft
    torch.cuda.empty_cache()
else:
    print(f"  SFT checkpoint not found at {SFT_CKPT}, skipping.")

# ── 输出结果对比表 ──────────────────────────────────

print()
print("=" * 68)
print("  SFT vs Zero-shot 评估对比")
print("=" * 68)

# 按测试样本分组
by_id = {}
for r in results:
    by_id.setdefault(r["id"], {})[r["model"]] = r

header = f"{'样本':<16} {'指标':<10} {'Zero-shot':<12} {'SFT':<12} {'Δ':<10}"
print(header)
print("-" * len(header))

for sid in TEST_IDS:
    row = by_id.get(sid, {})
    z = row.get("zero-shot", {})
    s = row.get("sft", {})

    if not z and not s:
        continue

    def v(r, key, fmt=".3f"):
        if r and key in r:
            return r[key]
        return 0

    for metric, key in [("Valid", "valid"), ("SSIM", "ssim"), ("Elements", "elements"),
                        ("Format", "format"), ("Len(tok)", "response_len")]:
        zv = v(z, key)
        sv = v(s, key)
        if isinstance(zv, bool):
            diff = "→" if sv else "—"
        else:
            diff = f"+{sv - zv:.3f}" if (isinstance(sv, (int, float)) and isinstance(zv, (int, float))) else "—"
        print(f"  {sid:<14} {metric:<10} {str(zv):<12} {str(sv):<12} {diff:<10}")
    print()

# 汇总统计
print("-" * 68)
valid_z = sum(1 for r in results if r["model"] == "zero-shot" and r["valid"])
valid_s = sum(1 for r in results if r["model"] == "sft" and r["valid"])
total_z = sum(1 for r in results if r["model"] == "zero-shot")
total_s = sum(1 for r in results if r["model"] == "sft")

ssim_z = [r["ssim"] for r in results if r["model"] == "zero-shot"]
ssim_s = [r["ssim"] for r in results if r["model"] == "sft"]
fmt_z = [r["format"] for r in results if r["model"] == "zero-shot"]
fmt_s = [r["format"] for r in results if r["model"] == "sft"]

print(f"  SVG Valid Rate:    zero-shot={valid_z}/{total_z}  SFT={valid_s}/{total_s}")
print(f"  Avg SSIM:          zero-shot={np.mean(ssim_z):.3f}  SFT={np.mean(ssim_s):.3f}  Δ={np.mean(ssim_s)-np.mean(ssim_z):+.3f}")
print(f"  Avg Format Score:  zero-shot={np.mean(fmt_z):.2f}/3  SFT={np.mean(fmt_s):.2f}/3")
print("=" * 68)

# 保存结果
out_path = "eval_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
