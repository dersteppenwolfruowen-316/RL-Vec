"""诊断实验 2：离散网格坐标（256×256 grid）

目标：用同样的 200 条数据，但 SVG 坐标从连续浮点数改成离散整数网格。
验证模型学不会 SVG 是不是因为坐标表示太难（回归问题），
换成离散化后（分类问题）是否能学会。

用法：
  # Mac 本地验证数据（CPU）
  python scripts/train_diag_grid.py --dry-run

  # Colab A100 训练（200 条数据，比较结果）
  python scripts/train_diag_grid.py --max-samples 200 --epochs 3
"""
import sys, os, json, argparse, gc, re, math
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─── 256×256 网格量化 ───
GRID_SIZE = 256


def quantize_coord(value: float, max_dim: float) -> int:
    """将连续坐标映射到 0-255 的离散网格。"""
    ratio = value / max_dim if max_dim > 0 else 0
    return max(0, min(GRID_SIZE - 1, int(ratio * GRID_SIZE)))


def quantize_svg(svg_content: str, max_dim: float = 500.0) -> str:
    """将 SVG 中的所有坐标值量化为 0-255 整数。"""
    # 替换所有浮点数（保留负号、小数点和数字）
    def replace_coord(m):
        val = float(m.group(0))
        return str(quantize_coord(abs(val), max_dim) if val >= 0 else str(quantize_coord(abs(val), max_dim)))

    # 匹配数字：整数或浮点数
    result = re.sub(r"-?\d+\.?\d*", replace_coord, svg_content)
    return result


def load_svg_content(svg_path: str) -> str:
    with open(svg_path, "r") as f:
        return f.read()


# ═══════════════════════════════════════
# 数据准备
# ═══════════════════════════════════════

def prepare_grid_data(
    bitmap_dir: str,
    svg_dir: str,
    max_samples: int = 200,
    grid_size: int = 256,
) -> list:
    """准备网格坐标训练数据。

    对每个样本：
    - 输入：floor plan 图片
    - 输出：简化版 SVG（坐标量化为 0-255 整数）
    """
    svg_files = sorted(Path(svg_dir).glob("*.svg"))
    if max_samples:
        svg_files = svg_files[:max_samples]

    # 先扫描所有 SVG 估算最大尺寸
    max_dim = 500.0  # 默认

    samples = []
    for svg_path in tqdm(svg_files, desc="Processing SVGs"):
        sid = svg_path.stem
        bitmap = Path(bitmap_dir) / f"{sid}.png"
        if not bitmap.exists():
            continue

        svg_raw = load_svg_content(str(svg_path))

        # 估算 SVG 的尺寸
        dims = re.findall(r'viewBox="\d+\.?\d*\s+\d+\.?\d*\s+(\d+\.?\d*)\s+(\d+\.?\d*)"', svg_raw)
        if dims:
            w, h = float(dims[0][0]), float(dims[0][1])
            max_dim = max(max_dim, w, h)

        # 量化坐标
        svg_quantized = quantize_svg(svg_raw, max_dim)

        # 提取简化的路径数据（去掉元数据、注释等）
        # 保留核心路径
        lines = []
        in_path = False
        for line in svg_quantized.split("\n"):
            stripped = line.strip()
            # 保留 path、line、rect 等核心元素
            if any(tag in stripped for tag in ["<path", "<line", "<rect", "<circle", "<polygon", "<polyline"]):
                lines.append(stripped)
            elif "</svg>" in stripped:
                lines.append(stripped)

        simplified_svg = "\n".join(lines)

        samples.append({
            "id": f"resplan_{sid}",
            "image": str(bitmap),
            "target": f"<svg_grid>\n{simplified_svg}\n</svg_grid>",
            "metadata": {"source_svg": str(svg_path), "max_dim": max_dim, "grid_size": grid_size},
        })

    print(f"  Prepared {len(samples)} grid-based samples (grid={grid_size}×{grid_size})")
    return samples


def validate_grid_svg(svg_text: str) -> bool:
    """验证网格 SVG 的基本合法性。"""
    # 至少包含 <svg 和 </svg>
    if "<svg" not in svg_text or "</svg>" not in svg_text:
        return False
    # 所有坐标应该在 0-255 范围内
    nums = re.findall(r"\b\d+\b", svg_text)
    for n in nums:
        val = int(n)
        if val > 255:
            return False
    return True


def extract_element_count(svg_text: str) -> int:
    """统计 SVG 中元素数量。"""
    return len(re.findall(r"<(path|line|rect|circle|polygon|polyline|ellipse)", svg_text))


# ═══════════════════════════════════════
# 训练
# ═══════════════════════════════════════

def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        return Image.new("RGB", (64, 64), "white")
    return Image.open(path).convert("RGB").resize((64, 64))


def process_sample(sample: dict, processor) -> dict:
    """处理单个网格坐标样本。"""
    img = load_image(sample["image"])
    target = sample["target"]

    image_inputs = processor.image_processor(
        [img], return_tensors="pt",
        min_pixels=64 * 64,
        max_pixels=64 * 64 * 2,
    )
    pixel_values = image_inputs["pixel_values"]
    image_grid_thw = image_inputs["image_grid_thw"]

    t, h, w = image_grid_thw[0].tolist()
    merge_size = 2
    num_patches = int(t * (h // merge_size) * (w // merge_size))
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None:
        image_token_id = processor.tokenizer.added_tokens_encoder.get("<|image_pad|>", 151889)

    prompt = "Convert this floor plan to SVG with 256x256 grid coordinates."
    text = (
        f"<|im_start|>user\n<image>\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{target}<|im_end|>"
    )

    token_ids = processor.tokenizer.encode(text, add_special_tokens=False)
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    nl_id = processor.tokenizer.encode("\n", add_special_tokens=False)[0]
    user_ids = processor.tokenizer.encode("user", add_special_tokens=False)

    insert_pos = None
    for j in range(len(token_ids) - len(user_ids) - 1):
        if (token_ids[j] == im_start_id
                and token_ids[j + 1:j + 1 + len(user_ids)] == user_ids
                and token_ids[j + len(user_ids) + 1] == nl_id):
            insert_pos = j + len(user_ids) + 2
            break
    if insert_pos is None:
        insert_pos = 3

    input_ids = token_ids[:insert_pos] + [image_token_id] * num_patches + token_ids[insert_pos:]
    attention_mask = [1] * len(input_ids)

    labels = input_ids.copy()
    im_start_positions = [k for k, tid in enumerate(input_ids) if tid == im_start_id]
    if len(im_start_positions) >= 2:
        for k in range(im_start_positions[1]):
            labels[k] = -100
    for k in range(len(labels)):
        if labels[k] == image_token_id:
            labels[k] = -100

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


def train(args):
    if args.dry_run:
        _dry_run(args)
        return

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if device == "cuda":
        cap = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if cap >= (8, 0) else torch.float16
    else:
        cap = (0, 0)
        dtype = torch.float32
        print(f"⚠️  CPU mode")

    print(f"Device: {device}, dtype: {dtype}")

    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    # 找数据目录
    for base in ["data/resplan", os.path.join(os.path.dirname(args.data_path), "..")]:
        bm = os.path.join(base, "bitmaps")
        sv = os.path.join(base, "svgs")
        if os.path.isdir(bm):
            bitmap_dir, svg_dir = bm, sv
            break

    print("Preparing grid-based SVG data...")
    samples = prepare_grid_data(bitmap_dir, svg_dir, args.max_samples)
    if len(samples) == 0:
        print("❌ 没有找到数据！")
        return

    # 量化
    quant_kwargs = {}
    if args.quantization == "4bit":
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if device == "cuda":
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.enabled:
            try:
                import flash_attn  # noqa
                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"
        else:
            attn_impl = "eager"
    else:
        attn_impl = "eager"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="cuda:0" if device == "cuda" else "cpu",
        attn_implementation=attn_impl,
        **quant_kwargs,
    )
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=False)

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    try:
        vision = model.base_model.model.model.visual
        for p in vision.parameters():
            p.requires_grad = False
        print("Vision encoder frozen")
    except Exception:
        pass

    trainable = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable, lr=args.lr, weight_decay=0.01)
    except ImportError:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    global_step = 0
    accum_steps = args.grad_accum_steps

    print("Pre-processing samples...")
    processed = []
    for s in tqdm(samples, desc="Pre-process"):
        processed.append(process_sample(s, processor))
    print(f"Pre-processed {len(processed)} samples")

    torch.cuda.empty_cache() if device == "cuda" else None
    print(f"Training: {len(samples)} samples, {args.epochs} epochs, accum={accum_steps}")

    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(processed, desc=f"Epoch {epoch+1}/{args.epochs}")

        for i, batch in enumerate(pbar):
            model_kwargs = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    model_kwargs[k] = v.to(model.device, non_blocking=True)
                else:
                    model_kwargs[k] = v

            outputs = model(**model_kwargs)
            raw_loss = outputs.loss
            loss = raw_loss / accum_steps
            loss.backward()
            total_loss += raw_loss.item()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(processed):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            pbar.set_postfix({"loss": f"{raw_loss.item():.4f}"})
            del batch, model_kwargs, outputs, loss, raw_loss
            gc.collect()

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        avg_loss = total_loss / len(processed)
        print(f"Epoch {epoch+1} done — Avg loss: {avg_loss:.4f}")

    # ─── 评估 ───
    print("\n" + "=" * 60)
    print("Evaluation: Grid SVG Generation")
    print("=" * 60)
    model.eval()

    eval_samples = samples[: min(10, len(samples))]
    valid_count = 0
    gt_elems = 0
    pred_elems = 0

    for s in eval_samples:
        batch = process_sample(s, processor)
        inp = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        inp.pop("labels", None)
        with torch.no_grad():
            generated = model.generate(
                **inp,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.6,
            )
        pred_text = processor.decode(generated[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

        is_valid = validate_grid_svg(pred_text)
        n_pred = extract_element_count(pred_text)
        n_gt = extract_element_count(s["target"])

        mark = "✓" if is_valid else "✗"
        print(f"  {mark} {s['id']}: valid={is_valid} elems_pred={n_pred} elems_gt={n_gt}")
        if is_valid:
            valid_count += 1
        gt_elems += n_gt
        pred_elems += n_pred

    valid_rate = valid_count / len(eval_samples) * 100
    print(f"\n  SVG valid rate: {valid_count}/{len(eval_samples)} = {valid_rate:.1f}%")
    if valid_count > 0:
        print(f"  Avg elements: GT={gt_elems/len(eval_samples):.1f} Pred={pred_elems/len(eval_samples):.1f}")

    if args.save_dir:
        path = os.path.join(args.save_dir, "diag_grid_final")
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        print(f"Saved to {path}")
    print("Done!")


def _dry_run(args):
    """本地 Mac 验证数据。"""
    print("=== DRY RUN: Grid Coordinate Diagnostic ===")
    for base in ["data/resplan"]:
        if os.path.isdir(os.path.join(base, "svgs")):
            svg_dir = os.path.join(base, "svgs")
            bitmap_dir = os.path.join(base, "bitmaps")
            break
    else:
        print("❌ 未找到 SVG 数据")
        return

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)

    samples = prepare_grid_data(bitmap_dir, svg_dir, min(args.max_samples or 5, 5))
    print(f"\n{'='*60}")
    print("Sample data preview:")
    for i, s in enumerate(samples[:3]):
        print(f"\n--- Sample {i}: {s['id']} ---")
        print(f"Image: {s['image']}")
        target_preview = s["target"][:300]
        print(f"Target (first 300 chars):\n{target_preview}...")
        print(f"Is valid grid SVG: {validate_grid_svg(s['target'])}")
        print(f"Element count: {extract_element_count(s['target'])}")
        batch = process_sample(s, processor)
        print(f"input_ids: {batch['input_ids'].shape}")
    print("\n✅ Dry run complete!")


def main():
    parser = argparse.ArgumentParser(description="Diagnostic: Grid-based SVG Training")
    parser.add_argument("--data-path", default="data/resplan")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", None])
    parser.add_argument("--save-dir", default="checkpoints/diag_grid")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.cpu:
        args.quantization = None

    if args.dry_run:
        _dry_run(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
