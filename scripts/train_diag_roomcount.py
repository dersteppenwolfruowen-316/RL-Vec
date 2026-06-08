"""诊断实验 1：房间计数 + 大致位置预测

目标：用 50 条数据，训练模型预测房间数量和大致位置。
验证 VLM 能不能学会最基本的"看图 → 结构化空间感知"。

用法：
  # Mac 本地验证数据（CPU，只跑 dry-run）
  python scripts/train_diag_roomcount.py --dry-run

  # Colab A100 训练
  python scripts/train_diag_roomcount.py --max-samples 50 --epochs 3

  # 指定 CPU（Mac Intel 可以跑数据准备，训练会很慢）
  python scripts/train_diag_roomcount.py --cpu --max-samples 2
"""
import sys, os, json, argparse, gc, re, math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─── 颜色到房间类型的映射（与 prepare_sft_data.py 一致）───
FILL_TO_ROOM = {
    "#90ee90": "bedroom",
    "#add8e6": "bathroom",
    "#ffb6c1": "kitchen",
    "#ffffe0": "living_room",
    "#98fb98": "balcony",
    "#d3d3d3": "storage",
    "#deb887": "stair",
}

STROKE_TO_ELEM = {
    "#333333": "wall",
    "#8b4513": "door",
    "#4169e1": "window",
    "#ff0000": "front_door",
}


def parse_svg_rooms(svg_path: str) -> dict:
    """从 SVG 文件中提取房间信息和大致位置。"""
    import xml.etree.ElementTree as ET

    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    rooms = []  # [{"type": "bedroom", "cx": float, "cy": float}]
    elem_counts = defaultdict(int)

    for path in root.findall(".//svg:path", ns):
        style = (path.get("style") or path.get("fill") or "").lower()
        d = path.get("d", "")

        # 找填充色对应的房间
        for color, room_type in FILL_TO_ROOM.items():
            if color in style:
                elem_counts[room_type] += 1
                # 从 path d 中估算中心
                coords = re.findall(r"[-+]?\d*\.?\d+", d)
                nums = [float(c) for c in coords if c and c != "-"]
                if len(nums) >= 2:
                    xs = nums[0::2]
                    ys = nums[1::2]
                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)
                    rooms.append({"type": room_type, "cx": round(cx, 1), "cy": round(cy, 1)})
                break

        # 找描边对应的元素（门、窗等）
        for color, elem_type in STROKE_TO_ELEM.items():
            if color in style:
                elem_counts[elem_type] += 1
                break

    return {
        "rooms": rooms,
        "counts": dict(elem_counts),
        "total_rooms": sum(1 for r in rooms if r["type"] not in ("wall", "door", "window", "front_door")),
    }


def build_room_summary(rooms: list, counts: dict, total_rooms: int) -> str:
    """构建简化的房间摘要文本。"""
    lines = ["<summary>"]
    lines.append(f"Total rooms: {total_rooms}")

    # 按类型汇总
    type_counts = defaultdict(int)
    for r in rooms:
        if r["type"] not in ("wall", "door", "window", "front_door"):
            type_counts[r["type"]] += 1
    if type_counts:
        type_str = ", ".join(f"{v} {k}" for k, v in sorted(type_counts.items()))
        lines.append(f"Room types: {type_str}")

    # 每个房间的位置
    lines.append("Approximate layout:")
    for r in rooms:
        if r["type"] not in ("wall", "door", "window", "front_door"):
            lines.append(f"- {r['type']}: center ({r['cx']}, {r['cy']})")

    lines.append("</summary>")
    return "\n".join(lines)


def prepare_roomcount_data(
    bitmap_dir: str, svg_dir: str, max_samples: int = 50, dry_run: bool = False
) -> list:
    """准备房间计数训练数据。"""
    svg_files = sorted(Path(svg_dir).glob("*.svg"))
    if max_samples:
        svg_files = svg_files[:max_samples]

    samples = []
    for svg_path in tqdm(svg_files, desc="Parsing SVGs"):
        sid = svg_path.stem
        bitmap = Path(bitmap_dir) / f"{sid}.png"

        try:
            info = parse_svg_rooms(str(svg_path))
        except Exception as e:
            print(f"  ✗ {sid}: parse error: {e}")
            continue

        if info["total_rooms"] == 0:
            continue  # skip invalid

        summary = build_room_summary(info["rooms"], info["counts"], info["total_rooms"])

        samples.append({
            "id": f"resplan_{sid}",
            "image": str(bitmap),
            "target": summary,
            "metadata": {
                "total_rooms": info["total_rooms"],
                "room_counts": info["counts"],
                "room_positions": info["rooms"],
            },
        })

    print(f"  Prepared {len(samples)} samples (total rooms: 1-{max(r['metadata']['total_rooms'] for r in samples)})")
    return samples


def eval_room_count_accuracy(pred: str, gt_metadata: dict) -> dict:
    """评估房间数量预测准确率。"""
    total_gt = gt_metadata.get("total_rooms", 0)

    # 从输出中提取预测的房间数
    m = re.search(r"Total rooms:\s*(\d+)", pred)
    total_pred = int(m.group(1)) if m else -1

    return {
        "total_correct": total_pred == total_gt,
        "total_pred": total_pred,
        "total_gt": total_gt,
        "exact_match": total_pred == total_gt,
    }


# ═══════════════════════════════════════
# 训练部分（复用 train_sft.py 的模式）
# ═══════════════════════════════════════

def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        return Image.new("RGB", (64, 64), "white")
    return Image.open(path).convert("RGB").resize((64, 64))


def process_sample(sample: dict, processor) -> dict:
    """处理单个样本，生成简化任务输入。"""
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

    # 简化 prompt：不需要生成完整 SVG，只要房间摘要
    prompt = "Describe the rooms in this floor plan. List each room type and its approximate position."
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

    # labels：mask user + image 部分
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
        print(f"⚠️  CPU mode: 训练将非常慢，仅用于数据验证")

    print(f"Device: {device}, dtype: {dtype}")

    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    # 准备数据
    print("Preparing room count data...")
    data_root = os.path.join(os.path.dirname(args.data_path), "..") if args.data_path != "data/resplan" else "data/resplan"
    bitmap_dir = os.path.join(data_root, "bitmaps") if not os.path.isabs("data/resplan/bitmaps") else os.path.join(data_root, "bitmaps")
    svg_dir = os.path.join(data_root, "svgs") if not os.path.isabs("data/resplan/svgs") else os.path.join(data_root, "svgs")

    # 尝试从相对路径找
    for base in ["data/resplan", os.path.join(os.path.dirname(args.data_path), "..")]:
        bm = os.path.join(base, "bitmaps")
        sv = os.path.join(base, "svgs")
        if os.path.isdir(bm):
            bitmap_dir, svg_dir = bm, sv
            break

    samples = prepare_roomcount_data(bitmap_dir, svg_dir, args.max_samples)
    if len(samples) == 0:
        print("❌ 没有找到有效样本！请先运行 convert_resplan.py 生成 SVG 数据。")
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

    # 选择 attention backend
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

    # LoRA rank=8（省显存）
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
        vision_fwd = vision.forward
        def vision_no_grad(*a, **kw):
            with torch.no_grad():
                return vision_fwd(*a, **kw)
        vision.forward = vision_no_grad
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

    # 预处理
    print("Pre-processing samples...")
    processed = []
    for s in tqdm(samples, desc="Pre-process"):
        processed.append(process_sample(s, processor))
    print(f"Pre-processed {len(processed)} samples")

    torch.cuda.empty_cache() if device == "cuda" else None
    print(f"Training: {len(samples)} samples, {args.epochs} epochs, accum={accum_steps}")

    # ─── 训练循环 ───
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

    # ─── 简单评估 ───
    print("\n" + "=" * 60)
    print("Evaluation: Room Count Prediction")
    print("=" * 60)
    model.eval()

    eval_samples = samples[: min(10, len(samples))]
    correct = 0
    for s in eval_samples:
        # 用模型生成预测
        batch = process_sample(s, processor)
        inp = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        # 去掉 labels 进行生成
        inp.pop("labels", None)
        with torch.no_grad():
            generated = model.generate(
                **inp,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.6,
            )
        pred_text = processor.decode(generated[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

        result = eval_room_count_accuracy(pred_text, s["metadata"])
        mark = "✓" if result["exact_match"] else "✗"
        print(f"  {mark} {s['id']}: pred={result['total_pred']} gt={result['total_gt']}")
        if not result["exact_match"]:
            print(f"       pred_text: {pred_text[:100]}...")
        if result["exact_match"]:
            correct += 1

    acc = correct / len(eval_samples) * 100
    print(f"\n  Room count accuracy: {correct}/{len(eval_samples)} = {acc:.1f}%")

    if args.save_dir:
        path = os.path.join(args.save_dir, "diag_roomcount_final")
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        print(f"Saved to {path}")
    print("Done!")


def _dry_run(args):
    """本地 Mac 验证数据。"""
    print("=== DRY RUN: Room Count Diagnostic ===")
    # 只需要找数据目录
    for base in ["data/resplan"]:
        if os.path.isdir(os.path.join(base, "svgs")):
            svg_dir = os.path.join(base, "svgs")
            bitmap_dir = os.path.join(base, "bitmaps")
            break
    else:
        print("❌ 未找到 SVG 数据。先运行 convert_resplan.py")
        return

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)

    samples = prepare_roomcount_data(bitmap_dir, svg_dir, min(args.max_samples or 5, 5), dry_run=True)
    print(f"\n{'='*60}")
    print("Sample data preview:")
    for i, s in enumerate(samples[:3]):
        print(f"\n--- Sample {i}: {s['id']} ---")
        print(f"Image: {s['image']}")
        print(f"Target:\n{s['target']}")
        print(f"Metadata: {json.dumps(s['metadata'], indent=2)}")

        batch = process_sample(s, processor)
        print(f"input_ids: {batch['input_ids'].shape}")
        print(f"pixel_values: {batch['pixel_values'].shape}")
    print("\n✅ Dry run complete!")


def main():
    parser = argparse.ArgumentParser(description="Diagnostic: Room Count Prediction")
    parser.add_argument("--data-path", default="data/resplan")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", None])
    parser.add_argument("--save-dir", default="checkpoints/diag_roomcount")
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--cpu", action="store_true", help="强制 CPU（Mac 本地用）")
    parser.add_argument("--dry-run", action="store_true", help="只验证数据，不训练")
    args = parser.parse_args()

    if args.cpu:
        args.quantization = None

    if args.dry_run:
        _dry_run(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
