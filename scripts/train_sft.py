"""SFT 训练脚本 — 使用中间指令数据微调 Qwen-VL。

用法:
  # Dry run（验证数据）
  python scripts/train_sft.py --dry-run

  # Colab / GPU
  python scripts/train_sft.py --max-samples 200 --batch-size 1 --epochs 3 --lr 1e-4

  # CPU 调试（极小数据）
  python scripts/train_sft.py --max-samples 2 --epochs 1 --cpu
"""
import sys, os, json, argparse, gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image
from tqdm import tqdm

# ── Environment tweaks ─────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ── Data ────────────────────────────────────────────
def load_samples(jsonl_path: str, max_samples: int = None) -> list:
    samples = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            samples.append(json.loads(line.strip()))
    return samples


def load_image(sample: dict, data_root: str = "data/resplan") -> Image.Image:
    """加载图像并缩放到 112x112 极低显存模式。"""
    img_path = sample["image"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(data_root, "bitmaps", os.path.basename(img_path))
    try:
        return Image.open(img_path).convert("RGB").resize((112, 112))
    except Exception:
        return Image.new("RGB", (112, 112), "white")


def process_sample(sample: dict, processor, user_prompt: str):
    """处理单个样本，返回模型输入 dict。

    先通过 image_processor 获取正确 patch 数，
    再用 tokenizer 手动插入 <|image_pad|> token IDs。
    """
    img = load_image(sample)
    asst_text = sample["conversations"][1]["value"]

    # 1) 提取图像特征和 grid 信息
    # 传入 min_pixels 防止被强制放大（112x112=12544 像素）
    image_inputs = processor.image_processor([img], return_tensors="pt", min_pixels=112*112)
    pixel_values = image_inputs["pixel_values"]  # [1, C, H, W]
    image_grid_thw = image_inputs["image_grid_thw"]  # [1, 3]

    # 2) 计算 patch 数
    t, h, w = image_grid_thw[0].tolist()
    # Qwen2.5-VL 的 vision encoder 有 2x2 patch merge，
    # image_grid_thw 给的是 merge 前的，需要除以 spatial_merge_size
    merge_size = 2  # Qwen2.5-VL-3B 默认
    num_patches = int(t * (h // merge_size) * (w // merge_size))
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None:
        # fallback: 从 added_tokens_encoder 找
        image_token_id = processor.tokenizer.added_tokens_encoder.get("<|image_pad|>", 151889)

    # 3) 构建文本（不含 image tokens 占位符）
    text = (
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{asst_text}<|im_end|>"
    )

    # 4) tokenize 文本
    token_ids = processor.tokenizer.encode(text, add_special_tokens=False)

    # 5) 在 "user\n" 之后插入 image tokens
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    nl_id = processor.tokenizer.encode("\n", add_special_tokens=False)[0]
    user_ids = processor.tokenizer.encode("user", add_special_tokens=False)

    insert_pos = None
    for j in range(len(token_ids) - len(user_ids) - 1):
        if (token_ids[j] == im_start_id
                and token_ids[j + 1:j + 1 + len(user_ids)] == user_ids
                and token_ids[j + len(user_ids) + 1] == nl_id):
            insert_pos = j + len(user_ids) + 2  # "\n" 之后
            break
    if insert_pos is None:
        insert_pos = 3  # fallback

    input_ids = token_ids[:insert_pos] + [image_token_id] * num_patches + token_ids[insert_pos:]
    attention_mask = [1] * len(input_ids)

    # 6) 计算 labels（mask user + image 部分）
    labels = input_ids.copy()
    im_start_positions = [k for k, tid in enumerate(input_ids) if tid == im_start_id]
    if len(im_start_positions) >= 2:
        # mask 第二个 <|im_start|> 之前的所有 token
        for k in range(im_start_positions[1]):
            labels[k] = -100
    # image tokens 也 mask
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


# ── Training ────────────────────────────────────────
def train(args):
    if args.dry_run:
        _dry_run(args)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # A100 (sm_80+) 原生支持 bf16；T4 (sm_75) 用 fp16
    cap = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if cap >= (8, 0) else torch.float16
    print(f"Device: {device} (sm_{cap[0]}.{cap[1]}), dtype: {dtype}")

    # ── Load model ──────────────────────────────────
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    quant_kwargs = {}
    # A100 40GB 足够跑全 bf16，不用 4bit（避免反量化缓存泄漏）
    # args.quantization 保持为 "4bit" 以便 T4 上使用
    # 但 A100 上用 4bit 会导致反量化权重在 checkpoint 中持续累积
    if args.quantization == "4bit" and torch.cuda.get_device_capability() < (8, 0):
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("Using 4bit quantization (T4 mode)")
    elif args.quantization == "4bit":
        # A100 上忽略 4bit 请求，全 bf16 更稳定
        print("A100 detected: using full bf16 instead of 4bit quantization")
    elif args.quantization == "8bit":
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print("Using 8bit quantization")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="cuda:0",  # 显式指定，避免 accelerate 分配问题
        **quant_kwargs,
    )
    # 在 PEFT 包装前关掉 cache（否则 KV cache + 4bit 反量化会吃满显存）
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=False)

    # LoRA
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
    print(f"Training ready: use_cache={model.config.use_cache}, checkpointing=enabled")

    # freeze vision encoder（省 ~4GB 显存）
    vision = model.base_model.model.model.visual
    for p in vision.parameters():
        p.requires_grad = False
    print("Vision encoder frozen")

    # ── Load data ───────────────────────────────────
    samples = load_samples(args.data_path, args.max_samples)
    print(f"Dataset: {len(samples)} samples")

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable)}")

    # ── Optimizer ──────────────────────────────────
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable, lr=args.lr, weight_decay=0.01)
        print("Using 8-bit AdamW")
    except ImportError:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # ── Train loop ─────────────────────────────────
    user_prompt = "Convert this architectural floor plan to SVG format. First analyze its structure, then generate the SVG step by step."
    global_step = 0

    # 提前处理所有样本，避免训练时反复调用 processor.image_processor 累积缓存
    print("Pre-processing all samples...")
    processed_samples = []
    for sample in tqdm(samples, desc="Pre-process"):
        processed_samples.append(process_sample(sample, processor, user_prompt))
    print(f"Pre-processed {len(processed_samples)} samples")

    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(processed_samples, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for i, batch in enumerate(pbar):
            # 放到设备
            model_kwargs = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    model_kwargs[k] = v.to(model.device)
                else:
                    model_kwargs[k] = v

            outputs = model(**model_kwargs)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # 用 set_to_none 比默认更快释放内存

            global_step += 1
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if (i + 1) % args.log_interval == 0:
                print(f"  Step {i + 1}/{len(samples)}  loss={loss.item():.4f}  avg={total_loss / (i + 1):.4f}")

            # 清理：只 gc 不 empty_cache（避免碎片化）
            del batch, model_kwargs, outputs, loss
            gc.collect()

        # epoch 结束时清理一次显存
        gc.collect()
        torch.cuda.empty_cache()
        avg_loss = total_loss / len(samples)
        print(f"Epoch {epoch + 1} done — Avg loss: {avg_loss:.4f}")

        if args.save_dir:
            path = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
            os.makedirs(path, exist_ok=True)
            model.save_pretrained(path)
            print(f"Saved to {path}")

    print("Training complete!")
    if args.save_dir:
        path = os.path.join(args.save_dir, "final")
        model.save_pretrained(path)
        print(f"Final model saved to {path}")


# ── Dry run ────────────────────────────────────────
def _dry_run(args):
    """在不加载模型的情况下验证数据处理流水线。"""
    from transformers import AutoProcessor

    print("=== DRY RUN ===")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)

    samples = load_samples(args.data_path, max_samples=2)
    print(f"Loaded {len(samples)} samples")

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    print(f"<|image_pad|> token ID: {image_token_id}")

    user_prompt = "Convert this architectural floor plan to SVG format."

    for i, sample in enumerate(samples):
        img = load_image(sample)
        print(f"\nSample {i}: {sample['id']} — image size: {img.size}")

        result = process_sample(sample, processor, user_prompt)
        print(f"  input_ids:  {result['input_ids'].shape}")
        print(f"  labels:     {result['labels'].shape}")
        print(f"  pixel_vals: {result['pixel_values'].shape}")
        print(f"  grid_thw:   {result['image_grid_thw'].shape}")

        # 验证 image token 数量
        n_img_tokens = (result["input_ids"] == image_token_id).sum().item()
        t, h, w = result["image_grid_thw"][0].tolist()
        merge_size = 2
        expected = int(t * (h // merge_size) * (w // merge_size))
        print(f"  image_tokens: {n_img_tokens} (expected {expected}) {'✅' if n_img_tokens == expected else '❌'}")

    print("\n✅ Dry run complete!")


# ── Entry ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SFT training for RL Vectorizer")
    parser.add_argument("--data-path", default="data/resplan/sft_train.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--batch-size", type=int, default=1)  # batch_size > 1 需要额外实现
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", None])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-dir", default="checkpoints/sft")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.cpu:
        args.quantization = None

    train(args)


if __name__ == "__main__":
    main()
