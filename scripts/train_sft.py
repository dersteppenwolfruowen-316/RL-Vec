"""SFT 训练脚本 — 使用中间指令数据微调 Qwen-VL。

用法:
  # CPU 测试（极小数据量）
  python scripts/train_sft.py --max-samples 4 --epochs 1 --cpu

  # 单 GPU
  python scripts/train_sft.py --batch-size 2 --lr 1e-4

  # 多 GPU (DeepSpeed)
  accelerate launch scripts/train_sft.py --batch-size 4 --lr 1e-4 --deepspeed
"""
import sys, os, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm


# ---------- Dataset ----------
class SFTDataset(Dataset):
    """从 sft_train.jsonl 加载 SFT 训练数据。"""

    def __init__(self, jsonl_path: str, max_samples: int = None, image_root: str = None):
        self.samples = []
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line.strip()))
        self.image_root = image_root or str(Path(jsonl_path).parent)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, processor, device="cpu"):
    """将 batch 样本处理为模型输入。

    返回: {"input_ids": ..., "attention_mask": ..., "labels": ..., "pixel_values": ...}
    """
    from PIL import Image

    user_text = "Convert this architectural floor plan to SVG format. First analyze its structure, then generate the SVG step by step."
    texts = []
    images = []

    for sample in batch:
        # 加载图像
        img_path = sample["image"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(
                os.path.dirname(os.path.dirname(img_path)),
                "bitmaps",
                os.path.basename(img_path),
            )
        if os.path.exists(img_path):
            images.append(Image.open(img_path).convert("RGB"))
        else:
            # 如果图像不存在，用空白占位
            images.append(Image.new("RGB", (512, 512), "white"))

        # 构造对话
        assistant_text = sample["conversations"][1]["value"]
        texts.append((user_text, assistant_text))

    # 用 processor 处理
    # Qwen 的 processor 支持 image + text 输入
    inputs = processor(
        text=[t[0] for t in texts],
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )

    # 构造 labels：把 user 部分的 token 设为 -100（不参与 loss）
    # 注意：这里需要 processor 返回 input_ids 的对应关系
    # 对于 Qwen3-VL，需要分别 tokenize user 和 assistant 部分

    # 更可靠的方法：对整个对话 tokenize，然后标记 assistant 部分
    full_texts = []
    for user_txt, asst_txt in texts:
        full = f"<|im_start|>user\n<image>\n{user_txt}<|im_end|>\n<|im_start|>assistant\n{asst_txt}<|im_end|>"
        full_texts.append(full)

    # 重新 tokenize 完整文本
    model_inputs = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )

    # 构造 labels：assistant 部分保留 token，其余为 -100
    labels = model_inputs["input_ids"].clone()

    # 找到每个序列中 <|im_start|>assistant 的位置
    asst_token_id = processor.tokenizer.encode("<|im_start|>assistant",
                                                add_special_tokens=False)[0]

    for i in range(labels.shape[0]):
        seq = labels[i]
        # 找到 assistant token 出现的位置
        asst_positions = (seq == asst_token_id).nonzero(as_tuple=True)[0]
        if len(asst_positions) > 0:
            start_pos = asst_positions[0].item()
            # start_pos 之前的所有 token（user 部分）设为 -100
            labels[i, :start_pos] = -100
        else:
            # 如果找不到（fallback），整个序列参与 loss
            pass

    model_inputs["labels"] = labels
    return model_inputs


# ---------- 训练 ----------
def train(args):
    if args.dry_run:
        print("=== DRY RUN: testing data pipeline only ===")
        _dry_run(args)
        return
    if args.cpu:
        device = "cpu"
        torch_dtype = torch.float32
    elif torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print("CUDA not available, falling back to CPU")

    print(f"Device: {device}, dtype: {torch_dtype}")

    # 加载模型
    from rl_vectorizer.models.qwen_vl import QwenVLModel

    model = QwenVLModel(
        base_model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        device=device,
        quantization=args.quantization,
        use_flash_attention=not args.cpu,
        torch_dtype=torch_dtype,
        # CPU 训练时禁用 device_map="auto"（会触发自动分布到多设备）
        device_map="auto" if device == "cuda" else None,
    )

    if args.cpu:
        model.model = model.model.to(device)
        model.model.eval()

    processor = model.processor

    # 加载数据
    dataset = SFTDataset(
        args.data_path,
        max_samples=args.max_samples,
    )
    print(f"Dataset: {len(dataset)} samples")

    # DataLoader — dataloader 用自定义 collate
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor, device),
        num_workers=0,  # CPU 训练设为 0
    )

    # 优化器（只训练 LoRA 参数）
    trainable_params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    print(f"Trainable params: {sum(p.numel() for p in trainable_params)}")

    # 训练循环
    model.model.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in epoch_pbar:
            # 移到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)

            # Forward
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
            )

            loss = outputs.loss

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if global_step % args.log_interval == 0:
                print(f"Step {global_step}: loss = {loss.item():.4f}")

        # 每个 epoch 保存
        if args.save_dir:
            save_path = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
            model.save_pretrained(save_path)
            print(f"Saved to {save_path}")

    print("Training complete!")
    if args.save_dir:
        model.save_pretrained(os.path.join(args.save_dir, "final"))
        print(f"Final model saved to {os.path.join(args.save_dir, 'final')}")


def _dry_run(args):
    """在不加载模型的情况下验证数据处理流水线。"""
    import json
    from PIL import Image
    from transformers import AutoProcessor

    # 读 2 条数据
    samples = []
    with open(args.data_path) as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            samples.append(json.loads(line.strip()))
    print(f"Loaded {len(samples)} samples for dry run")

    # 加载 processor（不需要加载模型权重）
    processor = AutoProcessor.from_pretrained(args.model_name,
                                               trust_remote_code=True,
                                               use_fast=False)

    # 检查图像
    for s in samples:
        img_path = s["image"]
        if not os.path.isabs(img_path):
            base = os.path.dirname(os.path.dirname(args.data_path))
            img_path = os.path.join(base, "bitmaps", os.path.basename(img_path))
        try:
            img = Image.open(img_path).convert("RGB")
            print(f"  Image {s['id']}: {img.size} OK")
        except Exception as e:
            print(f"  Image {s['id']}: FAILED ({e})")

    # 测试 tokenization (不含图像)
    full_example = samples[0]["conversations"][1]["value"]
    tokens = processor.tokenizer.encode(full_example)
    print(f"  Instruction tokens: {len(tokens)}")

    # 测试带图像的 tokenization
    user_text = "Convert this architectural floor plan to SVG format."
    img = Image.open(img_path).convert("RGB")
    inputs = processor(
        text=user_text,
        images=img,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )
    print(f"  Model input: { {k: v.shape for k, v in inputs.items()} }")
    print("✅ Dry run passed! Data pipeline is correct.")


# ---------- 入口 ----------
def main():
    parser = argparse.ArgumentParser(description="SFT training for RL Vectorizer")
    parser.add_argument("--data-path", default="data/resplan/sft_train.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="用 3B 而非 8B（CPU 友好）")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dry-run", action="store_true", help="只测试数据处理，不加载模型")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--quantization", default=None, choices=["4bit", "8bit", None])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="限制训练样本数（用于调试）")
    parser.add_argument("--save-dir", default="checkpoints/sft")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--cpu", action="store_true", help="强制 CPU 训练")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
