#!/usr/bin/env python3
"""GRPO 强化学习训练脚本 — 建筑平面图 SVG 生成优化。

用法:
  # 最小测试（仅 R_validity，最快验证训练循环）
  python scripts/train_rl_grpo.py --max-samples 10 --epochs 2 --rollout-n 4 \
      --reward-mode validity --save-dir checkpoints/rl_test

  # 完整训练（多组件 reward）
  python scripts/train_rl_grpo.py --max-samples 200 --epochs 5 --rollout-n 6 \
      --reward-mode all --visual-weight 0.25 --geometry-weight 0.25 \
      --save-dir checkpoints/rl_grpo

  # 从检查点继续训练
  python scripts/train_rl_grpo.py --load-rl-ckpt checkpoints/rl_grpo/epoch_3

算法: GRPO (Group Relative Policy Optimization)
  - 对每个 prompt 采样一组响应
  - reward 归一化作为 advantage
  - 策略比率裁剪 + KL 散度惩罚
  - 无需 critic 网络
"""
import sys, os, json, argparse, gc, re, math
from pathlib import Path
from typing import Optional, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ─── 工具函数 ───────────────────────────────────────────────────────────────

def load_jsonl(path: str, max_samples: Optional[int] = None) -> list:
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            samples.append(json.loads(line.strip()))
    return samples


def build_prompt(sample: dict) -> str:
    prompt = sample.get("conversations", [{}])[0].get("value", "")
    prompt = re.sub(r"<image>\s*", "", prompt)
    return prompt


def extract_svg(text: str) -> str:
    patterns = [
        r"<svg_output>\s*(.*?)\s*</svg_output>",
        r"```(?:svg)?\s*\n?(.*?)\n?\s*```",
        r"(<svg[\s\S]*?</svg>)",
    ]
    for p in patterns:
        m = re.search(p, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    svg_start = text.find("<svg")
    svg_end = text.find("</svg>")
    if svg_start != -1 and svg_end != -1:
        return text[svg_start:svg_end + 6]
    return text.strip()


def extract_intermediate(text: str) -> str:
    """提取 assistant 回复中的中间指令部分（<analysis>...<svg_output> 之前）。"""
    svg_pos = text.find("<svg_output>")
    if svg_pos != -1:
        return text[:svg_pos].strip()
    # fallback: 提取 analysis 到第一个 <svg
    parts = re.split(r"(<svg[\s>])", text)
    return parts[0].strip() if parts else ""


def load_image(sample: dict, data_root: str = "data/resplan") -> Image.Image:
    img_path = sample["image"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(data_root, "bitmaps", os.path.basename(img_path))
    try:
        return Image.open(img_path).convert("RGB")
    except Exception:
        return Image.new("RGB", (64, 64), "white")


# ─── Reward 计算 ────────────────────────────────────────────────────────────

def compute_rewards_batch(
    svg_codes: List[str],
    target_tensor: torch.Tensor,
    intermediate_xmls: Optional[List[str]] = None,
    reward_mode: str = "validity",
    renderer=None,
    device: str = "cuda",
) -> List[float]:
    """对一批 SVG 计算 reward。

    reward_mode:
      - "validity": 仅 SVG 是否可解析 (0/1)
      - "geo":      validity + 几何约束
      - "all":      validity + geometry + visual (DiffVG)
    """
    from lxml import etree

    rewards = []
    for i, svg in enumerate(svg_codes):
        # R_validity
        try:
            etree.fromstring(svg.encode())
            is_valid = True
        except Exception:
            is_valid = False

        if not is_valid:
            rewards.append(0.0)
            continue

        if reward_mode == "validity":
            rewards.append(1.0)
            continue

        # R_geometry (简单版本)
        if reward_mode in ("geo", "all"):
            try:
                geo_score = _geometry_score(svg)
            except Exception:
                geo_score = 0.5
        else:
            geo_score = 1.0

        # R_visual (DiffVG 或 cairosvg)
        if reward_mode == "all" and renderer is not None:
            try:
                vis_score = _visual_score(svg, target_tensor, renderer, device)
            except Exception:
                vis_score = 0.0
        else:
            vis_score = 0.0

        # 组合: validity(0.3) + geometry(0.35) + visual(0.35)
        if reward_mode == "all":
            r = 0.30 * 1.0 + 0.35 * geo_score + 0.35 * vis_score
        elif reward_mode == "geo":
            r = 0.40 * 1.0 + 0.60 * geo_score
        else:
            r = 1.0

        rewards.append(max(0.0, min(1.0, r)))

    return rewards


def _geometry_score(svg_code: str) -> float:
    """简单几何评分: 检查元素数量、比例。"""
    from lxml import etree
    root = etree.fromstring(svg_code.encode())
    ns = {"svg": "http://www.w3.org/2000/svg"}

    n_lines = len(root.xpath("//svg:line", namespaces=ns))
    n_paths = len(root.xpath("//svg:path", namespaces=ns))
    n_rects = len(root.xpath("//svg:rect", namespaces=ns))
    total = n_lines + n_paths + n_rects

    if total < 2:
        return 0.2  # 太少元素
    if total > 200:
        return 0.5  # 太多元素可能过度复杂
    return min(1.0, 0.3 + total / 50 * 0.7)


def _visual_score(
    svg_code: str,
    target_tensor: torch.Tensor,
    renderer,
    device: str,
) -> float:
    """渲染 SVG 并与目标计算视觉相似度。"""
    w, h = target_tensor.shape[3], target_tensor.shape[2]  # [1,3,H,W]
    # 限制渲染尺寸为 112x112 以加速
    render_size = min(w, h, 112)
    rendered = renderer.render(svg_code, width=render_size, height=render_size)
    if rendered is None or rendered.numel() == 0:
        return 0.0

    # 对齐格式
    if rendered.shape[1] != 3:
        rendered = rendered.permute(0, 3, 1, 2)
    rendered = rendered.to(device)

    # resize target
    target = F.interpolate(
        target_tensor, size=(render_size, render_size),
        mode="bilinear", align_corners=False,
    )

    # 简单的 MSE → 相似度
    mse = F.mse_loss(rendered, target).item()
    return max(0.0, 1.0 - mse * 5.0)  # 缩放, 使 MSE=0.2 时 ≈ 0


# ─── GRPO 损失 ──────────────────────────────────────────────────────────────

def compute_grpo_loss(
    log_probs_theta: torch.Tensor,   # [batch, seq_len] — 当前策略 log P
    log_probs_ref: torch.Tensor,     # [batch, seq_len] — 参考策略 log P
    advantages: torch.Tensor,        # [batch] — 归一化后的 advantage
    response_mask: torch.Tensor,     # [batch, seq_len] — response 部分为 1
    epsilon: float = 0.2,
    beta: float = 0.04,
) -> Tuple[torch.Tensor, dict]:
    """GRPO 损失函数。

    L_GRPO = -E[ min(r * A, clip(r, 1-ε, 1+ε) * A) - β * KL ]

    其中 r = exp(log π_θ - log π_ref) 是重要性采样比率。
    """
    # 策略比率: r = π_θ / π_ref = exp(log π_θ - log π_ref)
    log_ratio = log_probs_theta - log_probs_ref
    ratio = torch.exp(log_ratio)

    # 对 response 部分 mask
    mask = response_mask.float()
    seq_len = mask.sum(dim=1, keepdim=True).clamp(min=1)
    ratio_masked = (ratio * mask).sum(dim=1) / seq_len.squeeze(1)
    log_ratio_masked = (log_ratio * mask).sum(dim=1) / seq_len.squeeze(1)

    # 裁剪的策略比率
    clipped_ratio = torch.clamp(ratio_masked, 1.0 - epsilon, 1.0 + epsilon)

    # 策略梯度损失
    pg_loss = -torch.min(
        ratio_masked * advantages,
        clipped_ratio * advantages,
    ).mean()

    # KL 散度惩罚 (近似形式)
    # KL(π_θ || π_ref) = exp(log_ratio) - log_ratio - 1
    # 只在 response 部分计算
    kl_penalty = (torch.exp(log_ratio_masked) - log_ratio_masked - 1.0).mean()

    total_loss = pg_loss + beta * kl_penalty

    info = {
        "loss": total_loss.item(),
        "pg_loss": pg_loss.item(),
        "kl_penalty": kl_penalty.item(),
        "mean_ratio": ratio_masked.mean().item(),
        "mean_advantage": advantages.mean().item(),
        "approx_kl": (0.5 * (log_ratio_masked ** 2)).mean().item(),
    }

    return total_loss, info


# ─── 主训练函数 ─────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cap = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if cap >= (8, 0) else torch.float16
        print(f"Device: {device} (sm_{cap[0]}.{cap[1]}), dtype: {dtype}")
    else:
        dtype = torch.float32
        print(f"Device: {device} (CPU mode, float32)")

    # ── 加载模型 ──────────────────────────────────────────────────────────
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, PeftModel, get_peft_model, TaskType

    print(f"Loading base model: {args.model_name}")
    quant_kwargs = {}
    if args.quantization == "4bit":
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.quantization == "8bit":
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="cuda:0" if device == "cuda" else "cpu",
        attn_implementation=attn_impl,
        **quant_kwargs,
    )
    base_model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=False)

    # ── 加载 SFT checkpoint 作为参考策略 ───────────────────────────────
    sft_ckpt_path = args.sft_ckpt
    if sft_ckpt_path and os.path.exists(sft_ckpt_path):
        print(f"Loading SFT checkpoint: {sft_ckpt_path}")
        # 先应用 SFT LoRA 作为 'sft' adapter
        model = PeftModel.from_pretrained(
            base_model, sft_ckpt_path, adapter_name="sft",
            is_trainable=True,
        )
        # SFT adapter 就是我们的参考策略
        # 添加 RL 训练 adapter
        lora_cfg_rl = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model.add_adapter("rl", lora_cfg_rl)
        model.set_adapter("rl")
        print("Added RL LoRA adapter on top of SFT")
    else:
        print("No SFT checkpoint found, starting from base model")
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, lora_cfg)
        # 无参考策略时，使用当前策略作为参考（KL=0，退化为 REINFORCE）
        model.add_adapter("ref", lora_cfg)
        model.set_adapter("ref")
        # 再添加训练 adapter
        model.add_adapter("rl", lora_cfg)
        model.set_adapter("rl")

    # 可选项: 从 RL 检查点继续训练
    if args.load_rl_ckpt and os.path.exists(args.load_rl_ckpt):
        print(f"Loading RL checkpoint: {args.load_rl_ckpt}")
        # 加载到 'rl' adapter
        try:
            model.load_adapter(args.load_rl_ckpt, adapter_name="rl")
        except Exception as e:
            print(f"  Warning: could not load RL adapter: {e}")

    model.train()
    model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 冻结视觉编码器
    vision = model.base_model.model.model.visual
    for p in vision.parameters():
        p.requires_grad = False
    vision_fwd = vision.forward
    def vision_no_grad(*a, **kw):
        with torch.no_grad():
            return vision_fwd(*a, **kw)
    vision.forward = vision_no_grad

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"Trainable params: {n_trainable:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable, lr=args.lr, weight_decay=0.01)
        print("Using 8-bit AdamW")
    except ImportError:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # ── Data ──────────────────────────────────────────────────────────────
    samples = load_jsonl(args.data_path, args.max_samples)
    print(f"Dataset: {len(samples)} samples")

    # ── Renderer ─────────────────────────────────────────────────────────
    renderer = None
    if args.reward_mode in ("visual", "all"):
        try:
            from rl_vectorizer.rl.diffvg_renderer import DiffVGRenderer
            renderer = DiffVGRenderer(device)
            if renderer.is_available():
                print("Using DiffVG renderer (GPU)")
            else:
                print("DiffVG not available, using cairosvg fallback")
        except ImportError:
            print("DiffVG renderer not found, visual reward disabled")
            args.reward_mode = "geo" if args.reward_mode == "all" else args.reward_mode

    # ── Training Loop ─────────────────────────────────────────────────────
    user_prompt = (
        "Convert this architectural floor plan to SVG format. "
        "First analyze its structure, then generate the SVG step by step."
    )

    global_step = 0
    accum_steps = getattr(args, 'grad_accum_steps', 1)

    for epoch in range(args.epochs):
        epoch_losses = []
        epoch_rewards = []
        pbar = tqdm(samples, desc=f"RL Epoch {epoch+1}/{args.epochs}")

        for batch_idx, sample in enumerate(pbar):
            # ── Prepare input ──────────────────────────────────────────
            pil_img = load_image(sample)
            # 训练时用小图加速
            pil_img = pil_img.resize((args.image_size, args.image_size))

            target_tensor = (
                torch.from_numpy(np.array(pil_img)).float()
                .permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            )

            prompt_text = build_prompt(sample)
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt_text},
            ]}]
            txt = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            inputs = processor(text=[txt], images=[pil_img], return_tensors="pt")
            inputs = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]

            # ── Generate responses (with π_θ) ─────────────────────────
            model.set_adapter("rl")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_svg_tokens,
                    num_return_sequences=args.rollout_n,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.95,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # ── Extract and score responses ────────────────────────────
            responses = []
            generated_tokens = []
            for j in range(args.rollout_n):
                seq = outputs.sequences[j * inputs["input_ids"].shape[0]:
                                        (j + 1) * inputs["input_ids"].shape[0]]
                gen_tokens = seq[0][prompt_len:]
                generated_tokens.append(gen_tokens)
                resp = processor.decode(gen_tokens, skip_special_tokens=True)
                responses.append(resp)

            svg_codes = [extract_svg(r) for r in responses]
            interm_xmls = [extract_intermediate(r) for r in responses]

            # ── Compute rewards ────────────────────────────────────────
            rewards = compute_rewards_batch(
                svg_codes, target_tensor, interm_xmls,
                reward_mode=args.reward_mode,
                renderer=renderer, device=device,
            )
            rewards_t = torch.tensor(rewards, device=device)
            epoch_rewards.extend(rewards)

            # ── Normalize advantages ───────────────────────────────────
            if args.rollout_n >= 2 and rewards_t.std() > 1e-6:
                advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)
            else:
                advantages = torch.zeros_like(rewards_t)

            # ── Compute log-probs (π_θ and π_ref) ─────────────────────
            # 准备完整的输入序列（prompt + generation）
            batch_input_ids = []
            batch_attn_mask = []
            batch_labels = []  # 用于生成 loss mask

            for j in range(args.rollout_n):
                full_ids = torch.cat([
                    inputs["input_ids"][0],
                    generated_tokens[j],
                ])
                full_mask = torch.ones_like(full_ids)
                batch_input_ids.append(full_ids)
                batch_attn_mask.append(full_mask)

                # Labels: -100 for prompt, token IDs for response
                labels = full_ids.clone()
                labels[:prompt_len] = -100
                batch_labels.append(labels)

            # Pad batch
            max_len = max(ids.shape[0] for ids in batch_input_ids)
            padded_ids = torch.stack([
                F.pad(ids, (0, max_len - ids.shape[0]), value=processor.tokenizer.pad_token_id or 0)
                for ids in batch_input_ids
            ])
            padded_mask = torch.stack([
                F.pad(mask, (0, max_len - mask.shape[0]), value=0)
                for mask in batch_attn_mask
            ])
            padded_labels = torch.stack([
                F.pad(labels, (0, max_len - labels.shape[0]), value=-100)
                for labels in batch_labels
            ])

            # 复用 pixel_values, image_grid_thw (batch 中所有 rollout 共用)
            # Qwen2.5-VL processor 返回的 pixel_values 是 list[tensor]，需要 stack 成 [1, C, H, W]
            raw_pv = inputs["pixel_values"]
            if isinstance(raw_pv, (list, tuple)):
                raw_pv = torch.stack(raw_pv)
            batch_pixel_values = raw_pv.expand(args.rollout_n, -1, -1, -1).contiguous()

            raw_gt = inputs["image_grid_thw"]
            if isinstance(raw_gt, (list, tuple)):
                raw_gt = torch.stack(raw_gt)
            batch_grid_thw = raw_gt.expand(args.rollout_n, -1).contiguous()

            # ── π_θ log probs (trainable) ─────────────────────────────
            model.set_adapter("rl")
            model.train()
            outputs_theta = model(
                input_ids=padded_ids,
                attention_mask=padded_mask,
                pixel_values=batch_pixel_values,
                image_grid_thw=batch_grid_thw,
                labels=padded_labels,
            )
            # 从 logits 提取 response 部分的 log-probs
            logits_theta = outputs_theta.logits  # [B, L, V]
            log_probs_theta = _get_response_log_probs(
                logits_theta, padded_ids, padded_labels,
            )

            # ── π_ref log probs (no grad) ─────────────────────────────
            has_ref = False
            if sft_ckpt_path:
                try:
                    model.set_adapter("sft")
                    with torch.no_grad():
                        outputs_ref = model(
                            input_ids=padded_ids,
                            attention_mask=padded_mask,
                            pixel_values=batch_pixel_values,
                            image_grid_thw=batch_grid_thw,
                        )
                    logits_ref = outputs_ref.logits
                    log_probs_ref = _get_response_log_probs(
                        logits_ref, padded_ids, padded_labels,
                    )
                    has_ref = True
                except Exception:
                    pass

            if not has_ref:
                # 无参考策略时，将参考 log-probs 设为 detach 的 θ 版本
                log_probs_ref = log_probs_theta.detach()

            model.set_adapter("rl")

            # 构建 response mask
            response_mask = (padded_labels != -100).float()

            # ── GRPO Loss ──────────────────────────────────────────────
            loss, loss_info = compute_grpo_loss(
                log_probs_theta=log_probs_theta,
                log_probs_ref=log_probs_ref,
                advantages=advantages,
                response_mask=response_mask,
                epsilon=args.clip_epsilon,
                beta=args.kl_beta,
            )

            # 梯度累积
            loss = loss / accum_steps
            loss.backward()

            epoch_losses.append(loss_info["loss"])

            # 每 accum_steps 步更新参数
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(samples):
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            # 清理
            del outputs, outputs_theta
            if has_ref:
                del outputs_ref
            del inputs, batch_input_ids, padded_ids, padded_mask, padded_labels
            gc.collect()

            avg_r = np.mean(rewards) if rewards else 0.0
            pbar.set_postfix({
                "reward": f"{avg_r:.3f}",
                "loss": f"{loss_info['loss']:.3f}",
                "kl": f"{loss_info['approx_kl']:.4f}",
            })

            # 每 N 步释放缓存
            if (batch_idx + 1) % 10 == 0 and device == "cuda":
                torch.cuda.empty_cache()

        # ── Epoch end ─────────────────────────────────────────────────
        avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        print(f"\nEpoch {epoch+1} done — avg reward: {avg_epoch_reward:.4f}, "
              f"avg loss: {avg_epoch_loss:.4f}")

        # 生成示例展示
        if args.show_example and samples:
            _show_example(model, processor, samples[0], device, args)

        # 保存检查点
        if args.save_dir:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch+1}")
            os.makedirs(ckpt_path, exist_ok=True)
            model.set_adapter("rl")
            model.save_pretrained(ckpt_path)
            print(f"Saved RL checkpoint: {ckpt_path}")

    print("\n===== GRPO Training Complete! =====")


def _get_response_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """提取 response 部分每个 token 的 log-probability。

    Args:
        logits: [B, L, V]
        input_ids: [B, L]
        labels: [B, L] (prompt 部分为 -100, response 部分为 token id)

    Returns:
        [B, L] — 每个位置的 log-prob, prompt 位置为 0
    """
    log_probs = F.log_softmax(logits, dim=-1)  # [B, L, V]
    # 收集实际 token 的 log-prob
    token_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    # Mask out prompt 部分
    response_mask = (labels != -100).float()
    return token_log_probs * response_mask


def _show_example(model, processor, sample, device, args):
    """展示一个生成示例。"""
    model.set_adapter("rl")
    model.eval()

    pil_img = load_image(sample).resize((args.image_size, args.image_size))
    prompt_text = build_prompt(sample)
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil_img},
        {"type": "text", "text": prompt_text},
    ]}]
    txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[txt], images=[pil_img], return_tensors="pt")
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=args.max_svg_tokens,
            do_sample=False, temperature=0.1,  # 确定性生成用于展示
        )
    resp = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n  ── Example generation ──")
    print(f"  Response (first 300 chars): {resp[:300]}...")
    model.train()


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Floorplan SVG")
    parser.add_argument("--data-path", default="data/resplan/sft_train.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--sft-ckpt", default="checkpoints/sft/final")
    parser.add_argument("--load-rl-ckpt", default=None,
                        help="从 RL checkpoint 继续训练")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--rollout-n", type=int, default=4,
                        help="每组采样的响应数 (G)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-svg-tokens", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--quantization", default="4bit",
                        choices=["4bit", "8bit", "none"])
    parser.add_argument("--reward-mode", default="validity",
                        choices=["validity", "geo", "all"],
                        help="validity: 仅检有效性; geo: +几何; all: +视觉")
    parser.add_argument("--clip-epsilon", type=float, default=0.2,
                        help="GRPO 裁剪 ε")
    parser.add_argument("--kl-beta", type=float, default=0.04,
                        help="KL 散度惩罚系数 β")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--save-dir", default="checkpoints/rl_grpo")
    parser.add_argument("--show-example", action="store_true",
                        help="每 epoch 结束后展示一个生成示例")
    args = parser.parse_args()

    if args.quantization == "none":
        args.quantization = None

    train(args)


if __name__ == "__main__":
    main()
