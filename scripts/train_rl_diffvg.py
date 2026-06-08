"""RL training with DiffVG rewards.
Usage:
  python scripts/train_rl_diffvg.py --sft-ckpt checkpoints/sft/final --max-samples 200
  python scripts/train_rl_diffvg.py --epochs 5 --rollout-n 4 --lr 1e-6
"""
import sys, os, json, argparse, gc, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image
from tqdm import tqdm
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None


def compute_ssim(pred_img: torch.Tensor, target_img: torch.Tensor) -> float:
    if ssim is None:
        return 0.0
    p = pred_img.detach().cpu().numpy().transpose(1, 2, 0)
    t = target_img.detach().cpu().numpy().transpose(1, 2, 0)
    min_dim = min(p.shape[0], t.shape[0], p.shape[1], t.shape[1])
    if min_dim < 7:
        return 0.0
    return float(ssim(p[:min_dim, :min_dim], t[:min_dim, :min_dim], channel_axis=2, data_range=1.0))


def load_sample(jsonl_path: str, idx: int):
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line.strip())
    return None


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


def build_prompt(sample: dict) -> str:
    prompt = sample.get("conversations", [{}])[0].get("value", "")
    prompt = re.sub(r"<image>\s*", "", prompt)
    return prompt


def compute_reward(pred_svg: str, target_img: torch.Tensor, renderer) -> dict:
    from lxml import etree
    try:
        etree.fromstring(pred_svg.encode())
        valid = True
    except Exception:
        valid = False

    if not valid:
        return {"valid": False, "ssim": 0.0, "reward": -1.0}

    rendered = renderer.render(pred_svg, target_img.shape[2], target_img.shape[1])
    ssim_val = compute_ssim(rendered[0], target_img[0])
    reward = ssim_val - 0.5
    return {"valid": True, "ssim": ssim_val, "reward": reward}


def train(args):
    from transformers import (
        Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from rl_vectorizer.rl.diffvg_renderer import DiffVGRenderer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability() >= (8, 0) else torch.float16

    quant_kwargs = {}
    if args.quantization == "4bit":
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )

    if device == "cuda":
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
    else:
        attn_impl = "eager"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype,
        device_map="cuda:0" if device == "cuda" else "cpu",
        attn_implementation=attn_impl, **quant_kwargs,
    )

    if args.sft_ckpt and os.path.exists(args.sft_ckpt):
        model = PeftModel.from_pretrained(model, args.sft_ckpt)
        print(f"Loaded SFT checkpoint: {args.sft_ckpt}")

    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    if not args.sft_ckpt:
        model = get_peft_model(model, lora_cfg)

    model.config.use_cache = False
    model.train()
    model.enable_input_require_grads()
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=False)

    vision_fwd = model.base_model.model.model.visual.forward
    def vision_no_grad(*a, **kw):
        with torch.no_grad():
            return vision_fwd(*a, **kw)
    model.base_model.model.model.visual.forward = vision_no_grad

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    renderer = DiffVGRenderer(device)
    print(f"Renderer: {'diffvg' if renderer.is_available() else 'cairosvg'}")

    samples = []
    with open(args.data_path) as f:
        for i, line in enumerate(f):
            if args.max_samples and i >= args.max_samples:
                break
            samples.append(json.loads(line.strip()))
    print(f"Dataset: {len(samples)} samples")

    for epoch in range(args.epochs):
        epoch_rewards = []
        pbar = tqdm(samples, desc=f"RL Epoch {epoch+1}/{args.epochs}")

        for batch_idx, sample in enumerate(pbar):
            img_path = sample["image"]
            if not os.path.isabs(img_path):
                img_path = os.path.join(os.path.dirname(args.data_path), "..", "bitmaps", os.path.basename(img_path))
            if not os.path.exists(img_path):
                continue

            pil_img = Image.open(img_path).convert("RGB").resize((112, 112))
            target_tensor = torch.from_numpy(np.array(pil_img)).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

            prompt_text = build_prompt(sample)
            msgs = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt_text}]}]
            txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[txt], images=[pil_img], return_tensors="pt")
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=args.max_svg_tokens,
                    num_return_sequences=args.rollout_n,
                    do_sample=True, temperature=args.temperature, top_p=0.95,
                    output_scores=True, return_dict_in_generate=True,
                )

            rollout_rewards = []
            for j in range(args.rollout_n):
                seq = outputs.sequences[j * inputs["input_ids"].shape[0]:(j + 1) * inputs["input_ids"].shape[0]]
                resp = processor.decode(seq[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                svg = extract_svg(resp)
                result = compute_reward(svg, target_tensor, renderer)
                rollout_rewards.append(result["reward"])
                if batch_idx == 0 and j == 0:
                    print(f"  Sample: valid={result['valid']}, ssim={result['ssim']:.4f}")

            rollout_rewards_t = torch.tensor(rollout_rewards, device=device)
            reward_mean = rollout_rewards_t.mean()
            reward_std = max(rollout_rewards_t.std(), 1e-6)
            normalized_rewards = (rollout_rewards_t - reward_mean) / reward_std

            # REINFORCE update
            logits = outputs.scores
            all_log_probs = []
            seq_len = outputs.sequences.shape[1] - inputs["input_ids"].shape[1]
            for t in range(seq_len):
                logits_t = logits[t]
                token_id = outputs.sequences[:, inputs["input_ids"].shape[1] + t]
                log_probs = torch.log_softmax(logits_t, dim=-1)
                token_log_prob = log_probs.gather(1, token_id.unsqueeze(-1)).squeeze(-1)
                all_log_probs.append(token_log_prob)
            log_probs_per_seq = torch.stack(all_log_probs, dim=1).sum(dim=1)

            loss = -(log_probs_per_seq * normalized_rewards).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_rewards.extend(rollout_rewards)
            avg_r = np.mean(rollout_rewards)
            pbar.set_postfix({"reward": f"{avg_r:.4f}"})

            del inputs, outputs, loss
            gc.collect()
            if (batch_idx + 1) % 10 == 0 and device == "cuda":
                torch.cuda.empty_cache()

        avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else -1.0
        print(f"Epoch {epoch+1} done — avg reward: {avg_epoch_reward:.4f}")

        if args.save_dir:
            ckpt = os.path.join(args.save_dir, f"rl_epoch_{epoch+1}")
            os.makedirs(ckpt, exist_ok=True)
            model.save_pretrained(ckpt)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="RL + DiffVG Training")
    parser.add_argument("--data-path", default="data/resplan/sft_train.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--sft-ckpt", default="checkpoints/sft/final")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-svg-tokens", type=int, default=1024)
    parser.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", None])
    parser.add_argument("--save-dir", default="checkpoints/rl_diffvg")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
