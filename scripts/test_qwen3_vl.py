#!/usr/bin/env python3
"""Smoke-test Qwen-VL integration for rl_vectorizer.

Default mode is intentionally lightweight: it verifies package imports and local
Python dependencies without downloading/loading a multi-GB model. Use
``--load-model`` when you explicitly want to instantiate the HuggingFace model.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
REQUIRED_IMPORTS = [
    ("torch", "PyTorch runtime"),
    ("transformers", "HuggingFace model/processor runtime"),
    ("peft", "LoRA adapter runtime"),
    ("accelerate", "device_map='auto' model loading"),
    ("PIL", "image input support"),
]


def dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def test_imports(model_name: str = DEFAULT_MODEL) -> bool:
    print("=" * 60)
    print("Qwen-VL Import / Environment Smoke Test")
    print("=" * 60)
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)

    ok = True

    print("\n1. Checking Python dependencies...")
    for module_name, description in REQUIRED_IMPORTS:
        available = dependency_available(module_name)
        mark = "✓" if available else "✗"
        print(f"  {mark} {module_name:<14} {description}")
        ok = ok and available

    print("\n2. Checking rl_vectorizer package exports...")
    try:
        import rl_vectorizer
        import rl_vectorizer.models as models
        from rl_vectorizer.models import QwenVLModel

        print(f"  ✓ rl_vectorizer: {rl_vectorizer.__file__}")
        print(f"  ✓ models module: {models.__file__}")
        if QwenVLModel is None:
            ok = False
            print("  ✗ QwenVLModel export is None")
            if getattr(models, "QWEN_IMPORT_ERROR", None):
                print(f"    import error: {models.QWEN_IMPORT_ERROR}")
        else:
            print(f"  ✓ QwenVLModel export: {QwenVLModel}")
    except Exception as exc:
        ok = False
        print(f"  ✗ package import failed: {type(exc).__name__}: {exc}")

    print("\n3. Checking transformers model class availability...")
    try:
        from rl_vectorizer.models.qwen_vl import _resolve_vlm_class

        model_cls = _resolve_vlm_class(model_name)
        print(f"  ✓ {model_name}: {model_cls.__name__}")
    except Exception as exc:
        ok = False
        print(f"  ✗ {model_name}: {type(exc).__name__}: {exc}")

    print("\nResult:", "PASS" if ok else "FAIL")
    if not ok:
        print("\nInstall missing dependencies with:")
        print("  pip install -r requirements.txt")
        print("If Qwen3-VL is unavailable, upgrade transformers or use --model Qwen/Qwen2.5-VL-3B-Instruct.")
    else:
        print("\nImport smoke test passed. Use --load-model to test actual model loading.")

    return ok


def test_model_loading(model_name: str, use_flash_attention: bool = True):
    print("=" * 60)
    print("Testing Qwen-VL Model Loading")
    print("=" * 60)

    try:
        from rl_vectorizer.models import QwenVLModel

        if QwenVLModel is None:
            raise RuntimeError("QwenVLModel is not exported; run the import smoke test for details.")

        print("\n1. Loading model with config...")
        model = QwenVLModel(
            base_model_name=model_name,
            lora_rank=16,
            use_flash_attention=use_flash_attention,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("✓ Model loaded successfully!")

        print("\n2. Getting model info...")
        info = model.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")

        print("\n3. Testing image generation...")
        test_image = Image.new("RGB", (512, 512), color="white")

        response = model.generate(
            image=test_image,
            prompt="Describe this image briefly.",
            max_new_tokens=50,
        )
        print(f"✓ Generated response: {response[:100]}...")

        print("\n4. Testing SVG generation...")
        svg_code = model.generate_svg(
            image=test_image,
            svg_format="simple",
            max_new_tokens=100,
        )
        print(f"✓ Generated SVG (length: {len(svg_code)} chars)")

        print("\n5. Testing batch generation...")
        images = [test_image, test_image]
        responses = model.generate_batch(
            images=images,
            prompts=["Describe this.", "What do you see?"],
            max_new_tokens=50,
        )
        print(f"✓ Generated {len(responses)} responses")

        print("\n" + "=" * 60)
        print("Model loading test passed!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantization(model_name: str):
    print("\n" + "=" * 60)
    print("Testing Quantization Options")
    print("=" * 60)

    try:
        from rl_vectorizer.models import QwenVLModel

        if QwenVLModel is None:
            raise RuntimeError("QwenVLModel is not exported; run the import smoke test for details.")

        print("\n1. Testing 4-bit quantization...")
        model = QwenVLModel(
            base_model_name=model_name,
            quantization="4bit",
        )
        print("✓ 4-bit quantization loaded!")
        return True

    except Exception as e:
        print(f"\n✗ Quantization test failed: {e}")
        return False


def test_inference_pipeline(model_name: str):
    print("\n" + "=" * 60)
    print("Testing Complete Inference Pipeline")
    print("=" * 60)

    try:
        from rl_vectorizer.models import QwenVLModel
        from rl_vectorizer.reward import EnsembleReward

        if QwenVLModel is None:
            raise RuntimeError("QwenVLModel is not exported; run the import smoke test for details.")

        print("\n1. Loading model...")
        model = QwenVLModel(
            base_model_name=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        print("\n2. Loading reward function...")
        reward_fn = EnsembleReward(
            weights={
                "ssim": 0.40,
                "clip": 0.30,
                "keypoint": 0.20,
                "complexity": 0.10,
            }
        )

        print("\n3. Running inference...")
        test_image = Image.new("RGB", (512, 512), color="white")

        svg_code = model.generate_svg(
            image=test_image,
            svg_format="detailed",
            max_new_tokens=256,
        )

        print(f"✓ Generated SVG (length: {len(svg_code)} chars)")

        if svg_code:
            print("\n4. Computing reward...")
            import numpy as np
            target_bmp = np.array(test_image)
            result = reward_fn.compute(svg_code, target_bmp)
            print(f"✓ Reward computed: {result.total:.4f}")
            print(f"  Components: {result.components}")

        print("\n" + "=" * 60)
        print("Inference pipeline test passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Test rl_vectorizer Qwen-VL integration.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id to check/load.")
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Actually instantiate the model. This may download weights and require GPU/RAM.",
    )
    parser.add_argument("--full", action="store_true", help="Run quantization and reward pipeline tests too.")
    parser.add_argument("--no-flash-attention", action="store_true", help="Disable flash attention request.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    success = test_imports(args.model)

    if args.load_model or args.full:
        success = test_model_loading(
            model_name=args.model,
            use_flash_attention=not args.no_flash_attention,
        ) and success

    if args.full:
        test_quantization(args.model)
        test_inference_pipeline(args.model)

    sys.exit(0 if success else 1)
