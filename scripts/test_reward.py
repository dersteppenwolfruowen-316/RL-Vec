#!/usr/bin/env python3
"""
Reward 函数测试脚本
测试所有 Reward 组件
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image


def create_test_svg(valid: bool = True, simple: bool = False) -> str:
    """创建测试 SVG"""
    if not valid:
        return "<invalid>"

    if simple:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><line x1="10" y1="10" x2="90" y2="90" stroke="black"/></svg>'

    lines = [
        '<line x1="10" y1="10" x2="90" y2="90" stroke="black"/>',
        '<line x1="10" y1="90" x2="90" y2="10" stroke="black"/>',
        '<line x1="50" y1="10" x2="50" y2="90" stroke="black"/>',
        '<line x1="10" y1="50" x2="90" y2="50" stroke="black"/>',
    ]

    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
    svg += "".join(lines)
    svg += "</svg>"
    return svg


def test_geometric_reward():
    """测试几何约束 Reward"""
    print("\n" + "=" * 60)
    print("测试 1: GeometricConstraintReward")
    print("=" * 60)

    from rl_vectorizer.reward.geometric_reward import GeometricConstraintReward

    reward_fn = GeometricConstraintReward(weight=0.20)

    test_cases = [
        ("valid", create_test_svg(valid=True)),
        ("simple", create_test_svg(simple=True)),
        ("invalid", create_test_svg(valid=False)),
    ]

    for name, svg in test_cases:
        result = reward_fn.compute(svg)

        print(f"\n{name} SVG:")
        print(f"  Total: {result.total:.4f}")
        print(f"  Valid: {result.is_valid}")
        print(f"  Components: {result.components}")

        assert isinstance(result.total, float)
        assert isinstance(result.is_valid, bool)

    print("\n✓ GeometricConstraintReward 测试通过!")


def test_adversarial_reward():
    """测试对抗性 Reward"""
    print("\n" + "=" * 60)
    print("测试 2: AdversarialReward")
    print("=" * 60)

    from rl_vectorizer.reward.adversarial_reward import AdversarialReward

    reward_fn = AdversarialReward(weight=0.10)

    test_cases = [
        ("valid", create_test_svg(valid=True)),
        ("simple", create_test_svg(simple=True)),
        ("invalid", create_test_svg(valid=False)),
    ]

    for name, svg in test_cases:
        result = reward_fn.compute(svg)

        print(f"\n{name} SVG:")
        print(f"  Total: {result.total:.4f}")
        print(f"  Valid: {result.is_valid}")
        print(f"  Components: {result.components}")

        assert isinstance(result.total, float)

    print("\n✓ AdversarialReward 测试通过!")


def test_ensemble_reward():
    """测试集成 Reward"""
    print("\n" + "=" * 60)
    print("测试 3: EnsembleReward")
    print("=" * 60)

    from rl_vectorizer.reward.ensemble import EnsembleReward

    reward_fn = EnsembleReward(
        weights={
            "ssim": 0.30,
            "clip": 0.20,
            "keypoint": 0.15,
            "complexity": 0.10,
            "self_reward": 0.00,
            "geometric": 0.15,
            "adversarial": 0.10,
        }
    )

    svg = create_test_svg(valid=True)
    target = np.array(Image.new("RGB", (100, 100), color="white"))

    result = reward_fn.compute(svg, target)

    print(f"\n集成 Reward 结果:")
    print(f"  Total: {result.total:.4f}")
    print(f"  Valid: {result.is_valid}")
    print(f"  Weights: {reward_fn.weights}")

    for key, value in result.components.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    assert isinstance(result.total, float)
    assert result.is_valid == True

    print("\n✓ EnsembleReward 测试通过!")


def test_weight_validation():
    """测试权重验证"""
    print("\n" + "=" * 60)
    print("测试 4: 权重验证")
    print("=" * 60)

    from rl_vectorizer.reward.ensemble import EnsembleReward

    print("\n1. 测试不合法权重 (总和 != 1):")
    reward_fn = EnsembleReward(
        weights={
            "ssim": 0.5,
            "clip": 0.5,
            "keypoint": 0.5,
            "complexity": 0.5,
            "self_reward": 0.0,
            "geometric": 0.0,
            "adversarial": 0.0,
        }
    )
    print(f"  原始权重总和: 2.5")
    print(f"  归一化后总和: {sum(reward_fn.weights.values()):.4f}")

    print("\n2. 测试 Self-Reward 警告:")
    reward_fn = EnsembleReward(
        weights={
            "ssim": 0.40,
            "clip": 0.30,
            "keypoint": 0.20,
            "complexity": 0.10,
            "self_reward": 0.10,
            "geometric": 0.0,
            "adversarial": 0.0,
        }
    )

    print("\n✓ 权重验证测试完成!")


def test_component_control():
    """测试组件控制"""
    print("\n" + "=" * 60)
    print("测试 5: 组件控制")
    print("=" * 60)

    from rl_vectorizer.reward.ensemble import EnsembleReward

    reward_fn = EnsembleReward()

    print(f"\n1. 初始权重: {reward_fn.weights}")
    print(f"  Self-Reward: {reward_fn.weights.get('self_reward', 0):.4f}")

    reward_fn.disable_component("ssim")
    print(f"\n2. 禁用 SSIM 后: {reward_fn.weights}")
    assert reward_fn.weights.get("ssim") == 0.0

    reward_fn.set_weight("clip", 0.5)
    print(f"\n3. 设置 CLIP 权重为 0.5: {reward_fn.weights}")

    reward_fn.enable_all_components()
    print(f"\n4. 启用所有组件: {reward_fn.weights}")

    print("\n✓ 组件控制测试通过!")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Reward 函数测试套件")
    print("=" * 60)

    tests = [
        test_geometric_reward,
        test_adversarial_reward,
        test_ensemble_reward,
        test_weight_validation,
        test_component_control,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
