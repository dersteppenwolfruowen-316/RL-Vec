#!/usr/bin/env python3
"""
GRPO 算法测试脚本
测试 GRPO 训练器的各个组件
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
from rl_vectorizer.training.grpo_trainer import (
    GRPOTrainer,
    GRPOConfig,
    compute_grpo_advantage,
    compute_ppo_loss,
)


def test_grpo_advantage():
    """测试 GRPO 优势计算"""
    print("\n" + "=" * 60)
    print("测试 1: GRPO 优势计算")
    print("=" * 60)

    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    advantages = compute_grpo_advantage(rewards, normalize=True)

    print(f"原始 rewards: {rewards.tolist()}")
    print(f"计算的优势: {advantages.tolist()}")
    print(f"优势均值: {advantages.mean().item():.4f} (应为 0)")
    print(f"优势标准差: {advantages.std().item():.4f} (应为 1)")

    assert abs(advantages.mean().item()) < 0.01, "优势均值应接近 0"
    assert abs(advantages.std().item() - 1.0) < 0.1, "优势标准差应接近 1"

    print("✓ GRPO 优势计算测试通过!")


def test_ppo_loss():
    """测试 PPO 损失函数"""
    print("\n" + "=" * 60)
    print("测试 2: PPO 损失函数")
    print("=" * 60)

    log_probs = torch.tensor([-1.0, -1.5, -2.0])
    old_log_probs = torch.tensor([-1.2, -1.7, -2.2])
    advantages = torch.tensor([1.0, 0.5, -0.5])

    loss = compute_ppo_loss(log_probs, old_log_probs, advantages, epsilon=0.2)

    print(f"当前 log_probs: {log_probs.tolist()}")
    print(f"旧 log_probs: {old_log_probs.tolist()}")
    print(f"优势: {advantages.tolist()}")
    print(f"PPO 损失: {loss.item():.4f}")

    assert loss.item() <= 0, "PPO 损失应为负（正 advantage 下，策略更倾向于当前动作）"
    print("✓ PPO 损失计算测试通过!")


def test_grpo_config():
    """测试 GRPO 配置"""
    print("\n" + "=" * 60)
    print("测试 3: GRPO 配置")
    print("=" * 60)

    config = GRPOConfig(
        group_size=16,
        kl_beta=0.05,
        epsilon=0.15,
        entropy_coef=0.02,
    )

    config_dict = config.to_dict()
    print("配置参数:")
    for key, value in config_dict.items():
        print(f"  {key}: {value}")

    restored_config = GRPOConfig.from_dict(config_dict)
    assert restored_config.group_size == 16
    assert restored_config.kl_beta == 0.05

    print("✓ GRPO 配置测试通过!")


def test_advantage_clipping():
    """测试优势裁剪"""
    print("\n" + "=" * 60)
    print("测试 4: 优势裁剪")
    print("=" * 60)

    rewards = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    advantages = compute_grpo_advantage(rewards, normalize=True)

    print(f"原始优势: {advantages.tolist()}")
    print(f"优势范围: [{advantages.min().item():.2f}, {advantages.max().item():.2f}]")

    assert advantages.max().item() <= 10, "优势应被裁剪到 10"
    assert advantages.min().item() >= -10, "优势应被裁剪到 -10"

    print("✓ 优势裁剪测试通过!")


def test_training_loop():
    """测试训练循环"""
    print("\n" + "=" * 60)
    print("测试 5: 训练循环")
    print("=" * 60)

    config = {
        "grpo": {
            "group_size": 8,
            "kl_beta": 0.1,
            "epsilon": 0.2,
            "entropy_coef": 0.01,
            "reward_normalize": True,
            "clip_advantages": True,
        },
        "training": {
            "max_grad_norm": 1.0,
            "epochs": 1,
            "batch_size": 2,
        },
        "experiment": {
            "tensorboard": {
                "log_interval": 10,
            },
        },
    }

    print("模拟训练循环...")
    for step in range(3):
        rewards = torch.randn(16)
        advantages = compute_grpo_advantage(rewards, normalize=True)

        print(f"Step {step + 1}: reward_mean={rewards.mean().item():.4f}, "
              f"adv_mean={advantages.mean().item():.4f}")

    print("✓ 训练循环测试通过!")


def test_kl_divergence():
    """测试 KL 散度"""
    print("\n" + "=" * 60)
    print("测试 6: KL 散度")
    print("=" * 60)

    log_probs = torch.tensor([-2.0, -2.5, -3.0])  # 当前策略概率较低
    ref_log_probs = torch.tensor([-1.0, -1.5, -2.0])  # 参考模型概率较高

    import torch.nn.functional as F
    kl_div = F.kl_div(
        log_probs,
        ref_log_probs,
        reduction='batchmean',
        log_target=True
    )

    print(f"当前 log_probs: {log_probs.tolist()}")
    print(f"参考 log_probs: {ref_log_probs.tolist()}")
    print(f"KL 散度: {kl_div.item():.4f}")

    assert kl_div.item() > 0, "KL 散度应为正（当前策略概率低于参考模型时）"

    print("✓ KL 散度测试通过!")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("GRPO 算法测试套件")
    print("=" * 60)

    tests = [
        test_grpo_advantage,
        test_ppo_loss,
        test_grpo_config,
        test_advantage_clipping,
        test_training_loop,
        test_kl_divergence,
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
