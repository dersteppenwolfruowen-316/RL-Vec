# SFT 训练错误汇总与解决方案

> 日期: 2026-06-05
> 环境: Colab T4 (14.5 GB), Qwen2.5-VL-3B-Instruct, 4bit LoRA
> 数据: ResPlan 200 samples

---

## 一、环境依赖类

### 1. torchao 版本冲突

**错误信息**
```
ImportError: Found an incompatible version of torchao.
Found version 0.10.0, but only versions above 0.16.0 are supported
```

**原因**
Colab 预装 torchao 0.10.0，PEFT 新版本要求 ≥0.16.0 才能使用 torchao 的 LoRA dispatch。

**解决**
```bash
pip uninstall -y torchao
```
卸载后 PEFT 自动 fallback 到标准 LoRA 实现，不影响训练效果。

---

## 二、数据处理类

### 2. image_grid_thw 未传递

**错误信息**
```
TypeError: 'NoneType' object is not iterable
  File "...modeling_qwen2_5_vl.py", line 354, in rot_pos_emb
    for t, h, w in grid_thw:
```

**原因**
Qwen2.5-VL 的 vision encoder 需要 `image_grid_thw` 参数来计算图像 patch 的位置编码。训练循环只传了 `pixel_values`，遗漏了 processor 输出的其他字段。

**解决**
将 processor 返回的所有 tensor 字段（`input_ids`、`attention_mask`、`pixel_values`、`image_grid_thw`、`labels`）动态传递给 model.forward：

```python
model_kwargs = {}
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        model_kwargs[k] = v.to(device)
    else:
        model_kwargs[k] = v
outputs = model(**model_kwargs)
```

---

### 3. Image token 缺失（tokens: 0）

**错误信息**
```
ValueError: Image features and image tokens do not match, tokens: 0, features: 1369
```

**原因**
文本中没有任何 `<|image_pad|>` token，但图像处理产生了 1369 个 patch。`processor(text=..., images=...)` 在部分 transformers 版本中没有正确插入 image tokens。

**解决**
手动控制 image token 的插入：

1. 先用 `processor.image_processor` 处理图像，获取 `image_grid_thw`
2. 计算需要的 image token 数量
3. 在 token ID 层面插入 `<|image_pad|>` token（不是字符串拼接，是直接操作 token ID 列表）

---

### 4. Image token 过量（tokens: 2045）

**错误信息**
```
ValueError: Image features and image tokens do not match, tokens: 2045, features: 1369
```

**原因**
字符串拼接方式插入 `<|image_pad|>` 时，tokenizer 可能没有正确将其识别为单个 token，导致数量不匹配。

**解决**
不在文本层面操作，改用 `tokenizer.encode()` 得到 token ID 后，用列表切片插入：

```python
input_ids = ids[:pos] + [image_token_id] * num_patches + ids[pos:]
```

---

### 5. Patch merge 未考虑（tokens: 256, features: 64）

**错误信息**
```
ValueError: Image features and image tokens do not match, tokens: 256, features: 64
```

**原因**
Qwen2.5-VL 的 vision encoder 内部有 **2×2 patch merge** 层。`image_grid_thw` 返回的是 merge **前**的 grid（16×16=256），但模型期望的是 merge **后**的数量（8×8=64）。

```
image_grid_thw = [1, 16, 16]  ← merge 前
                     ↓  2×2 merge
                    [1, 8, 8] ← model 实际需要的
```

**解决**
计算 image token 数量时除以 `spatial_merge_size=2`：

```python
merge_size = 2  # Qwen2.5-VL-3B 默认
num_patches = int(t * (h // merge_size) * (w // merge_size))
```

---

### 6. text 格式错误

**错误信息**
```
ValueError: text input must be of type `str` (single example),
`list[str]` (batch or single pretokenized example) ...
```

**原因**
将 conversation dict 直接传给了 processor 的 `text=` 参数，但 processor 期望的是字符串。

**解决**
先用 `processor.apply_chat_template()` 转为字符串：

```python
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
inputs = processor(text=[text], images=[image], ...)
```

---

## 三、显存类（OOM）

### 7. 初始 OOM（gradient checkpointing 未开）

**错误信息**
```
CUDA out of memory. Tried to allocate 72.00 MiB
GPU 0 has a total capacity of 14.56 GiB
```

**原因**
模型 7.5GB 加载 + vision encoder 激活值 + 4096 tokens 序列 + LoRA 梯度，超出 T4 14.5GB。

**初始解决**
- 开启 gradient checkpointing
- 冻结 vision encoder
- 4bit 量化
- 将 max_length 从 4096 降到 2048

---

### 8. backward OOM（图像被放大）

**错误信息**
```
OOM during loss.backward(). Tried to allocate 1.79 GiB
```

**原因**
即使 resize 到 224×224，Qwen2.5-VL 的 `image_processor` 默认 `min_pixels=262144`（~512×512），会将图像放大到 ~518×518，产生 **1369 个 patch**，backward 时显存超限。

**解决**
传入 `min_pixels` 阻止放大：

```python
processor.image_processor([img], return_tensors="pt", min_pixels=224*224)
```

并将图像缩放到 224×224（patch 数降至 64）。

---

### 9. ⭐ 核心根因：bf16 在 T4 上模拟导致显存翻倍 ⭐

**错误信息**
```
OOM during loss.backward(). Tried to allocate 3.97 GiB
（即使 112×112 图像 + 所有优化全开，仍然 OOM）
```

**原因**
T4 显卡**不支持 bfloat16 原生计算**。`torch.cuda.is_bf16_supported()` 在 PyTorch 2.x 上返回 True，但实际上是通过 **fp32 软件模拟** 实现的。所有激活值、梯度、中间变量以 fp32（4 字节）存储而非 bf16（2 字节），**显存占用翻倍**。

| 数据类型 | T4 支持 | 每元素大小 |
|---------|---------|-----------|
| float16 | ✅ 原生 | 2 bytes |
| bfloat16 | ❌ 模拟（实际 fp32） | **4 bytes** |
| float32 | ✅ 原生 | 4 bytes |

对于一个 3B 模型 + ~500 tokens 的 forward + backward，fp32 模拟会导致显存多出 ~4-6 GB，恰好超出 T4 极限。

**解决**
强制使用 `torch.float16`：

```python
dtype = torch.float16  # 不要用 bfloat16
```

**⚠️ 这是我们碰到的所有 OOM 的根本原因，之前的各种优化只是推迟了它在 backward 时崩溃。** 去掉 bf16 模拟的开销后，其他优化（112px、gradient checkpointing、freeze vision）叠加，应该能在 T4 上跑通。

---

## 四、最终解决方案清单

所有修复已整合到 `scripts/train_sft.py`：

| # | 优化措施 | 省显存估计 |
|---|---------|-----------|
| 1 | **torch.float16（非 bfloat16）** | **~4-6 GB** |
| 2 | 4bit 量化 | ~3 GB |
| 3 | 冻结 vision encoder | ~1.5 GB |
| 4 | gradient checkpointing (use_reentrant=True) | ~2 GB |
| 5 | 图像 112×112 + min_pixels=12544 | ~1 GB |
| 6 | 8-bit AdamW optimizer | ~0.5 GB |
| 7 | expandable_segments + max_split_size_mb:128 | 减少碎片 |
| 8 | 每步 gc.collect() + empty_cache() | 防止泄漏 |
| 9 | 手动插入 image tokens（除以 merge_size=2） | 正确性修复 |
| 10 | 全量传递 model_kwargs（含 image_grid_thw） | 正确性修复 |

> **预期效果**：以上优化全部叠加后，T4 上 200 samples × 3 epoch 的 SFT 训练应能跑通。如果仍然 OOM，建议换 **A100**（40GB）或 **L4**（22.5GB）。

---

## 五、代码文件对应关系

| 文件 | 功能 | 状态 |
|------|------|------|
| `scripts/train_sft.py` | SFT 训练主脚本（含全部修复） | ✅ 已整合 |
| `prepare_sft_data.py` | ResPlan → 中间指令 JSONL | ✅ |
| `colab_sft_training.ipynb` | Colab notebook（安装+数据+训练+评估） | ✅ |
| `config/easyr1/qwen2_5_vl_3b_grpo.yaml` | GRPO 训练配置 | ✅ 待测试 |
| `scripts/easyr1_reward.py` | GRPO reward 函数 | ✅ 待测试 |
| `src/rl_vectorizer/reward/refinement_reward.py` | DiffVG 增量奖励 | ✅ 待集成 |
| `src/rl_vectorizer/utils/diff_line_rasterizer.py` | 可微线段光栅化器 | ✅ 待测试 |
