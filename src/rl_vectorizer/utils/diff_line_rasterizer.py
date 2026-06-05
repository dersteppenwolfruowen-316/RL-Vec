"""可微线段光栅化器 — 纯 PyTorch 实现，不依赖 CUDA。

专为建筑平面图（ResPlan）的直线结构设计。
支持:
  - 线段集的可微渲染
  - 端点到坐标的梯度回传
  - 多线段批量渲染
  - 抗锯齿边缘
"""
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F


class DifferentiableLineRasterizer(torch.nn.Module):
    """纯 PyTorch 可微线段光栅化器。

    用法:
        rasterizer = DifferentiableLineRasterizer(H=256, W=256)

        lines = torch.tensor([[x1, y1, x2, y2], ...], requires_grad=True)
        img = rasterizer(lines)                    # [H, W, 3] 渲染图

        loss = ((img - target) ** 2).mean()
        loss.backward()                            # -> 梯度传到 lines
    """

    def __init__(
        self,
        H: int = 1024,
        W: int = 1024,
        stroke_width: float = 2.0,
        stroke_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        sharpness: float = 50.0,
        mode: str = "gaussian",
    ):
        """mode: 'gaussian' (梯度平滑) 或 'sigmoid' (边缘锐利)"""
        super().__init__()
        self.H = H
        self.W = W
        self.base_stroke_width = stroke_width
        self.register_buffer("stroke_color", torch.tensor(stroke_color))
        self.register_buffer("bg_color", torch.tensor(bg_color))
        self.sharpness = sharpness
        self.mode = mode

        # 像素网格坐标 [H, W, 2]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("grid", torch.stack([grid_x, grid_y], dim=-1))  # [H, W, 2]

    def forward(
        self,
        lines: torch.Tensor,
        stroke_widths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """渲染线段集。

        Args:
            lines:    [N, 4] 线段坐标 (x1, y1, x2, y2)
            stroke_widths: [N] 可选，每条线的宽度 (默认用 self.base_stroke_width)

        Returns:
            img: [H, W, 3] RGB 图像，取值范围 [0, 1]
        """
        N = lines.shape[0]
        device = lines.device

        # 像素网格 [H, W, 2] -> [H, W, 1, 2]
        pixels = self.grid.to(device).unsqueeze(2)  # [H, W, 1, 2]

        # 线段参数
        a = lines[:, :2]   # [N, 2] 起点
        b = lines[:, 2:]   # [N, 2] 终点
        ab = b - a          # [N, 2]
        ab_sq = (ab * ab).sum(dim=-1, keepdim=True)  # [N, 1]

        sw = stroke_widths if stroke_widths is not None else self.base_stroke_width

        # --- 计算每个像素到每条线段的距离 ---
        # t = clamp(dot(p - a, ab) / dot(ab, ab), 0, 1)
        # closest = a + t * ab
        # dist = ||p - closest||

        # [H, W, N, 2] 每个像素到每条线段起点的向量
        ap = pixels - a.unsqueeze(0).unsqueeze(0)  # [H, W, N, 2]

        # 计算 t
        # 防止除零：ab 长度为 0 的线段（点）单独处理
        t_numer = (ap * ab.unsqueeze(0).unsqueeze(0)).sum(dim=-1)       # [H, W, N]
        t_denom = ab_sq.squeeze(-1).unsqueeze(0).unsqueeze(0)           # [H, W, N]
        t_denom = t_denom + 1e-8  # 防除零
        t = torch.clamp(t_numer / t_denom, 0.0, 1.0)                   # [H, W, N]

        # 最近点坐标
        closest = a.unsqueeze(0).unsqueeze(0) + t.unsqueeze(-1) * ab.unsqueeze(0).unsqueeze(0)

        # 距离
        dist = torch.norm(pixels - closest, dim=-1)                     # [H, W, N]

        # 取每条线段的最小距离 (每个像素到线段集的最短距离)
        dist, _ = dist.min(dim=-1)                                      # [H, W]

        # --- 用 sigmoid 或 gaussian 做抗锯齿混合 ---
        if isinstance(sw, torch.Tensor):
            sw_val = sw.mean().item()
        else:
            sw_val = sw

        if self.mode == "gaussian":
            # 高斯管模型：alpha = exp(-dist² / (2*σ²))
            # 梯度永远不会消失，适合优化
            sigma = sw_val / 2.0
            alpha = torch.exp(-(dist ** 2) / (2 * sigma * sigma + 1e-8))  # [H, W]
            alpha = torch.clamp(alpha, 0.0, 1.0)
        else:
            # Sigmoid 抗锯齿模式：边缘锐利，适合最终渲染
            alpha = torch.sigmoid((sw_val / 2.0 - dist) * self.sharpness)

        # 混合前景和背景
        img = (
            self.bg_color.to(device).unsqueeze(0).unsqueeze(0) * (1.0 - alpha.unsqueeze(-1))
            + self.stroke_color.to(device).unsqueeze(0).unsqueeze(0) * alpha.unsqueeze(-1)
        )

        return img  # [H, W, 3]

    def render_from_svg(
        self,
        svg_code: str,
        lines: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """从 SVG 代码中提取线段并渲染 (方便集成到现有 pipeline)。

        Args:
            svg_code: SVG 文本
            lines: 可选，预提取的线段张量。None 则自动从 SVG 解析

        Returns:
            img: [H, W, 3]
        """
        if lines is None:
            lines = self._extract_lines_from_svg(svg_code)
        return self.forward(lines)

    @staticmethod
    def _extract_lines_from_svg(svg_code: str) -> torch.Tensor:
        """从 SVG 中提取所有 line/path 元素的端点坐标。"""
        from lxml import etree
        import re
        nsmap = {"svg": "http://www.w3.org/2000/svg"}

        lines = []
        try:
            tree = etree.fromstring(svg_code.encode())
        except Exception:
            return torch.zeros((0, 4))

        for elem in tree.xpath("//svg:line", namespaces=nsmap):
            x1 = float(elem.get("x1", 0))
            y1 = float(elem.get("y1", 0))
            x2 = float(elem.get("x2", 0))
            y2 = float(elem.get("y2", 0))
            lines.append([x1, y1, x2, y2])

        for elem in tree.xpath("//svg:path", namespaces=nsmap):
            d = elem.get("d", "")
            nums = re.findall(r"-?\d+\.?\d*", d)
            for i in range(0, len(nums) - 3, 2):
                if i + 3 < len(nums):
                    lines.append([
                        float(nums[i]), float(nums[i + 1]),
                        float(nums[i + 2]), float(nums[i + 3]),
                    ])

        return torch.tensor(lines, dtype=torch.float32) if lines else torch.zeros((0, 4))


def optimize_lines(
    init_lines: torch.Tensor,
    target_img: torch.Tensor,
    num_steps: int = 300,
    lr: float = 50.0,
    H: int = 1024,
    W: int = 1024,
    stroke_width: float = 2.0,
    verbose: bool = True,
) -> Tuple[torch.Tensor, list]:
    """优化线段坐标以匹配目标图像（coarse-to-fine 策略）。

    先用模糊高斯管找到大致位置，再逐渐变细精确对齐。
    """
    from rl_vectorizer.utils.diff_line_rasterizer import DifferentiableLineRasterizer

    rasterizer = DifferentiableLineRasterizer(H=H, W=W, stroke_width=stroke_width, mode="gaussian")
    lines = init_lines.clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([lines], lr=lr, momentum=0.9)
    history = []

    # coarse-to-fine: sigma 从宽到窄
    sigmas = torch.linspace(stroke_width * 3, stroke_width * 0.6, num_steps)

    for step in range(num_steps):
        optimizer.zero_grad()
        # 更新当前的 sigma
        sigma = sigmas[step].item()
        img = rasterizer(lines, stroke_widths=torch.tensor([sigma]))
        diff = (img - target_img.to(img.device)).abs()

        # top-k loss: 只关注误差最大的 20% 像素（即线段区域）
        k = int(0.2 * diff.numel())
        topk_loss = diff.view(-1).topk(k)[0].mean()

        topk_loss.backward()
        optimizer.step()
        history.append(topk_loss.item())

        if verbose and (step + 1) % 60 == 0:
            print(f"  Step {step + 1}/{num_steps}: loss={topk_loss.item():.6f}, "
                  f"sigma={sigma:.1f}")

    return lines.detach(), history
