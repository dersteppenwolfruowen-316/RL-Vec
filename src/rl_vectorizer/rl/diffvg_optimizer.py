import torch
import numpy as np
from .diffvg_renderer import svg_to_scene, parse_svg_path_commands, commands_to_diffvg_shapes


class DiffVGOptimizer:
    def __init__(self, device: str = "cuda", num_iter: int = 100, lr: float = 1.0):
        self.device = device
        self.num_iter = num_iter
        self.lr = lr

    def optimize_from_svg(self, svg_text: str, target_img: torch.Tensor) -> str:
        import diffvg
        shapes, shape_groups = svg_to_scene(svg_text)
        if not shapes:
            return svg_text

        params = []
        for shape in shapes:
            shape.points.requires_grad_(True)
            params.append(shape.points)

        optimizer = torch.optim.Adam(params, lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        target = target_img.to(self.device)

        for i in range(self.num_iter):
            render_fn = diffvg.RenderFunction.apply
            img = render_fn(
                target.shape[3], target.shape[2], 2, 2, 0, None,
                tuple(shapes), tuple(shape_groups),
            )
            img_t = img.permute(2, 0, 1).unsqueeze(0)
            loss = loss_fn(img_t, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self._shapes_to_svg(shapes, shape_groups)

    def _shapes_to_svg(self, shapes, shape_groups) -> str:
        lines = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">']
        for i, shape in enumerate(shapes):
            pts = shape.points.detach().cpu().tolist()
            d_parts = [f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"]
            idx = 1
            num_ctrl = shape.num_control_points.tolist()
            for ncp in num_ctrl:
                if ncp == 0:
                    if idx < len(pts):
                        d_parts.append(f"L {pts[idx][0]:.1f} {pts[idx][1]:.1f}")
                        idx += 1
                elif ncp == 3:
                    if idx + 2 < len(pts):
                        d_parts.append(f"C {pts[idx][0]:.1f} {pts[idx][1]:.1f} "
                                       f"{pts[idx+1][0]:.1f} {pts[idx+1][1]:.1f} "
                                       f"{pts[idx+2][0]:.1f} {pts[idx+2][1]:.1f}")
                        idx += 3
                elif ncp == 2:
                    if idx + 1 < len(pts):
                        d_parts.append(f"Q {pts[idx][0]:.1f} {pts[idx][1]:.1f} "
                                       f"{pts[idx+1][0]:.1f} {pts[idx+1][1]:.1f}")
                        idx += 2
            if shape.is_closed:
                d_parts.append("Z")
            color = (0.2, 0.2, 0.2)
            lines.append(f'  <path d="{" ".join(d_parts)}" fill="none" stroke="rgb{color}" stroke-width="1"/>')
        lines.append("</svg>")
        return "\n".join(lines)
