from __future__ import annotations

from collections import deque
import heapq
from typing import Optional, Union

import numpy as np

try:
    import torch
except Exception:  # Allows NumPy-only topology validation without neural deps.
    torch = None


def connected_components(mask: np.ndarray, value: bool = True) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or bool(mask[y, x]) != value:
                continue
            q = deque([(y, x)])
            seen[y, x] = True
            comp: list[tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and bool(mask[ny, nx]) == value:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            comps.append(comp)
    return comps


def betti_numbers(mask: np.ndarray) -> tuple[int, int]:
    fg = connected_components(mask, True)
    bg = connected_components(mask, False)
    h, w = mask.shape
    holes = 0
    for comp in bg:
        if not any(y == 0 or x == 0 or y == h - 1 or x == w - 1 for y, x in comp):
            holes += 1
    return len(fg), holes


def fill_excess_holes(mask: np.ndarray, target_beta1: int) -> np.ndarray:
    out = mask.copy()
    bg = connected_components(out, False)
    h, w = out.shape
    holes = []
    for comp in bg:
        if not any(y == 0 or x == 0 or y == h - 1 or x == w - 1 for y, x in comp):
            holes.append(comp)
    holes = sorted(holes, key=len, reverse=True)
    for hole in holes[target_beta1:]:
        for y, x in hole:
            out[y, x] = True
    return out


def keep_largest_components(mask: np.ndarray, target_beta0: int) -> np.ndarray:
    comps = sorted(connected_components(mask, True), key=len, reverse=True)
    out = np.zeros_like(mask, dtype=bool)
    for comp in comps[:max(1, target_beta0)]:
        for y, x in comp:
            out[y, x] = True
    return out


def project_to_betti_anchor(mask: np.ndarray, target_beta0: int, target_beta1: int) -> np.ndarray:
    """Hard topology anchor used during guided reverse diffusion.

    The operation preserves the largest foreground structures and removes excess
    holes so generated masks follow the dataset-level Betti anchor. It is applied
    intermittently, then re-encoded into latent space.
    """
    out = keep_largest_components(mask, target_beta0)
    out = fill_excess_holes(out, target_beta1)
    return out


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    yd0, yd1 = max(0, dy), min(h, h + dy)
    xd0, xd1 = max(0, dx), min(w, w + dx)
    out[yd0:yd1, xd0:xd1] = mask[ys0:ys1, xs0:xs1]
    return out


def dilate_np(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                out |= _shift(mask, dy, dx)
    return out


def erode_np(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                out &= _shift(mask, dy, dx)
    return out


def project_to_topology_area_anchor(
    mask: np.ndarray,
    target_beta0: int,
    target_beta1: int,
    target_area: float,
    tolerance: float = 0.025,
    max_steps: int = 8,
) -> np.ndarray:
    """Topology projection that also preserves the real lesion area scale."""
    out = project_to_betti_anchor(mask, target_beta0, target_beta1)
    target_area = float(np.clip(target_area, 0.001, 0.85))
    for _ in range(max_steps):
        area = float(out.mean())
        if abs(area - target_area) <= tolerance:
            break
        if area < target_area:
            grown = dilate_np(out, 1)
            out = project_to_betti_anchor(grown, target_beta0, target_beta1)
        else:
            shrunk = erode_np(out, 1)
            if shrunk.any():
                out = project_to_betti_anchor(shrunk, target_beta0, target_beta1)
            else:
                break
    return out


def area_adaptive_threshold(prob: np.ndarray, target_area: float, min_area: float = 0.015) -> np.ndarray:
    """Threshold a probability map so foreground area follows a target area.

    Fixed 0.5 thresholding caused CCLD collapse when probabilities were calibrated
    low. This keeps the generated anatomy in the real area regime before topology
    projection is applied.
    """
    target_area = float(np.clip(target_area, min_area, 0.75))
    flat = prob.reshape(-1)
    k = int(round(target_area * flat.size))
    k = max(1, min(k, flat.size))
    if k == flat.size:
        return np.ones_like(prob, dtype=bool)
    idx = np.argpartition(flat, -k)[-k:]
    out = np.zeros(flat.size, dtype=bool)
    out[idx] = True
    return out.reshape(prob.shape)


def border_distance_score(shape: tuple[int, int], margin: int) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    dist = np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx]).astype(float)
    return np.clip(dist / max(float(margin), 1.0), 0.0, 1.0)


def connected_probability_region(
    prob: np.ndarray,
    target_area: float,
    border_margin: int = 8,
    min_area: float = 0.015,
) -> np.ndarray:
    """Grow one connected component from an interior high-probability seed.

    This avoids the previous top-k projection failure where disconnected high
    probabilities near image borders became rectangular border-attached masks.
    """
    h, w = prob.shape
    target_area = float(np.clip(target_area, min_area, 0.65))
    target_pixels = max(1, min(int(round(target_area * h * w)), h * w))
    interior = border_distance_score(prob.shape, border_margin)
    seed_score = prob * (0.25 + 0.75 * interior)
    sy, sx = np.unravel_index(int(np.argmax(seed_score)), prob.shape)

    selected = np.zeros_like(prob, dtype=bool)
    seen = np.zeros_like(prob, dtype=bool)
    heap: list[tuple[float, int, int]] = []

    def push(y: int, x: int) -> None:
        if 0 <= y < h and 0 <= x < w and not seen[y, x]:
            seen[y, x] = True
            edge_penalty = 0.35 * (1.0 - interior[y, x])
            score = float(prob[y, x] - edge_penalty)
            heapq.heappush(heap, (-score, y, x))

    push(int(sy), int(sx))
    while heap and int(selected.sum()) < target_pixels:
        _, y, x = heapq.heappop(heap)
        selected[y, x] = True
        push(y - 1, x)
        push(y + 1, x)
        push(y, x - 1)
        push(y, x + 1)

    return selected


def project_probability_to_topology_area(
    prob: np.ndarray,
    target_beta0: int,
    target_beta1: int,
    target_area: float,
    tolerance: float = 0.025,
    border_margin: int = 8,
) -> np.ndarray:
    if target_beta0 <= 1:
        seeded = connected_probability_region(prob, target_area, border_margin=border_margin)
    else:
        seeded = area_adaptive_threshold(prob, target_area)
    return project_to_topology_area_anchor(
        seeded,
        target_beta0=target_beta0,
        target_beta1=target_beta1,
        target_area=target_area,
        tolerance=tolerance,
    )


def forman_ricci_profile(mask: np.ndarray) -> np.ndarray:
    def degree(y: int, x: int) -> int:
        return sum(
            0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx]
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
        )

    values = []
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            for ny, nx in ((y + 1, x), (y, x + 1)):
                if ny < h and nx < w and mask[ny, nx]:
                    values.append(4 - degree(y, x) - degree(ny, nx))
    hist, _ = np.histogram(values if values else [0], bins=[-4, -3, -2, -1, 0, 1, 2, 3])
    return hist.astype(np.float64) / max(float(hist.sum()), 1.0)


NumberOrNone = Union[float, int, None]


def persistence_diagram(mask: np.ndarray, dim: int) -> Optional[np.ndarray]:
    try:
        import gudhi as gd
    except Exception:
        return None

    filt = 1.0 - mask.astype(float)
    complex_ = gd.CubicalComplex(top_dimensional_cells=filt)
    complex_.persistence()
    return complex_.persistence_intervals_in_dimension(dim)


def finite_diagram(diagram: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if diagram is None:
        return None
    if len(diagram) == 0:
        return np.zeros((0, 2), dtype=float)
    return diagram[np.isfinite(diagram[:, 1])].astype(float)


def diagram_distances(mask: np.ndarray, ref_pd0: Optional[np.ndarray], ref_pd1: Optional[np.ndarray]) -> dict[str, NumberOrNone]:
    try:
        import gudhi as gd
        from gudhi.wasserstein import wasserstein_distance
    except Exception:
        return {
            "ph_available": 0,
            "pd0_wasserstein": None,
            "pd1_wasserstein": None,
            "pd0_bottleneck": None,
            "pd1_bottleneck": None,
        }

    pd0 = finite_diagram(persistence_diagram(mask, 0))
    pd1 = finite_diagram(persistence_diagram(mask, 1))
    ref0 = finite_diagram(ref_pd0)
    ref1 = finite_diagram(ref_pd1)
    b0, b1 = betti_numbers(mask)

    def safe_wasserstein(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
        if a is None or b is None:
            return None
        if b0 == 0 and len(b) == 0:
            return 1.0
        return float(wasserstein_distance(a, b, order=1.0, internal_p=2.0))

    def safe_bottleneck(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
        if a is None or b is None:
            return None
        if b0 == 0 and len(b) == 0:
            return 1.0
        return float(gd.bottleneck_distance(a, b))

    return {
        "ph_available": 1,
        "pd0_wasserstein": safe_wasserstein(pd0, ref0),
        "pd1_wasserstein": safe_wasserstein(pd1, ref1),
        "pd0_bottleneck": safe_bottleneck(pd0, ref0),
        "pd1_bottleneck": safe_bottleneck(pd1, ref1),
    }


def try_persistence_summary(mask: np.ndarray) -> dict[str, NumberOrNone]:
    pd0 = finite_diagram(persistence_diagram(mask, 0))
    pd1 = finite_diagram(persistence_diagram(mask, 1))
    if pd0 is None or pd1 is None:
        return {"ph_available": 0, "h0_count": None, "h1_count": None, "total_persistence": None}

    total = 0.0
    for intervals in (pd0, pd1):
        for birth, death in intervals:
            total += float(death - birth)
    return {
        "ph_available": 1,
        "h0_count": int(len(pd0)),
        "h1_count": int(len(pd1)),
        "total_persistence": total
    }


def shape_moments(mask_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, _, h, w = mask_probs.shape
    yy = torch.linspace(0, 1, h, device=mask_probs.device)[None, None, :, None]
    xx = torch.linspace(0, 1, w, device=mask_probs.device)[None, None, None, :]
    mass = mask_probs.sum(dim=(1, 2, 3)).clamp_min(1e-5)
    cx = (mask_probs * xx).sum(dim=(1, 2, 3)) / mass
    cy = (mask_probs * yy).sum(dim=(1, 2, 3)) / mass
    varx = (mask_probs * (xx - cx[:, None, None, None]) ** 2).sum(dim=(1, 2, 3)) / mass
    vary = (mask_probs * (yy - cy[:, None, None, None]) ** 2).sum(dim=(1, 2, 3)) / mass
    return cx, cy, varx + vary


def soft_shape_preservation_loss(mask_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(mask_logits)
    cx, cy, spread = shape_moments(probs)
    tx, ty, tspread = shape_moments(target_mask)
    return torch.mean(torch.abs(cx - tx) + torch.abs(cy - ty) + torch.abs(spread - tspread))


def soft_curvature_loss(mask_logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(mask_logits)
    dx = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1]).mean()
    dy = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :]).mean()
    right_support = (probs[:, :, :, 1:] * probs[:, :, :, :-1]).mean()
    down_support = (probs[:, :, 1:, :] * probs[:, :, :-1, :]).mean()
    area = probs.mean().clamp_min(1e-4)
    return (dx + dy) / torch.sqrt(area) + 0.05 * torch.relu(0.03 - area)


def soft_topology_anchor_loss(mask_logits: torch.Tensor, area_target: torch.Tensor) -> torch.Tensor:
    area = torch.sigmoid(mask_logits).mean(dim=(1, 2, 3))
    return torch.mean(torch.abs(area - area_target))
