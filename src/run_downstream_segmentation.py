from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from dataset import KvasirZipDataset


PROJECT = Path(__file__).resolve().parents[1]
METHODS = ["Baseline Latent DDPM", "Curvature Only", "Topology Only", "Full CCLD"]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "results" / ".matplotlib_cache"))


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT / p


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = self.block(3, 24)
        self.e2 = self.block(24, 48)
        self.e3 = self.block(48, 96)
        self.up2 = nn.ConvTranspose2d(96, 48, 2, 2)
        self.d2 = self.block(96, 48)
        self.up1 = nn.ConvTranspose2d(48, 24, 2, 2)
        self.d1 = self.block(48, 24)
        self.out = nn.Conv2d(24, 1, 1)

    @staticmethod
    def block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, 1, 1),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        d2 = self.d2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class GeneratedPairDataset(Dataset):
    def __init__(self, images: torch.Tensor, masks: torch.Tensor, max_items: int | None = None):
        if max_items is not None:
            images = images[:max_items]
            masks = masks[:max_items]
        self.images = images.float().clamp(-1, 1)
        self.masks = masks.float()
        if self.masks.ndim == 3:
            self.masks = self.masks[:, None]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        return {"image": self.images[idx], "mask": self.masks[idx]}


def split_dataset(dataset, train_fraction: float, seed: int):
    train_size = int(len(dataset) * train_fraction)
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))


def image_mask_collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
    }


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()


def train_model(train_loader: DataLoader, cfg: dict, device: torch.device) -> SmallUNet:
    model = SmallUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("seg_lr", 2e-4)))
    epochs = int(cfg.get("seg_epochs", 12))
    model.train()
    for epoch in range(epochs):
        losses = []
        pbar = tqdm(train_loader, desc=f"Segmentation train {epoch + 1}/{epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, masks) + dice_loss(logits, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")
    return model


def boundary_map(mask: torch.Tensor) -> torch.Tensor:
    pooled = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    return (pooled - eroded).clamp(0, 1)


@torch.no_grad()
def evaluate_model(model: SmallUNet, loader: DataLoader, device: torch.device, return_samples: bool = False) -> dict:
    model.eval()
    totals = {"dice": [], "iou": [], "precision": [], "recall": [], "boundary_f1": []}
    for batch in tqdm(loader, desc="Segmentation eval"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        pred = (torch.sigmoid(model(images)) > 0.5).float()
        inter = (pred * masks).sum(dim=(1, 2, 3))
        pred_sum = pred.sum(dim=(1, 2, 3))
        mask_sum = masks.sum(dim=(1, 2, 3))
        union = pred_sum + mask_sum - inter
        dice = (2 * inter + 1) / (pred_sum + mask_sum + 1)
        iou = (inter + 1) / (union + 1)
        precision = (inter + 1) / (pred_sum + 1)
        recall = (inter + 1) / (mask_sum + 1)
        pb = boundary_map(pred)
        tb = boundary_map(masks)
        b_inter = (pb * tb).sum(dim=(1, 2, 3))
        b_precision = (b_inter + 1) / (pb.sum(dim=(1, 2, 3)) + 1)
        b_recall = (b_inter + 1) / (tb.sum(dim=(1, 2, 3)) + 1)
        boundary_f1 = 2 * b_precision * b_recall / (b_precision + b_recall + 1e-6)
        for key, vals in [
            ("dice", dice),
            ("iou", iou),
            ("precision", precision),
            ("recall", recall),
            ("boundary_f1", boundary_f1),
        ]:
            totals[key].extend(float(v) for v in vals.cpu())
    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in totals.items()}
    if return_samples:
        summary["_per_sample"] = {k: [float(x) for x in v] for k, v in totals.items()}
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_results(results: dict) -> list[dict]:
    rows = []
    for condition, metrics in results.items():
        row = {"condition": condition}
        for metric, stats in metrics.items():
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        rows.append(row)
    return rows


def require_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def tensor_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    arr = ((image.detach().cpu().clamp(-1, 1) + 1.0) * 0.5).permute(1, 2, 0).numpy()
    return np.clip(arr, 0.0, 1.0)


@torch.no_grad()
def predict_examples(model: SmallUNet, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    preds = torch.sigmoid(model(images.to(device))).cpu()
    return (preds > 0.5).float()


def save_prediction_panel(
    out_path: Path,
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    max_items: int = 4,
) -> None:
    plt = require_matplotlib()
    n = min(max_items, images.shape[0])
    rows = [("Real Image", None), ("Ground Truth", masks[:n])]
    label_map = {
        "Real Only": "Pred: Real Only",
        "Real + Baseline Latent DDPM": "Pred: + Baseline",
        "Real + Curvature Only": "Pred: + Curvature",
        "Real + Topology Only": "Pred: + Topology",
        "Real + Full CCLD": "Pred: + Full CCLD",
    }
    for condition, pred in predictions.items():
        rows.append((label_map.get(condition, condition), pred[:n]))

    fig, axes = plt.subplots(len(rows), n, figsize=(3.1 * n, 2.6 * len(rows)))
    if n == 1:
        axes = axes[:, None]
    for row_idx, (label, data) in enumerate(rows):
        for col in range(n):
            ax = axes[row_idx, col]
            if row_idx == 0:
                ax.imshow(tensor_image_to_numpy(images[col]))
            else:
                ax.imshow(data[col, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.text(
                    -0.08,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0},
                )
    fig.subplots_adjust(left=0.18, right=0.99, top=0.99, bottom=0.01, wspace=0.08, hspace=0.08)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chapter_pipeline_v2.json")
    parser.add_argument("--generated", default=None)
    args = parser.parse_args()
    cfg = json.loads(resolve_path(args.config).read_text())
    seed_all(int(cfg.get("seed", 42)))

    out_dir = resolve_path(cfg["out_dir"])
    generated_path = resolve_path(args.generated) if args.generated else out_dir / "generated_samples.pt"
    if not generated_path.exists():
        raise FileNotFoundError(f"Missing generated sample tensor: {generated_path}. Run train_chapter_pipeline_v2.py first.")

    dataset = KvasirZipDataset(resolve_path(cfg["zip_path"]), int(cfg["image_size"]))
    train_ds, val_ds = split_dataset(dataset, float(cfg["train_fraction"]), int(cfg["seed"]))
    max_real = int(cfg.get("seg_real_train_limit", min(len(train_ds), 220)))
    train_real = Subset(train_ds, list(range(max_real)))
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("seg_batch_size", cfg["batch_size"])),
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=image_mask_collate,
    )

    payload = torch.load(generated_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    example_batch = next(iter(val_loader))
    example_images = example_batch["image"][: int(cfg.get("seg_visual_count", 4))]
    example_masks = example_batch["mask"][: int(cfg.get("seg_visual_count", 4))]
    prediction_examples = {}
    conditions = {"Real Only": train_real}
    aug_limit = int(cfg.get("seg_aug_limit", 256))
    for method in METHODS:
        if method in payload:
            gen_ds = GeneratedPairDataset(payload[method]["images"], payload[method]["masks"], max_items=aug_limit)
            conditions[f"Real + {method}"] = ConcatDataset([train_real, gen_ds])

    for condition, train_data in conditions.items():
        loader = DataLoader(
            train_data,
            batch_size=int(cfg.get("seg_batch_size", cfg["batch_size"])),
            shuffle=True,
            num_workers=cfg["num_workers"],
            collate_fn=image_mask_collate,
        )
        model = train_model(loader, cfg, device)
        results[condition] = evaluate_model(model, val_loader, device, return_samples=True)
        prediction_examples[condition] = predict_examples(model, example_images, device)

    out_path = out_dir / "downstream_segmentation.json"
    out_path.write_text(json.dumps(results, indent=2))
    summary_results = {
        condition: {metric: stats for metric, stats in metrics.items() if metric != "_per_sample"}
        for condition, metrics in results.items()
    }
    write_csv(out_dir / "downstream_segmentation.csv", flatten_results(summary_results))
    per_sample_rows = []
    for condition, metrics in results.items():
        per_sample = metrics.get("_per_sample", {})
        if not per_sample:
            continue
        n = len(next(iter(per_sample.values())))
        for i in range(n):
            row = {"condition": condition, "sample_id": i}
            for metric, values in per_sample.items():
                row[metric] = values[i]
            per_sample_rows.append(row)
    if per_sample_rows:
        write_csv(out_dir / "downstream_segmentation_per_sample.csv", per_sample_rows)
    save_prediction_panel(
        out_dir / "downstream_segmentation_predictions.png",
        example_images,
        example_masks,
        prediction_examples,
        max_items=int(cfg.get("seg_visual_count", 4)),
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
