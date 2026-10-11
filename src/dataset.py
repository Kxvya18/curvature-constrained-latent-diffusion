from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class KvasirZipDataset(Dataset):
    def __init__(self, zip_path: Path, image_size: int):
        self.zip_path = Path(zip_path)
        self.image_size = image_size
        self._zip = None
        with zipfile.ZipFile(self.zip_path) as zf:
            names = zf.namelist()
        images = [n for n in names if n.startswith("segmented-images/images/") and n.lower().endswith((".jpg", ".jpeg", ".png"))]
        masks = [n for n in names if n.startswith("segmented-images/masks/") and n.lower().endswith((".jpg", ".jpeg", ".png"))]
        image_by_stem = {Path(n).stem: n for n in images}
        mask_by_stem = {Path(n).stem: n for n in masks}
        stems = sorted(set(image_by_stem) & set(mask_by_stem))
        if not stems:
            raise RuntimeError(f"No image/mask pairs found in {self.zip_path}")
        self.pairs = [(image_by_stem[s], mask_by_stem[s]) for s in stems]

    def __len__(self) -> int:
        return len(self.pairs)

    def _read_image(self, member: str, mode: str, resample: int) -> Image.Image:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path)
        data = self._zip.read(member)
        return Image.open(io.BytesIO(data)).convert(mode).resize((self.image_size, self.image_size), resample)

    def __getitem__(self, idx: int):
        image_member, mask_member = self.pairs[idx]
        image = self._read_image(image_member, "RGB", Image.Resampling.BILINEAR)
        mask = self._read_image(mask_member, "L", Image.Resampling.NEAREST)
        image_arr = np.asarray(image).copy().transpose(2, 0, 1)
        mask_arr = (np.asarray(mask).copy() > 127).astype(np.float32)
        image_tensor = torch.from_numpy(image_arr).float() / 127.5 - 1.0
        mask_tensor = torch.from_numpy(mask_arr)[None]
        return {"image": image_tensor, "mask": mask_tensor, "id": Path(image_member).stem}
