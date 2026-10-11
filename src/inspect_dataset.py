from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", required=True)
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

    images = sorted(n for n in names if n.startswith("segmented-images/images/") and n.lower().endswith((".jpg", ".jpeg", ".png")))
    masks = sorted(n for n in names if n.startswith("segmented-images/masks/") and n.lower().endswith((".jpg", ".jpeg", ".png")))
    image_stems = {Path(n).stem for n in images}
    mask_stems = {Path(n).stem for n in masks}

    print(json.dumps({
        "zip_path": str(zip_path),
        "image_count": len(images),
        "mask_count": len(masks),
        "paired_count": len(image_stems & mask_stems),
        "unpaired_images": len(image_stems - mask_stems),
        "unpaired_masks": len(mask_stems - image_stems),
        "example_image": images[0] if images else None,
        "example_mask": masks[0] if masks else None
    }, indent=2))


if __name__ == "__main__":
    main()
