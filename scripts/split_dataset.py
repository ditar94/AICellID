"""
Split a class-folder image dataset into train/val sets.

Input layout (each class is a subfolder of images):
    <src>/
        basophil/*.jpg
        eosinophil/*.jpg
        ...

Output layout:
    <dst>/train/<class>/...
    <dst>/val/<class>/...

Usage:
    python scripts/split_dataset.py --src data/raw/PBC --dst data --val-frac 0.2
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def split_class(src_class: Path, dst_root: Path, val_frac: float, rng: random.Random) -> tuple[int, int]:
    images = sorted(p for p in src_class.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    rng.shuffle(images)
    n_val = max(1, int(len(images) * val_frac)) if images else 0
    val_imgs = images[:n_val]
    train_imgs = images[n_val:]

    for split, imgs in (("train", train_imgs), ("val", val_imgs)):
        out_dir = dst_root / split / src_class.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, out_dir / img.name)

    return len(train_imgs), len(val_imgs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="Source dir containing one subfolder per class")
    ap.add_argument("--dst", type=Path, required=True, help="Destination dir; train/ and val/ will be created inside")
    ap.add_argument("--val-frac", type=float, default=0.2, help="Fraction of images per class to use for validation")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    args = ap.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"src not found or not a directory: {args.src}")

    rng = random.Random(args.seed)
    class_dirs = sorted(p for p in args.src.iterdir() if p.is_dir())
    if not class_dirs:
        raise SystemExit(f"no class subfolders found under {args.src}")

    total_train = total_val = 0
    print(f"splitting {len(class_dirs)} classes (val_frac={args.val_frac}, seed={args.seed})")
    for cls in class_dirs:
        n_tr, n_va = split_class(cls, args.dst, args.val_frac, rng)
        print(f"  {cls.name:20s}  train={n_tr:5d}  val={n_va:5d}")
        total_train += n_tr
        total_val += n_va
    print(f"done: {total_train} train, {total_val} val written under {args.dst}/{{train,val}}")


if __name__ == "__main__":
    main()
