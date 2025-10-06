#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhance a YOLO dataset tree (images/* with sibling labels/*) WITHOUT changing geometry.
- Keeps the same folder structure (train/val/test OR flat)
- Copies labels unchanged
- Re-encodes images (RGB, optional resize), autocontrast/denoise/sharpen/gamma

Example:
  python enhance_yolo_dataset.py \
    --in data/raw/HTP2_multiclass \
    --out data/raw/HTP2_multiclass_enh \
    --autocontrast --median 3 --sharpen 0.8 --gamma 1.0 --resize 640
"""
import argparse, shutil
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

def enhance(im, autocontrast=False, equalize=False, median=0, sharpen=0.0, gamma=1.0, resize=None):
    im = im.convert("RGB")
    if autocontrast: im = ImageOps.autocontrast(im)
    if equalize:     im = ImageOps.equalize(im)
    if median and median > 0: im = im.filter(ImageFilter.MedianFilter(size=max(3, int(median))))
    if sharpen and sharpen > 0:
        percent = int(150 * float(sharpen))
        im = im.filter(ImageFilter.UnsharpMask(radius=1, percent=percent, threshold=3))
    if gamma and abs(gamma-1.0) > 1e-3:
        inv = 1.0 / float(gamma)
        lut = [int((i/255.0) ** inv * 255.0 + 0.5) for i in range(256)]
        im = im.point(lut * 3)
    if resize: im = im.resize((int(resize), int(resize)), Image.BICUBIC)
    return im

def copy_labels(src_root: Path, dst_root: Path):
    for lab_dir in [p for p in src_root.rglob("labels") if p.is_dir()]:
        rel = lab_dir.relative_to(src_root)
        out_lab = dst_root / rel
        out_lab.mkdir(parents=True, exist_ok=True)
        for txt in lab_dir.glob("*.txt"):
            shutil.copy2(txt, out_lab / txt.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--autocontrast", action="store_true")
    ap.add_argument("--equalize", action="store_true")
    ap.add_argument("--median", type=int, default=0)
    ap.add_argument("--sharpen", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--resize", type=int, default=None)
    args = ap.parse_args()

    src = Path(args.inp); dst = Path(args.out)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Copy any non-image files (e.g., data.yaml) first
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() not in IMG_EXTS and not p.name.endswith(".txt"):
            rel = p.relative_to(src)
            (dst/rel).parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(p, dst/rel)
            except Exception:
                pass

    # Copy labels verbatim
    copy_labels(src, dst)

    # Enhance all images
    count = 0
    for img in src.rglob("*"):
        if img.is_file() and img.suffix.lower() in IMG_EXTS:
            rel = img.relative_to(src)
            out_img = (dst / rel).with_suffix(".jpg")
            out_img.parent.mkdir(parents=True, exist_ok=True)
            try:
                im = Image.open(img)
                im = enhance(im, args.autocontrast, args.equalize, args.median, args.sharpen, args.gamma, args.resize)
                im.save(out_img, format="JPEG", quality=95, optimize=True)
                count += 1
            except Exception as e:
                print(f"[WARN] {img}: {e}")

    print(f"Enhanced {count} images & copied labels -> {dst}")

if __name__ == "__main__":
    main()
