#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhance images safely (photometric ops only).

Usage examples:
  # Enhance raw emotions then combine
  python enhance_images.py --in data/raw/emotions --out data/raw/emotions_enh \
    --autocontrast --median 3 --sharpen 1.0 --gamma 0.95 --resize 224

  # Enhance HTP/DAP images after combining (safe for YOLO labels)
  python enhance_images.py --in data/processed/htp_dap_combined/images \
    --out data/processed/htp_dap_enhanced/images \
    --autocontrast --median 3 --sharpen 0.8 --gamma 1.0 --resize 640

Then point your next step to the new images folder (and copy labels if needed).
"""

import argparse, os
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

def enhance(im, autocontrast=False, equalize=False, median=0, sharpen=0.0, gamma=1.0, resize=None):
    # Convert to RGB (defensive)
    im = im.convert("RGB")

    if autocontrast:
        im = ImageOps.autocontrast(im)

    if equalize:  # global hist equalization; skip if drawings get too “harsh”
        im = ImageOps.equalize(im)

    if median and median > 0:
        im = im.filter(ImageFilter.MedianFilter(size=max(3, int(median))))

    if sharpen and sharpen > 0:
        # radius ~1, percent scaled by sharpen (0.0–2.0 typical)
        percent = int(150 * float(sharpen))
        im = im.filter(ImageFilter.UnsharpMask(radius=1, percent=percent, threshold=3))

    if gamma and gamma > 0 and abs(gamma - 1.0) > 1e-3:
        # gamma < 1 brightens midtones; >1 darkens
        inv = 1.0 / float(gamma)
        lut = [int((i/255.0) ** inv * 255.0 + 0.5) for i in range(256)]
        im = im.point(lut * 3)

    if resize:
        im = im.resize((int(resize), int(resize)), Image.BICUBIC)

    return im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input images folder")
    ap.add_argument("--out", dest="out", required=True, help="Output images folder")
    ap.add_argument("--autocontrast", action="store_true")
    ap.add_argument("--equalize", action="store_true")
    ap.add_argument("--median", type=int, default=0, help="Median filter size (0=off, 3/5 recommended)")
    ap.add_argument("--sharpen", type=float, default=0.0, help="Unsharp strength (0.0–2.0)")
    ap.add_argument("--gamma", type=float, default=1.0, help="Gamma correction (e.g., 0.9=brighter)")
    ap.add_argument("--resize", type=int, default=None, help="Square resize (e.g., 224 or 640)")
    args = ap.parse_args()

    src = Path(args.inp); dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            rel = p.relative_to(src)
            out_p = dst / rel
            out_p.parent.mkdir(parents=True, exist_ok=True)
            try:
                im = Image.open(p)
                im = enhance(im,
                    autocontrast=args.autocontrast,
                    equalize=args.equalize,
                    median=args.median,
                    sharpen=args.sharpen,
                    gamma=args.gamma,
                    resize=args.resize
                )
                # Always save as JPEG to standardize (change to PNG if you prefer)
                out_p = out_p.with_suffix(".jpg")
                im.save(out_p, format="JPEG", quality=95, optimize=True)
                count += 1
            except Exception as e:
                print(f"[WARN] {p}: {e}")

    print(f"Enhanced {count} images -> {dst}")

if __name__ == "__main__":
    main()
