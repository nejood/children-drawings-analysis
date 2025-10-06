#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhance ANY nested YOLO-like dataset tree.
- Enhances every image found under --in recursively (RGB, autocontrast/median/sharpen/gamma, optional resize)
- Copies ALL .txt files (labels) with the same relative subpaths
- Copies any data.yaml if present
Safe for YOLO because we do not change geometry semantics (uniform resize is OK for model input later).
"""
import argparse, shutil
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True)
    ap.add_argument("--out", dest="out",  required=True)
    ap.add_argument("--autocontrast", action="store_true")
    ap.add_argument("--equalize", action="store_true")
    ap.add_argument("--median", type=int, default=0)
    ap.add_argument("--sharpen", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--resize", type=int, default=None)
    args = ap.parse_args()

    src = Path(args.inp); dst = Path(args.out)
    if dst.exists(): shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # copy yaml + labels first
    for p in src.rglob("*"):
        if p.is_file() and (p.suffix.lower()==".yaml" or p.suffix.lower()==".yml" or p.suffix.lower()==".txt"):
            rel = p.relative_to(src); (dst/rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst/rel)

    # enhance images
    enhanced = 0
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            rel = p.relative_to(src)
            out_img = (dst/rel).with_suffix(".jpg")
            out_img.parent.mkdir(parents=True, exist_ok=True)
            try:
                im = Image.open(p)
                im = enhance(im, args.autocontrast, args.equalize, args.median, args.sharpen, args.gamma, args.resize)
                im.save(out_img, format="JPEG", quality=95, optimize=True)
                enhanced += 1
            except Exception as e:
                print(f"[WARN] {p}: {e}")

    print(f"Enhanced {enhanced} images & copied labels/yaml -> {dst}")

if __name__ == "__main__":
    main()
