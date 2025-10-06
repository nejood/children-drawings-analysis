#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Cleaner & Normalizer for Children's Drawings Project
------------------------------------------------------------
Supports two modes:
  1) Emotions (classification): --mode emotions --in data/processed/emotions_combined --out data/processed/emotions_clean
  2) HTP/DAP (YOLO detection):  --mode detection --in data/processed/htp_dap_combined --out data/processed/htp_dap_clean

What it does (common):
  - Walk images, drop unreadable/corrupt files, convert to RGB, optionally resize and JPEG/PNG re-encode.
  - Sanitize filenames (ascii-safe, no spaces, lowercase + unique suffix).
  - Optional near-duplicate removal via perceptual hash (requires imagehash).

Emotions-specific:
  - Rebuilds labels.csv using only kept images; preserves columns: filename,source,raw_label,label

Detection-specific (YOLO):
  - Validates/repairs YOLO label files (.txt per image).
  - Normalizes boxes to [0,1], clips to image frame, drops tiny boxes (min area threshold).
  - Drops objects with invalid class IDs; deletes empty label files (or keeps empty based on flag).
  - (Optional) Drops images with no remaining labels.
  - Writes a fresh data.yaml and a summary JSON with counts.

Usage examples:
  Emotions:
    python scripts/clean_normalize.py --mode emotions --in data/processed/emotions_combined --out data/processed/emotions_clean --resize 224
  Detection:
    python scripts/clean_normalize.py --mode detection --in data/processed/htp_dap_combined --out data/processed/htp_dap_clean --min_box_area 0.0005 --drop_unlabeled

Install (if using dup removal):
    pip install imagehash
"""

import argparse, os, re, shutil, json, hashlib, unicodedata
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from collections import Counter, defaultdict

try:
    import imagehash
    HAS_IMAGEHASH = True
except Exception:
    HAS_IMAGEHASH = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YOLO_OK = set("0123456789. -\n\t")

DETECTION_CLASSES = [
    "house","roof","door","window","chimney","fence","path",
    "tree","trunk","branches","leaves","roots","fruit",
    "person","head","eyes","mouth","nose","ears","neck","arms","hands","fingers",
    "legs","feet","body","shoulders","sad","fear","happy","angry","natural"
]
CLASS_TO_ID = {c:i for i,c in enumerate(DETECTION_CLASSES)}

def ascii_safe(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s.lower()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def img_iter(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def load_image(path: Path):
    with Image.open(path) as im:
        im.load()
        return im.convert("RGB")

def save_image(im: Image.Image, path: Path, ext=".jpg", quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    if ext.lower() in [".jpg",".jpeg"]:
        im.save(path.with_suffix(".jpg"), format="JPEG", quality=quality, optimize=True)
        return path.with_suffix(".jpg")
    elif ext.lower() == ".png":
        im.save(path.with_suffix(".png"), format="PNG", optimize=True)
        return path.with_suffix(".png")
    else:
        im.save(path, optimize=True)
        return path

def perceptual_key(im: Image.Image):
    if not HAS_IMAGEHASH:
        # Simple fallback: md5 of resized bytes
        tmp = im.resize((64,64))
        return hashlib.md5(tmp.tobytes()).hexdigest()
    return str(imagehash.phash(im))

# ---------- Emotions Cleaning ----------
def clean_emotions(src: Path, dst: Path, resize: int = None, dedup: bool = True):
    """
    src expects:
      images/ + labels.csv (filename,source,raw_label,label)
    """
    import csv
    src_images = src / "images"
    labels_csv = src / "labels.csv"
    assert src_images.exists() and labels_csv.exists(), "emotions mode requires images/ and labels.csv"

    out_images = dst / "images"
    ensure_dir(out_images)

    # Load labels
    rows = []
    with open(labels_csv, "r", encoding="utf-8") as f:
        import csv
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    by_name = {r["filename"]: r for r in rows}
    kept = []
    dup_keys = set()
    seen_keys = set()

    for p in img_iter(src_images):
        rel = p.name
        meta = by_name.get(rel)
        if meta is None:
            # unknown file; ignore gracefully
            continue
        try:
            im = load_image(p)
        except (UnidentifiedImageError, OSError):
            continue

        if resize:
            im = im.resize((resize, resize))

        key = perceptual_key(im) if dedup else None
        if key and key in seen_keys:
            dup_keys.add(key)
            continue
        if key:
            seen_keys.add(key)

        # sanitize filename & ensure uniqueness
        base = ascii_safe(Path(rel).stem) or "img"
        ext = ".jpg"
        out_name = base + ext
        i = 1
        while (out_images / out_name).exists():
            out_name = f"{base}_{i}{ext}"
            i += 1

        save_image(im, out_images / out_name, ext=".jpg")
        kept.append({
            "filename": out_name,
            "source": meta.get("source",""),
            "raw_label": meta.get("raw_label",""),
            "label": meta.get("label","natural")
        })

    # Write cleaned labels.csv
    ensure_dir(dst)
    with open(dst / "labels.csv", "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=["filename","source","raw_label","label"])
        writer.writeheader()
        for r in kept:
            writer.writerow(r)

    # Summary
    lab_counts = Counter([r["label"] for r in kept])
    summary = {
        "images_in": len(rows),
        "images_kept": len(kept),
        "deduplicated": len(dup_keys),
        "by_label": dict(lab_counts)
    }
    with open(dst / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Emotions cleaned. Kept:", len(kept), "->", dst)

# ---------- YOLO helpers ----------
def read_yolo_file(txt: Path):
    if not txt.exists():
        return []
    # quick sanitation (remove weird chars)
    bad = sum(ch not in YOLO_OK for ch in txt.read_text(errors="ignore"))
    if bad:
        txt.write_text("".join(ch for ch in txt.read_text(errors="ignore") if ch in YOLO_OK))
    rows = []
    for line in txt.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5: 
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            rows.append([cls, x, y, w, h])
        except Exception:
            continue
    return rows

def clip01(v): 
    return max(0.0, min(1.0, v))

def clean_boxes(rows, min_area=0.0):
    out = []
    for cls, x, y, w, h in rows:
        if cls < 0 or cls >= len(DETECTION_CLASSES):
            continue
        # clip to [0,1]
        x = clip01(x); y = clip01(y); w = clip01(w); h = clip01(h)
        # repair zero/neg
        if w <= 0 or h <= 0:
            continue
        if w*h < min_area:
            continue
        out.append([cls, x, y, w, h])
    return out

def write_yolo_file(txt: Path, rows):
    txt.parent.mkdir(parents=True, exist_ok=True)
    with open(txt, "w") as f:
        for cls, x, y, w, h in rows:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# ---------- Detection Cleaning ----------
def clean_detection(src: Path, dst: Path, resize: int = None, min_box_area: float = 0.0,
                    drop_unlabeled: bool = False, dedup: bool = False):
    """
    src expects:
      images/ + labels/ + (data.yaml)
    """
    src_images = src / "images"
    src_labels = src / "labels"
    assert src_images.exists() and src_labels.exists(), "detection mode requires images/ and labels/"

    out_images = dst / "images"
    out_labels = dst / "labels"
    ensure_dir(out_images); ensure_dir(out_labels)

    kept_imgs = 0
    dropped_imgs = 0
    dup_keys = set()
    seen_keys = set()
    class_counts = Counter()

    for p in img_iter(src_images):
        try:
            im = load_image(p)
        except (UnidentifiedImageError, OSError):
            dropped_imgs += 1
            continue

        # Clean label
        txt = src_labels / (p.stem + ".txt")
        rows = read_yolo_file(txt)
        rows = clean_boxes(rows, min_area=min_box_area)

        if drop_unlabeled and not rows:
            # nothing to keep
            dropped_imgs += 1
            continue

        if resize:
            im = im.resize((resize, resize))

        # dedup
        if dedup:
            key = perceptual_key(im)
            if key in seen_keys:
                dropped_imgs += 1
                dup_keys.add(key)
                continue
            seen_keys.add(key)

        # save image with clean filename
        base = ascii_safe(p.stem) or "img"
        out_name = base + ".jpg"
        i = 1
        while (out_images / out_name).exists():
            out_name = f"{base}_{i}.jpg"; i += 1
        save_image(im, out_images / out_name, ext=".jpg")

        # write labels (renamed consistently)
        out_txt = out_labels / (Path(out_name).stem + ".txt")
        write_yolo_file(out_txt, rows)
        for r in rows:
            class_counts[DETECTION_CLASSES[r[0]]] += 1

        kept_imgs += 1

    # write data.yaml
    (dst / "data.yaml").write_text(
        "path: {}\ntrain: images\nval: images\nnc: {}\nnames: {}\n".format(
            dst.as_posix(), len(DETECTION_CLASSES), json.dumps(DETECTION_CLASSES)
        )
    )

    summary = {
        "images_kept": kept_imgs,
        "images_dropped": dropped_imgs,
        "deduplicated": len(dup_keys),
        "class_counts": dict(class_counts),
        "min_box_area": min_box_area,
        "resize": resize
    }
    with open(dst / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Detection cleaned. Kept:", kept_imgs, "->", dst)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["emotions","detection"], required=True)
    ap.add_argument("--in", dest="inp", type=str, required=True, help="Input dataset root")
    ap.add_argument("--out", dest="out", type=str, required=True, help="Output (clean) dataset root")
    ap.add_argument("--resize", type=int, default=None, help="Optional square resize (e.g., 224)")
    ap.add_argument("--min_box_area", type=float, default=0.0, help="Min normalized area for YOLO boxes (e.g., 0.0005)")
    ap.add_argument("--drop_unlabeled", action="store_true", help="Drop detection images that end up with 0 boxes")
    ap.add_argument("--dedup", action="store_true", help="Remove near-duplicate images (uses imagehash if installed)")
    args = ap.parse_args()

    src = Path(args.inp)
    dst = Path(args.out)
    if dst.exists():
        shutil.rmtree(dst)

    if args.mode == "emotions":
        assert (src / "images").exists() and (src / "labels.csv").exists(), "emotions input must have images/ and labels.csv"
        clean_emotions(src, dst, resize=args.resize, dedup=args.dedup)
    else:
        assert (src / "images").exists() and (src / "labels").exists(), "detection input must have images/ and labels/"
        clean_detection(src, dst, resize=args.resize, min_box_area=args.min_box_area,
                        drop_unlabeled=args.drop_unlabeled, dedup=args.dedup)

if __name__ == "__main__":
    main()
