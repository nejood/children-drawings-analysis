#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine & normalize the two emotions datasets.

Inputs (default; auto-detected if present):
  data/raw/emotions/
  data/raw/Emotion_Recognition_multiclass/

Outputs:
  data/processed/emotions_combined/
    ├─ images/               (all copied images)
    ├─ labels.csv            (filename,source,raw_label,label)
    └─ summary.json          (counts and stats)
"""

import os
import re
import csv
import json
import shutil
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(".")
RAW_DIRS_DEFAULT = [
    PROJECT_ROOT / "data" / "raw" / "emotions",
    PROJECT_ROOT / "data" / "raw" / "Emotion_Recognition_multiclass",
]
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "emotions_combined"
IMAGES_OUT = OUT_DIR / "images"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def normalize_emotion_label(lbl_raw: str) -> str:
    s = re.sub(r"[^a-zA-Z]", "", str(lbl_raw)).lower()
    if s in ["happy","happiness","joy","joyful","delight"]: return "happy"
    if s in ["sad","sadness","blue","depress"]: return "sad"
    if s in ["angry","anger","mad","furious","rage"]: return "angry"
    if s in ["neutral","natural","normal","calm"]: return "natural"
    if "positive" in s: return "happy"
    if "negative" in s: return "angry" if "angr" in s else "sad"
    if "fear" in s: return "sad"
    return "natural"

def harvest_images_from_source(src_root: Path, records: list, copy_to: Path):
    num_images = 0
    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            raw_label = p.parent.name
            norm_label = normalize_emotion_label(raw_label)
            rel_name = f"{src_root.name}_{p.stem}{p.suffix.lower()}"
            dst = copy_to / rel_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(p, dst)
                records.append({"filename": rel_name, "source": src_root.name,
                                "raw_label": raw_label, "label": norm_label})
                num_images += 1
            except Exception as e:
                print(f"[WARN] copy failed: {p} -> {dst} ({e})")
    return num_images

def autodetect_sources(raw_root: Path) -> list:
    cands = []
    if raw_root.exists():
        for d in raw_root.iterdir():
            if d.is_dir() and any(k in d.name.lower() for k in ["emotion", "emotions", "emotion_recognition"]):
                cands.append(d)
    return cands

def main(raw_dirs=None):
    if raw_dirs is None:
        chosen = [d for d in RAW_DIRS_DEFAULT if d.exists()]
        if not chosen:
            chosen = autodetect_sources(PROJECT_ROOT / "data" / "raw")
        raw_dirs = chosen

    if not raw_dirs:
        raise SystemExit("No emotion dataset folders found under data/raw/. Place them there and rerun.")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    IMAGES_OUT.mkdir(parents=True, exist_ok=True)

    records = []
    for src in raw_dirs:
        print(f"[INFO] Scanning: {src}")
        harvest_images_from_source(src, records, IMAGES_OUT)

    if not records:
        raise SystemExit("No images found in emotion datasets.")

    labels_csv = OUT_DIR / "labels.csv"
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename","source","raw_label","label"])
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    by_label = Counter([r["label"] for r in records])
    by_source = Counter([r["source"] for r in records])
    mapping_examples = defaultdict(set)
    for r in records:
        if len(mapping_examples[r["raw_label"]]) < 5:
            mapping_examples[r["raw_label"]].add(r["label"])

    summary = {
        "images_total": len(records),
        "labels_by_class": dict(by_label),
        "counts_by_source": dict(by_source),
        "mapping_examples": {k: sorted(list(v)) for k,v in mapping_examples.items()},
        "note": "Mapping: positive→happy; negative→sad (default) or angry (if 'angr'); keep natural; fear→sad."
    }
    with open(OUT_DIR/"summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== EMOTIONS COMBINE COMPLETE ===")
    print("Images copied:", len(records))
    print("labels.csv   :", labels_csv)
    print("summary.json :", OUT_DIR / "summary.json")

if __name__ == "__main__":
    import sys
    user_dirs = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else None
    main(user_dirs)
