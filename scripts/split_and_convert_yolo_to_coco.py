# See chat for full content; this placeholder indicates script location.#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split YOLO dataset into train/val and convert to COCO JSON for DETR training.

Input:
  --root data/processed/htp_dap_combined

Output:
  data/processed/htp_dap_combined_coco/{train,val}/images + annotations.json
"""

import os, json, random, shutil, argparse
from pathlib import Path
from PIL import Image
import yaml
from tqdm import tqdm

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

def load_names_from_yaml(yaml_path: Path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return [str(n) for n in data.get("names", [])]

def parse_yolo_label(txt_path: Path):
    objs = []
    if not txt_path.exists():
        return objs
    for line in txt_path.read_text().splitlines():
        if not line.strip(): continue
        parts = line.split()
        cid = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        objs.append((cid, x, y, w, h))
    return objs

def yolo_to_xywh(yolo_box, W, H):
    x_c, y_c, w, h = yolo_box
    x = (x_c - w/2) * W
    y = (y_c - h/2) * H
    return [x, y, w * W, h * H]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.root)
    img_dir = root/"images"; lab_dir = root/"labels"; data_yaml = root/"data.yaml"
    assert img_dir.exists() and lab_dir.exists() and data_yaml.exists(), "Missing images/labels/data.yaml"

    class_names = load_names_from_yaml(data_yaml)
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    random.shuffle(images)
    n_val = max(1, int(args.val_ratio * len(images)))
    val_set = set([p.name for p in images[:n_val]])
    train_set = set([p.name for p in images[n_val:]])

    out_root = root.with_name(root.name + "_coco")
    for split in ["train","val"]:
        (out_root/split/"images").mkdir(parents=True, exist_ok=True)

    def build_coco(split_names):
        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)],
        }
        ann_id = 1
        for fname in tqdm(split_names, desc=f"COCO {len(split_names)} imgs"):
            src_img = img_dir/fname
            dst_img = out_root/("train" if fname in train_set else "val")/"images"/fname
            shutil.copy2(src_img, dst_img)
            with Image.open(src_img) as im:
                W, H = im.size
            img_id = len(coco["images"]) + 1
            coco["images"].append({"id": img_id, "file_name": fname, "width": W, "height": H})

            txt = lab_dir/(Path(fname).stem + ".txt")
            objs = parse_yolo_label(txt)
            for (cid, x, y, w, h) in objs:
                bbox = yolo_to_xywh((x, y, w, h), W, H)
                area = bbox[2] * bbox[3]
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": int(cid),
                    "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                    "area": float(area), "iscrowd": 0,
                })
                ann_id += 1
        return coco

    train_names = [n for n in train_set]
    val_names = [n for n in val_set]
    with open(out_root/"train"/"annotations.json", "w") as f:
        json.dump(build_coco(train_names), f)
    with open(out_root/"val"/"annotations.json", "w") as f:
        json.dump(build_coco(val_names), f)

    print("Done. COCO folders at:", out_root)

if __name__ == "__main__":
    main()
