# See chat for full content; this placeholder indicates script location.#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune DETR (Hugging Face) on COCO-formatted HTP/DAP data.

Prereq:
  Run split_and_convert_yolo_to_coco.py to create:
    <root>_coco/train/images + annotations.json
    <root>_coco/val/images + annotations.json
"""

import os, argparse, time, json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection, get_cosine_schedule_with_warmup

class CocoDetDataset(Dataset):
    def __init__(self, split_dir, image_processor):
        self.split_dir = Path(split_dir)
        self.images_dir = self.split_dir/"images"
        self.ann_path = self.split_dir/"annotations.json"
        self.processor = image_processor
        with open(self.ann_path, "r") as f: coco = json.load(f)
        self.images = coco["images"]; self.annotations = coco["annotations"]; self.categories = coco["categories"]
        self.anns_by_img = {}
        for ann in self.annotations:
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        info = self.images[idx]
        img_path = self.images_dir / info["file_name"]
        image = Image.open(img_path).convert("RGB")
        anns = self.anns_by_img.get(info["id"], [])
        boxes = [ann["bbox"] for ann in anns]
        labels = [ann["category_id"] for ann in anns]
        inputs = self.processor(images=image, annotations={"bbox": boxes, "category_id": labels}, return_tensors="pt")
        return {k: v.squeeze(0) for k,v in inputs.items()}

def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [{"class_labels": b["labels"], "boxes": b["boxes"]} for b in batch]
    return {"pixel_values": pixel_values, "labels": labels}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", type=str, required=True)
    ap.add_argument("--model_id", type=str, default="facebook/detr-resnet-50")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = Path(args.coco_root)/"train"; val_dir = Path(args.coco_root)/"val"
    assert train_dir.exists() and val_dir.exists(), "Run the YOLO->COCO converter first."

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForObjectDetection.from_pretrained(args.model_id, ignore_mismatched_sizes=True).to(device)

    train_ds = CocoDetDataset(train_dir, processor)
    val_ds   = CocoDetDataset(val_dir,   processor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val = 1e9
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.coco_root)/"models"/f"detr_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train(); tr_loss=0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        model.eval(); val_loss=0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(out_dir)

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | best_val_loss {best_val:.4f}")

    print("Training complete. Best model saved to:", out_dir)

if __name__ == "__main__":
    main()
