#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train ViT (Hugging Face) for emotion classification on the unified dataset.

Input:
  data/processed/emotions_combined/images + labels.csv

Output:
  data/processed/emotions_combined/models/vit_emotions_YYYYmmdd_HHMMSS/
"""

import os, argparse, time, random
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms
from torch import nn
from transformers import AutoImageProcessor, AutoModelForImageClassification, get_cosine_schedule_with_warmup

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

class EmotionDataset(Dataset):
    def __init__(self, root, df, transform):
        self.root = Path(root)
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.classes = sorted(self.df['label'].unique())
        self.cls_to_id = {c:i for i,c in enumerate(self.classes)}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / "images" / row['filename']).convert("RGB")
        img = self.transform(img)
        y = self.cls_to_id[row['label']]
        return img, y

def set_seed(seed=42):
    random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--model_id", type=str, default="google/vit-base-patch16-224-in21k")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.data_root)
    df = pd.read_csv(root / "labels.csv")
    classes = sorted(df["label"].unique())
    num_labels = len(classes)

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_val = max(1, int(args.val_ratio * len(df)))
    df_val = df.iloc[:n_val].copy()
    df_train = df.iloc[n_val:].copy()

    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    train_ds = EmotionDataset(root, df_train, train_tf)
    val_ds   = EmotionDataset(root, df_val,   val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    y_train_ids = [train_ds.cls_to_id[lbl] for lbl in df_train['label']]
    import numpy as np
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array(list(range(num_labels))), y=np.array(y_train_ids))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageClassification.from_pretrained(args.model_id, num_labels=num_labels, ignore_mismatched_sizes=True)

    if args.lora and PEFT_AVAILABLE:
        peft_cfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05,
                              target_modules=["query","key","value"], bias="none", task_type="SEQ_CLS")
        model = get_peft_model(model, peft_cfg)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = root / "models" / f"vit_emotions_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train(); tr_loss=0.0; tr_correct=0; tr_count=0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(pixel_values=imgs); logits = outputs.logits
            loss = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
            tr_loss += loss.item()*imgs.size(0)
            tr_correct += (logits.argmax(1)==labels).sum().item()
            tr_count += imgs.size(0)
        train_loss = tr_loss/max(1,tr_count); train_acc = tr_correct/max(1,tr_count)

        model.eval(); v_loss=0.0; v_correct=0; v_count=0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(pixel_values=imgs); logits = outputs.logits
                loss = criterion(logits, labels)
                v_loss += loss.item()*imgs.size(0)
                v_correct += (logits.argmax(1)==labels).sum().item()
                v_count += imgs.size(0)
        val_loss = v_loss/max(1,v_count); val_acc = v_correct/max(1,v_count)
        print(f"Epoch {epoch+1}/{args.epochs} | loss {train_loss:.4f} acc {train_acc:.4f} | val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(out_dir)

    with open(out_dir/"classes.txt", "w") as f:
        f.write("\n".join(classes))
    print("Best val acc:", best_val_acc, "| Saved to:", out_dir)

if __name__ == "__main__":
    main()
