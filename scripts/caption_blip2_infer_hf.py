# See chat for full content; this placeholder indicates script location.#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate captions for images with BLIP-2 (HF).

Example:
python caption_blip2_infer_hf.py \
  --images_dir data/processed/htp_dap_combined/images \
  --out_csv data/processed/captions.csv \
  --model_id Salesforce/blip2-opt-2.7b
"""

import os, argparse, csv
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--model_id", type=str, default="Salesforce/blip2-opt-2.7b")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--prompt", type=str, default="Describe the drawing briefly.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = [p for p in Path(args.images_dir).iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]]
    images.sort()

    processor = AutoProcessor.from_pretrained(args.model_id)
    dtype = torch.float16 if device=="cuda" else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=dtype).to(device)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["filename","caption"])
        for img_path in tqdm(images, desc="Captioning"):
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(device, dtype=dtype)
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            writer.writerow([img_path.name, caption])

    print("Saved captions to:", args.out_csv)

if __name__ == "__main__":
    main()
