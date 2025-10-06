#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified inference + PDF report:
- Emotions classification (ViT)
- Object detection (DETR)
- Captioning (BLIP-2)
- Outputs: JSON per image + PDF per image

Example:
python infer_and_report.py \
  --images_dir data/processed/htp_dap_combined/images \
  --out_dir outputs \
  --emotions_model_dir data/processed/emotions_combined/models/vit_emotions_YYYYmmdd_HHMMSS \
  --detr_model_dir data/processed/htp_dap_combined_coco/models/detr_YYYYmmdd_HHMMSS \
  --blip2_model_id Salesforce/blip2-opt-2.7b
"""

import os, json, argparse
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import AutoProcessor as HF_AutoProcessor
from transformers import Blip2ForConditionalGeneration
from transformers import AutoImageProcessor as HF_ImageProcessor
from transformers import AutoModelForObjectDetection
from report_utils import build_pdf

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

def softmax(x): e = torch.exp(x - x.max()); return e / e.sum()

def run_classification(model_dir, image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(device)
    inputs = image_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs); logits = out.logits.squeeze(0).cpu()
        probs = softmax(logits)
    id2label = model.config.id2label
    top_id = int(torch.argmax(probs).item())
    top_label = id2label.get(top_id, str(top_id))
    top_prob = float(probs[top_id].item())
    return {
        "probs": {id2label.get(i,str(i)): float(probs[i].item()) for i in range(len(probs))},
        "top1_label": top_label, "top1_prob": top_prob,
        "source_model": Path(model_dir).name
    }

def run_detection(model_dir, image, det_conf=0.25):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = HF_ImageProcessor.from_pretrained(model_dir)
    model = AutoModelForObjectDetection.from_pretrained(model_dir).to(device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])  # (h,w)
    results = processor.post_process_object_detection(outputs, threshold=det_conf, target_sizes=target_sizes)[0]
    id2label = model.config.id2label
    dets = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [float(x) for x in box.tolist()]
        dets.append({"label": id2label.get(int(label_id), str(int(label_id))),
                     "score": float(score), "bbox": [round(x,2) for x in box]})
    return dets, Path(model_dir).name

def run_captioning(model_id, image, prompt, max_new_tokens=40):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = HF_AutoProcessor.from_pretrained(model_id)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=dtype)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption, model_id

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--emotions_model_dir", type=str, required=True)
    ap.add_argument("--detr_model_dir", type=str, required=True)
    ap.add_argument("--blip2_model_id", type=str, default="Salesforce/blip2-opt-2.7b")
    ap.add_argument("--det_conf", type=float, default=0.25)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--prompt", type=str, default="Describe the drawing briefly.")
    args = ap.parse_args()

    images = [p for p in Path(args.images_dir).iterdir() if p.suffix.lower() in IMG_EXTS]
    images.sort()
    out_root = Path(args.out_dir); (out_root/"json").mkdir(parents=True, exist_ok=True); (out_root/"pdf").mkdir(parents=True, exist_ok=True)

    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        cls_res = run_classification(args.emotions_model_dir, image)
        dets, det_model_name = run_detection(args.detr_model_dir, image, det_conf=args.det_conf)
        caption, cap_model_name = run_captioning(args.blip2_model_id, image, args.prompt, args.max_new_tokens)

        payload = {"filename": img_path.name, "classification": cls_res, "detections": dets,
                   "caption": caption, "source_model_names": [cls_res.get("source_model"), det_model_name, cap_model_name]}
        json_path = out_root/"json"/(img_path.stem + ".json")
        with open(json_path, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2, ensure_ascii=False)

        pdf_path = out_root/"pdf"/(img_path.stem + ".pdf")
        meta = {"filename": img_path.name, "source_model_names": payload["source_model_names"]}
        build_pdf(str(pdf_path), meta, str(img_path),
                  {"top1_label": cls_res.get("top1_label"), "top1_prob": cls_res.get("top1_prob")},
                  dets, caption)

    print("Done. JSON ->", out_root/"json", "PDF ->", out_root/"pdf")

if __name__ == "__main__":
    main()
