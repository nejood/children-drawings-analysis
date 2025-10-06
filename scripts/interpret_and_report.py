#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply rule-based interpretation to existing JSON results and generate PDFs with an Interpretation section.

Usage:
python interpret_and_report.py \
  --images_dir data/processed/htp_dap_clean/images \
  --json_dir outputs/json \
  --pdf_out outputs/pdf_interpreted
"""

import json, argparse
from pathlib import Path
from PIL import Image
from rules_engine import interpret
from report_utils import build_pdf_with_interpretation

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--json_dir", type=str, required=True)
    ap.add_argument("--pdf_out", type=str, required=True)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    json_dir   = Path(args.json_dir)
    pdf_out    = Path(args.pdf_out); pdf_out.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files in {json_dir}")

    for jpath in json_files:
        data = json.loads(jpath.read_text())
        filename = data.get("filename")
        img_path = images_dir / filename
        if not img_path.exists():
            # try any image with same stem
            stem = Path(filename).stem
            matches = list(images_dir.glob(stem + ".*"))
            if matches:
                img_path = matches[0]
        if not img_path.exists():
            print(f"[WARN] image not found for {filename}"); continue

        # Inputs for rules
        dets = data.get("detections", [])
        cls = data.get("classification", {})
        top_emotion = cls.get("top1_label", "natural")
        caption = data.get("caption", "")

        intr = interpret(dets, top_emotion, caption)

        # Build PDF
        meta = {
            "filename": filename,
            "source_model_names": data.get("source_model_names", [])
        }
        out_pdf = pdf_out / (Path(filename).stem + "_interpreted.pdf")
        build_pdf_with_interpretation(
            str(out_pdf),
            meta,
            str(img_path),
            {"top1_label": top_emotion, "top1_prob": cls.get("top1_prob", 0.0)},
            dets,
            caption,
            intr
        )
        print("Wrote:", out_pdf)

    print("All done ->", pdf_out)

if __name__ == "__main__":
    main()
