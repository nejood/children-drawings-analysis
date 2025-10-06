#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

def wrap_text(text, max_chars=90):
    words = str(text).split()
    lines, cur, count = [], [], 0
    for w in words:
        if count + len(w) + 1 > max_chars:
            lines.append(" ".join(cur)); cur=[w]; count=len(w)
        else:
            cur.append(w); count += len(w) + 1
    if cur: lines.append(" ".join(cur))
    return lines

def _draw_header(c, title, w, y):
    c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, y, title)
    c.setLineWidth(0.5); c.line(2*cm, y-0.2*cm, w-2*cm, y-0.2*cm)

def _draw_kv(c, x, y, key, value, key_w=3.0*cm, val_w=12.0*cm, leading=14):
    c.setFont("Helvetica", 10); c.setFillColor(colors.black)
    c.drawString(x, y, f"{key}:")
    tx = c.beginText(x + key_w, y); tx.setFont("Helvetica", 10)
    for line in wrap_text(value, max_chars=int(val_w/4.5)):
        tx.textLine(line)
    c.drawText(tx); return y - leading

def add_image(c, img_path, x, y, max_w=8*cm, max_h=8*cm):
    try:
        img = ImageReader(img_path)
        iw, ih = img.getSize(); ratio = min(max_w/iw, max_h/ih)
        w = iw*ratio; h = ih*ratio
        c.drawImage(img, x, y-h, width=w, height=h)
        return h
    except Exception:
        return 0

def build_pdf(output_path, meta, image_path, classification, detections, caption):
    # Original (no interpretation)
    w, h = A4; c = canvas.Canvas(output_path, pagesize=A4); y = h - 2*cm
    _draw_header(c, "Children's Drawing Analysis Report", w, y); y -= 1.2*cm
    img_h = add_image(c, image_path, 2*cm, y-8*cm, max_w=8*cm, max_h=8*cm)
    right_x = 11*cm; y_right = y
    y_right = _draw_kv(c, right_x, y_right, "File", meta.get("filename",""))
    y_right = _draw_kv(c, right_x, y_right, "Models", ", ".join(meta.get("source_model_names", [])))
    y_right = _draw_kv(c, right_x, y_right, "Top Emotion",
                       f"{classification.get('top1_label','-')} ({classification.get('top1_prob',0):.2f})")
    y_caption = min(y - (img_h + 1.0*cm), y_right - 1.0*cm)
    _draw_header(c, "Caption", w, y_caption); y_caption -= 1.0*cm
    c.setFont("Helvetica", 10); tx = c.beginText(2*cm, y_caption)
    for line in wrap_text(caption or "", max_chars=100):
        tx.textLine(line)
    c.drawText(tx); c.showPage(); c.save()

def build_pdf_with_interpretation(output_path, meta, image_path, classification, detections, caption, interpretation):
    w, h = A4; c = canvas.Canvas(output_path, pagesize=A4); y = h - 2*cm
    _draw_header(c, "Children's Drawing Analysis Report", w, y); y -= 1.2*cm
    img_h = add_image(c, image_path, 2*cm, y-8*cm, max_w=8*cm, max_h=8*cm)
    right_x = 11*cm; y_right = y
    y_right = _draw_kv(c, right_x, y_right, "File", meta.get("filename",""))
    y_right = _draw_kv(c, right_x, y_right, "Models", ", ".join(meta.get("source_model_names", [])))
    y_right = _draw_kv(c, right_x, y_right, "Top Emotion",
                       f"{classification.get('top1_label','-')} ({classification.get('top1_prob',0):.2f})")

    # Caption
    y_caption = min(y - (img_h + 1.0*cm), y_right - 1.0*cm)
    _draw_header(c, "Caption", w, y_caption); y_caption -= 1.0*cm
    c.setFont("Helvetica", 10)
    tx = c.beginText(2*cm, y_caption)
    for line in wrap_text(caption or "", max_chars=100):
        tx.textLine(line)
    c.drawText(tx)

    # Interpretation
    y_inter = y_caption - 2.0*cm
    _draw_header(c, "Psychological Interpretation (Heuristic)", w, y_inter); y_inter -= 1.0*cm
    c.setFont("Helvetica", 10)
    # Summary
    tx2 = c.beginText(2*cm, y_inter)
    for line in wrap_text(interpretation.get("summary",""), max_chars=100):
        tx2.textLine(line)
    c.drawText(tx2)

    # (Optional) list indicators and risk flags
    y_list = y_inter - (14 * (len(wrap_text(interpretation.get("summary",""), 100)) + 1))
    c.setFont("Helvetica-Bold", 11); c.drawString(2*cm, y_list, "Indicators:")
    c.setFont("Helvetica", 10)
    y_list -= 14
    for s in interpretation.get("indicators", [])[:10]:
        for line in wrap_text(f"- {s}", max_chars=100):
            c.drawString(2*cm, y_list, line); y_list -= 12

    y_list -= 8; c.setFont("Helvetica-Bold", 11); c.drawString(2*cm, y_list, "Risk cues:")
    c.setFont("Helvetica", 10); y_list -= 14
    for s in interpretation.get("risk_flags", [])[:10]:
        for line in wrap_text(f"- {s}", max_chars=100):
            c.drawString(2*cm, y_list, line); y_list -= 12

    c.showPage(); c.save()
