#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flexible HTP/DAP combiner: no assumptions about folder names.
- Walks each --sources root recursively
- For every image found, looks for a label .txt with same stem:
    priority: same dir -> sibling 'labels' -> nearest 'labels' upward
- If a data.yaml is found within the source, use it to map class names → unified IDs
- Output is a flat YOLO dataset in data/processed/htp_dap_combined
"""
import argparse, os, re, json, shutil
from pathlib import Path
from collections import Counter

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
DETECTION_CLASSES = [
    "house","roof","door","window","chimney","fence","path",
    "tree","trunk","branches","leaves","roots","fruit",
    "person","head","eyes","mouth","nose","ears","neck","arms","hands","fingers",
    "legs","feet","body","shoulders","sad","fear","happy","angry","natural"
]
CLASS_TO_ID = {c:i for i,c in enumerate(DETECTION_CLASSES)}
SYN = {"human":"person","kid":"person","boy":"person","girl":"person","leaf":"leaves","branch":"branches",
       "pathway":"path","road":"path","windowpane":"window","doorway":"door","ear":"ears","finger":"fingers",
       "mouths":"mouth","rooftop":"roof","trouser":"legs"}

def find_yaml(root: Path):
    cands = list(root.rglob("data.yaml"))
    return cands[0] if cands else None

def parse_names(yaml_path: Path):
    if not yaml_path or not yaml_path.exists(): return None
    txt = yaml_path.read_text(errors="ignore")
    m = re.search(r"names\s*:\s*\[([^\]]+)\]", txt, re.IGNORECASE)
    if m: return [p.strip().strip("'\"") for p in m.group(1).split(",")]
    block = re.search(r"names\s*:\s*\n((?:\s*-\s*[^\n]+\n)+)", txt, re.IGNORECASE)
    if block:
        items = re.findall(r"-\s*([^\n]+)", block.group(1))
        return [i.strip().strip("'\"") for i in items]
    return None

def norm_key(name: str): return re.sub(r"\s+","_", name.strip().lower())

def build_id_map(source_root: Path):
    names = parse_names(find_yaml(source_root))
    if not names:
        # pass-through; will trust numeric ids to already match unified scheme
        return lambda cid: cid if 0 <= cid < len(DETECTION_CLASSES) else None
    name_to_id = {}
    for n in names:
        k = norm_key(n)
        if k in CLASS_TO_ID: name_to_id[k]=CLASS_TO_ID[k]
        elif k.endswith("s") and k[:-1] in CLASS_TO_ID: name_to_id[k]=CLASS_TO_ID[k[:-1]]
        elif k in SYN and SYN[k] in CLASS_TO_ID: name_to_id[k]=CLASS_TO_ID[SYN[k]]
        else: name_to_id[k]=None
    def map_fn(cid_source: int):
        if cid_source<0 or cid_source>=len(names): return None
        k = norm_key(names[cid_source])
        return name_to_id.get(k, None)
    return map_fn

def nearest_label_for(image_path: Path, source_root: Path):
    stem = image_path.stem
    # 1) same directory
    cand = image_path.with_suffix(".txt")
    if cand.exists(): return cand
    # 2) sibling 'labels' under same parent
    lab = image_path.parent.parent / "labels" / (stem + ".txt")
    if lab.exists(): return lab
    # 3) look upward for a 'labels' dir
    cur = image_path.parent
    while cur != source_root.parent:
        lab_dir = cur / "labels"
        if lab_dir.exists():
            p = lab_dir / (stem + ".txt")
            if p.exists(): return p
        cur = cur.parent
    return None  # unlabeled image

def remap_yolo_txt(src_txt: Path, id_map, dst_txt: Path):
    if not src_txt or not src_txt.exists():
        return 0
    out_lines = []
    for line in src_txt.read_text().splitlines():
        if not line.strip(): continue
        parts = line.split()
        try:
            old_id = int(float(parts[0])); new_id = id_map(old_id)
            if new_id is None: continue
            out_lines.append(" ".join([str(new_id)] + parts[1:]))
        except Exception:
            continue
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    dst_txt.write_text("\n".join(out_lines))
    return len(out_lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, help="One or more dataset roots (enhanced or original)")
    ap.add_argument("--out", type=str, default="data/processed/htp_dap_combined")
    args = ap.parse_args()

    OUT = Path(args.out); IMG_OUT = OUT/"images"; LAB_OUT = OUT/"labels"
    if OUT.exists(): shutil.rmtree(OUT)
    IMG_OUT.mkdir(parents=True, exist_ok=True); LAB_OUT.mkdir(parents=True, exist_ok=True)

    total_imgs = 0; total_labels = 0
    for src_str in args.sources:
        src = Path(src_str)
        if not src.exists():
            print(f"[WARN] missing source: {src}"); continue
        id_map = build_id_map(src)

        for p in src.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                rel_name = f"{src.name}_{p.stem}{p.suffix.lower()}"
                shutil.copy2(p, IMG_OUT/rel_name)
                lbl = nearest_label_for(p, src)
                dst_txt = LAB_OUT/(Path(rel_name).stem + ".txt")
                wrote = remap_yolo_txt(lbl, id_map, dst_txt)
                # allow empty .txt to exist (some images might be unlabeled)
                total_imgs += 1
                total_labels += wrote

    # write data.yaml
    (OUT/"data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images\nval: images\nnc: {len(DETECTION_CLASSES)}\n"
        f"names: {json.dumps(DETECTION_CLASSES)}\n"
    )
    print("\n=== REBUILD COMPLETE ===")
    print("Images copied:", total_imgs)
    print("Labels written (boxes):", total_labels)
    print("Output path  :", OUT)

if __name__ == "__main__":
    main()
