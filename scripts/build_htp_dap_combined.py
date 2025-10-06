#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild the unified HTP/DAP YOLO dataset from selected sources.
If --sources is omitted, scans data/raw/* for 'htp'/'dap'.
"""
import os, re, json, shutil, argparse
from pathlib import Path
from collections import Counter

BASE = Path(".")
RAW  = BASE / "data" / "raw"
OUT  = BASE / "data" / "processed" / "htp_dap_combined"
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

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

def find_data_yaml(root: Path):
    cands = list(root.rglob("data.yaml"))
    return cands[0] if cands else None

def parse_names_from_yaml(yaml_path: Path):
    txt = yaml_path.read_text(errors="ignore")
    m = re.search(r"names\s*:\s*\[([^\]]+)\]", txt, re.IGNORECASE)
    if m: return [p.strip().strip("'\"") for p in m.group(1).split(",")]
    block = re.search(r"names\s*:\s*\n((?:\s*-\s*[^\n]+\n)+)", txt, re.IGNORECASE)
    if block:
        import re as _re
        items = _re.findall(r"-\s*([^\n]+)", block.group(1))
        return [i.strip().strip("'\"") for i in items]
    return None

def norm_key(name: str): return re.sub(r"\s+","_", name.strip().lower())

def build_id_map(source_root: Path):
    yaml_path = find_data_yaml(source_root)
    names = parse_names_from_yaml(yaml_path) if yaml_path else None
    if not names:
        return lambda cid: cid if 0 <= cid < len(DETECTION_CLASSES) else None, None
    name_to_unified = {}
    for n in names:
        key = norm_key(n)
        if key in CLASS_TO_ID: name_to_unified[key] = CLASS_TO_ID[key]
        elif key.endswith("s") and key[:-1] in CLASS_TO_ID: name_to_unified[key] = CLASS_TO_ID[key[:-1]]
        elif key in SYN and SYN[key] in CLASS_TO_ID: name_to_unified[key] = CLASS_TO_ID[SYN[key]]
        else: name_to_unified[key] = None
    def map_fn(cid_source: int):
        if cid_source < 0 or cid_source >= len(names): return None
        src_name = norm_key(names[cid_source])
        return name_to_unified.get(src_name, None)
    return map_fn, names

def collect_yolo_pairs(root: Path):
    pairs = []
    for split in ["train","valid","val","test"]:
        img_dir = root / split / "images"; lab_dir = root / split / "labels"
        if img_dir.exists() and lab_dir.exists(): pairs.append((img_dir, lab_dir))
    if not pairs:
        for img_dir in root.rglob("images"):
            lab_dir = img_dir.parent / "labels"
            if img_dir.exists() and lab_dir.exists(): pairs.append((img_dir, lab_dir))
    return pairs

def remap_yolo_label_file(src_txt: Path, dst_txt: Path, id_map_fn):
    lines_out = []
    if not src_txt.exists():
        dst_txt.write_text(""); return
    for line in src_txt.read_text().splitlines():
        if not line.strip(): continue
        comps = line.split()
        try:
            old_id = int(float(comps[0])); new_id = id_map_fn(old_id)
            if new_id is None: continue
            lines_out.append(" ".join([str(new_id)] + comps[1:]))
        except Exception:
            continue
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    dst_txt.write_text("\n".join(lines_out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", help="Specific raw HTP/DAP roots to use (e.g., ..._enh)")
    args = ap.parse_args()

    if args.sources:
        candidates = [Path(s) for s in args.sources]
    else:
        if not RAW.exists(): raise SystemExit("data/raw/ not found.")
        candidates = [d for d in RAW.iterdir() if d.is_dir() and any(k in d.name.lower() for k in ["htp","dap"])]

    if not candidates:
        raise SystemExit("No HTP/DAP sources found.")

    if OUT.exists(): shutil.rmtree(OUT)
    (OUT/"images").mkdir(parents=True, exist_ok=True)
    (OUT/"labels").mkdir(parents=True, exist_ok=True)

    det_counts = Counter(); img_count = 0
    for src in candidates:
        id_map_fn, _ = build_id_map(src)
        for (img_dir, lab_dir) in collect_yolo_pairs(src):
            for img_path in img_dir.glob("*"):
                if img_path.suffix.lower() not in IMG_EXTS: continue
                rel_name = f"{src.name}_{img_path.stem}{img_path.suffix.lower()}"
                shutil.copy2(img_path, OUT/"images"/rel_name)
                src_txt = lab_dir / (img_path.stem + ".txt")
                dst_txt = OUT/"labels"/(Path(rel_name).stem + ".txt")
                remap_yolo_label_file(src_txt, dst_txt, id_map_fn)
                if dst_txt.exists():
                    for line in dst_txt.read_text().splitlines():
                        if line.strip():
                            try:
                                cid = int(line.split()[0])
                                if 0 <= cid < len(DETECTION_CLASSES): det_counts[DETECTION_CLASSES[cid]] += 1
                            except: pass
                img_count += 1

    (OUT/"data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images\nval: images\nnc: {len(DETECTION_CLASSES)}\n"
        f"names: {json.dumps(DETECTION_CLASSES)}\n"
    )
    print("\n=== REBUILD COMPLETE ===")
    print("Images copied:", img_count)
    print("Labels files :", len(list((OUT/'labels').glob('*.txt'))))
    print("Output path  :", OUT)

if __name__ == "__main__":
    main()
