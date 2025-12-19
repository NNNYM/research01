
# 把ANTI-UAV的数据标注格式编程coco格式
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert Anti-UAV dataset (visible.mp4 + visible.json) to COCO-VID format
for TransVOD-Lite training.

NO command line arguments.
You only need to set SRC_ROOT and DST_ROOT below.

Expected Anti-UAV structure:
    SRC_ROOT/
        train/
            SEQ_1/
                visible.mp4
                visible.json
            SEQ_2/
                ...
        val/
            ...
        test/
            ...

Output structure:
    DST_ROOT/
        Data/
            train/SEQ_x/00000001.jpg ...
            val/SEQ_x/...
            test/SEQ_x/...
        annotations/
            anti_uav_train.json
            anti_uav_val.json
            anti_uav_test.json
"""

import json
import cv2
import os
from pathlib import Path


# =========================================================
# 视频转图片，并改成coco格式标注
# =========================================================
SRC_ROOT = Path("/mnt/e/important/datasets/Anti-UAV-RGBT")   # 修改为你的 Anti-UAV 数据根目录,里边包含文件（train,val,test）
DST_ROOT = Path("data_test/anti_uav_coco")  # 输出目录
# =========================================================



def extract_frames(video_path: Path, out_dir: Path):
    """Extract all frames from visible.mp4 → jpg list"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    idx = 1
    frame_files = []
    H = W = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if H is None:
            H, W = frame.shape[:2]

        fname = f"{idx:08d}.jpg"
        cv2.imwrite(str(out_dir / fname), frame)
        frame_files.append(fname)
        idx += 1

    cap.release()

    if H is None:
        raise RuntimeError(f"No frames extracted from {video_path}")

    return frame_files, (H, W)



def load_visible_json(path: Path):
    data = json.load(open(path, "r"))
    return data["exist"], data["gt_rect"]



def process_split(split: str):
    """
    Convert Anti-UAV split (train/val/test)
    """
    src_split_dir = SRC_ROOT / split
    if not src_split_dir.exists():
        print(f"[INFO] Split does not exist: {src_split_dir}, skip.")
        return

    print(f"[INFO] Processing split: {split}")

    dst_data_root = DST_ROOT / "Data" / split
    dst_data_root.mkdir(parents=True, exist_ok=True)

    dst_anno_root = DST_ROOT / "annotations"
    dst_anno_root.mkdir(parents=True, exist_ok=True)

    videos = []
    images = []
    annotations = []
    categories = [{"id": 1, "name": "uav"}]

    video_id = 1
    image_id = 1
    anno_id = 1

    seq_dirs = sorted([d for d in src_split_dir.iterdir() if d.is_dir()], key=lambda x: x.name)

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        mp4_path = seq_dir / "visible.mp4"
        js_path = seq_dir / "visible.json"

        if not mp4_path.exists() or not js_path.exists():
            print(f"[WARN] Skip {seq_name}: missing visible.mp4 or visible.json")
            continue

        print(f"[INFO] Extracting {seq_name} ...")

        # 1. extract frames
        out_dir = dst_data_root / seq_name
        frame_files, (H, W) = extract_frames(mp4_path, out_dir)

        # 2. load annotations
        exist, rects = load_visible_json(js_path)

        n = min(len(frame_files), len(exist), len(rects))

        videos.append({"id": video_id, "name": seq_name})

        # 3. images + annotations
        for i in range(n):
            frame_id = i + 1
            fname = frame_files[i]

            images.append({
                "id": image_id,
                "file_name": f"{split}/{seq_name}/{fname}",
                "height": H,
                "width": W,
                "video_id": video_id,
                "frame_id": frame_id,
            })

            # has target?
            if exist[i] == 1:
                x, y, w, h = rects[i]
                annotations.append({
                    "id": anno_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0
                })
                anno_id += 1

            image_id += 1

        video_id += 1

    # Write annotation json
    out_json = DST_ROOT / "annotations" / f"anti_uav_{split}.json"
    with open(out_json, "w") as f:
        json.dump({
            "videos": videos,
            "images": images,
            "annotations": annotations,
            "categories": categories
        }, f)

    print(f"[DONE] {split}: videos={len(videos)}, images={len(images)}, annos={len(annotations)}")
    print(f"Saved: {out_json}\n")



def main():
    print(f"[ROOT] SRC_ROOT = {SRC_ROOT}")
    print(f"[ROOT] DST_ROOT = {DST_ROOT}")

    for split in ["train", "val", "test"]:
        process_split(split)

    print("[ALL DONE] Anti-UAV → COCO-VID conversion completed.")



if __name__ == "__main__":
    main()
