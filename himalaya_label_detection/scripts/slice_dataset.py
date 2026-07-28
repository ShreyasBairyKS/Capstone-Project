import argparse
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np

def convert_yolo_to_abs(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    c = int(parts[0])
    cx, cy, w, h = map(float, parts[1:5])
    
    abs_w = w * img_w
    abs_h = h * img_h
    abs_cx = cx * img_w
    abs_cy = cy * img_h
    
    x_min = abs_cx - (abs_w / 2)
    y_min = abs_cy - (abs_h / 2)
    x_max = abs_cx + (abs_w / 2)
    y_max = abs_cy + (abs_h / 2)
    return c, x_min, y_min, x_max, y_max

def convert_abs_to_yolo(c, x_min, y_min, x_max, y_max, slice_w, slice_h):
    abs_w = x_max - x_min
    abs_h = y_max - y_min
    abs_cx = x_min + (abs_w / 2)
    abs_cy = y_min + (abs_h / 2)
    
    cx = abs_cx / slice_w
    cy = abs_cy / slice_h
    w = abs_w / slice_w
    h = abs_h / slice_h
    
    # Clip just in case
    cx = max(0.0, min(cx, 1.0))
    cy = max(0.0, min(cy, 1.0))
    w = max(0.0, min(w, 1.0))
    h = max(0.0, min(h, 1.0))
    
    return f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def slice_image_and_labels(img_path, label_path, out_img_dir, out_label_dir, slice_h, slice_w, overlap):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Cannot read {img_path}")
        return
        
    img_h, img_w = img.shape[:2]
    
    # Read labels
    boxes = []
    if label_path and label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                box = convert_yolo_to_abs(line, img_w, img_h)
                if box:
                    boxes.append(box)

    stride_x = int(slice_w * (1 - overlap))
    stride_y = int(slice_h * (1 - overlap))
    
    x_starts = list(range(0, img_w - slice_w + 1, stride_x))
    if x_starts[-1] + slice_w < img_w:
        x_starts.append(img_w - slice_w)
        
    y_starts = list(range(0, img_h - slice_h + 1, stride_y))
    if y_starts[-1] + slice_h < img_h:
        y_starts.append(img_h - slice_h)
        
    for y in y_starts:
        for x in x_starts:
            slice_boxes = []
            
            # Check overlap for each box
            for (c, bx1, by1, bx2, by2) in boxes:
                # Intersection
                ix1 = max(x, bx1)
                iy1 = max(y, by1)
                ix2 = min(x + slice_w, bx2)
                iy2 = min(y + slice_h, by2)
                
                if ix1 < ix2 and iy1 < iy2:
                    # The box is at least partially in the slice
                    # Shift coordinates relative to slice
                    nx1 = ix1 - x
                    ny1 = iy1 - y
                    nx2 = ix2 - x
                    ny2 = iy2 - y
                    
                    # Convert to YOLO format
                    yolo_str = convert_abs_to_yolo(c, nx1, ny1, nx2, ny2, slice_w, slice_h)
                    slice_boxes.append(yolo_str)
            
            # Only save slices that have bounding boxes (to speed up training)
            # OR optionally save empty slices to train on negative examples.
            # We will save 10% of background slices to prevent false positives.
            save_slice = False
            if slice_boxes:
                save_slice = True
            else:
                if np.random.rand() < 0.1:
                    save_slice = True
                    
            if save_slice:
                slice_name = f"{img_path.stem}_{y}_{x}"
                slice_img = img[y:y+slice_h, x:x+slice_w]
                cv2.imwrite(str(out_img_dir / f"{slice_name}{img_path.suffix}"), slice_img)
                
                out_lbl = out_label_dir / f"{slice_name}.txt"
                if slice_boxes:
                    with open(out_lbl, 'w') as f:
                        f.write("\n".join(slice_boxes) + "\n")
                else:
                    # Empty file for negative sample
                    out_lbl.touch()

def main():
    parser = argparse.ArgumentParser(description="Slice massive images and YOLO labels into overlapping square patches")
    parser.add_argument("--data", default="data/yolo_dataset", help="Path to YOLO dataset")
    parser.add_argument("--out", default="data/yolo_dataset_sliced", help="Output path")
    parser.add_argument("--size", type=int, default=1504, help="Slice size")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio")
    args = parser.parse_args()

    in_dir = Path(args.data)
    out_dir = Path(args.out)
    
    if out_dir.exists():
        print(f"Removing existing {out_dir}...")
        shutil.rmtree(out_dir)
        
    out_dir.mkdir(parents=True)
    
    # Copy data.yaml
    yaml_src = in_dir / "data.yaml"
    if yaml_src.exists():
        shutil.copy(yaml_src, out_dir / "data.yaml")
        
    for split in ["train", "val", "test"]:
        split_img_dir = in_dir / "images" / split
        if not split_img_dir.exists():
            continue
            
        print(f"Slicing {split} set...")
        out_img_dir = out_dir / "images" / split
        out_lbl_dir = out_dir / "labels" / split
        out_img_dir.mkdir(parents=True)
        out_lbl_dir.mkdir(parents=True)
        
        for img_path in split_img_dir.iterdir():
            if img_path.suffix.lower() not in [".bmp", ".png", ".jpg", ".jpeg"]:
                continue
                
            label_path = in_dir / "labels" / split / f"{img_path.stem}.txt"
            slice_image_and_labels(img_path, label_path, out_img_dir, out_lbl_dir, args.size, args.size, args.overlap)
            
    print(f"\nDone! Sliced dataset saved to {out_dir}")
    print("You can now train YOLO with:")
    print("python himalaya_label_detection/scripts/train_yolo_roi.py --data data/yolo_dataset_sliced/data.yaml")

if __name__ == "__main__":
    main()
