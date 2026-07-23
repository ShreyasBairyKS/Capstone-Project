"""
crop_roi.py  —  Interactive ROI Cropping Tool (Horizontal Scroll Fix)
═══════════════════════════════════════════════════════════════════════════════

The full BMP images are 8000x1504 with text oriented sideways.
This tool ROTATES the image 90 degrees clockwise for display, so the text is
upright and readable. The image becomes 1504 tall and 8000 wide.

It then scales the height to fit your screen (e.g. 700px tall) and lets you
scroll left/right to find the ROI and draw your box.

CONTROLS
─────────
  Mouse wheel       = scroll left / right through the image
  A / D keys        = scroll left / right (alternative)
  Click + drag      = draw crop rectangle
  G                 = save crop → good/
  B                 = save crop → bad/
  S                 = skip this image
  R                 = reset / redraw selection
  Q                 = quit (progress saved)
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path

# ── Viewport config ────────────────────────────────────────────────────────────
VIEWPORT_H = 700   # display height (scaled to fit screen vertically)
VIEWPORT_W = 1200  # visible width of viewport (scroll to see the rest)

state = {
    "scroll_x":   0,
    "scaled_img": None,
    "scaled_w":   0,
    "draw_start": None,
    "draw_end":   None,
    "dragging":   False,
    "rect_done":  False,
}

def clamp_scroll():
    max_scroll = max(0, state["scaled_w"] - VIEWPORT_W)
    state["scroll_x"] = max(0, min(state["scroll_x"], max_scroll))

def get_viewport():
    x0 = state["scroll_x"]
    x1 = min(state["scaled_w"], x0 + VIEWPORT_W)
    
    # In case the scaled image is smaller than viewport width
    vp = state["scaled_img"][:, x0:x1].copy()
    
    # Pad to VIEWPORT_W if we are at the edge
    if vp.shape[1] < VIEWPORT_W:
        pad_w = VIEWPORT_W - vp.shape[1]
        vp = cv2.copyMakeBorder(vp, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=[0,0,0])
    return vp

def draw_rect_on_viewport(vp, color=(0, 255, 0)):
    if state["draw_start"] and state["draw_end"]:
        x1, y1 = state["draw_start"]
        x2, y2 = state["draw_end"]
        cv2.rectangle(vp, (x1, y1), (x2, y2), color, 2)
    return vp

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEWHEEL:
        # Scroll horizontally
        delta = -100 if flags > 0 else 100
        state["scroll_x"] += delta
        clamp_scroll()
        vp = get_viewport()
        draw_rect_on_viewport(vp, color=(0,255,0) if state["rect_done"] else (0,0,255))
        add_hud(vp)
        cv2.imshow("NSC Crop Tool", vp)

    elif event == cv2.EVENT_LBUTTONDOWN:
        state["dragging"]   = True
        state["rect_done"]  = False
        state["draw_start"] = (x, y)
        state["draw_end"]   = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
        state["draw_end"] = (x, y)
        vp = get_viewport()
        draw_rect_on_viewport(vp, color=(0, 0, 255))
        add_hud(vp)
        cv2.imshow("NSC Crop Tool", vp)

    elif event == cv2.EVENT_LBUTTONUP:
        state["dragging"]  = False
        state["rect_done"] = True
        state["draw_end"]  = (x, y)
        vp = get_viewport()
        draw_rect_on_viewport(vp, color=(0, 255, 0))
        add_hud(vp, done=True)
        cv2.imshow("NSC Crop Tool", vp)

def add_hud(vp, done=False):
    h, w = vp.shape[:2]
    scroll_pct = int(100 * state["scroll_x"] / max(1, state["scaled_w"] - VIEWPORT_W))
    line1 = state.get("hud_line1", "")
    line2 = "Scroll: Mouse Wheel (or A/D)   Draw: Click+drag   G=good  B=bad  S=skip  R=redo  Q=quit"
    if done:
        line2 = "Box drawn!  Press G (good) or B (bad) to save, R to redo"

    # Dark background bar for text
    cv2.rectangle(vp, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.rectangle(vp, (0, h-30), (w, h), (0, 0, 0), -1)

    cv2.putText(vp, line1,  (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vp, line2,  (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(vp, f"Horizontal Scroll: {scroll_pct}%", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

def load_and_scale(img_path: Path):
    """Load image, rotate to upright, and scale to fit VIEWPORT_H."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    # Rotate 90 degrees clockwise to make text upright
    # Shape goes from (8000, 1504) -> (1504, 8000)
    rot_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    h, w = rot_img.shape[:2]
    scale = VIEWPORT_H / h
    new_h = VIEWPORT_H
    new_w = int(w * scale)
    scaled = cv2.resize(rot_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return rot_img, scaled, scale

def viewport_to_full_coords(vx1, vy1, vx2, vy2, scale: float):
    # Viewport x is relative to scroll_x
    sx1 = vx1 + state["scroll_x"]
    sx2 = vx2 + state["scroll_x"]
    
    x1 = int(min(sx1, sx2) / scale)
    y1 = int(min(vy1, vy2) / scale)
    x2 = int(max(sx1, sx2) / scale)
    y2 = int(max(vy1, vy2) / scale)
    return x1, y1, x2, y2

def save_crop(rot_img, x1, y1, x2, y2, out_path: Path):
    h, w = rot_img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = rot_img[y1:y2, x1:x2]
    if crop.size == 0:
        return False, (0, 0, 0, 0)
    cv2.imwrite(str(out_path), crop)
    return True, (x1, y1, x2 - x1, y2 - y1)

def load_progress(f: Path) -> set:
    if f.exists():
        return set(json.load(open(f)).get("done", []))
    return set()

def save_progress(f: Path, done: set):
    json.dump({"done": sorted(done)}, open(f, "w"), indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi",        default="ROI_1")
    parser.add_argument("--images",     default="dataset/NSC")
    parser.add_argument("--out-dir",    default=None)
    parser.add_argument("--start-from", default=None)
    parser.add_argument("--good-only",  action="store_true")
    parser.add_argument("--bad-only",   action="store_true")
    args = parser.parse_args()

    roi_name    = args.roi
    images_root = Path(args.images)
    out_dir     = Path(args.out_dir) if args.out_dir else Path(f"data/rois/{roi_name}")
    good_dir    = out_dir / "good"
    bad_dir     = out_dir / "bad"
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    progress_file = out_dir / "progress.json"
    done = load_progress(progress_file)

    folders = []
    if not args.bad_only:  folders.append((images_root / "NSC GOOD IMAGES", "GOOD"))
    if not args.good_only: folders.append((images_root / "NSC BAD IMAGES",  "BAD"))

    all_files = []
    for folder, label in folders:
        if folder.exists():
            for f in sorted(folder.glob("*.bmp")):
                all_files.append((f, label))

    if not all_files:
        print(f"No images found. Check --images path: {images_root}")
        return

    if args.start_from:
        names = [f.name for f, _ in all_files]
        if args.start_from in names:
            all_files = all_files[names.index(args.start_from):]

    remaining = [(f, lbl) for f, lbl in all_files if f.name not in done]
    total = len(all_files)

    print(f"\n{'='*60}")
    print(f"  {roi_name} — Horizontal Cropping Tool")
    print(f"  {len(remaining)} remaining  ({len(done)} done / {total} total)")
    print(f"{'='*60}")
    
    # IMPORTANT: Use AUTOSIZE so OpenCV doesn't stretch the image!
    cv2.namedWindow("NSC Crop Tool", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("NSC Crop Tool", mouse_callback)

    good_count = len(list(good_dir.glob("*")))
    bad_count  = len(list(bad_dir.glob("*")))

    for file_idx, (img_path, orig_label) in enumerate(remaining):
        result = load_and_scale(img_path)
        if result is None:
            continue

        rot_img, scaled_img, scale = result

        state["scaled_img"] = scaled_img
        state["scaled_w"]   = scaled_img.shape[1]
        state["scroll_x"]   = 0
        state["draw_start"] = None
        state["draw_end"]   = None
        state["dragging"]   = False
        state["rect_done"]  = False
        
        pct = int(100 * file_idx / max(1, len(remaining)))
        state["hud_line1"] = (
            f"{roi_name}  |  {img_path.name}  [{orig_label}]  "
            f"({file_idx+1}/{len(remaining)})  {pct}%  "
            f"good={good_count}  bad={bad_count}"
        )

        vp = get_viewport()
        add_hud(vp)
        cv2.imshow("NSC Crop Tool", vp)

        action = None
        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == ord('g') and state["rect_done"]:
                action = 'good'; break
            elif key == ord('b') and state["rect_done"]:
                action = 'bad'; break
            elif key == ord('s'):
                action = 'skip'; break
            elif key == ord('q'):
                action = 'quit'; break
            elif key == ord('r'):
                state["draw_start"] = None
                state["draw_end"]   = None
                state["rect_done"]  = False
                vp = get_viewport(); add_hud(vp)
                cv2.imshow("NSC Crop Tool", vp)
            elif key == ord('a') or key == 81: # A or Left Arrow
                state["scroll_x"] -= 200; clamp_scroll()
                vp = get_viewport(); draw_rect_on_viewport(vp); add_hud(vp)
                cv2.imshow("NSC Crop Tool", vp)
            elif key == ord('d') or key == 83: # D or Right Arrow
                state["scroll_x"] += 200; clamp_scroll()
                vp = get_viewport(); draw_rect_on_viewport(vp); add_hud(vp)
                cv2.imshow("NSC Crop Tool", vp)

        if action == 'quit':
            save_progress(progress_file, done); break

        if action == 'skip':
            done.add(img_path.name)
            save_progress(progress_file, done)
            continue

        vx1, vy1 = state["draw_start"]
        vx2, vy2 = state["draw_end"]
        x1, y1, x2, y2 = viewport_to_full_coords(vx1, vy1, vx2, vy2, scale)

        out_subdir = good_dir if action == 'good' else bad_dir
        out_path   = out_subdir / (img_path.stem + ".png")
        
        ok, (cx, cy, cw, ch) = save_crop(rot_img, x1, y1, x2, y2, out_path)

        if ok:
            if action == 'good': good_count += 1
            else:                bad_count  += 1
        else:
            print(f"  ERROR: empty crop on {img_path.name} — press R and try again")
            continue

        done.add(img_path.name)
        save_progress(progress_file, done)

    cv2.destroyAllWindows()
    print(f"\nDone.  good={good_count}  bad={bad_count}")

if __name__ == "__main__":
    main()
