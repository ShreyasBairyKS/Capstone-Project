"""
Diagnostic: verifies that multi-camera batched inference keeps each camera's detections isolated.

For every camera image it compares
  (a) single-image inference,
  (b) batched inference (same code path as the API), and
  (c) batched inference executed from worker threads (how the API now runs it),
and reports any camera whose detections differ or contain another camera's boxes.

Usage (run on the machine/engine you want to test, e.g. the CUDA PC or inside Docker):
  python pipeline/diagnose_batch_isolation.py img_front.jpg img_rear.jpg img_side.jpg
  docker exec -it visionqai_backend python pipeline/diagnose_batch_isolation.py /app/simulation/cap/a.jpg ...
Exit code 0 = isolated, 1 = contamination found.
"""
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
from pipeline.inference.engine import InferenceEngine  # noqa: E402


def signature(detections):
    """Order-independent, rounding-tolerant fingerprint of a detection list."""
    return sorted((d.class_name, round(d.box[0] / 4), round(d.box[1] / 4), round(d.box[2] / 4), round(d.box[3] / 4)) for d in detections)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="2-4 distinct images, one per camera")
    ap.add_argument("--rounds", type=int, default=5, help="repetitions of the threaded batched pass")
    args = ap.parse_args()

    engine = InferenceEngine(warmup=True)
    print(f"Engine: {engine.model_path.name} | {engine.hardware_desc} | batch={engine.batch_size}")

    names = [f"camera_{i + 1}" for i in range(len(args.images))]
    frames = {}
    for n, p in zip(names, args.images):
        img = cv2.imread(p)
        if img is None:
            print(f"Cannot read {p}")
            return 2
        frames[n] = img

    expected = {n: signature(engine.predict_single(f, n).detections) for n, f in frames.items()}
    for n in names:
        print(f"  single  {n}: {expected[n]}")

    problems = 0

    def check(label, result):
        nonlocal problems
        for n in names:
            got = signature(result.views[n].detections)
            ok = got == expected[n]
            others = [m for m in names if m != n and any(g in expected[m] for g in got) and g not in expected[n]] if not ok else []
            print(f"  {label:<14}{n}: {'OK' if ok else 'MISMATCH ' + str(got)}{'  <- contains boxes from ' + str(others) if others else ''}")
            problems += 0 if ok else 1

    check("batched", engine.predict_multi_camera(frames))
    with ThreadPoolExecutor(max_workers=1) as pool:  # same as FastAPI's worker-thread execution
        for r in range(args.rounds):
            check(f"threaded #{r + 1}", pool.submit(engine.predict_multi_camera, frames).result())

    print("RESULT:", "ISOLATED (no cross-camera contamination)" if problems == 0 else f"{problems} MISMATCH(ES) FOUND")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
