"""
app.py
──────
Streamlit operator dashboard for the Himalaya label inspection system.

Displays:
  - Live inspection results for a selected folder
  - Pass / Fail counts
  - Heatmap overlays for rejected labels
  - ROI anomaly scores per rejection
  - Downloadable rejection log (JSON)

Usage:
    cd himalaya_label_detection
    streamlit run dashboard/app.py
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.image_utils import load_image, list_images, save_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"


# ── Helpers ───────────────────────────────────────────────────────────────────

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return image[:, :, ::-1]


def load_results(results_dir: Path) -> list[dict]:
    records = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            with open(f) as fp:
                data = json.load(fp)
            records.extend(data if isinstance(data, list) else [data])
        except Exception:
            pass
    return records


# ── Page ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Himalaya Label Inspection",
        layout="wide",
    )

    st.title("Himalaya Label Defect Detection")
    st.caption("Winter Defense Moisturizing Cream — Inline inspection dashboard")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Run Inspection")
        inspect_folder = st.text_input(
            "Image folder",
            value=str(PROJECT_ROOT / "data" / "raw" / "bad"),
        )
        save_heatmaps = st.checkbox("Save heatmap images", value=True)
        run_btn = st.button("▶  Inspect", type="primary", use_container_width=True)

        st.divider()

        st.header("Load Saved Results")
        results_folder = st.text_input(
            "Results folder",
            value=str(RESULTS_DIR),
        )
        load_btn = st.button("Load", use_container_width=True)

    # ── Trigger inference ─────────────────────────────────────────────────────
    if run_btn:
        json_out = RESULTS_DIR / "latest_run.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_inference.py"),
            "--folder", inspect_folder,
            "--json-out", str(json_out),
        ]
        if save_heatmaps:
            cmd += ["--save-output", str(RESULTS_DIR / "heatmaps")]

        with st.spinner("Running inspection pipeline..."):
            proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode == 0:
            st.success("Inspection complete!")
        else:
            st.error("Inspection failed — check terminal for details.")
            st.code(proc.stderr[-2000:])

        st.rerun()

    # ── Load results ──────────────────────────────────────────────────────────
    latest_json = RESULTS_DIR / "latest_run.json"
    records: list[dict] = []

    if load_btn:
        records = load_results(Path(results_folder))
    elif latest_json.exists():
        records = load_results(RESULTS_DIR)

    if not records:
        st.info(
            "No results yet.  "
            "Place images in `data/raw/bad/` or `data/raw/good/` "
            "and click **Inspect**."
        )
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    n_total = len(records)
    n_pass  = sum(1 for r in records if r.get("verdict") == "PASS")
    n_fail  = n_total - n_pass
    avg_ms  = (sum(r.get("latency_ms", 0) for r in records) / n_total
               if n_total else 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inspected",    n_total)
    c2.metric("PASS",         n_pass)
    c3.metric("FAIL",         n_fail)
    c4.metric("Avg latency",  f"{avg_ms:.1f} ms")

    st.divider()

    # ── All results table ─────────────────────────────────────────────────────
    with st.expander(f"All results ({n_total})", expanded=False):
        table_rows = []
        for r in records:
            table_rows.append({
                "File":         r["file"],
                "Verdict":      r["verdict"],
                "Latency (ms)": r.get("latency_ms"),
                "LR SSIM":      r.get("lr_score"),
                "GM Similarity":r.get("gm_score"),
                "Barcode":      r.get("barcode", "—"),
            })
        st.dataframe(table_rows, use_container_width=True)

    # ── Rejection details ─────────────────────────────────────────────────────
    fail_records = [r for r in records if r.get("verdict") == "FAIL"]
    st.subheader(f"Rejections  ({len(fail_records)})")

    if not fail_records:
        st.success("No rejections in this batch.")
    else:
        for rec in fail_records:
            fname = rec["file"]
            stem  = Path(fname).stem
            suffix = Path(fname).suffix

            with st.expander(f"❌  {fname}   —   {rec.get('latency_ms', '?')} ms"):
                col_img, col_detail = st.columns([1, 2])

                # Heatmap
                heatmap_path = (RESULTS_DIR / "heatmaps"
                                / f"{stem}_FAIL{suffix}")
                with col_img:
                    if heatmap_path.exists():
                        img = load_image(heatmap_path)
                        st.image(bgr_to_rgb(img), caption="Anomaly heatmap",
                                 use_container_width=True)
                    else:
                        st.caption("Heatmap not saved (re-run with 'Save heatmap images')")

                # Failure detail
                with col_detail:
                    st.markdown("**Failure reasons:**")
                    for reason in rec.get("reasons", []):
                        st.markdown(f"- `{reason}`")

                    roi_scores = rec.get("roi_scores", {})
                    if roi_scores:
                        st.markdown("**ROI Anomaly Scores:**")
                        rows = [{"ROI": k, "Score": round(v, 4)}
                                for k, v in sorted(roi_scores.items(),
                                                   key=lambda x: -x[1])]
                        st.dataframe(rows, use_container_width=True, height=200)

                    extra = []
                    if rec.get("barcode"):
                        extra.append(f"**Barcode decoded:** `{rec['barcode']}`")
                    if rec.get("lr_score") is not None:
                        extra.append(f"**L-R SSIM:** {rec['lr_score']:.4f}")
                    if rec.get("gm_score") is not None:
                        extra.append(f"**GM similarity:** {rec['gm_score']:.4f}")
                    if extra:
                        st.markdown("  \n".join(extra))

    st.divider()

    # ── Export ────────────────────────────────────────────────────────────────
    st.subheader("Export")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.download_button(
            label="Download all results (JSON)",
            data=json.dumps(records, indent=2),
            file_name="inspection_results.json",
            mime="application/json",
        )
    with col_exp2:
        st.download_button(
            label="Download rejections only (JSON)",
            data=json.dumps(fail_records, indent=2),
            file_name="rejection_log.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
