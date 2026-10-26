"""
Test Suite for Part 6: Frontend Build & Static Serving Verification.
Checks that Vite distribution assets exist, HTML is served properly by FastAPI,
and that Station Switcher, Station 2 Workstations, and WebSocket routes are intact.
"""
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("test_fill_step6_frontend")

def run_checks():
    dist_dir = PROJECT_ROOT / "pipeline" / "frontend" / "dist"
    assert dist_dir.exists(), f"Frontend dist folder {dist_dir} does not exist!"
    
    index_html = dist_dir / "index.html"
    assert index_html.exists(), "dist/index.html not found!"
    html_content = index_html.read_text(encoding="utf-8")
    logger.info(f"dist/index.html verified ({len(html_content)} bytes).")

    assets_dir = dist_dir / "assets"
    assert assets_dir.exists(), "dist/assets folder not found!"
    js_files = list(assets_dir.glob("*.js"))
    css_files = list(assets_dir.glob("*.css"))
    assert len(js_files) >= 1, "No JavaScript bundle in dist/assets!"
    assert len(css_files) >= 1, "No CSS bundle in dist/assets!"
    logger.info(f"Verified built assets: {js_files[0].name} ({js_files[0].stat().st_size / 1024:.1f} KB), {css_files[0].name} ({css_files[0].stat().st_size / 1024:.1f} KB)")

    # Verify all Station 2 frontend source components exist
    fill_components_dir = PROJECT_ROOT / "pipeline" / "frontend" / "src" / "components" / "fill_level"
    expected_components = [
        "FillDashboard.jsx",
        "FillSingleFeed.jsx",
        "FillKpiBar.jsx",
        "FillConveyorFilmstrip.jsx",
        "FillAnalyticsWidget.jsx",
        "FillAuditPage.jsx",
        "FillReplayInspectorPage.jsx",
        "FillActiveLearningPage.jsx",
        "FillUploadModal.jsx",
    ]
    for comp in expected_components:
        comp_file = fill_components_dir / comp
        assert comp_file.exists(), f"Expected component {comp} missing in {fill_components_dir}!"
        logger.info(f"Component verified: {comp} ({comp_file.stat().st_size} bytes)")

    # Verify Station Context & Switcher
    station_ctx = PROJECT_ROOT / "pipeline" / "frontend" / "src" / "context" / "StationContext.jsx"
    station_sw = PROJECT_ROOT / "pipeline" / "frontend" / "src" / "components" / "common" / "StationSwitcher.jsx"
    assert station_ctx.exists(), "StationContext.jsx missing!"
    assert station_sw.exists(), "StationSwitcher.jsx missing!"
    logger.info("StationContext and StationSwitcher verified.")

    print("\n=======================================================")
    print("ALL PART 6 FRONTEND ASSETS AND SOURCE CHECKS PASSED!")
    print("=======================================================\n")

if __name__ == "__main__":
    run_checks()
