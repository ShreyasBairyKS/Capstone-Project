r"""
build_exe.py
─────────────────────────────────────────────────────────────────────────────
Builds the portable Windows app:

    <out>\Himalaya\Himalaya.exe          the program (and _internal\, its libraries)
    <out>\Himalaya\models\...            the models the default profile uses  (editable, not inside the exe)
    <out>\Himalaya\config\...            the ROI templates and roi_onnx_thresholds.json (editable)
    <out>\Himalaya\README.txt

    python himalaya_label_detection/template_roi_pipeline/desktop/build_exe.py --work D:\build\himalaya [--zip]

--work   scratch folder for the build environment and PyInstaller's files. Keep it OUT of OneDrive: it holds
         thousands of small files that would be synced for nothing.
--out    where the finished folder goes (default <work>\dist)
Takes a few minutes the first time (it installs the packages into a clean virtual environment, so nothing
unrelated ends up in the bundle) and about one minute after that.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DESKTOP = Path(__file__).resolve().parent
PKG_ROOT = DESKTOP.parent
REPO_ROOT = PKG_ROOT.parents[1]

README = """HIMALAYA LABEL INSPECTION  -  version {version}
================================================

Start it
    Double-click Himalaya.exe. A window opens after a few seconds (the four models are loading).
    Drop a full label scan on the window, or use "Inspect a folder" to check a whole folder and get a CSV.
    Close the window to quit. Nothing needs to be installed.

First start on a new PC
    Windows may say it "protected your PC" because the program is not signed. Choose More info, then Run anyway.
    The window needs the Microsoft Edge WebView2 runtime, which comes with Windows 10/11. Without it the app
    opens in your web browser instead.

Where things are
    models\\     one folder per region (ROI_1 ... ROI_4), each with weights\\model.onnx. To use another model,
                replace the file or point config\\roi_onnx_thresholds.json at another folder ("model_dirs").
    config\\     roi_onnx_thresholds.json holds the pass/fail limit of each region and which models to load;
                roi_templates\\ holds the pictures used to find the regions on a scan.
    Results     saved in %LOCALAPPDATA%\\Himalaya\\history (the app's "i" button shows the size and can clear it).
    Log         %LOCALAPPDATA%\\Himalaya\\logs\\app.log - send this file if something does not start.

Good to know
    A scan takes about 35 seconds on an ordinary CPU, one after another.
    The limits in config\\roi_onnx_thresholds.json are PROVISIONAL (fitted on 40 scans). Check them on scans the
    models were not trained on before relying on the verdicts.
"""


def run(cmd: list, **kw) -> None:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def models_to_copy(profile: dict) -> list[Path]:
    """The model folders the profile loads, relative to the repository root."""
    root = Path(profile.get("models_root", "models/gray_scale_rois"))
    names = ("ROI_1", "ROI_2", "ROI_3", "ROI_4")
    dirs = profile.get("model_dirs", {})
    out = []
    for n in names:
        rel = (root / dirs.get(n, n))
        resolved = Path(os.path.normpath(REPO_ROOT / rel)).relative_to(REPO_ROOT)   # collapses any "../x"
        if resolved not in out:
            out.append(resolved)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--zip", action="store_true", help="also write Himalaya-<version>.zip next to the folder")
    p.add_argument("--skip-install", action="store_true", help="reuse the environment as it is")
    args = p.parse_args()

    sys.path.insert(0, str(PKG_ROOT.parent))
    from template_roi_pipeline.desktop.version import __version__

    work = args.work.resolve()
    if "onedrive" in str(work).lower():
        print("warning: --work is inside OneDrive; the build creates many small files there.")
    out = (args.out or work / "dist").resolve()
    venv = work / "venv"
    py = venv / "Scripts" / "python.exe"
    work.mkdir(parents=True, exist_ok=True)

    print("1/4 build environment")
    if not py.exists():
        run([sys.executable, "-m", "venv", venv])
    if not args.skip_install:
        run([py, "-m", "pip", "install", "--disable-pip-version-check", "-r", DESKTOP / "requirements-desktop.txt"])

    print("2/4 PyInstaller")
    run([py, "-m", "PyInstaller", DESKTOP / "himalaya.spec", "--noconfirm", "--clean",
         "--distpath", out, "--workpath", work / "build"], cwd=work)
    app = out / "Himalaya"
    if not (app / "Himalaya.exe").is_file():
        print("the exe was not produced")
        return 1

    print("3/4 models and config next to the exe")
    cfg_src = PKG_ROOT / "config"
    profile = json.loads((cfg_src / "roi_onnx_thresholds.json").read_text(encoding="utf-8"))
    for rel in models_to_copy(profile):
        src, dst = REPO_ROOT / rel, app / rel
        if not (src / "weights" / "model.onnx").is_file():
            print(f"missing model: {src}")
            return 1
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"  {rel}  ({(dst / 'weights' / 'model.onnx').stat().st_size / 1e6:.0f} MB)")
    (app / "config").mkdir(exist_ok=True)
    shutil.copytree(cfg_src / "roi_templates", app / "config" / "roi_templates", dirs_exist_ok=True)
    shutil.copy2(cfg_src / "roi_onnx_thresholds.json", app / "config" / "roi_onnx_thresholds.json")
    (app / "README.txt").write_text(README.format(version=__version__), encoding="utf-8")

    print("4/4 summary")
    total = sum(f.stat().st_size for f in app.rglob("*") if f.is_file())
    print(f"  {app}\n  {total / 1e6:.0f} MB in {sum(1 for f in app.rglob('*') if f.is_file())} files")
    if args.zip:
        archive = out / f"Himalaya-{__version__}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in sorted(app.rglob("*")):
                if f.is_file():
                    z.write(f, Path("Himalaya") / f.relative_to(app))
        print(f"  {archive}  ({archive.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
