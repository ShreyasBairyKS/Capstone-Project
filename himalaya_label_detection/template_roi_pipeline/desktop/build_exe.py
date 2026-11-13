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
import hashlib
import json
import math
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
    Extract the entire ZIP first. Keep Himalaya.exe, _internal, models and config together.
    Do not run the program from inside the ZIP or send only the EXE.
    Double-click Himalaya.exe. A window opens after a few seconds (the four models are loading).
    Drop a full label scan on the window, or use "Inspect a folder" to check a whole folder and get a CSV.
    Close the window to quit. Nothing needs to be installed.

First start on a new PC
    Use 64-bit Windows 10/11. No Python, model download, administrator access or package installation is needed.
    The program is unsigned, so Windows SmartScreen or your organization's security policy may block it.
    Ask your IT team to approve the supplied checksum; do not disable antivirus or bypass organization policy.
    The native window uses Microsoft Edge WebView2, which is commonly preinstalled but not guaranteed.
    If it is unavailable the app opens in your existing web browser instead; keep the small running dialog open
    until you finish. The app does not install or download a runtime.

Where things are
    models\\     one folder per region (ROI_1 ... ROI_4), each with weights\\model.onnx. To use another model,
                replace the file or point config\\roi_onnx_thresholds.json at another folder ("model_dirs").
    config\\     roi_onnx_thresholds.json holds the pass/fail limit of each region and which models to load;
                rejection_margin is the safety band below each limit (currently {rejection_margin:.4f}).
                roi_templates\\ holds the pictures used to find the regions on a scan.
    Results     saved in %LOCALAPPDATA%\\Himalaya\\history (the app's "i" button shows the size and can clear it).
    Log         %LOCALAPPDATA%\\Himalaya\\logs\\app.log - send this file if something does not start.

Good to know
    A region is rejected above its original threshold. With a positive rejection margin it is also rejected
    at or above (threshold - rejection_margin). For example, 0.0033 below a threshold is FAIL with margin 0.0050.
    The margin is a conservative acceptance rule, not a confidence probability. Thresholds and weights are unchanged.
    Scans run one after another on the CPU; processing time depends on the computer.
    Upload full original label scans, normally 1504 x 8000 pixels, not isolated crops or phone photos.
    The application works locally. Images are not sent to a cloud service; saved results contain image data.
    The current models and limits are for EVALUATION, not production quality approval.
    Base limits were fitted on 40 training scans, not held-out data. PASS does not guarantee no defect.
    JSON and CSV exports include the provisional-limit warning. Validate on independently labelled, unseen scans
    and obtain the organization's acceptance before relying on automated pass/fail decisions.
    checksums.sha256 lists the exact delivered files; release.json records the runtime versions and model setup.
"""


def run(cmd: list, **kw) -> None:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def models_to_copy(profile: dict) -> list[Path]:
    """The model folders the profile loads, relative to the repository root."""
    root = Path(profile.get("models_root", "models/gray_scale_rois"))
    names = ("ROI_1", "ROI_2", "ROI_3", "ROI_4")
    dirs = profile.get("model_dirs", {})
    if root.is_absolute() or any(Path(d).is_absolute() for d in dirs.values()):
        raise ValueError("Portable builds require relative models_root and model_dirs paths.")
    out = []
    for n in names:
        rel = (root / dirs.get(n, n))
        resolved = Path(os.path.normpath(REPO_ROOT / rel)).relative_to(REPO_ROOT)   # collapses any "../x"
        if resolved not in out:
            out.append(resolved)
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--zip", action="store_true", help="also write Himalaya-<version>.zip next to the folder")
    p.add_argument("--skip-install", action="store_true", help="reuse the environment as it is")
    p.add_argument("--venv", type=Path, help="reuse an existing virtual environment, e.g. the repository .venv")
    p.add_argument("--evaluation", type=Path, help="include a run_template_inference.py JSON evaluation in the handoff summary")
    args = p.parse_args()
    if sys.platform != "win32":
        p.error("Build the Windows executable on Windows.")

    sys.path.insert(0, str(PKG_ROOT.parent))
    from template_roi_pipeline.desktop.version import __version__

    work = args.work.resolve()
    if "onedrive" in str(work).lower():
        print("warning: --work is inside OneDrive; the build creates many small files there.")
    out = (args.out or work / "dist").resolve()
    venv = args.venv.resolve() if args.venv else work / "venv"
    py = venv / "Scripts" / "python.exe"
    if args.venv and not py.is_file():
        p.error(f"The supplied virtual environment has no Python interpreter: {py}")

    cfg_src = PKG_ROOT / "config"
    profile = json.loads((cfg_src / "roi_onnx_thresholds.json").read_text(encoding="utf-8"))
    model_dirs = models_to_copy(profile)
    if set(profile.get("thresholds", {})) != {"ROI_1", "ROI_2", "ROI_3", "ROI_4"} or any(
            not math.isfinite(float(v)) for v in profile["thresholds"].values()):
        p.error("The release profile must have four finite thresholds.")
    rejection_margin = float(profile.get("rejection_margin", 0.0))
    if not math.isfinite(rejection_margin) or rejection_margin < 0:
        p.error("The rejection margin must be finite and non-negative.")
    for rel in model_dirs:
        model = REPO_ROOT / rel / "weights" / "model.onnx"
        if not model.is_file():
            p.error(f"Missing model: {model}")
        meta = REPO_ROOT / rel / "model_meta.json"
        if meta.is_file():
            expected = json.loads(meta.read_text(encoding="utf-8")).get("onnx_export", {}).get("onnx_sha256")
            if expected and sha256(model) != expected:
                p.error(f"The model does not match its provenance checksum: {model}")
    for name in ("roi_1", "roi_2", "roi_3", "roi_4"):
        if not (cfg_src / "roi_templates" / f"{name}_template.png").is_file():
            p.error(f"Missing template: {name}")
    evaluation = None
    if args.evaluation:
        records = json.loads(args.evaluation.read_text(encoding="utf-8"))
        if not records or any(r.get("label") not in ("good", "bad") for r in records):
            p.error("The evaluation must contain labelled good/bad inspection results.")
        if any(abs(v["threshold"] - profile["thresholds"][n]) > 0.00000051 for r in records for n, v in r["rois"].items()):
            p.error("The evaluation used different thresholds than this release.")
        if any(v.get("rejection_margin", 0.0) != rejection_margin for r in records for v in r["rois"].values()):
            p.error("The evaluation used a different rejection margin than this release.")
        evaluation = {
            "source_results_sha256": sha256(args.evaluation),
            "rejection_margin": rejection_margin,
            "dataset_note": "Local NSC folders with image-level labels; includes training/calibration data, NOT held-out validation.",
            "true_positive": sum(r["label"] == "bad" and r["verdict"] == "FAIL" for r in records),
            "false_negative": sum(r["label"] == "bad" and r["verdict"] == "PASS" for r in records),
            "false_positive": sum(r["label"] == "good" and r["verdict"] == "FAIL" for r in records),
            "true_negative": sum(r["label"] == "good" and r["verdict"] == "PASS" for r in records),
            "missing_rois": sum(v["status"] == "MISSING" for r in records for v in r["rois"].values()),
            "near_threshold_rejections": sum(v.get("rejection_reason") == "near_threshold" for r in records for v in r["rois"].values()),
            "missed_bad_images": [Path(r["image_path"]).name for r in records if r["label"] == "bad" and r["verdict"] == "PASS"],
            "rejected_good_images": [Path(r["image_path"]).name for r in records if r["label"] == "good" and r["verdict"] == "FAIL"],
        }
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
    for rel in model_dirs:
        src, dst = REPO_ROOT / rel, app / rel
        if not (src / "weights" / "model.onnx").is_file():
            print(f"missing model: {src}")
            return 1
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.ckpt", "*.pt", "*.pth", "__pycache__"))
        print(f"  {rel}  ({(dst / 'weights' / 'model.onnx').stat().st_size / 1e6:.0f} MB)")
    (app / "config").mkdir(exist_ok=True)
    shutil.copytree(cfg_src / "roi_templates", app / "config" / "roi_templates", dirs_exist_ok=True)
    shutil.copy2(cfg_src / "roi_onnx_thresholds.json", app / "config" / "roi_onnx_thresholds.json")
    readme = README.format(version=__version__, rejection_margin=rejection_margin)
    if evaluation:
        readme += (f"\nLatest local scan check\n"
                   f"    {evaluation['true_positive']} defective scans detected; {evaluation['false_negative']} defective scans MISSED.\n"
                   f"    {evaluation['true_negative']} good scans passed; {evaluation['false_positive']} good scans incorrectly rejected.\n"
                   "    These scans include training/calibration data. This is not independent accuracy validation.\n"
                   "    Do not use the current build for automated production acceptance. Details are in release.json.\n")
    (app / "README.txt").write_text(readme, encoding="utf-8")

    packages = ["numpy", "opencv-python-headless", "onnxruntime", "fastapi", "starlette", "uvicorn",
                "python-multipart", "pydantic", "pywebview", "pythonnet", "clr_loader", "pyinstaller"]
    environment = json.loads(subprocess.check_output([
        str(py), "-c", "import json, platform, importlib.metadata as m; print(json.dumps({'python': platform.python_version(), "
        "'architecture': platform.machine(), 'packages': {p: m.version(p) for p in " + repr(packages) + "}}))"], text=True))
    (app / "release.json").write_text(json.dumps({"version": __version__, "platform": "Windows x64",
                                                "environment": environment, "profile": profile,
                                                "evaluation": evaluation}, indent=2), encoding="utf-8")
    files = sorted(f for f in app.rglob("*") if f.is_file() and f.name != "checksums.sha256")
    (app / "checksums.sha256").write_text("".join(f"{sha256(f)}  {f.relative_to(app).as_posix()}\n" for f in files), encoding="utf-8")

    print("4/4 summary")
    total = sum(f.stat().st_size for f in app.rglob("*") if f.is_file())
    print(f"  {app}\n  {total / 1e6:.0f} MB in {sum(1 for f in app.rglob('*') if f.is_file())} files")
    if args.zip:
        archive = out / f"Himalaya-{__version__}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in sorted(app.rglob("*")):
                if f.is_file():
                    z.write(f, Path("Himalaya") / f.relative_to(app))
        with zipfile.ZipFile(archive) as z:
            corrupt = z.testzip()
            if corrupt:
                raise RuntimeError(f"The release archive failed its integrity check: {corrupt}")
        archive.with_suffix(".zip.sha256").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
        print(f"  {archive}  ({archive.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
