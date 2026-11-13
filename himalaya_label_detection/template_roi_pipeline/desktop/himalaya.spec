# -*- mode: python ; coding: utf-8 -*-
# PyInstaller recipe for Himalaya.exe. Run through build_exe.py, which prepares the clean environment,
# calls this, then puts models/ and config/ next to the exe (they stay outside the bundle on purpose).
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

SPEC_DIR = Path(SPECPATH)                       # .../template_roi_pipeline/desktop
PKG = SPEC_DIR.parent                           # .../template_roi_pipeline
SRC = PKG.parent                                # .../himalaya_label_detection (holds the package)
sys.path.insert(0, str(SRC))

hidden = (
    collect_submodules("template_roi_pipeline", filter=lambda n: ".tests" not in n and ".scripts" not in n)
    + collect_submodules("uvicorn")             # uvicorn picks its loops/protocols by name at run time
    + ["multipart", "python_multipart"]
)
datas = [(str(PKG / "app" / "static"), "template_roi_pipeline/app/static")]

a = Analysis(
    [str(SPEC_DIR / "launcher.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "pandas", "PIL", "IPython", "pytest", "torch", "tensorflow",
              "PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy", "gi", "cefpython3", "wx"],
    noarchive=False,
)
# Only Edge WebView2 and CPU inference are used. A shared venv may also contain leftover GPU provider DLLs.
a.binaries = [b for b in a.binaries if not any(name in b[0].lower() for name in
              ("opencv_videoio_ffmpeg", "onnxruntime_providers_cuda", "onnxruntime_providers_tensorrt"))]
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Himalaya",
    console=False,                              # a window app: no terminal; the log is in %LOCALAPPDATA%\Himalaya\logs
    icon=str(SPEC_DIR / "himalaya.ico"),
    upx=False,                                  # UPX only slows start-up and makes antivirus tools suspicious
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Himalaya")
