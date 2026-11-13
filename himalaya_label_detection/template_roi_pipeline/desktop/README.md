# Himalaya desktop app (Windows)

The inspection server and page from `app/`, packaged as a program you double-click: no Python installation or terminal.
`Himalaya.exe` loads four models, starts a local server, and displays its own window using Microsoft Edge WebView2.
WebView2 is commonly preinstalled, but not guaranteed; without it, the app uses the existing web browser instead.
Extract the entire ZIP before starting. Send the ZIP, not the EXE alone. Closing the native window ends the app.

```
Himalaya\
  Himalaya.exe            the program
  _internal\              its libraries (Python, OpenCV, ONNX Runtime, ...)
  models\gray_scale_rois\ ROI_1, ROI_2, ROI_3, new_roi4   (weights\model.onnx each)   <- editable
  config\                 roi_onnx_thresholds.json and roi_templates\                 <- editable
  README.txt              the short version for whoever receives the folder
  release.json            runtime versions, decision profile and measured evaluation results
  checksums.sha256        hashes of the delivered files
```

`models/` and `config/` sit next to the exe on purpose: to use another model, replace its `weights/model.onnx` (or
point `config/roi_onnx_thresholds.json` at another folder with `model_dirs`) and restart. Nothing is rebuilt.

## Decision Rule

Version 1.0.2 retains all four original thresholds and model weights. The profile adds `"rejection_margin": 0.005`.
An ROI fails above its threshold, and also at or above `threshold - rejection_margin` when the margin is positive.
For example, a score `0.0033` below its original limit now fails. With margin zero, the original rule applies:
only a score strictly above the threshold fails. Missing ROIs still fail the label.

The UI identifies these cases as "too close to the limit", and displays both the original limit and acceptance
cutoff. This is an absolute score margin, not a confidence probability. The on-screen provisional-limit banner
has been removed; configuration provenance remains in JSON/CSV reports. Changing the margin invalidates cached
answers. Historical reports keep the settings and verdict originally used; re-upload to apply the new rule.

The unchanged-model evaluation on all 120 local NSC scans gave:

| Rule | Bad Scans Detected | Bad Scans Missed | Good Scans Rejected | Good Scans Passed |
|---|---:|---:|---:|---:|
| Original thresholds | 32/42 | 10/42 | 9/78 | 69/78 |
| With 0.005 margin | 37/42 | 5/42 | 18/78 | 60/78 |

All 480 ROIs were located. The remaining missed bad scans are `21.bmp`, `22.bmp`, `23.bmp`, `36.bmp`, and `37.bmp`.
These folders include training/calibration data; this is not independent validation. The margin reduces missed
defects at the cost of more false rejections. The ZIP is suitable for supervised evaluation, not production sign-off.

## What the desktop version adds to the web app

| | |
|---|---|
| **Saved results** | Every finished scan is written to `%LOCALAPPDATA%\Himalaya\history` (newest 100 are kept; `HIMALAYA_MAX_HISTORY` changes that), so the list and every result page survive a restart. Scans of the same file with the same models and limits are answered from it. The "i" button shows the size and can clear it. |
| **Inspect a folder** | Pick a folder and every BMP/PNG/JPEG/TIFF in it is inspected one after another. The page shows the rows filling in with each region's score, can stop the run, and downloads a CSV (`file, verdict, reasons, seconds, error`, then status/score/limit per region). A scan uploaded meanwhile waits for at most one folder item. A run cut short by closing the app is marked "Interrupted". |
| **About / models** | The "i" button lists the model and limit of every region, the limits file, the folders in use (with Open buttons) and the log. |
| **One copy at a time** | A second launch says the app is already running instead of loading another 320 MB of models. |
| **Guards** | The server listens on 127.0.0.1 only, refuses requests whose `Host` header is not `127.0.0.1`/`localhost` (DNS rebinding), and the routes that act on the machine (folder runs, opening folders, clearing results) need a header only the app's own page sends. |

If the window cannot open (WebView2 missing) the app opens in your web browser instead and says so.
The log, `%LOCALAPPDATA%\Himalaya\logs\app.log`, is the first place to look if it does not start; start-up problems
a user can fix (a missing models folder, say) are shown in the window instead of a traceback.

## Build it

From the repository root, use the existing `.venv` (tested with 64-bit Python 3.10 on Windows):

```powershell
& ".\.venv\Scripts\python.exe" "himalaya_label_detection\template_roi_pipeline\desktop\build_exe.py" --venv ".\.venv" --skip-install --work "$env:TEMP\kilo\himalaya-build" --out "himalaya_label_detection\template_roi_pipeline\desktop\dist" --zip
```

`--venv` selects the existing environment; no global packages are installed. Omit `--skip-install` to install
`requirements-desktop.txt` into that environment. `--work` holds PyInstaller's scratch files; keep it outside
OneDrive. The output is `dist/Himalaya` (about 536 MB) plus `Himalaya-1.0.2.zip` (about 386 MB) and its SHA-256 file.
Add `--evaluation <JSON>` to embed results generated by `scripts/run_template_inference.py --json-out <JSON>`.
The builder checks model hashes against their metadata, validates the decision profile, and checks ZIP integrity.

The script copies exactly the model folders the default profile loads (`models_to_copy`), so a different default
profile builds a different app. The result is far too large to commit; share the ZIP through an approved file-transfer
service or a release attachment. Do not publish model weights or client scans without permission.

Run it from source without building (same code, same window):

```powershell
& ".\.venv\Scripts\python.exe" "himalaya_label_detection\template_roi_pipeline\desktop\launcher.py"
& ".\.venv\Scripts\python.exe" "himalaya_label_detection\template_roi_pipeline\desktop\launcher.py" --no-window
& ".\.venv\Scripts\python.exe" "himalaya_label_detection\template_roi_pipeline\desktop\launcher.py" --no-browser --port 8010
```

`--home`, `--thresholds` and `--data-dir` change where models/config, the limits file and the saved results are
(`HIMALAYA_HOME` / `HIMALAYA_DATA_DIR` do the same from the environment).

## Files

| File | Role |
|---|---|
| `launcher.py` | entry point: models, server thread on a free port, the window, single instance, logging |
| `paths.py` | where models, config and saved data are, from source or packaged |
| `himalaya.spec`, `build_exe.py` | the PyInstaller recipe and the script that runs it and lays out the folder |
| `requirements-desktop.txt` | pinned packages for the build environment |
| `make_icon.py`, `himalaya.ico` | draws the app icon |
| `version.py` | the version shown in the About dialog |
| `../app/history.py` | the on-disk store for saved results |

## What has and has not been checked

* Checked: 102 model-free tests, including positive/negative thresholds, exact rejection-band boundaries,
  invalid numeric output, changed-margin cache invalidation, persistence, mixed/error folder runs, and CSV/JSON consistency.
* `../scripts/verify_desktop_release.py` verifies the ZIP and every shipped checksum, relocates the app to a Unicode
  path, removes Python paths from the child environment, and compares real model results against source before and
  after restart. It includes bad scan 24 changing from original-rule PASS to rejection-margin FAIL.
* A clean Windows installation without Python or WebView2, and Windows 10 specifically, still require a separate check.
* Windows SmartScreen or organization policy may block the unsigned EXE. Signing requires a code-signing certificate;
  ask IT to approve the artifact rather than disabling security controls.
