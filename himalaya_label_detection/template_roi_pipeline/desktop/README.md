# Himalaya desktop app (Windows)

The inspection server and page from `app/`, packaged as a program you double-click: no Python, no terminal, no
browser. `Himalaya.exe` loads the four models, starts the server on a free port of this machine, and shows the page in
a window of its own (Microsoft Edge WebView2, which ships with Windows 10/11). Closing the window ends everything.

```
Himalaya\
  Himalaya.exe            the program
  _internal\              its libraries (Python, OpenCV, ONNX Runtime, ...)
  models\gray_scale_rois\ ROI_1, ROI_2, ROI_3, new_roi4   (weights\model.onnx each)   <- editable
  config\                 roi_onnx_thresholds.json and roi_templates\                 <- editable
  README.txt              the short version for whoever receives the folder
```

`models/` and `config/` sit next to the exe on purpose: to use another model, replace its `weights/model.onnx` (or
point `config/roi_onnx_thresholds.json` at another folder with `model_dirs`) and restart. Nothing is rebuilt.

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

From the repository root (needs Python 3.11 and internet access the first time, for the packages):

```bash
python himalaya_label_detection/template_roi_pipeline/desktop/build_exe.py --work D:/build/himalaya --zip
```

`--work` is scratch space for a clean virtual environment (`requirements-desktop.txt`, pinned to the versions the
pipeline was verified with) and PyInstaller's files. Keep it out of OneDrive. The finished folder is
`<work>/dist/Himalaya` (about 520 MB), `--zip` adds `Himalaya-<version>.zip`. The first build takes several minutes,
later ones about one (`--skip-install` reuses the environment).

The script copies exactly the model folders the default profile loads (`models_to_copy`), so a different default
profile builds a different app. The build does not touch the repository, and the result is far too large to commit:
publish it as a GitHub Release asset.

Run it from source without building (same code, same window):

```bash
python himalaya_label_detection/template_roi_pipeline/desktop/launcher.py                 # window
python himalaya_label_detection/template_roi_pipeline/desktop/launcher.py --no-window     # default browser
python himalaya_label_detection/template_roi_pipeline/desktop/launcher.py --no-browser --port 8010   # just serve
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

* Checked: 52 model-free tests (the new ones cover saved results, folder runs, the guards, the launcher's start-up);
  the built exe starting, serving, opening its window, and scoring good scan 1 and bad scan 6 with exactly the
  scores of the source version; the folder-run, About and saved-results pages in a browser.
* Not checked: the native folder dialog and the report downloads inside the window (they need a person at the
  screen); a PC without Python or an existing WebView2 (Windows Sandbox or a second PC would do); Windows 10.
* Windows SmartScreen will warn about the unsigned exe on first run; signing needs a code-signing certificate.
