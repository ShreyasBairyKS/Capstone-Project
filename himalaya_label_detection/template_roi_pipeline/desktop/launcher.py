r"""
launcher.py
─────────────────────────────────────────────────────────────────────────────
Himalaya desktop app. Loads the models, starts the inspection server on a free port of this machine,
and shows its page in a window of its own (Edge WebView2). Closing the window ends everything.

    Himalaya.exe                      the packaged app
    python .../desktop/launcher.py    the same, from source

    --no-window      serve and open the default browser instead of the window (also the fallback when the
                     WebView2 runtime is missing); --no-browser serves without opening anything
    --port N         fixed port instead of a free one
    --home DIR       folder holding models/ and config/ (default: next to the exe, or the repository root)
    --thresholds F   a different profile than <home>/config/roi_onnx_thresholds.json
    --data-dir DIR   where saved results and logs go (default %LOCALAPPDATA%\Himalaya)

A log is written to <data dir>/logs/app.log: it is the first place to look when something does not start.
"""

from __future__ import annotations

import argparse
import ctypes
import html
import json
import logging
import logging.handlers
import multiprocessing
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

if not getattr(sys, "frozen", False):                       # running from source: put himalaya_label_detection/ on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Bound decoding before OpenCV is imported, including compressed oversized images.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "32000000")

from template_roi_pipeline.desktop import paths  # noqa: E402
from template_roi_pipeline.desktop.version import __version__  # noqa: E402

TITLE = "Himalaya Label Inspection"
log = logging.getLogger("himalaya")


class AppError(Exception):
    """A problem the user can fix (a missing folder, say); shown as it is, without a traceback."""


# ── Process-level setup ───────────────────────────────────────────────────────

class _LogStream:
    """stdout/stderr of a windowed exe are None; anything printed (or any library warning) goes to the log instead."""

    def write(self, text: str) -> int:
        for line in text.rstrip().splitlines():
            log.info(line)
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def setup_logging(data: Path) -> Path:
    (data / "logs").mkdir(parents=True, exist_ok=True)
    path = data / "logs" / "app.log"
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    if sys.stdout is None or sys.stderr is None:
        sys.stdout = sys.stderr = _LogStream()
    return path


def message_box(text: str, title: str = TITLE) -> None:
    if sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(0, text, title, 0x40)          # blocks until dismissed
    else:
        print(f"{title}: {text}")


_mutex = None


def already_running() -> bool:
    """One copy at a time: a second would load another 320 MB of models and fight for the CPU."""
    global _mutex
    if sys.platform != "win32":
        return False
    _mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\HimalayaLabInspection")
    return ctypes.windll.kernel32.GetLastError() == 183                 # ERROR_ALREADY_EXISTS


def open_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(str(path))                                         # noqa: S606 - a folder the launcher named
    else:
        webbrowser.open(path.as_uri())


# ── Models + server ───────────────────────────────────────────────────────────

class Runtime:
    """Loads the pipeline and runs the server on a socket of this machine."""

    def __init__(self, args: argparse.Namespace, data: Path):
        self.args, self.data = args, data
        self.app = None
        self.url: str | None = None
        self._server = None
        self._thread: threading.Thread | None = None

    def load(self):
        from template_roi_pipeline.app.server import create_app
        from template_roi_pipeline.inspector import TemplateROIInspector

        args = self.args
        home = Path(args.home).expanduser().resolve() if args.home else paths.app_home()
        cfg = paths.config_dir(home)
        thr_path = Path(args.thresholds) if args.thresholds else cfg / "roi_onnx_thresholds.json"
        if not thr_path.is_file():
            raise AppError(f"The limits file is missing:\n{thr_path}\n\nIt belongs in the 'config' folder next to the app.")
        try:
            doc = json.loads(thr_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AppError(f"The limits file could not be read:\n{thr_path}\n\n{exc}") from exc
        if not isinstance(doc, dict):
            raise AppError(f"The limits file must contain a JSON object:\n{thr_path}")
        models_root = home / doc.get("models_root", "models/gray_scale_rois")
        if not models_root.is_dir():
            raise AppError(f"The models folder is missing:\n{models_root}\n\n"
                           "It holds one folder per region (ROI_1 ... ROI_4) with weights/model.onnx inside.")
        log.info("home=%s config=%s profile=%s models=%s", home, cfg, thr_path.name, models_root)
        try:
            inspector = TemplateROIInspector.from_config(
                cfg, models_root, thresholds_file=thr_path, input_mode=doc.get("input_mode", "gray"),
                score_stat=doc.get("score_stat", "mean"))
        except FileNotFoundError as exc:
            raise AppError(f"A model or template file is missing:\n{exc}\n\nModels folder: {models_root}") from exc
        except ValueError as exc:
            raise AppError(f"The inspection settings are invalid:\n{exc}\n\nLimits file: {thr_path}") from exc
        for name, clf in inspector.classifiers.items():
            path = getattr(clf, "model_path", None)
            log.info("%s: model '%s'", name, path.parent.parent.name if path else clf)

        history = self.data / "history"
        self.app = create_app(
            inspector, doc, str(doc.get("models_root", "")), samples={},
            history_dir=history, max_history=int(os.environ.get("HIMALAYA_MAX_HISTORY", "100")),
            app_info={"version": __version__, "frozen": paths.FROZEN, "profile": thr_path.name},
            folders={"home": home, "models": models_root, "results": history, "logs": self.data / "logs"},
            open_folder=open_folder, allowed_hosts=["127.0.0.1", "localhost"])
        return self.app

    def start(self) -> str:
        import uvicorn

        if self.app is None:
            self.load()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", self.args.port))                         # port 0: the system picks a free one
        sock.listen(16)
        config = uvicorn.Config(self.app, log_config=None, log_level="warning", access_log=False)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, kwargs={"sockets": [sock]}, daemon=True, name="server")
        self._thread.start()
        deadline = time.time() + 30
        while not self._server.started:
            if not self._thread.is_alive() or time.time() > deadline:
                raise AppError("The inspection server did not start. See the log for details.")
            time.sleep(0.05)
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}/"
        log.info("serving %s", self.url)
        return self.url

    def busy(self) -> bool:
        return bool(self.app and self.app.state.is_busy())

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)


# ── Window ────────────────────────────────────────────────────────────────────

_PAGE_CSS = """
  :root { color-scheme: light dark; }
  body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #f3f5f9; color: #0e1520;
         font: 15px/1.55 "Segoe UI", system-ui, sans-serif; }
  @media (prefers-color-scheme: dark) { body { background: #090c11; color: #e8ecf3; } .sub { color: #8995a7 !important; } }
  .box { text-align: center; padding: 24px; max-width: 640px; }
  .mark { width: 64px; height: 64px; border-radius: 18px; margin: 0 auto 22px; display: grid; place-items: center;
          background: linear-gradient(140deg, #72b4ff, #2160e8 70%); box-shadow: 0 12px 28px -10px rgba(33,96,232,.75); }
  .mark svg { width: 38px; height: 38px; fill: #fff; }
  h1 { font-size: 22px; letter-spacing: -.02em; margin: 0 0 6px; }
  .sub { color: #647084; margin: 0; }
"""
_MARK = '<div class="mark"><svg viewBox="0 0 24 24"><path d="M2 20 9.5 6l4 7 2.5-4L22 20Z"/></svg></div>'


def splash_html() -> str:
    return f"""<!doctype html><meta charset="utf-8"><title>{TITLE}</title><style>{_PAGE_CSS}
  .bar {{ width: 220px; height: 5px; border-radius: 9px; background: rgba(120,130,150,.25); margin: 26px auto 0; overflow: hidden; }}
  .bar i {{ display: block; width: 40%; height: 100%; border-radius: 9px; background: #2160e8; animation: go 1.2s ease-in-out infinite; }}
  @keyframes go {{ from {{ margin-left: -40%; }} to {{ margin-left: 100%; }} }}
</style><div class="box">{_MARK}<h1>Himalaya label inspection</h1><p class="sub">Loading the models…</p><div class="bar"><i></i></div></div>"""


def error_html(message: str, log_path: Path) -> str:
    return f"""<!doctype html><meta charset="utf-8"><title>{TITLE}</title><style>{_PAGE_CSS}
  pre {{ text-align: left; white-space: pre-wrap; background: rgba(201,42,42,.09); border: 1px solid rgba(201,42,42,.32);
        border-radius: 12px; padding: 14px 16px; font: 13px/1.5 Consolas, monospace; }}
</style><div class="box">{_MARK}<h1>Himalaya could not start</h1><pre>{html.escape(message)}</pre>
<p class="sub">Details are in the log:<br>{html.escape(str(log_path))}</p></div>"""


class WindowApi:
    """What the page may ask of the desktop shell (the page is ours and loads only from 127.0.0.1)."""

    def __init__(self):
        self._window = None                     # underscore: pywebview exposes (and walks) every public attribute to the page

    def choose_folder(self):
        import webview

        kind = webview.FileDialog.FOLDER if hasattr(webview, "FileDialog") else webview.FOLDER_DIALOG
        picked = self._window.create_file_dialog(kind)
        return picked[0] if picked else None


def run_window(rt: Runtime, log_path: Path) -> int:
    import webview

    try:
        webview.settings["ALLOW_DOWNLOADS"] = True                       # the report buttons are <a download>
    except Exception:                                                    # older/newer pywebview without the setting
        log.warning("pywebview has no ALLOW_DOWNLOADS setting")
    api = WindowApi()
    window = webview.create_window(TITLE, html=splash_html(), js_api=api, width=1280, height=860,
                                   min_size=(900, 620), background_color="#090c11",
                                   text_select=True, zoomable=True)          # paths and scores must be copyable
    api._window = window

    def on_closing():
        if rt.busy() and not window.create_confirmation_dialog(TITLE, "A scan is still running. Quit anyway?"):
            return False
        return None

    window.events.closing += on_closing

    def boot():
        if webview.renderer != "edgechromium":
            log.warning("WebView2 is unavailable; refusing the legacy Internet Explorer renderer")
            window.destroy()
            return
        try:
            window.load_url(rt.start())
        except AppError as exc:
            log.error("start-up failed: %s", exc)
            window.load_html(error_html(str(exc), log_path))
        except Exception as exc:                                         # the log has the traceback
            log.exception("start-up failed")
            window.load_html(error_html(f"{type(exc).__name__}: {exc}", log_path))

    try:
        webview.start(boot, gui="edgechromium", storage_path=str(rt.data / "webview"), private_mode=False)
        if webview.renderer == "edgechromium":
            return 0
    except Exception:
        log.exception("could not open the window (is the Edge WebView2 runtime installed?)")
    return run_browser(rt, log_path, reason="The app window could not be opened (the Microsoft Edge WebView2 "
                       "runtime may be missing), so the app was opened in your web browser instead.\n\n")


def run_browser(rt: Runtime, log_path: Path, reason: str = "") -> int:
    try:
        url = rt.url or rt.start()
    except AppError as exc:
        message_box(str(exc))
        return 1
    if not rt.args.no_browser:
        webbrowser.open(url)
    if rt.args.no_browser:                                               # headless: keep serving until killed
        log.info("headless; waiting")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return 0
    message_box(f"{reason}Himalaya is running at {url}\n\nKeep this message open while inspecting images in your browser.\n"
                "Click OK only when finished to quit Himalaya.")
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--home", help="folder holding models/ and config/")
    p.add_argument("--thresholds", help="a profile other than <home>/config/roi_onnx_thresholds.json")
    p.add_argument("--data-dir", help="where saved results and logs go")
    p.add_argument("--port", type=int, default=0, help="default: any free port")
    p.add_argument("--no-window", action="store_true", help="use the default browser instead of the app window")
    p.add_argument("--no-browser", action="store_true", help="with --no-window: just serve")
    return p.parse_args(argv)


def main(argv=None) -> int:
    multiprocessing.freeze_support()
    args = parse_args(argv)
    data = paths.data_dir(args.data_dir)
    log_path = setup_logging(data)
    log.info("Himalaya %s starting (frozen=%s)", __version__, paths.FROZEN)
    if not (args.no_window or args.no_browser) and already_running():
        message_box("Himalaya is already running. Look for its window in the taskbar.")
        return 0
    rt = Runtime(args, data)
    try:
        code = run_browser(rt, log_path) if (args.no_window or args.no_browser) else run_window(rt, log_path)
    finally:
        rt.stop()
        logging.shutdown()
    os._exit(code)                                                       # do not wait for a scan still in the worker


if __name__ == "__main__":
    main()
