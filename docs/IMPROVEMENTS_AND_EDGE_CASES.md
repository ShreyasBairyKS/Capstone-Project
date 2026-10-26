# VisionQAI: Performance Optimizations and Edge-Case Analysis

Written after the Station 2 integration audit (2026-10-26). Based on reading the code and on CPU-only runs. **No performance claim here has been measured on the GPU.** Verify before optimizing.

Each item has an ID. `docs/TODO.md` tracks the work by those IDs.

Already done and not repeated here: threshold env wiring, load/unload locks, event-loop blocking (inference now runs in worker threads, serialized by `_infer_lock`), real replay boxes, honest fill histogram, Docker fill support, env-driven paths, per-station settings.

---

## Part 1. Performance optimizations

### High impact

| ID | Item | Why |
|---|---|---|
| P1 | **Find where Station 1 time actually goes** | Each cap inspection runs inference, then synchronous audit saving (3 x 12 MP JPEGs at quality 95, labels, SQLite), then renders overlays for all 3 cameras at full resolution, then JPEG-encodes and base64s each. The GPU is probably not the bottleneck. Add per-stage timing first. |
| P2 | **Async audit saving and cheaper rendering (cap)** | Turn on `async_saving` and stop waiting on it. Downscale to about 1280 px before drawing masks. Save at quality about 90. Render base64 only for the latest frame, not every frame. |
| P3 | **Same treatment for Station 2** | Per inspection it renders the overlay, encodes base64 and saves the raw JPEG. Same remedies as P2. |
| P4 | **Crop to the bottle before the water detector** | The meniscus detector runs on the full frame after the bottle detector. Running it on the bottle crop is faster and removes false hits from labels, reflections and background lines. |
| P5 | **KPI query on every inspection** | `get_kpis` does a full-table aggregate after every inspection, so it gets slower as the DB grows. Keep running counters, or cache for about 1 s. |
| P6 | **Extra REST call per cycle (frontend)** | `fetchDefects` runs after every inspection. Push the new defect over the WebSocket instead. |
| P7 | **Base64 images in WebSocket JSON** | About 33% larger than binary, and the frontend keeps 20 of them in React state. Send image URLs or binary frames. |

### Medium impact

| ID | Item | Why |
|---|---|---|
| P8 | **Persistent SQLite connections** | Both DB classes open a new connection and run 4 PRAGMAs per call. Use one connection per thread. |
| P9 | **Index-friendly search** | `LIKE '%x%'` cannot use an index. It will crawl at 100k+ rows. Use FTS5 or prefix search. |
| P10 | **Cacheable fill replay images** | `get_bottle_detail` re-reads, re-renders and base64-encodes both images on every call. Serve them as image endpoints with cache headers, the way cap does. |
| P11 | **TensorRT tuning** | INT8 calibration, CUDA graphs, pinned-memory transfers. None are used today. Only worth it if P1 shows the GPU is the bottleneck. |
| P12 | **Optional "keep both models loaded"** | Each station switch pays a full reload and engine deserialize. An opt-in `KEEP_BOTH_LOADED=1` for GPUs with enough VRAM removes it. Unloading stays the default. |
| P13 | **Frontend bundle splitting** | One 814 KB chunk. Lazy-load tab pages and Recharts. |
| P14 | **Completion-driven auto-stream** | The 3.5 s timer fires whether or not the last cycle finished. Trigger the next cycle when the previous one completes. |
| P15 | **Wire real camera ingestion** | `StreamManager` and `ThreadedCamera` exist but no API route runs a live capture loop. Downscale on the capture thread when added. |
| P16 | **Stream exports** | The export zip and CSV are built fully in memory. Stream them. |

---

## Part 2. Edge cases and failure modes

### GPU, Docker and models

| ID | Failure | Detail and fix |
|---|---|---|
| E1 | **VRAM may not actually free** | `empty_cache()` only affects PyTorch's allocator. TensorRT memory is freed only when the engine object is destroyed. `gc.collect()` was added but is unverified. Watch `nvidia-smi` across many switches. |
| E2 | **Corrupt or stale `.engine`** | An interrupted compile, an engine copied from another GPU, or a TensorRT version change leaves a file that exists but will not load. Station 2 then fails instead of recompiling. Fix: try the load, and on failure recompile, then fall back to ONNX or PT. |
| E3 | **FORCE_RECOMPILE footgun** | `VISIONQAI_FILL_FORCE_RECOMPILE=1` left on deletes and recompiles the engine on every load, so every station switch takes minutes. Make it one-shot. |
| E4 | **Missing cap weights kill the whole server** | The cap engine loads at startup, so nothing starts, not even Station 2. Make the failure non-fatal and show it in the UI. |
| E5 | **Docker silently on CPU** | If the NVIDIA runtime is missing, inference runs about 10x slower with only the hardware label showing it. Add a loud warning and a UI banner. |
| E6 | **Startup compile vs healthcheck** | The first TensorRT compile at boot exceeds the 60 s `start-period`, so Docker flags the container unhealthy. Lengthen the start period and report a "compiling" state. |
| E7 | **SQLite WAL on a Windows bind mount** | Docker Desktop/WSL2 mounts are a known source of "database is locked" and corruption. Running host and container against the same `audit/` at once is worse. Use a named volume, or `journal_mode=DELETE` on Windows. |
| E8 | **Version drift** | `ultralytics>=8.3.0` is unpinned, and the venv has torch 2.3.0 while Docker has 2.4.1. Export and TensorRT behavior can differ between machines. Pin versions. |
| E9 | **Multiple uvicorn workers** | Each would load its own models, and a station switch would only affect one of them. Document and enforce a single worker. |
| E10 | **Thread-safety on a real GPU** | Inference now runs in worker threads, serialized by a lock. Verified on CPU only. Confirm on CUDA under the stress test. |

### Multi-tab and UI state

| ID | Failure | Detail and fix |
|---|---|---|
| E11 | **Two tabs on different stations fight** | The active station lives in each tab, but GPU state is global. Tab A switching to Fill unloads the model Tab B uses, and Tab B's next request reloads it. They can ping-pong. Most likely demo accident. Fix: hold the active station on the server and broadcast it. |
| E12 | **Cap Simulate while on Fill** | It lazy-reloads the cap model, so both models are in VRAM again. Ties into E11. |
| E13 | **WebSocket retries forever** | Reconnects every 3 s with no backoff and no give-up state. |
| E14 | **Stale thresholds in the UI** | `fillHealth` refreshes only on mount and after saving settings. Changes from another client do not show. |

### Fill inspection quality

| ID | Failure | Detail and fix |
|---|---|---|
| E15 | **EXIF rotation ignored** | A sideways phone photo gives wrong ratios or `no_bottle`. Apply EXIF orientation on upload. |
| E16 | **Tilted, truncated or overlapping bottles** | Tilt skews the ratio. A bottle cut off at the frame edge gives a wrong height. With two bottles in frame the meniscus can be matched to the wrong one. Flag these as UNCERTAIN instead of guessing. |
| E17 | **Out-of-bounds meniscus silently clamped** | A false detection becomes a confident 0% or 100% instead of being flagged. |
| E18 | **`no_water` auto-rejects and fires the actuator** | Dark, opaque or low-contrast liquids give false rejects. Consider UNCERTAIN when the bottle is confident but no meniscus is found. |
| E19 | **Fill ratio is bottle-shape dependent** | Ratio is relative to bottle height with fixed 3% / 2% trims tuned to one shape. Tapered bottles are non-linear. Make trims configurable. |
| E20 | **Conveyor gaps count as inspections** | `INSPECTION_FAILED` frames count toward the total and lower the yield KPI. Check whether cap does the same. |

### Data and storage

| ID | Failure | Detail and fix |
|---|---|---|
| E21 | **Reused custom bottle IDs overwrite** | `INSERT OR REPLACE` replaces the old record and files. `a/b` and `a_b` also map to the same folder. Reject duplicates or suffix them. |
| E22 | **Audit failure aborts the verdict** | If disk is full or a write fails, the request errors after the decision was made. Log loudly and still return the verdict. |
| E23 | **No retention policy** | Every frame is saved at quality 95. Disk fills quickly at line speed. Add max size or age with cleanup. |
| E24 | **Active-learning edits shrink the label** | Saving a fill annotation rewrites the label as a 10 px-high box even if only the verdict changed. Operator notes are never saved. |
| E25 | **Unverified labels in the export** | The export zip includes every auto-generated label, not just operator-verified ones. Retraining on it feeds the model its own predictions. Add a `verified` flag and export only verified items. Check cap's export too. |
| E26 | **Actuator is only a flag** | `trigger_reject_actuator` is data. Nothing drives hardware. Document it as a simulation boundary. |

### Configuration and security

| ID | Failure | Detail and fix |
|---|---|---|
| E27 | **Inline comments in `.env`** | `KEY=1 # note` is read as `1 # note`. The Python loader, the bat loader and Docker compose each handle it differently. An integer setting such as the batch size would crash at startup. Strip inline comments in the loader. |
| E28 | **Out-of-range fill thresholds** | For example underfill 0.65 makes the app fail at boot with a pydantic traceback. Clamp or give a clear message. |
| E29 | **Bat `.env` parsing** | Breaks on values containing `&`, `%` or spaces. Prefer having Python load `.env`. |
| E30 | **No authentication** | CORS is `*` with credentials, and any LAN machine can call `/api/unload`. Fine for a lab, not on a shared network. |
