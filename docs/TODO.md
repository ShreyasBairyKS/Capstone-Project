# VisionQAI TODO Tracker

Source analysis: [`IMPROVEMENTS_AND_EDGE_CASES.md`](IMPROVEMENTS_AND_EDGE_CASES.md). IDs (P = performance, E = edge case) refer to it.

Items are batched by theme and complexity so related fixes ship together. Work them roughly in order. Tick the boxes as each lands and add a short note (date, what changed, how it was verified).

Legend: **S** = small and low risk, can be tested on CPU. **M** = moderate, touches several files. **L** = large or design change. **GPU** = needs the CUDA machine or Docker to verify.

---

## Batch 1: Config robustness (S)
Small, safe, CPU-testable. Prevents confusing setup failures on other PCs and co-developers' machines.

- [ ] E27: strip inline `# comments` in the Python `.env` loader (and in `start_standalone.bat`, or let Python do all loading)
- [ ] E28: clear startup error (or clamp with a warning) for out-of-range `VISIONQAI_FILL_*` thresholds
- [ ] E29: remove the fragile bat `.env` parser; rely on `env_loader.py`
- [ ] E3: make `*_FORCE_RECOMPILE` one-shot (recompile once, then ignore until the flag is toggled)
- [ ] E8: pin `ultralytics` and the torch/torchvision versions; align venv and Docker; document

## Batch 2: Resilience and failure handling (M)
Keep the system serving when something goes wrong.

- [ ] E22: audit save failures must not abort the verdict (log, flag the record, still return the decision), both stations
- [ ] E4: cap model missing or failing must not stop the server; report the failure in `/api/health` and the UI
- [ ] E2: engine load with fallback chain: try `.engine`, on failure delete and recompile, then ONNX, then PT (both stations) **GPU**
- [ ] E21: reject or suffix duplicate custom bottle IDs; make folder-name sanitising collision-safe
- [ ] E5: warn loudly (log + UI banner) when running on CPU although a GPU was expected (Docker without NVIDIA runtime) **GPU**
- [ ] E13: WebSocket reconnect with backoff and a visible "disconnected" state

## Batch 3: Station state and multi-tab safety (M, design change)
One shared source of truth for which station owns the GPU.

- [ ] E11/E12: server-held active station; load/unload driven by the server; broadcast station changes to all tabs
- [ ] E14: refresh `fillHealth`/`capHealth` on station-change broadcasts and after any settings save
- [ ] E9: refuse or warn when started with more than one uvicorn worker

## Batch 4: Inspection quality, Station 2 (M)
Fewer wrong verdicts.

- [ ] E17: flag out-of-bounds or clamped meniscus as UNCERTAIN instead of a confident ratio
- [ ] E16: flag truncated bottle (box touches frame edge), multiple bottle candidates and extreme tilt as UNCERTAIN
- [ ] E18: bottle confident but no meniscus found: route to UNCERTAIN (configurable) instead of auto-reject
- [ ] E19: make the 3% / 2% bottle-box trims configurable per station settings
- [ ] E15: apply EXIF orientation on image upload (both stations)
- [ ] E20: exclude `INSPECTION_FAILED` conveyor gaps from yield KPIs (check cap too)
- [ ] P4: run the water detector on the bottle crop and map coordinates back

## Batch 5: Active learning and data hygiene (M)
Trustworthy training data and bounded disk use.

- [ ] E25: add a `verified` flag set only by operator save; export only verified items (both stations)
- [ ] E24: keep the model's box extent when only the verdict changes; save the operator notes
- [ ] E23: retention policy (max GB or max age) with cleanup job and a setting in `.env`
- [ ] E26: document the actuator as a simulation boundary (and a hook point for real hardware) in the docs

## Batch 6: Backend performance (M)
**Do P1 first**; it decides which of the rest are worth doing.

- [ ] P1: add per-stage timing (inference, gate, audit, render, encode) to responses and logs; record a CPU baseline and a GPU baseline **GPU**
- [ ] P2: cap: async audit saving, downscale before drawing masks, lower JPEG quality, base64 only for the latest frame
- [ ] P3: fill: same rendering and saving optimizations
- [ ] P5: KPI counters or a short cache instead of a full-table aggregate per inspection (both stations)
- [ ] P8: persistent per-thread SQLite connections (both stations)
- [ ] P9: index-friendly bottle-ID search (FTS5 or prefix)
- [ ] P10: cacheable image endpoints for fill replay instead of base64 in JSON
- [ ] P16: stream the CSV and zip exports

## Batch 7: Frontend performance (S/M)

- [ ] P6: push new defects over the WebSocket; drop the per-cycle `fetchDefects`
- [ ] P7: send image URLs or binary frames; stop holding base64 in the history state
- [ ] P13: lazy-load tab pages and Recharts to split the bundle
- [ ] P14: completion-driven auto-stream instead of a fixed timer

## Batch 8: GPU, Docker and deployment (L, needs hardware)
Cannot be fully verified without the CUDA machine.

- [ ] E10: run both stations under the stress test with threaded inference on CUDA; confirm no errors or corrupted results **GPU**
- [ ] E1: switch Cap <-> Fill 20+ times and confirm VRAM returns to baseline in `nvidia-smi`; add engine cleanup if not **GPU**
- [ ] E6: raise the container healthcheck `start-period`; add a "compiling engines" state to `/api/health` **GPU**
- [ ] E7: move the SQLite DB to a named volume (or a Windows-safe journal mode) and test host + container **GPU**
- [ ] First-time TensorRT compile of both fill models in Docker; record timings in the target PC guide **GPU**
- [ ] P11: evaluate INT8 / CUDA graphs only if P1 shows the GPU is the bottleneck **GPU**
- [ ] P12: optional `KEEP_BOTH_LOADED` flag for GPUs with spare VRAM **GPU**
- [ ] P15: wire real camera ingestion into the API (needs actual hardware)

## Batch 9: Security (optional, lab dependent)

- [ ] E30: simple API token or restrict CORS and bind address when not on an isolated network

---

## Done
- [x] Fill thresholds read from `.env` (single source of truth)
- [x] Load/unload locks; `gc.collect()` before `empty_cache()`
- [x] Real boxes in fill Replay Inspector; honest fill histogram
- [x] Root `docker-compose.yml` fill support, nginx timeouts, `onnxruntime`
- [x] Station switch error handling and rollback; GPU state sync on page load
- [x] Active Learning scales to the real image size
- [x] Env-driven paths and simulation folders (`env_loader.py`)
- [x] Per-station settings API and modal (`/api/fill/config`)
- [x] CRLF batch files plus `.gitattributes`
- [x] Inference moved off the event loop; inspections serialized; health-check race fixed
- [x] Docs moved to `docs/`
