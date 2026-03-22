# MacTernaryVision – 1.58-bit YOLO face tracking on Mac M2

Real-time face detection using your ternary YOLO weights with OpenCV DNN on Apple Silicon.

## Prerequisites

- Homebrew: `brew install opencv cmake`
- Python (venv): `pip install ultralytics torch onnx`

## Quick start

1. **Export ONNX** (fuses `yolo_1.58b_epoch_*_weights.pt` and saves `yolov8s-seg.onnx`):
   ```bash
   cd MacTernaryVision
   python3 export_onnx.py
   # or: ../venv/bin/python export_onnx.py
   ```

2. **Build C++ app**:
   ```bash
   cmake .
   make
   ```

3. **Run** (allow camera when prompted):
   ```bash
   ./MacTernaryVision
   ```
   Press **ESC** to quit.

### Benchmarks (space + compute / energy proxies)

The app can record **reproducible** metrics for comparing your 1.58-bit ONNX against a full-precision baseline (same arch, same OpenCV backend).

| Metric | Meaning |
|--------|---------|
| **ONNX disk** | Size of the `.onnx` file on disk (storage / transfer cost). |
| **RSS / peak RSS** | Process memory from Mach (`phys_footprint` when available)—similar to Activity Monitor “Memory”. |
| **DNN forward (mean / p50 / p95)** | Wall time for `net.forward()` only (ms); correlates with compute work per frame. |
| **CPU proxy** | `(user+system CPU time) / wall time` between samples—can exceed 100% on multi-core (Activity Monitor style). **`cpu_per_core_est`** divides by logical CPU count. |

**Not** measured in-app: package **watts** or die **temperature** (no stable public API for per-app heat). For watts, use external sampling (below).

**CLI**

```bash
./MacTernaryVision --benchmark
./MacTernaryVision --benchmark --benchmark-log run.csv
./MacTernaryVision --model fp32_baseline.onnx --benchmark --bench-frames 600
./MacTernaryVision --help
```

- `--benchmark` — green overlay with disk, RSS, DNN timings, CPU proxy.  
- `--benchmark-log FILE` — append one CSV row about every **1 second** (header written on first open).  
- `--bench-frames N` — exit after `N` frames (steady run without manual stop).  
- `--model PATH` — ONNX path (default `yolov8s-seg.onnx`).

**CSV columns** (`--benchmark-log`):  
`unix_ms,onnx_bytes,rss_bytes,rss_peak_bytes,infer_mean_ms,infer_p50_ms,infer_p95_ms,cpu_proc_pct,cpu_per_core_est`

**Fair comparison (1.58 vs FP baseline)**

1. Export two ONNX files with the **same** YOLOv8s-seg head and input size (640).  
2. Run each with the **same** flags: `DNN_BACKEND_OPENCV`, `DNN_TARGET_CPU`, same machine load.  
3. Warm up ~30s, then log for 2+ minutes; compare **onnx_bytes**, **rss**, **infer_*** ms, **cpu_***.  
4. Report **relative** savings (e.g. “−X% model size, −Y% mean forward time”) under these conditions—not abstract claims without a run.

**Optional: package power (macOS, requires sudo)**

In a second terminal while the app runs:

```bash
sudo powermetrics --samplers cpu_power -i 1000 -n 60
```

Interpret vendor docs for your chip; correlate timestamps with your CSV `unix_ms`.

## Files

- `export_onnx.py` – Loads base YOLOv8s-seg, shrinks head to 1 class (face), injects QALoRA, fuses your `.pt` weights, then exports ONNX.
- `main.cpp` – OpenCV webcam + DNN inference; segmentation overlay + optional benchmark UI/log.
- `benchmark_mac.cpp` / `benchmark_mac.hpp` – ONNX file size, Mach RSS, `getrusage` CPU proxy, rolling DNN latency stats.
- `CMakeLists.txt` – Build config for Homebrew OpenCV on Apple Silicon.
- `yolo_1.58b_epoch_3_weights.pt` – Symlink to your weights in the parent folder.

## Notes

- The `.pt` file can be `yolo_1.58b_epoch_1_weights.pt` or `yolo_1.58b_epoch_3_weights.pt`; the script auto-detects it.
- If the camera window does not appear, run `./MacTernaryVision` from **Terminal.app** (not from an IDE terminal).
