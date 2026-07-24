# YOLOv8n vs YOLOv8s — Benchmark (Step 8)

**Status: COMPLETE.** Trained on Colab GPU via `training/train_yolov8.ipynb`; weights copied into `weights/` (`yolov8n_best.pt`, `yolov8s_best.pt`).

## Setup

- Dataset: `data/css-data/` (10-class Kaggle mirror of Roboflow's Construction Site Safety project), 2801 images, split train/valid/test.
- Both models trained with identical hyperparameters (`epochs=100`, `patience=20`, `imgsz=640`, `batch=16`, `seed=42`) — the only difference between runs is the base architecture (`yolov8n.pt` vs `yolov8s.pt`).
- mAP reported on the held-out **test** split, not the validation split used for early stopping.
- **Inference speed measured on CPU**, not the GPU the notebook trains on — this project's actual deployment target is CPU-only, so a GPU inference number would misrepresent the real tradeoff.

## Results

| model | params (M) | size (MB) | mAP50 | mAP50-95 | CPU inference (ms/image) |
|---|---|---|---|---|---|
| YOLOv8n | 3.0 | 6.0 | 0.7621 | 0.4634 | 131.2 |
| YOLOv8s | 11.1 | 21.5 | 0.8197 | 0.5536 | 403.9 |

Derived numbers worth naming directly:
- **Speed**: nano is **3.08x faster** (131.2ms vs 403.9ms/image) — ~7.6 FPS vs ~2.5 FPS raw single-image throughput on CPU.
- **Accuracy**: small is better everywhere — **+5.8 points mAP50** (+7.6% relative), **+9.0 points mAP50-95** (+19.5% relative). This is a real, non-trivial accuracy gain, not noise.
- **Size**: small is 3.7x more parameters and 3.6x larger on disk.

## Why nano

This system's business goal is **near-real-time** PPE monitoring on a **CPU-only** deployment target (`/stream/start`, continuous live-camera inference) — not offline batch analysis where extra latency is free. At small's 403.9ms/image, raw throughput caps out around 2.5 FPS *before* ByteTrack and violation-rule overhead are even added on top; combined with `sampling.stream_every_n_frames=3` already deliberately reducing how often we bother to run inference at all, small would leave very little headroom before the pipeline can no longer keep up with a live feed in any meaningful sense. Nano's ~7.6 FPS raw ceiling gives real breathing room for that same overhead.

Small's accuracy gain is genuine — a +19.5% relative improvement in mAP50-95 is not something to dismiss — but it is a gain that arrives 3x slower. For a system whose entire premise (per this project's business framing) is *continuous, near-real-time* intervention rather than an after-the-fact audit tool, a violation flagged 3x slower is a violation caught later than it needed to be. **Nano is the right choice for the live-stream path.**

This is not an unconditional verdict, though: if this system were repurposed for **offline batch review** of already-recorded footage (no live-stream latency constraint, e.g. `/detect/video` run overnight on a day's archived footage), small's accuracy advantage would matter more and its 3x slower throughput would matter less — worth stating honestly rather than pretending nano wins in every context. `config/config.yaml`'s `detection.model_path` is set to nano because the *live-stream* use case is this system's stated primary goal, not because small is categorically worse.

## Caveats

- Colab's assigned CPU may differ meaningfully from the local deployment machine's CPU — treat the **relative** 3.08x n-vs-s speedup as more trustworthy than the absolute ms/image figures.
- All 10 dataset classes were trained as given (not filtered to just the 2 this system's violation logic acts on: `NO-Hardhat`, `NO-Safety Vest`) — see `config/config.yaml`'s class-map comments for why filtering wasn't done.
- mAP figures are across all 10 trained classes, not isolated to the 2 classes (`NO-Hardhat`, `NO-Safety Vest`) this system actually acts on — a per-class breakdown would be a natural follow-up if this system's accuracy on those 2 specific classes ever needs closer scrutiny (e.g. before a real deployment decision), but wasn't necessary to make the nano-vs-small throughput/accuracy tradeoff call above.
