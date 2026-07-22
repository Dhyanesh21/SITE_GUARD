"""
scripts/demo_step2_tracking_video.py — a deeper, motion-based verification of
Step 2 tracking than the static-frame pytest tests.

WHY THIS SCRIPT EXISTS
The pytest tests for Step 2 feed the SAME static image repeatedly, which only
proves persist=True carries state forward — it can't prove anything about
ByteTrack's occlusion-survival claim, because nothing ever moves or gets
occluded. This script builds a small SYNTHETIC video (fully offline,
deterministic, reproducible — no downloaded footage) where a real detected
person crop walks across the frame and is partially covered by a fake
"machinery" block for a few consecutive frames in the middle. It then runs
the actual Tracker across every frame of that video (via frame_source's
VIDEO path) and prints the track_id + confidence for every frame, so you can
see with real numbers whether identity survives the occlusion.

Outputs (both under scripts/_demo_output/, gitignored — regenerate any time):
  synthetic_walk.avi          the raw synthetic input
  synthetic_walk_tracked.avi  same video with tracker boxes/IDs drawn on it
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils import ASSETS

from app.detection.detector import Detector
from app.detection.frame_source import SourceType, iter_frames
from app.tracking.tracker import Tracker

OUT_DIR = Path(__file__).parent / "_demo_output"
VIDEO_PATH = OUT_DIR / "synthetic_walk.avi"
ANNOTATED_PATH = OUT_DIR / "synthetic_walk_tracked.avi"

WIDTH, HEIGHT = 640, 480
N_FRAMES = 40
FPS = 15
OCCLUSION_FRAMES = range(15, 23)   # ~8 consecutive frames of partial occlusion
OCCLUSION_HEIGHT_COVERAGE = 0.75   # fraction of crop height hidden, TOP-DOWN
                                   # (head/torso is the informative region for
                                   # a person detector; hiding legs alone
                                   # barely dents confidence — proven by the
                                   # first attempt at this demo below)


def build_synthetic_walk_video() -> None:
    """Crop the highest-confidence real 'person' detection out of bus.jpg,
    then paste it at a shifting x-offset across N_FRAMES on a plain
    background, simulating a worker walking left to right. For the
    OCCLUSION_FRAMES window, a dark rectangle covers part of the crop,
    simulating the worker passing behind machinery.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bus = cv2.imread(str(ASSETS / "bus.jpg"))
    detector = Detector()
    people = sorted(
        (d for d in detector.infer(bus) if d.class_name == "person"),
        key=lambda d: d.confidence,
        reverse=True,
    )
    assert people, "expected at least one confidently-detected person in bus.jpg"
    top = people[0]
    x1, y1, x2, y2 = (int(v) for v in (top.bbox.x1, top.bbox.y1, top.bbox.x2, top.bbox.y2))
    crop = bus[y1:y2, x1:x2].copy()
    ch, cw = crop.shape[:2]

    # Scale the crop to fit the canvas with vertical margin (bbox from a
    # full-size photo can be taller than our small synthetic canvas).
    max_ch = int(HEIGHT * 0.8)
    if ch > max_ch:
        scale = max_ch / ch
        crop = cv2.resize(crop, (int(cw * scale), max_ch))
        ch, cw = crop.shape[:2]

    print(f"Using source person crop {cw}x{ch}px (post-scale), original detection confidence={top.confidence:.3f}")

    writer = cv2.VideoWriter(str(VIDEO_PATH), cv2.VideoWriter_fourcc(*"XVID"), FPS, (WIDTH, HEIGHT))
    start_x, end_x = 20, WIDTH - cw - 20
    y = (HEIGHT - ch) // 2

    for i in range(N_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), (200, 200, 200), dtype=np.uint8)
        x = int(start_x + (end_x - start_x) * (i / (N_FRAMES - 1)))
        frame[y : y + ch, x : x + cw] = crop

        if i in OCCLUSION_FRAMES:
            occ_y2 = y + int(ch * OCCLUSION_HEIGHT_COVERAGE)  # hide TOP portion
            cv2.rectangle(
                frame,
                (x - 15, y - 10),
                (x + cw + 15, occ_y2),
                (40, 40, 40),
                thickness=-1,
            )
        writer.write(frame)
    writer.release()


def run_tracking_demo() -> None:
    build_synthetic_walk_video()

    tracker = Tracker()
    annotated_writer = None
    trace: list[tuple[int, int | None, float | None, int]] = []  # (frame, track_id, conf, n_people_boxes)

    for frame in iter_frames(SourceType.VIDEO, str(VIDEO_PATH), every_n_frames=1):
        detections = tracker.track(frame.image)
        people = [d for d in detections if d.class_name == "person"]
        top = max(people, key=lambda d: d.confidence) if people else None
        trace.append((
            frame.index,
            top.track_id if top else None,
            round(top.confidence, 3) if top else None,
            len(people),
        ))

        if annotated_writer is None:
            h, w = frame.image.shape[:2]
            annotated_writer = cv2.VideoWriter(
                str(ANNOTATED_PATH), cv2.VideoWriter_fourcc(*"XVID"), FPS, (w, h)
            )
        vis = frame.image.copy()
        for d in detections:
            bx1, by1, bx2, by2 = (int(v) for v in (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2))
            color = (0, 200, 0) if d.class_name == "person" else (0, 0, 200)
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(
                vis, f"{d.class_name} id={d.track_id} {d.confidence:.2f}",
                (bx1, max(0, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )
        annotated_writer.write(vis)

    annotated_writer.release()

    print(f"\n{'frame':>5} {'track_id':>9} {'confidence':>10} {'n_boxes':>8}")
    for idx, tid, conf, n in trace:
        marker = "  <-- occlusion window" if idx in OCCLUSION_FRAMES else ""
        print(f"{idx:5d} {str(tid):>9} {str(conf):>10} {n:8d}{marker}")

    ids_seen = {t for _, t, _, _ in trace if t is not None}
    missed_frames = [idx for idx, tid, _, _ in trace if tid is None]
    print(f"\nDistinct track_ids seen across the whole synthetic walk: {ids_seen}")
    print(f"Frames with NO person track_id at all: {missed_frames}")
    if len(ids_seen) <= 1:
        print("RESULT: identity was STABLE — one track_id was reused across the entire walk, "
              "including through the occlusion window.")
    else:
        print("RESULT: an ID SWITCH occurred — more than one track_id was assigned during the walk.")

    print(f"\nWrote: {VIDEO_PATH}")
    print(f"Wrote: {ANNOTATED_PATH}  (bboxes + track_id + confidence overlaid)")


if __name__ == "__main__":
    run_tracking_demo()
