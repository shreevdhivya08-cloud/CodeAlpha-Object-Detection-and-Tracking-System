"""
Object Detection and Tracking using YOLOv8 + OpenCV
------------------------------------------------------
Features: webcam feed, bounding boxes, labels, confidence scores,
          object tracking IDs, FPS display, and video saving.
"""

import cv2
import time
import argparse
from ultralytics import YOLO

# ── Colour palette (one colour per class index, cycles if needed) ──────────────
PALETTE = [
    (56, 220, 130), (255, 100,  60), ( 60, 160, 255), (220,  60, 220),
    (255, 220,  40), ( 40, 220, 220), (180,  80, 255), (255, 140,  40),
]

def get_color(class_id: int) -> tuple:
    return PALETTE[class_id % len(PALETTE)]


def draw_box(frame, x1, y1, x2, y2, label: str, color: tuple, track_id=None):
    """Draw a bounding box with a filled label tag."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    display = f"#{track_id} {label}" if track_id is not None else label
    (tw, th), baseline = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

    tag_y1 = max(y1 - th - baseline - 6, 0)
    cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, display, (x1 + 4, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA)


def draw_hud(frame, fps: float, num_objects: int):
    """Overlay FPS and object count in the top-left corner."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (230, 62), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS : {fps:5.1f}", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, f"OBJ : {num_objects:3d}", (16, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 200, 255), 1, cv2.LINE_AA)


def run(
    source: int | str = 0,
    model_name: str = "yolov8n.pt",
    conf: float = 0.40,
    iou: float = 0.45,
    save_path: str | None = "output.mp4",
    show: bool = True,
    track: bool = True,
):
    """
    Main inference loop.

    Parameters
    ----------
    source     : camera index (int) or video file / RTSP URL (str)
    model_name : YOLOv8 weights file – downloaded automatically on first run
    conf       : minimum confidence threshold (0–1)
    iou        : NMS IoU threshold (0–1)
    save_path  : where to write the output video, or None to skip saving
    show       : whether to open a preview window
    track      : use ByteTrack (True) or plain detect (False)
    """

    print(f"\n[INFO] Loading model  : {model_name}")
    model = YOLO(model_name)
    names = model.names           # dict {id: 'class_name'}

    print(f"[INFO] Opening source : {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Resolution     : {fw}×{fh}  |  camera FPS: {cam_fps:.1f}")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, cam_fps, (fw, fh))
        print(f"[INFO] Saving output  : {save_path}")

    prev_time = time.perf_counter()
    frame_count = 0

    print("\n[INFO] Running — press  Q  to quit.\n")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Stream ended or camera disconnected.")
                break

            # ── Inference ──────────────────────────────────────────────────────
            if track:
                results = model.track(
                    frame,
                    conf=conf,
                    iou=iou,
                    persist=True,       # keep tracker state across frames
                    verbose=False,
                )
            else:
                results = model(frame, conf=conf, iou=iou, verbose=False)

            # ── Parse results ──────────────────────────────────────────────────
            num_objects = 0
            result = results[0]

            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                    cls_id          = int(boxes.cls[i].item())
                    conf_score      = float(boxes.conf[i].item())
                    track_id        = (
                        int(boxes.id[i].item())
                        if (track and boxes.id is not None)
                        else None
                    )

                    label  = f"{names[cls_id]} {conf_score:.0%}"
                    color  = get_color(cls_id)
                    draw_box(frame, x1, y1, x2, y2, label, color, track_id)
                    num_objects += 1

            # ── FPS calculation ────────────────────────────────────────────────
            now       = time.perf_counter()
            fps       = 1.0 / (now - prev_time + 1e-9)
            prev_time = now
            frame_count += 1

            draw_hud(frame, fps, num_objects)

            # ── Output ─────────────────────────────────────────────────────────
            if writer:
                writer.write(frame)

            if show:
                cv2.imshow("YOLOv8 Detection & Tracking  [Q to quit]", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Quit key pressed.")
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"\n[INFO] Saved → {save_path}")
        cv2.destroyAllWindows()
        print(f"[INFO] Processed {frame_count} frames. Done.")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv8 Object Detection & Tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source",  default=0,
                        help="Camera index (0,1,…) or path/URL to video file")
    parser.add_argument("--model",   default="yolov8n.pt",
                        help="YOLOv8 weights (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--conf",    type=float, default=0.40,
                        help="Confidence threshold (0–1)")
    parser.add_argument("--iou",     type=float, default=0.45,
                        help="NMS IoU threshold (0–1)")
    parser.add_argument("--save",    default="output.mp4",
                        help="Output video path (pass empty string to skip)")
    parser.add_argument("--no-show", action="store_true",
                        help="Disable the preview window (headless mode)")
    parser.add_argument("--no-track",action="store_true",
                        help="Disable ByteTrack; run plain detection only")

    args = parser.parse_args()

    # Convert source to int if it looks like a camera index
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    run(
        source     = source,
        model_name = args.model,
        conf       = args.conf,
        iou        = args.iou,
        save_path  = args.save or None,
        show       = not args.no_show,
        track      = not args.no_track,
    )
