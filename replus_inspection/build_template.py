#!/usr/bin/env python3
"""Build a silhouette/corner reference template using the same geometry as app.py.

Examples:
  python build_template.py --image /path/to/good_part.png --name "Bracket A"
  python build_template.py --video /path/to/reference.mp4 --name "Bracket A"
  python build_template.py --webcam 0 --name "Bracket A"
  python build_template.py --arena --arena-serial 123456789 --name "Bracket A"

Keys: S=save template, SPACE=pause/resume, Q/ESC=quit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from app import BASE_DIR, DEFAULT_PROCESSING, create_source, process_frame
    from geometry import extract_component_silhouette, make_template_payload
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import the inspection pipeline from app.py. Run this script from the "
        "hardware_angle_inspection directory and install requirements first.\n"
        f"Original error: {exc}"
    ) from exc

DEFAULT_OUTPUT = BASE_DIR / "data" / "angle_template.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively build a known-good corner-geometry template.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Still image of a known-good part")
    source.add_argument("--video", type=Path, help="Video containing a known-good part")
    source.add_argument("--webcam", type=int, nargs="?", const=0, help="Webcam index (default: 0)")
    source.add_argument("--arena", action="store_true", help="Use the first available LUCID Arena camera")
    parser.add_argument("--arena-serial", default="", help="Optional LUCID camera serial number")
    parser.add_argument("--name", default="Reference hardware", help="Reference template name")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output JSON (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--force", action="store_true", help="Overwrite the output without prompting")
    parser.add_argument("--no-loop", action="store_true", help="Do not loop a video at EOF")
    return parser.parse_args()


def open_source(args: argparse.Namespace):
    if args.image is not None:
        path = args.image.expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"Image file not found: {path}")
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is None or frame.size == 0:
            raise RuntimeError(f"OpenCV could not read image: {path}")
        return None, frame, f"image: {path.name}", True

    if args.video is not None:
        path = args.video.expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"Video file not found: {path}")
        source = create_source({"source_type": "video", "video_path": str(path), "loop_video": not args.no_loop})
        ok, frame = source.read()
        if not ok or frame is None:
            source.close()
            raise RuntimeError(f"Could not read a frame from video: {path}")
        return source, frame, getattr(source, "label", f"video: {path.name}"), False

    if args.webcam is not None:
        source = create_source({"source_type": "webcam", "webcam_index": int(args.webcam)})
        ok, frame = source.read()
        if not ok or frame is None:
            source.close()
            raise RuntimeError(f"Could not read webcam index {args.webcam}")
        return source, frame, getattr(source, "label", f"webcam {args.webcam}"), False

    source = create_source({"source_type": "arena", "arena_serial": args.arena_serial})
    ok, frame = source.read()
    if not ok or frame is None:
        source.close()
        raise RuntimeError("LUCID Arena camera did not return a frame.")
    return source, frame, getattr(source, "label", "LUCID Arena camera"), False


def create_control_windows() -> None:
    cv2.namedWindow("Template controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Template controls", 560, 430)
    noop = lambda _value: None
    cfg = DEFAULT_PROCESSING
    cv2.createTrackbar("Canny low", "Template controls", int(cfg["canny_low"]), 254, noop)
    cv2.createTrackbar("Canny high", "Template controls", int(cfg["canny_high"]), 255, noop)
    cv2.createTrackbar("Blur kernel (odd = 2n+1)", "Template controls", (int(cfg["blur_kernel"]) - 1) // 2, 7, noop)
    cv2.createTrackbar("Contour epsilon x1000", "Template controls", int(round(cfg["contour_epsilon_ratio"] * 1000)), 30, noop)
    cv2.createTrackbar("Morph kernel odd index", "Template controls", (int(cfg["silhouette_morph_kernel"]) - 1) // 2, 10, noop)
    cv2.createTrackbar("Min component area x1000", "Template controls", int(round(cfg["min_component_area_ratio"] * 1000)), 250, noop)
    cv2.createTrackbar("Angle tolerance x10", "Template controls", int(round(cfg["angle_tolerance_deg"] * 10)), 300, noop)


def read_control_values() -> dict:
    low = cv2.getTrackbarPos("Canny low", "Template controls")
    high = max(low + 1, cv2.getTrackbarPos("Canny high", "Template controls"))
    blur_index = cv2.getTrackbarPos("Blur kernel (odd = 2n+1)", "Template controls")
    epsilon = cv2.getTrackbarPos("Contour epsilon x1000", "Template controls") / 1000.0
    morph_index = cv2.getTrackbarPos("Morph kernel odd index", "Template controls")
    min_area = cv2.getTrackbarPos("Min component area x1000", "Template controls") / 1000.0
    tolerance = cv2.getTrackbarPos("Angle tolerance x10", "Template controls") / 10.0
    return {
        "canny_low": max(0, min(254, low)),
        "canny_high": max(1, min(255, high)),
        "blur_kernel": max(1, min(15, blur_index * 2 + 1)),
        "contour_epsilon_ratio": max(0.001, min(0.03, epsilon)),
        "silhouette_morph_kernel": max(1, min(21, morph_index * 2 + 1)),
        "min_component_area_ratio": max(0.001, min(0.5, min_area)),
        "max_component_area_ratio": float(DEFAULT_PROCESSING["max_component_area_ratio"]),
        "angle_tolerance_deg": max(0.1, min(30.0, tolerance)),
        "draw_edges": False,
        "draw_angle_labels": True,
    }


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def edge_preview(frame: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    gray = to_gray(frame)
    kernel = max(1, min(15, int(cfg["blur_kernel"])))
    if kernel % 2 == 0:
        kernel += 1
    smooth = cv2.GaussianBlur(gray, (kernel, kernel), 0) if kernel > 1 else gray
    canny = cv2.Canny(smooth, int(cfg["canny_low"]), int(cfg["canny_high"]))
    _contour, mask, _info = extract_component_silhouette(gray, cfg)
    return canny, mask


def save_template(payload: dict, output: Path, force: bool) -> bool:
    output = output.expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        answer = input(f"\n{output} already exists. Replace it? Type 'yes' to overwrite: ").strip().lower()
        if answer != "yes":
            print("Save cancelled; existing template was left unchanged.")
            return False
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(f"\nTemplate saved: {output}")
    print(f"Reference corners: {len(payload['corners'])}")
    print(f"Reference edges: {len(payload['edges'])}")
    print("The FastAPI application will load this template on next startup; restart it if already running.")
    return True


def main() -> int:
    args = parse_args()
    source = None
    try:
        source, current_frame, source_label, is_still = open_source(args)
        try:
            cv2.namedWindow("Template builder - silhouette corners", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Template builder - silhouette corners", 1280, 800)
            cv2.namedWindow("Canny edges", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Canny edges", 900, 550)
            cv2.namedWindow("Silhouette mask", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Silhouette mask", 900, 550)
            create_control_windows()
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV GUI support is unavailable. Install GUI-enabled opencv-python (not "
                "opencv-python-headless) in the active environment."
            ) from exc

        print(f"Source: {source_label}")
        print("Review that the green outer boundary follows the part, especially the central notch.")
        print("Tune contour epsilon if too many small corners appear or a notch corner disappears.")
        print("Keys: S=save template, SPACE=pause/resume, Q/ESC=quit.")
        active_template: Optional[dict] = None
        paused = is_still
        latest_response: dict = {}
        last_cfg: dict = {}

        while True:
            if not paused and source is not None:
                ok, next_frame = source.read()
                if ok and next_frame is not None:
                    current_frame = next_frame
                elif args.video is not None and args.no_loop:
                    paused = True
                    print("Video reached its end; paused on the last available frame.")
                elif args.webcam is not None or args.arena:
                    time.sleep(0.01)
                    continue

            cfg = read_control_values()
            last_cfg = cfg
            annotated, latest_response = process_frame(current_frame, cfg, active_template)
            corner_count = latest_response.get("corner_count", 0)
            edge_count = latest_response.get("edge_count", 0)
            silhouette = latest_response.get("silhouette", {})
            status = (
                f"SOURCE: {source_label} | corners: {corner_count} | edges: {edge_count} | "
                f"silhouette: {'OK' if silhouette.get('found') else 'NOT FOUND'} | {'PAUSED' if paused else 'LIVE'}"
            )
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 66), (20, 25, 35), -1)
            cv2.putText(annotated, status[:145], (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (240, 245, 250), 1, cv2.LINE_AA)
            cv2.putText(annotated, "S: SAVE CORNER TEMPLATE   SPACE: PAUSE/RESUME   Q/ESC: QUIT", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 235, 255), 1, cv2.LINE_AA)
            cv2.imshow("Template builder - silhouette corners", annotated)
            edges_img, silhouette_img = edge_preview(current_frame, cfg)
            cv2.imshow("Canny edges", edges_img)
            cv2.imshow("Silhouette mask", silhouette_img)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("Exited without saving a new template.")
                break
            if key == ord(" ") and not is_still:
                paused = not paused
            elif key in (ord("s"), ord("S")):
                try:
                    if not latest_response.get("silhouette", {}).get("found"):
                        raise ValueError("No component silhouette found. Tune thresholding inputs, position the part, and try again.")
                    payload = make_template_payload(latest_response, args.name, last_cfg, source_label)
                    if save_template(payload, args.output, args.force):
                        active_template = payload
                        args.force = True
                except (ValueError, OSError) as exc:
                    print(f"Cannot save template: {exc}", file=sys.stderr)

        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if source is not None:
            try:
                source.close()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
