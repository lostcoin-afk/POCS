#!/usr/bin/env python3
"""Calibrate planar pixel coordinates into millimetres from a known rectangle.

Select four corners in order around the rectangle:
  1 -> 2: known width, 2 -> 3: known height,
  3 -> 4: width return edge, 4 -> 1: height return edge.

Outputs data/calibration.json by default. The file contains a scalar mm/px
estimate and an image-to-plane homography. The live inspector will load it on
startup and use the homography for dimensions and circle diameters when the
camera frame resolution matches the calibration frame.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class RectanglePicker:
    def __init__(self, image: np.ndarray, max_width: int = 1800, max_height: int = 1000):
        self.image = image
        h, w = image.shape[:2]
        scale = min(1.0, max_width / float(w), max_height / float(h))
        self.display_w = max(1, int(round(w * scale)))
        self.display_h = max(1, int(round(h * scale)))
        self.scale_x = w / self.display_w
        self.scale_y = h / self.display_h
        self.preview = cv2.resize(image, (self.display_w, self.display_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        self.points: list[tuple[float, float]] = []
        self.window = "Rectangle scale calibration - click 4 corners clockwise"

    def _render(self) -> np.ndarray:
        view = self.preview.copy()
        text = [
            "Click reference corners in order: 1, 2, 3, 4 (around perimeter)",
            "Point 1->2 = known WIDTH; point 2->3 = known HEIGHT",
            "Enter/S = calculate and save  |  R = reset  |  Q/ESC = cancel",
        ]
        panel_h = 72
        canvas = np.zeros((self.display_h + panel_h, self.display_w, 3), dtype=np.uint8)
        canvas[:self.display_h] = view
        canvas[self.display_h:] = (20, 28, 38)
        for i, line in enumerate(text):
            cv2.putText(canvas, line, (12, self.display_h + 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 242, 250), 1, cv2.LINE_AA)
        colors = [(40, 240, 255), (40, 240, 120), (255, 180, 40), (245, 90, 220)]
        if self.points:
            display_points = [(int(round(x / self.scale_x)), int(round(y / self.scale_y))) for x, y in self.points]
            for index, pt in enumerate(display_points):
                cv2.circle(canvas, pt, 6, colors[index], -1, cv2.LINE_AA)
                cv2.putText(canvas, str(index + 1), (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[index], 2, cv2.LINE_AA)
            if len(display_points) >= 2:
                for i in range(1, len(display_points)):
                    cv2.line(canvas, display_points[i - 1], display_points[i], (70, 240, 100), 2, cv2.LINE_AA)
            if len(display_points) == 4:
                cv2.line(canvas, display_points[-1], display_points[0], (70, 240, 100), 2, cv2.LINE_AA)
        return canvas

    def _on_mouse(self, event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or y >= self.display_h or len(self.points) >= 4:
            return
        original_x = min(self.image.shape[1] - 1.0, max(0.0, x * self.scale_x))
        original_y = min(self.image.shape[0] - 1.0, max(0.0, y * self.scale_y))
        self.points.append((float(original_x), float(original_y)))

    def pick(self) -> list[tuple[float, float]]:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)
        try:
            while True:
                cv2.imshow(self.window, self._render())
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("r"), ord("R")):
                    self.points.clear()
                elif key in (10, 13, ord("s"), ord("S")):
                    if len(self.points) == 4:
                        return list(self.points)
                elif key in (27, ord("q"), ord("Q")):
                    raise RuntimeError("Calibration cancelled by user.")
        finally:
            cv2.destroyWindow(self.window)


def edge_lengths_px(points: np.ndarray) -> tuple[float, float, dict[str, float]]:
    """Average the opposite sides of the selected quadrilateral."""
    p = np.asarray(points, dtype=np.float64).reshape(4, 2)
    dist = lambda a, b: float(np.linalg.norm(p[a] - p[b]))
    lengths = {
        "top_px": dist(0, 1),
        "right_px": dist(1, 2),
        "bottom_px": dist(2, 3),
        "left_px": dist(3, 0),
    }
    width_px = (lengths["top_px"] + lengths["bottom_px"]) / 2.0
    height_px = (lengths["left_px"] + lengths["right_px"]) / 2.0
    return width_px, height_px, lengths


def calculate_calibration(points: np.ndarray, width_mm: float, height_mm: float, image_size: tuple[int, int]) -> dict:
    """Compute averaged scalar scales and perspective-correct image-to-mm homography."""
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("Known physical width and height must both be positive millimetres.")
    p = np.asarray(points, dtype=np.float64).reshape(4, 2)
    # Require the four manually selected corners to form a proper convex quad.
    contour = np.round(p).astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        raise ValueError("The four selected points do not form a convex rectangle. Reset and click its corners clockwise: top-left, top-right, bottom-right, bottom-left.")
    width_px, height_px, sides = edge_lengths_px(p)
    if min(width_px, height_px) < 10:
        raise ValueError("Selected rectangle is too small in the image; choose a larger reference.")
    area_px = abs(float(cv2.contourArea(contour)))
    if area_px < 500:
        raise ValueError("Selected reference area is too small; choose a larger rectangle for more accurate calibration.")
    scale_x = float(width_mm / width_px)
    scale_y = float(height_mm / height_px)
    # Least-squares shared scale over all four physical/pixel side observations.
    pixel_lengths = np.array([sides["top_px"], sides["bottom_px"], sides["left_px"], sides["right_px"]], dtype=np.float64)
    physical_lengths = np.array([width_mm, width_mm, height_mm, height_mm], dtype=np.float64)
    mm_per_px = float(np.dot(pixel_lengths, physical_lengths) / max(1e-12, np.dot(pixel_lengths, pixel_lengths)))
    average_scale = (scale_x + scale_y) / 2.0
    axis_difference = 100.0 * abs(scale_x - scale_y) / max(1e-12, average_scale)

    width_mm_f, height_mm_f = float(width_mm), float(height_mm)
    destination = np.array([[0, 0], [width_mm_f, 0], [width_mm_f, height_mm_f], [0, height_mm_f]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(p.astype(np.float32), destination.astype(np.float32))
    image_w, image_h = int(image_size[0]), int(image_size[1])
    return {
        "method": "known_rectangle_planar_homography",
        "mm_per_px": mm_per_px,
        "mm_per_px_x": scale_x,
        "mm_per_px_y": scale_y,
        "axis_scale_difference_percent": axis_difference,
        "known_width_mm": width_mm_f,
        "known_height_mm": height_mm_f,
        "average_width_px": width_px,
        "average_height_px": height_px,
        "selected_area_px": area_px,
        "measured_sides_px": sides,
        "corner_points_px": p.tolist(),
        "image_size": [image_w, image_h],
        "homography_image_to_mm": homography.tolist(),
        "homography_destination_units": "millimetres",
        "warning": (
            "Horizontal and vertical scalar scales disagree; prefer the saved homography for this plane."
            if axis_difference > 2.0 else
            "Scalar scale agreement is within 2%; homography remains preferred where perspective is present."
        ),
    }


def read_frame(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.image:
        image_path = Path(args.image).expanduser().resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        return image, str(image_path)
    if args.video:
        video_path = Path(args.video).expanduser().resolve()
        if not video_path.is_file():
            raise RuntimeError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(args.frame_index)))
        ok, image = cap.read()
        cap.release()
        if not ok or image is None:
            raise RuntimeError(f"Could not read frame {args.frame_index} from {video_path}")
        return image, f"{video_path} (frame {args.frame_index})"
    cap = cv2.VideoCapture(int(args.webcam))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open webcam index {args.webcam}")
    ok, image = cap.read()
    cap.release()
    if not ok or image is None:
        raise RuntimeError("Could not capture a calibration frame from webcam.")
    return image, f"webcam index {args.webcam}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a calibration image")
    group.add_argument("--video", help="Path to a video containing the known rectangle")
    group.add_argument("--webcam", type=int, help="Capture a single frame from webcam index")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index to read from video (default: 0)")
    parser.add_argument("--width-mm", type=float, help="Known reference width in mm (clicks 1->2 and 3->4)")
    parser.add_argument("--height-mm", type=float, help="Known reference height in mm (clicks 2->3 and 4->1)")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "data" / "calibration.json"), help="Output calibration JSON (default: this project data/calibration.json)")
    parser.add_argument("--max-display-width", type=int, default=1800)
    parser.add_argument("--max-display-height", type=int, default=1000)
    args = parser.parse_args()

    width_mm = args.width_mm if args.width_mm is not None else float(input("Known reference width (mm; edge 1->2): "))
    height_mm = args.height_mm if args.height_mm is not None else float(input("Known reference height (mm; edge 2->3): "))
    image, source = read_frame(args)
    points = np.asarray(RectanglePicker(image, args.max_display_width, args.max_display_height).pick(), dtype=np.float64)
    calibration = calculate_calibration(points, width_mm, height_mm, (image.shape[1], image.shape[0]))
    calibration["source"] = source
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    tmp.replace(output)

    preview = image.copy()
    pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(preview, [pts], True, (50, 240, 100), 3, cv2.LINE_AA)
    for i, (x, y) in enumerate(points, 1):
        cv2.circle(preview, (int(round(x)), int(round(y))), 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(preview, str(i), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, str(i), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    preview_path = output.with_name("calibration_preview.png")
    cv2.imwrite(str(preview_path), preview)

    print("\nCalibration saved:", output)
    print(f"Image resolution: {image.shape[1]} x {image.shape[0]}")
    print(f"Reference average width: {calibration['average_width_px']:.3f} px = {width_mm:.3f} mm")
    print(f"Reference average height: {calibration['average_height_px']:.3f} px = {height_mm:.3f} mm")
    print(f"X scale: {calibration['mm_per_px_x']:.8f} mm/px")
    print(f"Y scale: {calibration['mm_per_px_y']:.8f} mm/px")
    print(f"Recommended shared scale: {calibration['mm_per_px']:.8f} mm/px")
    print(f"Difference between axis scales: {calibration['axis_scale_difference_percent']:.3f}%")
    print("Homography: saved for perspective-correct planar measurements")
    print("Preview saved:", preview_path)
    print("\nImportant: mount the calibration rectangle and inspected feature in the same physical plane,")
    print("keep camera/lens/focus fixed, and use this calibration at the same image resolution.")
    if calibration["axis_scale_difference_percent"] > 2.0:
        print("WARNING: X and Y scales differ by >2%. Check corner selection, perspective, lens distortion, or plane alignment.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
