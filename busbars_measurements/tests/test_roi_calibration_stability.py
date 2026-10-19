import cv2
import numpy as np

from calibrate_scale import calculate_calibration
from geometry import process_measurement


def make_part_frame(intensity_delta: int = 0) -> np.ndarray:
    """High-contrast stepped bracket with one genuine circular opening."""
    h, w = 600, 800
    bg = int(np.clip(245 + intensity_delta, 0, 255))
    part = int(np.clip(25 + intensity_delta, 0, 255))
    frame = np.full((h, w, 3), bg, np.uint8)
    points = np.array(
        [[220, 110], [580, 110], [580, 210], [500, 210],
         [500, 330], [360, 330], [360, 480], [220, 480]],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [points], (part, part, part))
    cv2.circle(frame, (255, 425), 22, (bg, bg, bg), -1, cv2.LINE_AA)
    return frame


def config(roi_top_pct: float = 15.0) -> dict:
    return {
        "roi_top_pct": roi_top_pct,
        "blur_kernel": 5,
        "morph_kernel": 5,
        "min_area_ratio": 0.015,
        "max_area_ratio": 0.88,
        "canny_low": 45,
        "canny_high": 130,
        "line_hough_threshold": 55,
        "line_min_length": 75,
        "line_max_gap": 20,
        "axis_tolerance_deg": 10,
        "line_merge_px": 13,
        "max_axis_lines": 8,
        "min_dimension_px": 18,
        "circle_min_radius": 6,
        "circle_max_radius": 140,
        "circle_min_distance": 58,
        "circle_param1": 110,
        "circle_param2": 34,
        "circle_min_contrast": 16,
        "circle_min_edge_support": 0.18,
        "mm_per_px": None,
        "_temporal_state": {},
    }


def test_roi_excludes_rows_above_cutoff_and_reports_full_frame_coordinates():
    frame = make_part_frame()
    cfg = config(roi_top_pct=15.0)
    overlay, data, mask, edges = process_measurement(frame, cfg)
    assert data["roi"]["top_y"] == 90
    assert np.count_nonzero(mask[:90]) == 0
    assert np.count_nonzero(edges[:90]) == 0
    assert data["silhouette"]["found"]
    bbox = data["silhouette"]["bbox"]
    assert bbox["y"] >= 90
    assert overlay.shape == frame.shape


def test_guides_and_circle_detection_are_stable_under_small_intensity_changes():
    cfg = config(roi_top_pct=15.0)
    outputs = []
    for delta in (0, 1, -1, 2, 0):
        _, data, _, _ = process_measurement(make_part_frame(delta), cfg)
        outputs.append(data)
    last = outputs[-1]
    assert 3 <= len(last["horizontal_guides"]) <= 6
    assert 3 <= len(last["vertical_guides"]) <= 6
    assert len(last["circles"]) == 1
    # The main dimensions should not bounce when intensity changes by a few levels.
    x_spans = [d["length_px"] for d in outputs if d["length_px"] is not None]
    y_spans = [d["width_px"] for d in outputs if d["width_px"] is not None]
    assert max(x_spans) - min(x_spans) <= 2.0
    assert max(y_spans) - min(y_spans) <= 2.0


def test_four_corner_calibration_maps_reference_rectangle_to_millimetres():
    points = np.array([[100, 100], [500, 110], [490, 310], [110, 300]], dtype=np.float64)
    calibration = calculate_calibration(points, width_mm=80.0, height_mm=40.0, image_size=(800, 600))
    assert calibration["mm_per_px"] > 0
    assert calibration["axis_scale_difference_percent"] >= 0
    homography = np.asarray(calibration["homography_image_to_mm"], dtype=np.float64)
    transformed = cv2.perspectiveTransform(points.astype(np.float32).reshape(-1, 1, 2), homography)[..., 0, :]
    expected = np.array([[0, 0], [80, 0], [80, 40], [0, 40]], dtype=np.float32)
    assert np.max(np.abs(transformed - expected)) < 0.05


def test_calibration_rejects_invalid_corner_order():
    bad_points = np.array([[100, 100], [500, 110], [110, 300], [490, 310]], dtype=np.float64)
    try:
        calculate_calibration(bad_points, width_mm=80.0, height_mm=40.0, image_size=(800, 600))
    except ValueError:
        pass
    else:
        raise AssertionError("Non-convex corner order should be rejected")


def test_calibration_is_disabled_if_source_resolution_changes():
    frame = make_part_frame()
    cfg = config(roi_top_pct=15.0)
    cfg.update({
        "mm_per_px": 0.2,
        "homography_image_to_mm": [[0.2, 0, 0], [0, 0.2, 0], [0, 0, 1]],
        "calibration_image_size": [1920, 1200],
    })
    _, data, _, _ = process_measurement(frame, cfg)
    assert data["calibration_mode"] == "uncalibrated"
    assert data["calibration_resolution_matches"] is False
    assert data["length_label"].endswith("px")
