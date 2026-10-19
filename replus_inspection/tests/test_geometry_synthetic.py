"""Smoke test for silhouette extraction, notch corners, and reference matching."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import DEFAULT_PROCESSING, process_frame  # noqa: E402
from geometry import compare_corner_template, make_template_payload  # noqa: E402


def make_synthetic_bracket() -> np.ndarray:
    image = np.full((600, 1000, 3), 125, dtype=np.uint8)
    polygon = np.array(
        [
            [180, 120], [360, 120], [360, 310], [415, 345], [470, 310],
            [470, 120], [620, 120], [620, 165], [760, 165], [760, 300],
            [650, 300], [650, 420], [470, 420], [470, 340], [415, 370],
            [360, 340], [360, 420], [180, 420],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [polygon], (205, 205, 205))
    for center, radius in [((280, 235), 18), ((550, 235), 18), ((715, 235), 26)]:
        cv2.circle(image, center, radius, (80, 80, 80), -1)
    # A texture scratch must not become part of the filled external silhouette.
    cv2.line(image, (210, 360), (310, 350), (175, 175, 175), 2)
    return image


def main() -> None:
    image = make_synthetic_bracket()
    _annotated, data = process_frame(image, dict(DEFAULT_PROCESSING), None)

    assert data["silhouette"]["found"], "Silhouette should be found"
    assert data["corner_count"] >= 12, f"Expected boundary corners; found {data['corner_count']}"
    notch_angles = [
        float(corner["angle_deg"])
        for corner in data["corners"]
        if corner["is_concave"] and float(corner["angle_deg"]) < 90.0
    ]
    assert len(notch_angles) >= 2, f"Expected both V-notch diagonal corners, got {notch_angles}"

    reference = make_template_payload(data, "synthetic-known-good", dict(DEFAULT_PROCESSING), "unit test")
    comparison = compare_corner_template(data["corners"], reference, tolerance=3.0)
    assert comparison["overall_status"] == "PASS", comparison["overall_status"]
    assert not comparison["unmatched_current"]
    assert not comparison["unmatched_reference"]

    print(
        "PASS: silhouette found;",
        data["edge_count"], "boundary edges;",
        data["corner_count"], "corners; notch opening angles =",
        [round(angle, 2) for angle in notch_angles],
    )
    print("PASS: an identical contour matches its generated corner template.")


if __name__ == "__main__":
    main()
