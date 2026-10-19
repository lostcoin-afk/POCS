"""Contour/silhouette geometry for hardware inspection.

The component boundary is segmented first. Polygon vertices define individual edges and
physical corners; templates are matched by normalized position, corner opening angle,
and orientations of the adjacent boundary edges.
"""
from __future__ import annotations

import math
import time
import uuid
from typing import Any, Optional

import cv2
import numpy as np

FEATURE_TYPE = "silhouette_corners_v1"


def normalize_orientation(angle_deg: float) -> float:
    return float(angle_deg % 180.0)


def undirected_angle_delta(a: float, b: float) -> float:
    delta = abs(normalize_orientation(a) - normalize_orientation(b))
    return float(min(delta, 180.0 - delta))


def _odd(value: int, minimum: int = 1, maximum: int = 31) -> int:
    value = max(minimum, min(maximum, int(value)))
    return value if value % 2 else min(maximum, value + 1)


def _mask_candidates(gray: np.ndarray, cfg: dict[str, Any]) -> list[tuple[np.ndarray, str]]:
    """Generate plausible binary silhouettes with Otsu and adaptive thresholds."""
    height, width = gray.shape[:2]
    blur_kernel = _odd(int(cfg.get("blur_kernel", 3)), 1, 15)
    smooth = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0) if blur_kernel > 1 else gray

    _, otsu = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    candidates: list[tuple[np.ndarray, str]] = [
        (otsu, "otsu-bright"),
        (cv2.bitwise_not(otsu), "otsu-dark"),
    ]

    # Adaptive threshold is a fallback for mild illumination gradients. Its smaller
    # regions are filtered by component area and border-contact checks below.
    min_dim = min(height, width)
    block = min(81, max(31, (min_dim // 8) | 1))
    if block >= min_dim:
        block = max(3, (min_dim - 1) | 1)
    if block % 2 == 0:
        block -= 1
    if block >= 3:
        adaptive = cv2.adaptiveThreshold(
            smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 3
        )
        candidates.extend([
            (adaptive, "adaptive-bright"),
            (cv2.bitwise_not(adaptive), "adaptive-dark"),
        ])
    return candidates


def extract_component_silhouette(
    gray: np.ndarray, cfg: dict[str, Any]
) -> tuple[Optional[np.ndarray], np.ndarray, dict[str, Any]]:
    """Find the largest plausible, non-border-touching external component contour.

    Returns (contour, filled_mask, metadata). A missing contour is a normal detection
    result, not an exception: callers can keep streaming and surface the failure.
    """
    height, width = gray.shape[:2]
    image_area = float(height * width)
    min_area_ratio = float(cfg.get("min_component_area_ratio", 0.01))
    max_area_ratio = float(cfg.get("max_component_area_ratio", 0.85))
    morph_kernel = _odd(int(cfg.get("silhouette_morph_kernel", 5)), 1, 21)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    best: Optional[tuple[float, np.ndarray, str]] = None
    for candidate_mask, method in _mask_candidates(gray, cfg):
        cleaned = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            area_ratio = area / image_area
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            # A contour touching the image frame usually represents the background mask,
            # a cropped component, or a threshold region rather than a complete part.
            if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
                continue
            if w < 12 or h < 12:
                continue
            bbox_ratio = area / max(1.0, float(w * h))
            center_x = (x + 0.5 * w) / max(1.0, width)
            center_y = (y + 0.5 * h) / max(1.0, height)
            center_penalty = math.hypot(center_x - 0.5, center_y - 0.5)
            # Area selects the actual part over small texture regions; the small fill and
            # centering terms help break ties between similarly sized candidates.
            base_score = area_ratio + 0.025 * bbox_ratio - 0.02 * center_penalty
            # Prefer a plausible global-threshold silhouette when it exists; adaptive
            # thresholds often pick up the dark shadow halo around bright metal.
            priority = 0.12 if method.startswith("otsu-") else 0.0
            score = base_score + priority
            if best is None or score > best[0]:
                best = (score, contour, method)

    # Always evaluate an edge-ring contour as well. On metallic parts, object and
    # background intensities may overlap; the external contour of a closed Canny ring
    # can be a better silhouette than a local-threshold outline.
    blur_kernel = _odd(int(cfg.get("blur_kernel", 3)), 1, 15)
    smooth = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0) if blur_kernel > 1 else gray
    low = max(0, min(254, int(cfg.get("canny_low", 40))))
    high = max(low + 1, min(255, int(cfg.get("canny_high", 120))))
    edge_ring = cv2.Canny(smooth, low, high)
    edge_ring = cv2.morphologyEx(edge_ring, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    contours, _ = cv2.findContours(edge_ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        area_ratio = area / image_area
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
            continue
        if w < 12 or h < 12:
            continue
        bbox_ratio = area / max(1.0, float(w * h))
        center_x = (x + 0.5 * w) / max(1.0, width)
        center_y = (y + 0.5 * h) / max(1.0, height)
        center_penalty = math.hypot(center_x - 0.5, center_y - 0.5)
        base_score = area_ratio + 0.025 * bbox_ratio - 0.02 * center_penalty
        # Prefer a true closed edge ring over an adaptive-threshold halo, but prefer
        # Otsu above both whenever its contour does not touch the image frame.
        score = base_score + 0.04
        if best is None or score > best[0]:
            best = (score, contour, "canny-fallback")

    filled = np.zeros((height, width), dtype=np.uint8)
    if best is None:
        return None, filled, {
            "found": False,
            "method": "none",
            "area_px": 0.0,
            "area_ratio": 0.0,
            "bbox": None,
        }

    _, contour, method = best
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    x, y, w, h = cv2.boundingRect(contour)
    area = float(cv2.contourArea(contour))
    return contour, filled, {
        "found": True,
        "method": method,
        "area_px": round(area, 2),
        "area_ratio": round(area / image_area, 6),
        "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
    }


def _fit_edge_orientation(points: np.ndarray, fallback_angle: float, image_diagonal: float) -> tuple[float, int]:
    """Fit a boundary-edge orientation after rejecting contour-chain outliers.

    This is a lightweight robust fit: trim rounded corner samples, fit once, retain
    points near the fitted line, then refit. It avoids merging unrelated edges as the
    old Hough-segment grouping did.
    """
    if points is None or len(points) < 3:
        return normalize_orientation(fallback_angle), int(0 if points is None else len(points))
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) >= 10:
        trim = min(max(1, int(round(len(pts) * 0.08))), max(1, (len(pts) - 6) // 2))
        work = pts[trim:-trim] if len(pts) - 2 * trim >= 4 else pts
    else:
        work = pts
    if len(work) < 3:
        work = pts

    def estimate(sample: np.ndarray) -> tuple[float, float, float, float]:
        vx, vy, x0, y0 = [float(v) for v in cv2.fitLine(sample, cv2.DIST_L2, 0, 0.01, 0.01).flatten()]
        length = max(1e-9, math.hypot(vx, vy))
        return vx / length, vy / length, x0, y0

    vx, vy, x0, y0 = estimate(work)
    residuals = np.abs((work[:, 0] - x0) * vy - (work[:, 1] - y0) * vx)
    threshold = max(1.5, image_diagonal * 0.0018)
    inliers = work[residuals <= threshold]
    if len(inliers) >= max(3, int(0.35 * len(work))):
        vx, vy, _, _ = estimate(inliers)
        support = int(len(inliers))
    else:
        support = int(len(work))
    angle = normalize_orientation(math.degrees(math.atan2(vy, vx)))
    return angle, support


def _unwrap_contour_indices(indices: list[int], count: int) -> list[int]:
    if not indices:
        return []
    result = [int(indices[0])]
    for idx in indices[1:]:
        value = int(idx)
        while value <= result[-1]:
            value += count
        result.append(value)
    return result


def _angle_between_rays(previous: np.ndarray, vertex: np.ndarray, following: np.ndarray) -> float:
    first = previous.astype(float) - vertex.astype(float)
    second = following.astype(float) - vertex.astype(float)
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom < 1e-9:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denom, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def build_contour_geometry(
    contour: Optional[np.ndarray], image_shape: tuple[int, ...], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Convert an external contour into simplified boundary edges and corners."""
    height, width = image_shape[:2]
    empty = {"corners": [], "edges": [], "polygon": None, "bbox": None, "epsilon_px": 0.0}
    if contour is None or len(contour) < 10:
        return empty

    points = contour.reshape(-1, 2).astype(np.int32)
    perimeter = float(cv2.arcLength(contour, True))
    epsilon_ratio = float(cfg.get("contour_epsilon_ratio", 0.004))
    epsilon_ratio = min(0.03, max(0.001, epsilon_ratio))
    epsilon_px = max(1.0, perimeter * epsilon_ratio)
    approx = cv2.approxPolyDP(contour, epsilon_px, True).reshape(-1, 2).astype(np.int32)

    # Remove duplicated adjacent vertices if any were introduced by rounding.
    filtered: list[np.ndarray] = []
    for point in approx:
        if not filtered or float(np.linalg.norm(point.astype(float) - filtered[-1].astype(float))) >= 1.0:
            filtered.append(point)
    if len(filtered) > 2 and np.linalg.norm(filtered[0].astype(float) - filtered[-1].astype(float)) < 1.0:
        filtered.pop()
    approx = np.asarray(filtered, dtype=np.int32)
    if len(approx) < 3:
        return empty

    bx, by, bw, bh = cv2.boundingRect(contour)
    box_w = max(1.0, float(bw - 1))
    box_h = max(1.0, float(bh - 1))
    diagonal = max(1.0, math.hypot(box_w, box_h))
    contour_count = len(points)
    raw_indices = [int(np.argmin(np.sum((points.astype(float) - p.astype(float)) ** 2, axis=1))) for p in approx]
    indices = _unwrap_contour_indices(raw_indices, contour_count)

    edges: list[dict[str, Any]] = []
    edge_chains: list[np.ndarray] = []
    n_vertices = len(approx)
    signed_area = float(cv2.contourArea(approx.reshape(-1, 1, 2), oriented=True))
    for i in range(n_vertices):
        start_idx = indices[i]
        end_idx = indices[(i + 1) % n_vertices]
        if i == n_vertices - 1:
            while end_idx <= raw_indices[i]:
                end_idx += contour_count
        if end_idx <= start_idx:
            end_idx += contour_count
        chain_indices = [k % contour_count for k in range(start_idx, end_idx + 1)]
        chain = points[np.asarray(chain_indices, dtype=np.int32)] if chain_indices else np.empty((0, 2), dtype=np.int32)
        edge_chains.append(chain)

        p1 = approx[i].astype(float)
        p2 = approx[(i + 1) % n_vertices].astype(float)
        length_px = float(np.linalg.norm(p2 - p1))
        fallback = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        fitted_angle, support_points = _fit_edge_orientation(chain, fallback, diagonal)
        mid_x = float((p1[0] + p2[0]) / 2.0)
        mid_y = float((p1[1] + p2[1]) / 2.0)
        edges.append({
            "id": f"E{i + 1:02d}",
            "start_corner_id": f"C{i + 1:02d}",
            "end_corner_id": f"C{((i + 1) % n_vertices) + 1:02d}",
            "x1": float(p1[0]), "y1": float(p1[1]),
            "x2": float(p2[0]), "y2": float(p2[1]),
            "x1_norm": float((p1[0] - bx) / box_w),
            "y1_norm": float((p1[1] - by) / box_h),
            "x2_norm": float((p2[0] - bx) / box_w),
            "y2_norm": float((p2[1] - by) / box_h),
            "mid_x": mid_x, "mid_y": mid_y,
            "nx": float(np.clip((mid_x - bx) / box_w, 0.0, 1.0)),
            "ny": float(np.clip((mid_y - by) / box_h, 0.0, 1.0)),
            "angle_deg": float(fitted_angle),
            "length_px": length_px,
            "length_norm": float(length_px / diagonal),
            "support_points": int(support_points),
            # Retain this key for clients built against the old feature-table shape.
            "support_segments": int(support_points),
        })

    corners: list[dict[str, Any]] = []
    for i, vertex in enumerate(approx):
        previous = approx[(i - 1) % n_vertices]
        following = approx[(i + 1) % n_vertices]
        opening = _angle_between_rays(previous, vertex, following)
        incoming = vertex.astype(float) - previous.astype(float)
        outgoing = following.astype(float) - vertex.astype(float)
        turn_cross = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
        is_concave = bool(turn_cross * signed_area < -1e-6)
        polygon_interior = (360.0 - opening) if is_concave else opening
        prev_edge = edges[(i - 1) % n_vertices]
        next_edge = edges[i]
        x, y = float(vertex[0]), float(vertex[1])
        corners.append({
            "id": f"C{i + 1:02d}",
            "x": x, "y": y,
            "nx": float(np.clip((x - bx) / box_w, 0.0, 1.0)),
            "ny": float(np.clip((y - by) / box_h, 0.0, 1.0)),
            "angle_deg": round(opening, 4),
            "interior_angle_deg": round(polygon_interior, 4),
            "is_concave": is_concave,
            "prev_edge_id": prev_edge["id"],
            "next_edge_id": next_edge["id"],
            "prev_edge_angle_deg": round(float(prev_edge["angle_deg"]), 4),
            "next_edge_angle_deg": round(float(next_edge["angle_deg"]), 4),
            "prev_edge_length_norm": round(float(prev_edge["length_norm"]), 6),
            "next_edge_length_norm": round(float(next_edge["length_norm"]), 6),
        })

    return {
        "corners": corners,
        "edges": edges,
        "polygon": approx,
        "bbox": {"x": int(bx), "y": int(by), "width": int(bw), "height": int(bh)},
        "epsilon_px": float(epsilon_px),
    }


def _corner_orientation_cost(current: dict[str, Any], reference: dict[str, Any]) -> tuple[float, float, float]:
    cp, cn = float(current.get("prev_edge_angle_deg", 0.0)), float(current.get("next_edge_angle_deg", 0.0))
    rp, rn = float(reference.get("prev_edge_angle_deg", 0.0)), float(reference.get("next_edge_angle_deg", 0.0))
    direct_prev, direct_next = undirected_angle_delta(cp, rp), undirected_angle_delta(cn, rn)
    reverse_prev, reverse_next = undirected_angle_delta(cp, rn), undirected_angle_delta(cn, rp)
    if direct_prev + direct_next <= reverse_prev + reverse_next:
        return (direct_prev + direct_next) / 2.0, direct_prev, direct_next
    return (reverse_prev + reverse_next) / 2.0, reverse_prev, reverse_next


def compare_corner_template(
    current_corners: list[dict[str, Any]], template: Optional[dict[str, Any]], tolerance: float
) -> dict[str, Any]:
    """Match physical corners by normalized location and adjacent geometric features."""
    ref_corners = (template or {}).get("corners", []) if isinstance(template, dict) else []
    valid_template = bool(template and template.get("feature_type") == FEATURE_TYPE and ref_corners)
    tolerance = float(tolerance)
    orientation_tolerance = max(5.0, tolerance * 2.0)

    if not valid_template:
        rows = [{
            "current_corner_id": corner["id"],
            "reference_corner_id": None,
            "angle_deg": round(float(corner["angle_deg"]), 3),
            "reference_angle_deg": None,
            "delta_deg": None,
            "orientation_delta_deg": None,
            "position_delta": None,
            "status": "NO TEMPLATE",
        } for corner in current_corners]
        return {
            "template_saved": False, "matches": [], "mapping": {},
            "corner_pairs": rows, "angle_pairs": rows,
            "unmatched_current": [c["id"] for c in current_corners],
            "unmatched_reference": [], "tolerance_deg": tolerance,
            "orientation_tolerance_deg": orientation_tolerance,
            "overall_status": "REFERENCE NEEDED",
        }

    candidates: list[tuple[float, int, int, float, float, float, float]] = []
    for ci, current in enumerate(current_corners):
        for ri, reference in enumerate(ref_corners):
            dx = float(current.get("nx", 0.0)) - float(reference.get("nx", 0.0))
            dy = float(current.get("ny", 0.0)) - float(reference.get("ny", 0.0))
            position_delta = math.hypot(dx, dy) / math.sqrt(2.0)
            angle_delta = abs(float(current.get("angle_deg", 0.0)) - float(reference.get("angle_deg", 0.0)))
            orientation_avg, orientation_prev, orientation_next = _corner_orientation_cost(current, reference)
            cost = 0.52 * position_delta + 0.28 * min(1.0, angle_delta / 180.0) + 0.20 * (orientation_avg / 90.0)
            candidates.append((cost, ci, ri, position_delta, angle_delta, orientation_prev, orientation_next))
    candidates.sort(key=lambda row: row[0])

    used_current: set[int] = set()
    used_reference: set[int] = set()
    mapping: dict[str, str] = {}
    match_details: list[dict[str, Any]] = []
    match_metrics: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for cost, ci, ri, position_delta, angle_delta, orientation_prev, orientation_next in candidates:
        if ci in used_current or ri in used_reference:
            continue
        # Normalized distance rejects assignments to other features on the silhouette.
        if position_delta > 0.34 or cost > 0.60:
            continue
        current_id = str(current_corners[ci]["id"])
        reference_id = str(ref_corners[ri]["id"])
        used_current.add(ci)
        used_reference.add(ri)
        mapping[current_id] = reference_id
        match_metrics[(ci, ri)] = (position_delta, angle_delta, orientation_prev, orientation_next)
        match_details.append({
            "current_corner": current_id,
            "reference_corner": reference_id,
            "match_cost": round(cost, 4),
            "position_delta": round(position_delta, 4),
        })

    rows: list[dict[str, Any]] = []
    any_fail = False
    for ci, current in enumerate(current_corners):
        reference_id = mapping.get(str(current["id"]))
        if reference_id is None:
            rows.append({
                "current_corner_id": current["id"], "reference_corner_id": None,
                "angle_deg": round(float(current["angle_deg"]), 3),
                "reference_angle_deg": None, "delta_deg": None,
                "orientation_delta_deg": None, "position_delta": None,
                "status": "UNMATCHED",
            })
            continue
        ri = next(i for i, ref in enumerate(ref_corners) if str(ref["id"]) == reference_id)
        reference = ref_corners[ri]
        position_delta, _angle_delta, orientation_prev, orientation_next = match_metrics[(ci, ri)]
        measured_angle = float(current["angle_deg"])
        reference_angle = float(reference["angle_deg"])
        delta = measured_angle - reference_angle
        max_orientation_delta = max(orientation_prev, orientation_next)
        passed = abs(delta) <= tolerance and max_orientation_delta <= orientation_tolerance
        status = "PASS" if passed else "FAIL"
        any_fail |= not passed
        rows.append({
            "current_corner_id": current["id"],
            "reference_corner_id": reference_id,
            "angle_deg": round(measured_angle, 3),
            "reference_angle_deg": round(reference_angle, 3),
            "delta_deg": round(delta, 3),
            "prev_orientation_delta_deg": round(orientation_prev, 3),
            "next_orientation_delta_deg": round(orientation_next, 3),
            "orientation_delta_deg": round(max_orientation_delta, 3),
            "position_delta": round(position_delta, 4),
            "status": status,
        })

    matched_reference = set(mapping.values())
    unmatched_current = [str(c["id"]) for c in current_corners if str(c["id"]) not in mapping]
    unmatched_reference = [str(c["id"]) for c in ref_corners if str(c["id"]) not in matched_reference]
    matched_rows = [row for row in rows if row["reference_corner_id"] is not None]
    if len(matched_rows) < min(3, len(ref_corners)):
        overall = "INSUFFICIENT MATCHES"
    elif any_fail or unmatched_current or unmatched_reference:
        overall = "FAIL"
    else:
        overall = "PASS"
    return {
        "template_saved": True,
        "matches": match_details,
        "mapping": mapping,
        "corner_pairs": rows,
        # Compatibility alias for older clients while the UI transitions to corner pairs.
        "angle_pairs": rows,
        "unmatched_current": unmatched_current,
        "unmatched_reference": unmatched_reference,
        "tolerance_deg": tolerance,
        "orientation_tolerance_deg": orientation_tolerance,
        "overall_status": overall,
    }


def make_template_payload(
    response: dict[str, Any], name: str, cfg: dict[str, Any], source_label: str = "unknown source"
) -> dict[str, Any]:
    """Create the persisted template from current-frame geometry."""
    source_corners = response.get("corners", [])
    source_edges = response.get("lines", response.get("edges", []))
    if len(source_corners) < 4:
        raise ValueError(
            f"Only {len(source_corners)} boundary corners were found. At least 4 are required; "
            "verify the silhouette and contour approximation before saving."
        )

    corners: list[dict[str, Any]] = []
    for idx, corner in enumerate(source_corners, start=1):
        corners.append({
            "id": f"TC{idx:02d}",
            "nx": float(corner["nx"]), "ny": float(corner["ny"]),
            "angle_deg": float(corner["angle_deg"]),
            "interior_angle_deg": float(corner.get("interior_angle_deg", corner["angle_deg"])),
            "is_concave": bool(corner.get("is_concave", False)),
            "prev_edge_angle_deg": float(corner["prev_edge_angle_deg"]),
            "next_edge_angle_deg": float(corner["next_edge_angle_deg"]),
            "prev_edge_length_norm": float(corner["prev_edge_length_norm"]),
            "next_edge_length_norm": float(corner["next_edge_length_norm"]),
        })

    edges: list[dict[str, Any]] = []
    for idx, edge in enumerate(source_edges, start=1):
        edges.append({
            "id": f"TE{idx:02d}",
            "start_xy_norm": [float(edge.get("x1_norm", 0.0)), float(edge.get("y1_norm", 0.0))],
            "end_xy_norm": [float(edge.get("x2_norm", 0.0)), float(edge.get("y2_norm", 0.0))],
            "midpoint_norm": [float(edge.get("nx", 0.0)), float(edge.get("ny", 0.0))],
            "angle_deg": float(edge["angle_deg"]),
            "length_norm": float(edge["length_norm"]),
        })

    save_cfg = {key: value for key, value in cfg.items() if key in {
        "canny_low", "canny_high", "blur_kernel", "contour_epsilon_ratio",
        "silhouette_morph_kernel", "min_component_area_ratio", "max_component_area_ratio",
        "angle_tolerance_deg", "draw_edges", "draw_angle_labels",
    }}
    return {
        "schema_version": 2,
        "feature_type": FEATURE_TYPE,
        "template_id": str(uuid.uuid4()),
        "name": (name or "Reference hardware").strip()[:100],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_width": int(response.get("image_width", 0)),
        "image_height": int(response.get("image_height", 0)),
        "corner_count": len(corners),
        "edge_count": len(edges),
        "corners": corners,
        "edges": edges,
        "silhouette": response.get("silhouette", {}),
        "processing_config": save_cfg,
        "source": source_label,
        "note": (
            "Reference built from silhouette boundary corners and their adjacent edge orientations. "
            "Validate repeatability on multiple known-good parts before production use."
        ),
    }


def draw_geometry_overlay(
    frame: np.ndarray,
    canny_edges: np.ndarray,
    contour: Optional[np.ndarray],
    geometry: dict[str, Any],
    comparisons: dict[str, Any],
    cfg: dict[str, Any],
) -> np.ndarray:
    """Render the silhouette, simplified boundary edges, and only physical corner angles."""
    annotated = frame.copy()
    if cfg.get("draw_edges", False):
        green = np.zeros_like(annotated)
        green[:, :, 1] = canny_edges
        annotated = cv2.addWeighted(annotated, 1.0, green, 0.22, 0)

    if contour is not None:
        cv2.drawContours(annotated, [contour], -1, (60, 220, 150), 2, cv2.LINE_AA)

    polygon = geometry.get("polygon")
    edges = geometry.get("edges", [])
    corners = geometry.get("corners", [])
    h, w = annotated.shape[:2]
    palette = [
        (0, 220, 255), (255, 180, 0), (80, 220, 110), (255, 90, 190),
        (80, 150, 255), (200, 255, 70), (255, 110, 70), (180, 100, 255),
    ]

    if polygon is not None and len(polygon) >= 2:
        pts = polygon.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(annotated, [pts], True, (0, 230, 255), 2, cv2.LINE_AA)

    for idx, edge in enumerate(edges):
        p1 = (int(round(edge["x1"])), int(round(edge["y1"])))
        p2 = (int(round(edge["x2"])), int(round(edge["y2"])))
        cv2.line(annotated, p1, p2, palette[idx % len(palette)], 2, cv2.LINE_AA)

    if cfg.get("draw_angle_labels", True):
        # Always draw every corner point, but place text labels with collision avoidance.
        # This keeps the central notch measurable without recreating the old all-pairs
        # label clutter. Full values remain available in the corner table/API.
        for corner in corners:
            x, y = int(round(corner["x"])), int(round(corner["y"]))
            cv2.circle(annotated, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(annotated, (x, y), 2, (0, 20, 30), -1, cv2.LINE_AA)

        label_scale = 0.36
        placed_rectangles: list[tuple[int, int, int, int]] = []
        # Concave notch corners and non-right angles get first choice of label placement.
        ordered = sorted(
            corners,
            key=lambda c: (bool(c.get("is_concave")), abs(float(c.get("angle_deg", 90.0)) - 90.0)),
            reverse=True,
        )
        for corner in ordered:
            x, y = int(round(corner["x"])), int(round(corner["y"]))
            label = f'{corner["id"]} {corner["angle_deg"]:.1f} deg'
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1)
            text_w = min(text_w, max(30, w - 8))
            candidates_xy = [
                (x + 7, y - 7),
                (x - text_w - 7, y - 7),
                (x + 7, y + text_h + 7),
                (x - text_w - 7, y + text_h + 7),
                (x + 7, y + 2 * text_h + 9),
                (x - text_w - 7, y + 2 * text_h + 9),
            ]
            options = []
            for tx, ty in candidates_xy:
                tx = max(4, min(tx, w - text_w - 4))
                ty = max(text_h + 4, min(ty, h - baseline - 3))
                rect = (tx - 2, ty - text_h - 2, tx + text_w + 2, ty + baseline + 2)
                overlap_area = 0
                for rx1, ry1, rx2, ry2 in placed_rectangles:
                    ix = max(0, min(rect[2], rx2) - max(rect[0], rx1))
                    iy = max(0, min(rect[3], ry2) - max(rect[1], ry1))
                    overlap_area += ix * iy
                options.append((overlap_area, tx, ty, rect))
            overlap_area, tx, ty, rect = min(options, key=lambda item: item[0])
            # If every anchor collides, leave the point visible and its exact angle in the
            # table instead of drawing unreadable text over another corner's label.
            if overlap_area > 0:
                continue
            placed_rectangles.append(rect)
            cv2.putText(annotated, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (245, 250, 255), 1, cv2.LINE_AA)


    status = comparisons.get("overall_status", "REFERENCE NEEDED")
    status_color = (0, 220, 0) if status == "PASS" else ((0, 0, 255) if status == "FAIL" else (0, 190, 255))
    cv2.rectangle(annotated, (0, 0), (min(w, 510), 36), (24, 29, 39), -1)
    silhouette_text = "SILHOUETTE OK" if contour is not None else "SILHOUETTE NOT FOUND"
    cv2.putText(
        annotated,
        f"{status} | {silhouette_text} | {len(edges)} edges / {len(corners)} corners",
        (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 1, cv2.LINE_AA,
    )
    return annotated
