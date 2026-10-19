"""Silhouette, oriented-bounding-rectangle and circle measurements for planar parts.

Measurements are image-space pixels unless a mm_per_px calibration is supplied.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import cv2
import numpy as np


def _odd(value: int, maximum: int = 31) -> int:
    value = max(1, min(maximum, int(value)))
    return value if value % 2 else min(maximum, value + 1)


def _gray_bgr(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if frame.ndim == 2:
        gray = frame
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        bgr = frame.copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray, bgr


def _candidate_masks(gray: np.ndarray, cfg: dict[str, Any]) -> list[tuple[np.ndarray, str]]:
    blur = _odd(int(cfg.get("blur_kernel", 3)), 15)
    smooth = cv2.GaussianBlur(gray, (blur, blur), 0) if blur > 1 else gray
    _, otsu = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    candidates = [(otsu, "otsu-bright"), (cv2.bitwise_not(otsu), "otsu-dark")]
    # Canny's external closed edge ring often works when metal and backdrop overlap in brightness.
    lo = int(cfg.get("canny_low", 45))
    hi = max(lo + 1, int(cfg.get("canny_high", 130)))
    edges = cv2.Canny(smooth, lo, hi)
    k = _odd(int(cfg.get("morph_kernel", 5)), 31)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ring = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidates.append((ring, "canny-closed"))
    return candidates


def extract_silhouette(gray: np.ndarray, cfg: dict[str, Any]) -> tuple[Optional[np.ndarray], np.ndarray, dict[str, Any]]:
    """Choose a large plausible foreground contour and return contour + filled mask + metadata."""
    h, w = gray.shape[:2]
    frame_area = float(h * w)
    min_ratio = float(cfg.get("min_area_ratio", 0.015))
    max_ratio = float(cfg.get("max_area_ratio", 0.88))
    k = _odd(int(cfg.get("morph_kernel", 5)), 31)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    best: Optional[tuple[float, np.ndarray, str]] = None

    for candidate, method in _candidate_masks(gray, cfg):
        cleaned = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, open_kernel, iterations=1)
        # Closing repairs small gaps in a silhouette. Keep this kernel modest to preserve holes/notches.
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            ratio = area / frame_area
            if ratio < min_ratio or ratio > max_ratio:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            # The user-selected ROI may intentionally cut through the top of the part.
            # Still reject silhouettes that run into the full-frame left/right/bottom borders.
            top_is_roi_cut = bool(cfg.get("allow_top_touch", False))
            if x <= 1 or x + bw >= w - 1 or y + bh >= h - 1 or (y <= 1 and not top_is_roi_cut):
                continue
            if bw < 20 or bh < 20:
                continue
            bbox_fill = area / max(1.0, float(bw * bh))
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            center_penalty = math.hypot(cx - 0.5, cy - 0.5)
            # Largest object gets preference; bbox fill and centrality only break close ties.
            score = ratio + 0.035 * bbox_fill - 0.015 * center_penalty
            if method.startswith("otsu"):
                score += 0.01
            if best is None or score > best[0]:
                best = (score, contour, method)

    mask = np.zeros((h, w), np.uint8)
    if best is None:
        return None, mask, {"found": False, "method": "none", "area_px": 0.0, "area_ratio": 0.0, "bbox": None}
    _, contour, method = best
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    x, y, bw, bh = cv2.boundingRect(contour)
    area = float(cv2.contourArea(contour))
    return contour, mask, {
        "found": True,
        "method": method,
        "area_px": round(area, 2),
        "area_ratio": round(area / frame_area, 6),
        "bbox": {"x": int(x), "y": int(y), "width": int(bw), "height": int(bh)},
    }


def _measure_label(px: float, mm_per_px: Optional[float]) -> tuple[float, str]:
    if mm_per_px is not None and math.isfinite(mm_per_px) and mm_per_px > 0:
        mm = px * mm_per_px
        return float(mm), f"{mm:.2f} mm"
    return float(px), f"{px:.1f} px"


def _circle_contrast(gray: np.ndarray, mask: np.ndarray, edge_image: np.ndarray, x: float, y: float, r: float) -> tuple[float, float]:
    """Return absolute inner-vs-ring intensity contrast and approximate edge support."""
    h, w = gray.shape[:2]
    ix, iy, ir = int(round(x)), int(round(y)), max(2, int(round(r)))
    if ix - ir - 3 < 0 or iy - ir - 3 < 0 or ix + ir + 3 >= w or iy + ir + 3 >= h:
        return 0.0, 0.0
    yy, xx = np.ogrid[iy-ir-3:iy+ir+4, ix-ir-3:ix+ir+4]
    dist = np.sqrt((xx - ix) ** 2 + (yy - iy) ** 2)
    crop = gray[iy-ir-3:iy+ir+4, ix-ir-3:ix+ir+4]
    mask_crop = mask[iy-ir-3:iy+ir+4, ix-ir-3:ix+ir+4] > 0
    inner = (dist <= max(1.0, 0.62 * r)) & mask_crop
    annulus = (dist >= 0.82 * r) & (dist <= 1.12 * r) & mask_crop
    if int(inner.sum()) < 8 or int(annulus.sum()) < 8:
        return 0.0, 0.0
    contrast = abs(float(np.mean(crop[inner])) - float(np.mean(crop[annulus])))
    edge_crop = edge_image[iy-ir-3:iy+ir+4, ix-ir-3:ix+ir+4] > 0
    circumference = (dist >= r - 2.5) & (dist <= r + 2.5)
    circum_count = int(circumference.sum())
    support = float((edge_crop & circumference).sum() / max(1, circum_count))
    return contrast, support


def detect_circles(gray: np.ndarray, silhouette_mask: np.ndarray, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Find circular features, rank candidates by edge support, and suppress duplicates.

    HoughCircles commonly produces several nearby estimates for one physical circle.
    We use contrast + circumference-edge support for ranking and non-maximum
    suppression based on overlapping candidate disks. Temporal persistence is applied
    by the caller after this per-frame filter.
    """
    h, w = gray.shape[:2]
    ys, xs = np.where(silhouette_mask > 0)
    if len(xs) < 100:
        return []
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    roi = gray[y0:y1, x0:x1]
    min_dim = min(roi.shape[:2])
    min_r = int(cfg.get("circle_min_radius", max(5, int(min_dim * 0.008))))
    max_r = int(cfg.get("circle_max_radius", max(min_r + 1, int(min_dim * 0.16))))
    dp = max(1.0, float(cfg.get("circle_dp", 1.2)))
    min_dist = max(8, int(cfg.get("circle_min_distance", max(24, min_dim * 0.045))))
    param1 = max(30, int(cfg.get("circle_param1", 110)))
    param2 = max(8, int(cfg.get("circle_param2", 34)))
    smooth = cv2.GaussianBlur(roi, (5, 5), 1.2)
    # Local contrast normalization helps moderate brightness drift without changing
    # the actual source frame used for measurements.
    try:
        smooth = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8)).apply(smooth)
    except cv2.error:
        pass
    candidates = cv2.HoughCircles(
        smooth, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
        param1=param1, param2=param2, minRadius=min_r, maxRadius=max_r,
    )
    if candidates is None:
        return []
    edge_image = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 110)
    scored: list[dict[str, Any]] = []
    min_contrast = float(cfg.get("circle_min_contrast", 16.0))
    min_support = float(cfg.get("circle_min_edge_support", 0.18))
    for cxr, cyr, radius in candidates[0].tolist():
        cx, cy, r = float(cxr + x0), float(cyr + y0), float(radius)
        ix, iy = int(round(cx)), int(round(cy))
        if ix < 0 or iy < 0 or ix >= w or iy >= h or silhouette_mask[iy, ix] == 0:
            continue
        contrast, support = _circle_contrast(gray, silhouette_mask, edge_image, cx, cy, r)
        if contrast < min_contrast or support < min_support:
            continue
        # Rank strong, complete circumference support above contrast-only texture blobs.
        quality = 0.72 * support + 0.28 * min(1.0, contrast / 55.0)
        scored.append({
            "cx": cx, "cy": cy, "radius_px": r,
            "contrast": float(contrast), "edge_support": float(support),
            "quality": float(quality),
        })
    scored.sort(key=lambda item: (item["quality"], item["edge_support"], item["radius_px"]), reverse=True)

    selected: list[dict[str, Any]] = []
    for candidate in scored:
        cx, cy, r = candidate["cx"], candidate["cy"], candidate["radius_px"]
        duplicate = False
        for prior in selected:
            d = math.hypot(cx - prior["cx"], cy - prior["cy"])
            rp = prior["radius_px"]
            # Same physical circular boundary: candidate disks overlap strongly and
            # have comparable radii. Distinct, non-overlapping holes remain separate.
            comparable_radius = min(r, rp) / max(r, rp) >= 0.62
            if comparable_radius and d < 0.82 * (r + rp):
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
    # Stable ordering by image position makes IDs less likely to jump between frames.
    selected.sort(key=lambda item: (item["cx"], item["cy"]))
    output: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected, 1):
        diameter_px = 2.0 * candidate["radius_px"]
        diameter, diameter_label = _measure_label(diameter_px, cfg.get("mm_per_px"))
        output.append({
            "id": f"C{index:02d}",
            "cx": round(candidate["cx"], 2), "cy": round(candidate["cy"], 2),
            "radius_px": round(candidate["radius_px"], 2),
            "diameter_px": round(diameter_px, 2),
            "diameter": round(diameter, 4), "diameter_label": diameter_label,
            "contrast": round(candidate["contrast"], 2),
            "edge_support": round(candidate["edge_support"], 3),
        })
    return output


def _cluster_axis_segments(
    raw_segments: list[dict[str, Any]],
    coordinate_tolerance: float,
    min_span: float,
    max_lines: int,
    axis: str,
) -> list[dict[str, Any]]:
    """Merge Hough fragments that lie on the same nearly-axis-aligned boundary."""
    if not raw_segments:
        return []
    raw_segments = sorted(raw_segments, key=lambda item: item["coord"])
    groups: list[list[dict[str, Any]]] = []
    for seg in raw_segments:
        if not groups:
            groups.append([seg])
            continue
        current = groups[-1]
        weight = sum(max(1.0, x["length"]) for x in current)
        center = sum(x["coord"] * max(1.0, x["length"]) for x in current) / weight
        if abs(seg["coord"] - center) <= coordinate_tolerance:
            current.append(seg)
        else:
            groups.append([seg])

    merged: list[dict[str, Any]] = []
    for group in groups:
        weights = np.asarray([max(1.0, g["length"]) for g in group], dtype=np.float64)
        coords = np.asarray([g["coord"] for g in group], dtype=np.float64)
        coord = float(np.average(coords, weights=weights))
        start = float(min(g["start"] for g in group))
        end = float(max(g["end"] for g in group))
        span = end - start
        if span < min_span:
            continue
        merged.append({
            "axis": axis,
            "coord": round(coord, 2),
            "start": round(start, 2),
            "end": round(end, 2),
            "span_px": round(span, 2),
            "support_px": round(float(sum(g["length"] for g in group)), 2),
            "segments": int(len(group)),
            "angle_deg": round(float(np.average([g["angle_deg"] for g in group], weights=weights)), 3),
        })

    # Keep the longest-supported edge locations, then restore spatial ordering.
    merged.sort(key=lambda item: (item["span_px"], item["support_px"]), reverse=True)
    merged = merged[:max(2, int(max_lines))]
    merged.sort(key=lambda item: item["coord"])
    for i, line in enumerate(merged, 1):
        line["id"] = f"{'V' if axis == 'vertical' else 'H'}{i:02d}"
    return merged


def detect_axis_guides(silhouette_mask: np.ndarray, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    """Detect near-horizontal and near-vertical silhouette edges, not internal texture."""
    h, w = silhouette_mask.shape[:2]
    boundary = cv2.Canny(silhouette_mask, 40, 120)
    threshold = max(10, int(cfg.get("line_hough_threshold", 45)))
    min_line_length = max(15, int(cfg.get("line_min_length", 65)))
    max_gap = max(0, int(cfg.get("line_max_gap", 25)))
    axis_tolerance = max(1.0, min(30.0, float(cfg.get("axis_tolerance_deg", 12.0))))
    coordinate_tolerance = max(1.0, float(cfg.get("line_merge_px", 13.0)))
    min_merged_span = max(20.0, float(cfg.get("line_min_merged_span", min_line_length * 0.75)))
    max_lines = max(2, min(16, int(cfg.get("max_axis_lines", 8))))

    segments = cv2.HoughLinesP(
        boundary,
        rho=1,
        theta=np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_gap,
    )
    raw_h: list[dict[str, Any]] = []
    raw_v: list[dict[str, Any]] = []
    if segments is not None:
        for item in segments[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in item]
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < min_line_length:
                continue
            angle = math.degrees(math.atan2(dy, dx)) % 180.0
            horizontal_deviation = min(angle, 180.0 - angle)
            vertical_deviation = abs(angle - 90.0)
            if horizontal_deviation <= axis_tolerance:
                raw_h.append({
                    "coord": (y1 + y2) / 2.0,
                    "start": min(x1, x2), "end": max(x1, x2),
                    "length": length, "angle_deg": angle,
                })
            elif vertical_deviation <= axis_tolerance:
                raw_v.append({
                    "coord": (x1 + x2) / 2.0,
                    "start": min(y1, y2), "end": max(y1, y2),
                    "length": length, "angle_deg": angle,
                })

    horizontal = _cluster_axis_segments(raw_h, coordinate_tolerance, min_merged_span, max_lines, "horizontal")
    vertical = _cluster_axis_segments(raw_v, coordinate_tolerance, min_merged_span, max_lines, "vertical")
    return horizontal, vertical, boundary



def _prepare_temporal_state(cfg: dict[str, Any], shape: tuple[int, int]) -> Optional[dict[str, Any]]:
    """Initialize a small temporal tracker used to suppress one-frame false detections."""
    state = cfg.get("_temporal_state")
    if not isinstance(state, dict):
        return None
    key = (int(shape[0]), int(shape[1]), round(float(cfg.get("roi_top_pct", 0.0)), 2))
    if state.get("key") != key:
        state.clear()
        state.update({
            "key": key, "frame_index": 0,
            "tracks": {"horizontal": [], "vertical": [], "circles": []},
            "bbox_samples": [],
        })
    state["frame_index"] = int(state.get("frame_index", 0)) + 1
    return state


def _track_coordinate(track: dict[str, Any], kind: str) -> tuple[float, float, float]:
    items = [sample["item"] for sample in track["samples"]]
    if kind == "circles":
        return (
            float(np.median([item["cx"] for item in items])),
            float(np.median([item["cy"] for item in items])),
            float(np.median([item["radius_px"] for item in items])),
        )
    return (float(np.median([item["coord"] for item in items])), 0.0, 0.0)


def _stabilize_features(
    raw_items: list[dict[str, Any]], state: Optional[dict[str, Any]], kind: str,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Median-smooth guides/circles and require multi-frame persistence after warm-up."""
    if state is None:
        return raw_items
    frame_index = int(state["frame_index"])
    window = 7
    tracks: list[dict[str, Any]] = state["tracks"][kind]
    tracks[:] = [t for t in tracks if frame_index - int(t["last_frame"]) < window]
    if kind in {"horizontal", "vertical"}:
        raw_items = sorted(raw_items, key=lambda item: item["coord"])
        tolerance = max(7.0, float(cfg.get("line_merge_px", 13.0)) * 1.35)
    else:
        raw_items = sorted(raw_items, key=lambda item: (item["cx"], item["cy"]))
        tolerance = 1.0
    used_tracks: set[int] = set()
    for item in raw_items:
        best_track = None
        best_score = float("inf")
        for ti, track in enumerate(tracks):
            if ti in used_tracks or frame_index - int(track["last_frame"]) >= window:
                continue
            a, b, r = _track_coordinate(track, kind)
            if kind == "circles":
                dist = math.hypot(float(item["cx"]) - a, float(item["cy"]) - b)
                radius = float(item["radius_px"])
                center_tol = max(10.0, 0.72 * min(radius, r))
                radius_tol = max(10.0, 0.42 * max(radius, r))
                if dist > center_tol or abs(radius - r) > radius_tol:
                    continue
                score = dist / center_tol + abs(radius - r) / radius_tol
            else:
                dist = abs(float(item["coord"]) - a)
                if dist > tolerance:
                    continue
                score = dist / tolerance
            if score < best_score:
                best_score, best_track = score, ti
        if best_track is None:
            tracks.append({"samples": [], "last_frame": frame_index})
            best_track = len(tracks) - 1
        track = tracks[best_track]
        track["samples"].append({"frame": frame_index, "item": dict(item)})
        track["samples"] = [s for s in track["samples"] if frame_index - int(s["frame"]) < window]
        track["last_frame"] = frame_index
        used_tracks.add(best_track)

    required_hits = 1 if frame_index <= 1 else 2 if frame_index < 5 else 3
    stabilized: list[dict[str, Any]] = []
    for track in tracks:
        samples = track["samples"]
        # Keep a confirmed feature through at most one missed frame. This damps
        # flicker without allowing an old false line/circle to linger on screen.
        if frame_index - int(track.get("last_frame", frame_index)) > 1:
            continue
        if len(samples) < required_hits:
            continue
        items = [sample["item"] for sample in samples]
        fields = ("cx", "cy", "radius_px", "contrast", "edge_support") if kind == "circles" else ("coord", "start", "end", "span_px", "support_px", "angle_deg")
        item = dict(items[-1])
        for field in fields:
            values = [float(value[field]) for value in items if field in value]
            if values:
                item[field] = float(np.median(values))
        if kind == "circles":
            item["diameter_px"] = 2.0 * float(item["radius_px"])
            diameter, label = _measure_label(item["diameter_px"], cfg.get("mm_per_px"))
            item["diameter"] = round(diameter, 4)
            item["diameter_label"] = label
            item["cx"] = round(float(item["cx"]), 2)
            item["cy"] = round(float(item["cy"]), 2)
            item["radius_px"] = round(float(item["radius_px"]), 2)
        else:
            item["coord"] = round(float(item["coord"]), 2)
            item["start"] = round(float(item["start"]), 2)
            item["end"] = round(float(item["end"]), 2)
            item["span_px"] = round(max(0.0, item["end"] - item["start"]), 2)
        stabilized.append(item)
    if kind == "circles":
        stabilized.sort(key=lambda item: (item["cx"], item["cy"]))
        for i, item in enumerate(stabilized, 1):
            item["id"] = f"C{i:02d}"
    else:
        stabilized.sort(key=lambda item: item["coord"])
        prefix = "H" if kind == "horizontal" else "V"
        for i, item in enumerate(stabilized, 1):
            item["id"] = f"{prefix}{i:02d}"
    return stabilized


def _stabilize_bbox(bbox: dict[str, int], state: Optional[dict[str, Any]]) -> dict[str, int]:
    if state is None:
        return bbox
    samples = state.setdefault("bbox_samples", [])
    x0, y0 = int(bbox["x"]), int(bbox["y"])
    x1, y1 = x0 + int(bbox["width"]) - 1, y0 + int(bbox["height"]) - 1
    samples.append((x0, y0, x1, y1))
    del samples[:-7]
    values = np.median(np.asarray(samples, dtype=np.float64), axis=0).round().astype(int)
    sx0, sy0, sx1, sy1 = [int(v) for v in values]
    return {"x": sx0, "y": sy0, "width": max(1, sx1 - sx0 + 1), "height": max(1, sy1 - sy0 + 1)}


def _map_point_to_mm(x: float, y_local: float, cfg: dict[str, Any]) -> Optional[tuple[float, float]]:
    homography = cfg.get("homography_image_to_mm")
    if homography is None:
        return None
    try:
        matrix = np.asarray(homography, dtype=np.float64).reshape(3, 3)
        y_global = float(y_local) + float(cfg.get("_roi_top_y", 0))
        point = np.asarray([[[float(x), y_global]]], dtype=np.float64)
        mapped = cv2.perspectiveTransform(point, matrix)[0, 0]
        if not np.all(np.isfinite(mapped)):
            return None
        return float(mapped[0]), float(mapped[1])
    except (ValueError, TypeError, cv2.error):
        return None


def _distance_in_mm(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def _apply_homography_to_dimensions(dimensions: list[dict[str, Any]], bbox: dict[str, int], circles: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """Use a planar image-to-mm homography when available; return physical bbox X/Y spans."""
    if cfg.get("homography_image_to_mm") is None:
        return None, None
    for dimension in dimensions:
        p1_local, p2_local = dimension["p1"], dimension["p2"]
        p1 = _map_point_to_mm(p1_local["x"], p1_local["y"], cfg)
        p2 = _map_point_to_mm(p2_local["x"], p2_local["y"], cfg)
        if p1 is not None and p2 is not None:
            value = _distance_in_mm(p1, p2)
            dimension["distance"] = round(value, 4)
            dimension["distance_label"] = f"{value:.2f} mm"
            dimension["calibration_method"] = "planar_homography"
    bx, by, bw, bh = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])
    p_left = _map_point_to_mm(bx, by + (bh - 1) / 2.0, cfg)
    p_right = _map_point_to_mm(bx + bw - 1, by + (bh - 1) / 2.0, cfg)
    p_top = _map_point_to_mm(bx + (bw - 1) / 2.0, by, cfg)
    p_bottom = _map_point_to_mm(bx + (bw - 1) / 2.0, by + bh - 1, cfg)
    length_mm = _distance_in_mm(p_left, p_right) if p_left and p_right else None
    width_mm = _distance_in_mm(p_top, p_bottom) if p_top and p_bottom else None
    # Transform a sampled circle circumference; a homography corrects perspective for
    # circles lying on the calibrated plane, unlike a single global mm/px factor.
    for circle in circles:
        points = []
        for angle in np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False):
            x = circle["cx"] + circle["radius_px"] * math.cos(float(angle))
            y = circle["cy"] + circle["radius_px"] * math.sin(float(angle))
            mapped = _map_point_to_mm(x, y, cfg)
            if mapped is not None:
                points.append(mapped)
        if len(points) >= 20:
            pts = np.asarray(points, dtype=np.float64)
            a = np.column_stack((pts[:, 0], pts[:, 1], np.ones(len(pts))))
            b = -(pts[:, 0] ** 2 + pts[:, 1] ** 2)
            try:
                d, e, f = np.linalg.lstsq(a, b, rcond=None)[0]
                cx, cy = -d / 2.0, -e / 2.0
                radius_sq = cx * cx + cy * cy - f
                if radius_sq > 0 and math.isfinite(radius_sq):
                    diameter_mm = 2.0 * math.sqrt(radius_sq)
                    circle["diameter"] = round(diameter_mm, 4)
                    circle["diameter_label"] = f"{diameter_mm:.2f} mm"
                    circle["diameter_mm"] = round(diameter_mm, 4)
                    circle["calibration_method"] = "planar_homography"
            except np.linalg.LinAlgError:
                pass
    return length_mm, width_mm


def _format_distance(px: float, mm_per_px: Optional[float]) -> tuple[float, str]:
    value, label = _measure_label(px, mm_per_px)
    return round(value, 4), label


def _draw_double_arrow(image: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int], thickness: int = 2) -> None:
    """Draw a double-ended dimension arrow with robust tiny-span handling."""
    cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 8:
        return
    ux, uy = dx / length, dy / length
    head = max(5.0, min(12.0, length * 0.08))
    wing = head * 0.48
    for point, direction in ((p1, 1.0), (p2, -1.0)):
        bx = point[0] + direction * ux * head
        by = point[1] + direction * uy * head
        p_a = (int(round(bx - uy * wing)), int(round(by + ux * wing)))
        p_b = (int(round(bx + uy * wing)), int(round(by - ux * wing)))
        cv2.line(image, point, p_a, color, thickness, cv2.LINE_AA)
        cv2.line(image, point, p_b, color, thickness, cv2.LINE_AA)


def _draw_label(image: np.ndarray, text: str, anchor: tuple[int, int], color: tuple[int, int, int], scale: float = 0.54) -> None:
    h, w = image.shape[:2]
    x = max(3, min(w - 4, int(anchor[0])))
    y = max(16, min(h - 3, int(anchor[1])))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = max(3, min(w - tw - 5, x))
    y = max(th + 5, min(h - baseline - 3, y))
    cv2.rectangle(image, (x - 2, y - th - 4), (min(w - 1, x + tw + 3), min(h - 1, y + baseline + 2)), (15, 20, 25), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _build_dimensions(
    horizontal: list[dict[str, Any]],
    vertical: list[dict[str, Any]],
    bbox: dict[str, Any],
    image_width: int,
    image_height: int,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Calculate adjacent guide gaps plus global X/Y silhouette spans."""
    mm_per_px = cfg.get("mm_per_px")
    minimum = max(1.0, float(cfg.get("min_dimension_px", 18.0)))
    x0, y0 = float(bbox["x"]), float(bbox["y"])
    bw, bh = float(bbox["width"]), float(bbox["height"])
    rows: list[dict[str, Any]] = []

    def add(axis: str, id_: str, line_a: str, line_b: str, coord_a: float, coord_b: float,
            anchor: float, overall: bool = False) -> None:
        px = abs(coord_b - coord_a)
        if px < minimum:
            return
        value, label = _format_distance(px, mm_per_px)
        if axis == "X":
            p1 = {"x": round(min(coord_a, coord_b), 2), "y": round(anchor, 2)}
            p2 = {"x": round(max(coord_a, coord_b), 2), "y": round(anchor, 2)}
        else:
            p1 = {"x": round(anchor, 2), "y": round(min(coord_a, coord_b), 2)}
            p2 = {"x": round(anchor, 2), "y": round(max(coord_a, coord_b), 2)}
        rows.append({
            "id": id_, "axis": axis, "line_a": line_a, "line_b": line_b,
            "distance_px": round(px, 2), "distance": value, "distance_label": label,
            "overall": bool(overall), "p1": p1, "p2": p2,
            "midpoint": {"x": round((p1["x"] + p2["x"]) / 2, 2), "y": round((p1["y"] + p2["y"]) / 2, 2)},
        })

    # X measurements are distances between vertical guides. Spread arrow anchors to reduce label collisions.
    v_sorted = sorted(vertical, key=lambda item: item["coord"])
    for i, (a, b) in enumerate(zip(v_sorted, v_sorted[1:]), 1):
        overlap_lo = max(float(a["start"]), float(b["start"]))
        overlap_hi = min(float(a["end"]), float(b["end"]))
        if overlap_hi - overlap_lo >= 20:
            y_anchor = (overlap_lo + overlap_hi) / 2.0
        else:
            y_anchor = y0 + bh * (0.28 + 0.12 * ((i - 1) % 4))
        y_anchor = max(8.0, min(image_height - 8.0, y_anchor))
        add("X", f"X{i:02d}", a["id"], b["id"], float(a["coord"]), float(b["coord"]), y_anchor)
    if bw >= minimum:
        add("X", "X-OVERALL", "SILHOUETTE-LEFT", "SILHOUETTE-RIGHT", x0, x0 + bw - 1,
            min(image_height - 10.0, y0 + bh + 25.0) if y0 + bh + 25.0 < image_height else max(10.0, y0 - 18.0), True)

    # Y measurements are distances between horizontal guides.
    h_sorted = sorted(horizontal, key=lambda item: item["coord"])
    for i, (a, b) in enumerate(zip(h_sorted, h_sorted[1:]), 1):
        overlap_lo = max(float(a["start"]), float(b["start"]))
        overlap_hi = min(float(a["end"]), float(b["end"]))
        if overlap_hi - overlap_lo >= 20:
            x_anchor = (overlap_lo + overlap_hi) / 2.0
        else:
            x_anchor = x0 + bw * (0.28 + 0.12 * ((i - 1) % 4))
        x_anchor = max(8.0, min(image_width - 8.0, x_anchor))
        add("Y", f"Y{i:02d}", a["id"], b["id"], float(a["coord"]), float(b["coord"]), x_anchor)
    if bh >= minimum:
        # Place the full-height arrow outside the left edge when there is room; it avoids the right-side X-gap labels.
        overall_y_x = x0 - 24.0 if x0 >= 45.0 else (x0 + bw + 24.0 if x0 + bw + 24.0 < image_width else image_width - 20.0)
        add("Y", "Y-OVERALL", "SILHOUETTE-TOP", "SILHOUETTE-BOTTOM", y0, y0 + bh - 1,
            max(10.0, min(image_width - 10.0, overall_y_x)), True)
    return rows


def _draw_axis_guides(annotated: np.ndarray, horizontal: list[dict[str, Any]], vertical: list[dict[str, Any]]) -> None:
    h, w = annotated.shape[:2]
    # Red: horizontal edge guides; blue: vertical edge guides. These are extended through the frame.
    red = (50, 55, 245)
    blue = (245, 95, 35)
    for line in horizontal:
        y = int(round(line["coord"]))
        cv2.line(annotated, (0, y), (w - 1, y), red, 2, cv2.LINE_AA)
        cv2.circle(annotated, (max(0, min(w - 1, int(round((line["start"] + line["end"]) / 2)))), max(0, min(h - 1, y))), 3, (255, 255, 255), -1, cv2.LINE_AA)
        # Put labels at the upper-left edge, with slight vertical staggering.
        _draw_label(annotated, line["id"], (12 + (int(line["id"][1:]) % 4) * 52, y - 4), red, 0.48)
    for line in vertical:
        x = int(round(line["coord"]))
        cv2.line(annotated, (x, 0), (x, h - 1), blue, 2, cv2.LINE_AA)
        y_mid = max(0, min(h - 1, int(round((line["start"] + line["end"]) / 2))))
        cv2.circle(annotated, (max(0, min(w - 1, x)), y_mid), 3, (255, 255, 255), -1, cv2.LINE_AA)
        _draw_label(annotated, line["id"], (x + 4, 90 + (int(line["id"][1:]) % 5) * 18), blue, 0.48)


def _draw_dimension_arrows(annotated: np.ndarray, dimensions: list[dict[str, Any]]) -> None:
    green = (50, 220, 100)
    for dimension in dimensions:
        p1 = (int(round(dimension["p1"]["x"])), int(round(dimension["p1"]["y"])))
        p2 = (int(round(dimension["p2"]["x"])), int(round(dimension["p2"]["y"])))
        _draw_double_arrow(annotated, p1, p2, green, thickness=3 if dimension.get("overall") else 2)
        mid = dimension["midpoint"]
        cv2.circle(annotated, (int(round(mid["x"])), int(round(mid["y"]))), 3, (20, 20, 20), -1, cv2.LINE_AA)
        label = f'{dimension["id"]} {dimension["distance_label"]}'
        if dimension["axis"] == "X":
            anchor = (int(round(mid["x"] - 34)), int(round(mid["y"] - 8)))
        else:
            anchor = (int(round(mid["x"] - 84 if dimension.get("overall") else mid["x"] + 5)), int(round(mid["y"])))
        _draw_label(annotated, label, anchor, green, 0.52)


def _process_measurement_region(frame: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """Measure horizontal/vertical guide spacing and circle diameters from a silhouette.

    Horizontal red and vertical blue guides are extended across the frame. Green
    double arrows show adjacent guide spacing plus the overall silhouette X/Y spans.
    Values are pixels until a known planar reference calibrates mm_per_px.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty.")
    gray, bgr = _gray_bgr(frame)
    h, w = gray.shape[:2]
    temporal_state = _prepare_temporal_state(cfg, (h, w))
    blur = _odd(int(cfg.get("blur_kernel", 5)), 15)
    smooth = cv2.GaussianBlur(gray, (blur, blur), 0) if blur > 1 else gray
    canny_low = max(0, min(254, int(cfg.get("canny_low", 45))))
    canny_high = max(canny_low + 1, min(255, int(cfg.get("canny_high", 130))))
    # CLAHE normalizes moderate global lighting drift before edge preview/thresholding.
    try:
        normalized = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(smooth)
    except cv2.error:
        normalized = smooth
    edge_preview = cv2.Canny(normalized, canny_low, canny_high)
    contour, silhouette_mask, silhouette = extract_silhouette(gray, cfg)
    mm_per_px = cfg.get("mm_per_px")
    annotated = bgr.copy()
    measurements: dict[str, Any] = {
        "image_width": int(w), "image_height": int(h),
        "silhouette": silhouette, "edge_pixels": int(cv2.countNonZero(edge_preview)),
        "length_px": None, "width_px": None, "length": None, "width": None,
        "unit": "mm" if mm_per_px else "px", "mm_per_px": mm_per_px,
        "circles": [], "rectangles": [], "horizontal_guides": [], "vertical_guides": [],
        "linear_measurements": [], "processing_note": "",
    }
    if contour is None or len(contour) < 5:
        measurements["processing_note"] = "No convincing silhouette found. Improve contrast or adjust segmentation settings."
        cv2.putText(annotated, "NO SILHOUETTE DETECTED", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return annotated, measurements, silhouette_mask, edge_preview

    bbox_raw = cv2.boundingRect(contour)
    bx, by, bw, bh = [int(v) for v in bbox_raw]
    bbox = _stabilize_bbox({"x": bx, "y": by, "width": bw, "height": bh}, temporal_state)
    bx, by, bw, bh = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    silhouette["bbox"] = dict(bbox)
    silhouette["area_ratio"] = round(float(silhouette.get("area_px", 0.0)) / max(1.0, float(w * h)), 6)
    # Draw slightly bolder overlays so the guides remain visible on industrial footage.
    cv2.drawContours(annotated, [contour], -1, (60, 235, 80), 3, cv2.LINE_AA)
    x_value, x_label = _measure_label(float(max(0, bw - 1)), mm_per_px)
    y_value, y_label = _measure_label(float(max(0, bh - 1)), mm_per_px)
    cv2.rectangle(annotated, (bx, by), (min(w - 1, bx + bw - 1), min(h - 1, by + bh - 1)), (0, 155, 255), 2, cv2.LINE_AA)
    horizontal, vertical, silhouette_edges = detect_axis_guides(silhouette_mask, cfg)
    circles = detect_circles(gray, silhouette_mask, cfg)
    if temporal_state is not None:
        horizontal = _stabilize_features(horizontal, temporal_state, "horizontal", cfg)
        vertical = _stabilize_features(vertical, temporal_state, "vertical", cfg)
        circles = _stabilize_features(circles, temporal_state, "circles", cfg)
    measurements["length_px"] = round(float(max(0, bw - 1)), 3)
    measurements["width_px"] = round(float(max(0, bh - 1)), 3)
    measurements["length"] = round(x_value, 4)
    measurements["width"] = round(y_value, 4)
    measurements["length_label"] = x_label
    measurements["width_label"] = y_label
    measurements["horizontal_span_label"] = x_label
    measurements["vertical_span_label"] = y_label
    measurements["horizontal_guides"] = horizontal
    measurements["vertical_guides"] = vertical
    bbox_dict = {"x": int(bx), "y": int(by), "width": int(bw), "height": int(bh)}
    dimensions = _build_dimensions(horizontal, vertical, bbox_dict, w, h, cfg)
    length_mm_h, width_mm_h = _apply_homography_to_dimensions(dimensions, bbox_dict, circles, cfg)
    if length_mm_h is not None:
        measurements["length"] = round(length_mm_h, 4)
        measurements["length_label"] = f"{length_mm_h:.2f} mm"
        measurements["horizontal_span_label"] = measurements["length_label"]
    if width_mm_h is not None:
        measurements["width"] = round(width_mm_h, 4)
        measurements["width_label"] = f"{width_mm_h:.2f} mm"
        measurements["vertical_span_label"] = measurements["width_label"]
    measurements["linear_measurements"] = dimensions
    measurements["horizontal_guide_count"] = len(horizontal)
    measurements["vertical_guide_count"] = len(vertical)
    measurements["guide_edge_pixels"] = int(cv2.countNonZero(silhouette_edges))
    _draw_axis_guides(annotated, horizontal, vertical)
    _draw_dimension_arrows(annotated, dimensions)

    # Circle detections were filtered and temporally stabilized before drawing.
    measurements["circles"] = circles
    for circle in circles:
        cx, cy, r = int(round(circle["cx"])), int(round(circle["cy"])), int(round(circle["radius_px"]))
        cv2.circle(annotated, (cx, cy), r, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)
        label = f'{circle["id"]} DIA {circle["diameter_label"]}'
        tx = max(5, min(w - 300, cx - r))
        ty = max(22, cy - r - 8)
        _draw_label(annotated, label, (tx, ty), (0, 255, 255), 0.52)

    unit_text = "mm (calibrated)" if mm_per_px else "pixels (uncalibrated)"
    cv2.rectangle(annotated, (8, 8), (min(w - 8, 700), 50), (20, 25, 32), -1)
    cv2.putText(annotated, f"MEASUREMENT: {unit_text} | {w}x{h}", (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"GUIDES: H={len(horizontal)} V={len(vertical)} | X spans={sum(1 for d in dimensions if d['axis']=='X')} Y spans={sum(1 for d in dimensions if d['axis']=='Y')}", (20, min(h - 12, 76)), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (100, 240, 150), 2, cv2.LINE_AA)
    measurements["processing_note"] = "Red guides = horizontal boundary lines; blue guides = vertical boundary lines; green arrows = inter-guide distances and overall X/Y spans. These are image-plane measurements; inspect segmentation and calibration before metrology use."
    return annotated, measurements, silhouette_mask, edge_preview


def process_measurement(frame: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """Crop rows above the live ROI cutoff, then run the complete pipeline only below it.

    The ROI spans the full image width and extends to the frame bottom. All guide
    lines and dimensions are therefore clipped at the selected top boundary.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty.")
    gray, bgr = _gray_bgr(frame)
    full_h, full_w = gray.shape[:2]
    roi_pct = max(0.0, min(80.0, float(cfg.get("roi_top_pct", 0.0))))
    top_y = int(round(full_h * roi_pct / 100.0))
    top_y = min(full_h - 1, max(0, top_y))
    region = frame[top_y:, :]
    region_cfg = dict(cfg)
    region_cfg["allow_top_touch"] = top_y > 0
    region_cfg["_roi_top_y"] = top_y

    # Homography calibration is valid only at the original calibrated frame resolution.
    calibration_size = region_cfg.get("calibration_image_size")
    calibration_resolution_matches = True
    if calibration_size and len(calibration_size) >= 2:
        try:
            calibrated_w, calibrated_h = int(calibration_size[0]), int(calibration_size[1])
            calibration_resolution_matches = (calibrated_w, calibrated_h) == (full_w, full_h)
            if not calibration_resolution_matches:
                # A saved px/mm scale is also tied to source resolution. Do not
                # silently apply either it or the homography to a resized frame.
                region_cfg["homography_image_to_mm"] = None
                region_cfg["mm_per_px"] = None
        except (TypeError, ValueError):
            calibration_resolution_matches = False
            region_cfg["homography_image_to_mm"] = None
            region_cfg["mm_per_px"] = None

    overlay_region, data, mask_region, edge_region = _process_measurement_region(region, region_cfg)
    overlay = bgr.copy()
    if top_y > 0:
        # Dim excluded image rows to make the active processing region obvious.
        excluded = overlay[:top_y]
        tint = np.zeros_like(excluded)
        tint[:] = (18, 24, 33)
        overlay[:top_y] = cv2.addWeighted(excluded, 0.40, tint, 0.60, 0.0)
    overlay[top_y:, :] = overlay_region
    mask = np.zeros((full_h, full_w), dtype=np.uint8)
    edges = np.zeros((full_h, full_w), dtype=np.uint8)
    mask[top_y:, :] = mask_region
    edges[top_y:, :] = edge_region

    # The cutoff is the measurement limit: all processing/drawing below it only.
    cutoff_color = (65, 255, 125)
    cv2.line(overlay, (0, top_y), (full_w - 1, top_y), cutoff_color, 3, cv2.LINE_AA)
    cutoff_label = f"ROI TOP LIMIT y={top_y}px ({roi_pct:.1f}%)"
    _draw_label(overlay, cutoff_label, (12, max(20, top_y - 5) if top_y > 18 else top_y + 22), cutoff_color, 0.56)

    data["image_width"] = full_w
    data["image_height"] = full_h
    data["roi"] = {"top_y": top_y, "top_pct": round(roi_pct, 2), "x": 0, "y": top_y, "width": full_w, "height": full_h - top_y}
    if isinstance(data.get("silhouette"), dict):
        data["silhouette"]["area_ratio"] = round(float(data["silhouette"].get("area_px", 0.0)) / max(1.0, float(full_w * full_h)), 6)
        bbox = data["silhouette"].get("bbox")
        if bbox:
            bbox["y"] = int(bbox.get("y", 0)) + top_y
    for guide in data.get("horizontal_guides", []):
        guide["coord"] = round(float(guide.get("coord", 0)) + top_y, 2)
    for guide in data.get("vertical_guides", []):
        guide["start"] = round(float(guide.get("start", 0)) + top_y, 2)
        guide["end"] = round(float(guide.get("end", 0)) + top_y, 2)
    for dimension in data.get("linear_measurements", []):
        for key in ("p1", "p2", "midpoint"):
            if key in dimension and isinstance(dimension[key], dict):
                dimension[key]["y"] = round(float(dimension[key].get("y", 0)) + top_y, 2)
    for circle in data.get("circles", []):
        circle["cy"] = round(float(circle.get("cy", 0)) + top_y, 2)
    if region_cfg.get("homography_image_to_mm") is not None:
        data["calibration_mode"] = "planar homography"
    elif region_cfg.get("mm_per_px"):
        data["calibration_mode"] = "single planar scale"
    else:
        data["calibration_mode"] = "uncalibrated"
    data["calibration_resolution_matches"] = bool(calibration_resolution_matches)
    return overlay, data, mask, edges

