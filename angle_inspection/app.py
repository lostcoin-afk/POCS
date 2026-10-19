"""Silhouette/corner hardware inspection with FastAPI, OpenCV, and optional LUCID Arena SDK.

Run from this directory with:
    uvicorn app:app --host 0.0.0.0 --port 8000

For LUCID cameras, install the official Arena SDK Python package separately and run
this application in the same Python environment in which arena_api is available.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from geometry import (
    build_contour_geometry,
    compare_corner_template,
    draw_geometry_overlay,
    extract_component_silhouette,
    make_template_payload,
    FEATURE_TYPE,
)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
TEMPLATE_FILE = DATA_DIR / "angle_template.json"
for _directory in (TEMPLATES_DIR, DATA_DIR, UPLOAD_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

APP_TITLE = "Hardware Geometry Inspector"
DEFAULT_PROCESSING: dict[str, Any] = {
    "canny_low": 40,
    "canny_high": 120,
    "blur_kernel": 3,
    "contour_epsilon_ratio": 0.004,
    "silhouette_morph_kernel": 5,
    "min_component_area_ratio": 0.01,
    "max_component_area_ratio": 0.85,
    "angle_tolerance_deg": 3.0,
    "draw_edges": False,
    "draw_angle_labels": True,
}

DEFAULT_SOURCE: dict[str, Any] = {
    "source_type": "video",  # video | webcam | arena
    "video_path": "",
    "webcam_index": 0,
    "arena_serial": "",
    "loop_video": True,
}

app = FastAPI(title=APP_TITLE, version="2.0.0")
app.mount("/templates", StaticFiles(directory=str(TEMPLATES_DIR)), name="static-templates")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def process_frame(frame: np.ndarray, cfg: dict[str, Any], template: Optional[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    """Process a frame using silhouette extraction and physical corner geometry."""
    if frame is None or frame.size == 0:
        raise ValueError("The image source returned an empty frame.")
    if frame.ndim == 2:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        bgr = frame.copy()

    original_h, original_w = bgr.shape[:2]
    max_width = 1440
    if original_w > max_width:
        ratio = max_width / float(original_w)
        bgr = cv2.resize(
            bgr, (int(original_w * ratio), int(original_h * ratio)), interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = max(1, min(15, int(cfg.get("blur_kernel", 3))))
    if kernel % 2 == 0:
        kernel += 1
    smooth = cv2.GaussianBlur(gray, (kernel, kernel), 0) if kernel > 1 else gray
    low = max(0, min(254, int(cfg.get("canny_low", 40))))
    high = max(low + 1, min(255, int(cfg.get("canny_high", 120))))
    canny_edges = cv2.Canny(smooth, low, high)

    contour, silhouette_mask, silhouette_info = extract_component_silhouette(gray, cfg)
    geometry = build_contour_geometry(contour, bgr.shape, cfg)
    edges = geometry.get("edges", [])
    corners = geometry.get("corners", [])
    tolerance = float(cfg.get("angle_tolerance_deg", 3.0))
    comparisons = compare_corner_template(corners, template, tolerance)
    annotated = draw_geometry_overlay(bgr, canny_edges, contour, geometry, comparisons, cfg)

    safe_corners = []
    for corner in corners:
        safe_corners.append({
            key: (round(value, 4) if isinstance(value, float) else value)
            for key, value in corner.items()
        })
    safe_edges = []
    for edge in edges:
        safe_edges.append({
            key: (round(value, 4) if isinstance(value, float) else value)
            for key, value in edge.items()
        })

    response = {
        "timestamp": time.time(),
        "image_width": int(bgr.shape[1]),
        "image_height": int(bgr.shape[0]),
        "original_width": int(original_w),
        "original_height": int(original_h),
        "edge_pixels": int(cv2.countNonZero(canny_edges)),
        "raw_hough_segments": 0,
        "line_count": len(edges),  # retained for clients that still call this field lines
        "edge_count": len(edges),
        "corner_count": len(corners),
        "lines": safe_edges,
        "corners": safe_corners,
        "current_angle_pairs": comparisons.get("corner_pairs", []),
        "comparison": comparisons,
        "silhouette": silhouette_info,
        "silhouette_mask_pixels": int(cv2.countNonZero(silhouette_mask)),
        "contour_epsilon_px": round(float(geometry.get("epsilon_px", 0.0)), 3),
        "processing_ms": None,
    }
    return annotated, response


# ------------------------------- frame sources ------------------------------
class FrameSource:
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class OpenCVSource(FrameSource):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        source_type = config.get("source_type", "video")
        if source_type == "webcam":
            index = int(config.get("webcam_index", 0))
            self.capture = cv2.VideoCapture(index)
            label = f"webcam index {index}"
        else:
            video_path = str(config.get("video_path", "")).strip()
            if not video_path:
                raise RuntimeError("Enter a video path or upload a video before starting.")
            path = Path(video_path).expanduser()
            if not path.is_absolute():
                path = (BASE_DIR / path).resolve()
            if not path.exists() or not path.is_file():
                raise RuntimeError(f"Video file not found: {path}")
            self.path = str(path)
            self.capture = cv2.VideoCapture(self.path)
            label = f"video: {path.name}"
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open {label}. Check the source and codec support.")
        self.loop_video = bool(config.get("loop_video", True)) and source_type == "video"
        self.label = label
        reported_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_interval = 1.0 / min(120.0, max(1.0, reported_fps)) if 0.1 < reported_fps < 1000 else 1.0 / 25.0
        self.next_read_at: Optional[float] = None

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        # Keep prerecorded video close to its natural playback rate; camera sources
        # are not paced here because the camera controls its own acquisition rate.
        if self.next_read_at is not None:
            delay = self.next_read_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        ok, frame = self.capture.read()
        if not ok and self.loop_video:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        self.next_read_at = time.monotonic() + self.frame_interval
        return ok, frame

    def close(self) -> None:
        self.capture.release()


class ArenaSource(FrameSource):
    """LUCID Arena SDK adapter. The Python binding ships with LUCID's Arena SDK."""
    _COLOR_CODES = {
        "BayerRG8": cv2.COLOR_BAYER_RG2BGR,
        "BayerGR8": cv2.COLOR_BAYER_GR2BGR,
        "BayerGB8": cv2.COLOR_BAYER_GB2BGR,
        "BayerBG8": cv2.COLOR_BAYER_BG2BGR,
    }

    def __init__(self, config: dict[str, Any]):
        try:
            from arena_api.system import system  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Arena SDK Python package is unavailable. Install LUCID's Arena SDK, "
                "activate its Python environment, then restart this app."
            ) from exc
        self.system = system
        self.device = None
        self.streaming = False
        serial = str(config.get("arena_serial", "")).strip()
        try:
            if serial:
                infos = system.device_infos
                selected = next((info for info in infos if str(info.get("serial", "")) == serial), None)
                if selected is None:
                    available = ", ".join(str(info.get("serial", "unknown")) for info in infos) or "none"
                    raise RuntimeError(f"LUCID camera serial {serial} was not found. Available serials: {available}")
                devices = system.create_device(device_infos=selected)
            else:
                devices = system.create_device()
            if not devices:
                raise RuntimeError("No LUCID Arena cameras found. Check power, Ethernet, network adapter, and ArenaView.")
            self.device = devices[0]
            self.pixel_format = str(self.device.nodemap["PixelFormat"].value)
            # Avoid packed >8-bit formats here. Inspection can still work well with Mono8 or 8-bit color.
            allowed = {"Mono8", "BayerRG8", "BayerGR8", "BayerGB8", "BayerBG8", "RGB8", "BGR8", "RGBA8", "BGRA8"}
            if self.pixel_format not in allowed:
                try:
                    self.device.nodemap["PixelFormat"].value = "Mono8"
                    self.pixel_format = str(self.device.nodemap["PixelFormat"].value)
                except Exception as format_exc:
                    raise RuntimeError(
                        f"Camera pixel format '{self.pixel_format}' is not supported by this example. "
                        "Set the camera to Mono8, an 8-bit Bayer format, RGB8, or BGR8 in ArenaView."
                    ) from format_exc
            self.device.start_stream()
            self.streaming = True
            info = getattr(self.device, "nodemap", None)
            try:
                model = str(info["DeviceModelName"].value) if info is not None else "LUCID camera"
            except Exception:
                model = "LUCID camera"
            self.label = f"Arena SDK: {model} ({serial or 'first available camera'})"
        except Exception:
            self.close()
            raise

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        assert self.device is not None
        buffer = self.device.get_buffer()
        try:
            height = int(buffer.height)
            width = int(buffer.width)
            bits = int(buffer.bits_per_pixel)
            if bits not in (8, 24, 32):
                raise RuntimeError(f"Unsupported Arena image depth: {bits} bits per pixel. Configure an 8-bit pixel format.")
            channels = bits // 8
            flat = np.ctypeslib.as_array(buffer.pdata, shape=(height * width * channels,)).copy()
            image = flat.reshape((height, width) if channels == 1 else (height, width, channels))
            pixel_format = self.pixel_format
        finally:
            # Copy the data before requeueing: Arena may reuse the SDK buffer after this call.
            self.device.requeue_buffer(buffer)

        if pixel_format in self._COLOR_CODES and image.ndim == 2:
            image = cv2.cvtColor(image, self._COLOR_CODES[pixel_format])
        elif pixel_format == "RGB8" and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif pixel_format == "RGBA8" and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif pixel_format == "BGRA8" and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        return True, image

    def close(self) -> None:
        if self.device is not None:
            try:
                if self.streaming:
                    self.device.stop_stream()
            except Exception:
                pass
            try:
                self.system.destroy_device(self.device)
            except Exception:
                pass
            self.device = None
            self.streaming = False


def create_source(config: dict[str, Any]) -> FrameSource:
    source_type = config.get("source_type", "video")
    if source_type == "arena":
        return ArenaSource(config)
    if source_type in ("video", "webcam"):
        return OpenCVSource(config)
    raise RuntimeError(f"Unknown source type: {source_type}")


# ------------------------------ service/thread ------------------------------
def _placeholder_jpeg(message: str = "Select a source and click Start") -> bytes:
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    image[:] = (25, 30, 42)
    cv2.putText(image, "HARDWARE GEOMETRY INSPECTOR", (45, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (230, 238, 248), 2, cv2.LINE_AA)
    cv2.putText(image, message[:90], (45, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (150, 180, 210), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 84])[1].tobytes()


class InspectionService:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.source_config = dict(DEFAULT_SOURCE)
        self.processing_config = dict(DEFAULT_PROCESSING)
        self.template: Optional[dict[str, Any]] = self._load_template()
        # A template built with build_template.py stores its silhouette/contour settings;
        # reuse them so reference creation and live inspection use the same geometry rules.
        saved_processing = self.template.get("processing_config") if self.template else None
        if isinstance(saved_processing, dict):
            self.processing_config.update(
                {key: value for key, value in saved_processing.items() if key in DEFAULT_PROCESSING}
            )
        self.latest_jpeg = _placeholder_jpeg()
        self.latest_data: dict[str, Any] = {
            "timestamp": time.time(), "image_width": 960, "image_height": 540,
            "edge_pixels": 0, "raw_hough_segments": 0, "line_count": 0, "edge_count": 0,
            "corner_count": 0, "lines": [], "corners": [], "current_angle_pairs": [],
            "silhouette": {"found": False, "method": "none", "area_px": 0.0, "area_ratio": 0.0, "bbox": None},
            "comparison": compare_corner_template([], self.template, float(self.processing_config["angle_tolerance_deg"])),
            "processing_ms": None,
        }
        self.status: dict[str, Any] = {
            "running": False,
            "source_label": "Not connected",
            "source_type": self.source_config["source_type"],
            "message": "Choose a source and click Start.",
            "error": None,
            "fps": 0.0,
            "last_frame_time": None,
            "template_saved": bool(self.template),
            "template_name": self.template.get("name") if self.template else None,
        }

    def _load_template(self) -> Optional[dict[str, Any]]:
        try:
            if TEMPLATE_FILE.exists():
                payload = json.loads(TEMPLATE_FILE.read_text(encoding="utf-8"))
                # Old line-pair templates are intentionally not loaded: their identities and
                # pairwise angles are not compatible with corner-based reference matching.
                if (
                    isinstance(payload, dict)
                    and payload.get("feature_type") == FEATURE_TYPE
                    and isinstance(payload.get("corners"), list)
                    and payload.get("corners")
                ):
                    return payload
        except Exception:
            pass
        return None


    def start(self, source_config: dict[str, Any]) -> None:
        self.stop()
        with self.lock:
            self.source_config = {**DEFAULT_SOURCE, **source_config}
            self.status.update({
                "running": True,
                "source_label": "Connecting…",
                "source_type": self.source_config.get("source_type"),
                "message": "Connecting to image source…",
                "error": None,
            })
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run, name="inspection-capture", daemon=True)
            self.thread.start()

    def stop(self) -> None:
        with self.lock:
            thread = self.thread
            self.stop_event.set()
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        with self.lock:
            if not thread or not thread.is_alive():
                self.thread = None
                self.status["running"] = False
                if not self.status.get("error"):
                    self.status["message"] = "Stopped."

    def update_processing(self, updates: dict[str, Any]) -> None:
        allowed = set(DEFAULT_PROCESSING)
        clean: dict[str, Any] = {}
        float_keys = {"contour_epsilon_ratio", "min_component_area_ratio", "max_component_area_ratio", "angle_tolerance_deg"}
        bool_keys = {"draw_edges", "draw_angle_labels"}
        for key, value in updates.items():
            if key not in allowed:
                continue
            if key in bool_keys:
                clean[key] = bool(value)
            elif key in float_keys:
                clean[key] = float(value)
            else:
                clean[key] = int(value)
        with self.lock:
            self.processing_config.update(clean)
            self.processing_config["canny_low"] = max(0, min(254, self.processing_config["canny_low"]))
            self.processing_config["canny_high"] = max(self.processing_config["canny_low"] + 1, min(255, self.processing_config["canny_high"]))
            self.processing_config["blur_kernel"] = max(1, min(15, self.processing_config["blur_kernel"]))
            self.processing_config["silhouette_morph_kernel"] = max(1, min(21, self.processing_config["silhouette_morph_kernel"]))
            if self.processing_config["silhouette_morph_kernel"] % 2 == 0:
                self.processing_config["silhouette_morph_kernel"] += 1
            self.processing_config["contour_epsilon_ratio"] = max(0.001, min(0.03, self.processing_config["contour_epsilon_ratio"]))
            self.processing_config["min_component_area_ratio"] = max(0.001, min(0.5, self.processing_config["min_component_area_ratio"]))
            self.processing_config["max_component_area_ratio"] = max(0.1, min(0.95, self.processing_config["max_component_area_ratio"]))
            if self.processing_config["max_component_area_ratio"] <= self.processing_config["min_component_area_ratio"]:
                self.processing_config["max_component_area_ratio"] = min(0.95, self.processing_config["min_component_area_ratio"] + 0.1)
            self.processing_config["angle_tolerance_deg"] = max(0.1, min(30.0, self.processing_config["angle_tolerance_deg"]))


    def save_template(self, name: str = "Reference hardware") -> dict[str, Any]:
        with self.lock:
            if not self.latest_data.get("silhouette", {}).get("found"):
                raise ValueError("No complete component silhouette was detected. Improve lighting/background separation and try again.")
            try:
                template = make_template_payload(
                    self.latest_data,
                    name,
                    dict(self.processing_config),
                    str(self.status.get("source_label", "unknown source")),
                )
            except ValueError:
                raise
            temporary = TEMPLATE_FILE.with_suffix(TEMPLATE_FILE.suffix + ".tmp")
            temporary.write_text(json.dumps(template, indent=2), encoding="utf-8")
            temporary.replace(TEMPLATE_FILE)
            self.template = template
            self.status["template_saved"] = True
            self.status["template_name"] = template["name"]
            current_corners = [dict(corner) for corner in self.latest_data.get("corners", [])]
            self.latest_data["comparison"] = compare_corner_template(
                current_corners, self.template, float(self.processing_config["angle_tolerance_deg"])
            )
            self.latest_data["current_angle_pairs"] = self.latest_data["comparison"].get("corner_pairs", [])
            return template

    def _internal_lines_from_latest(self) -> list[dict[str, Any]]:
        return [dict(line) for line in self.latest_data.get("lines", [])]

    def _internal_corners_from_latest(self) -> list[dict[str, Any]]:
        return [dict(corner) for corner in self.latest_data.get("corners", [])]


    def clear_template(self) -> None:
        with self.lock:
            self.template = None
            try:
                TEMPLATE_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            self.status["template_saved"] = False
            self.status["template_name"] = None
            self.latest_data["comparison"] = compare_corner_template(
                self._internal_corners_from_latest(), None,
                float(self.processing_config["angle_tolerance_deg"])
            )
            self.latest_data["current_angle_pairs"] = self.latest_data["comparison"].get("corner_pairs", [])

    def get_snapshot(self) -> tuple[bytes, dict[str, Any], dict[str, Any], dict[str, Any]]:
        with self.lock:
            return self.latest_jpeg, dict(self.latest_data), dict(self.status), dict(self.processing_config)

    def _run(self) -> None:
        source: Optional[FrameSource] = None
        previous_frame_time: Optional[float] = None
        fps_samples: list[float] = []
        try:
            with self.lock:
                config = dict(self.source_config)
            source = create_source(config)
            with self.lock:
                self.status["source_label"] = getattr(source, "label", config.get("source_type", "source"))
                self.status["message"] = "Streaming and inspecting frames."
                self.status["error"] = None
            while not self.stop_event.is_set():
                ok, frame = source.read()
                if not ok or frame is None:
                    if config.get("source_type") == "video" and not config.get("loop_video", True):
                        with self.lock:
                            self.status["message"] = "Video reached its end."
                        break
                    time.sleep(0.01)
                    continue
                tick = time.perf_counter()
                with self.lock:
                    proc_cfg = dict(self.processing_config)
                    template = self.template
                annotated, data = process_frame(frame, proc_cfg, template)
                processing_ms = (time.perf_counter() - tick) * 1000.0
                data["processing_ms"] = round(processing_ms, 2)
                jpeg_quality = 84
                ok_jpeg, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                if not ok_jpeg:
                    continue
                now = time.time()
                if previous_frame_time is not None:
                    delta = now - previous_frame_time
                    if delta > 0:
                        fps_samples.append(1.0 / delta)
                        fps_samples = fps_samples[-20:]
                previous_frame_time = now
                with self.lock:
                    self.latest_jpeg = encoded.tobytes()
                    self.latest_data = data
                    self.status.update({
                        "running": True,
                        "source_label": getattr(source, "label", config.get("source_type", "source")),
                        "source_type": config.get("source_type"),
                        "message": "Streaming and inspecting frames.",
                        "error": None,
                        "fps": round(sum(fps_samples) / len(fps_samples), 1) if fps_samples else 0.0,
                        "last_frame_time": now,
                        "template_saved": bool(self.template),
                        "template_name": self.template.get("name") if self.template else None,
                    })
        except Exception as exc:
            with self.lock:
                self.status.update({
                    "running": False,
                    "message": "Could not start or process the selected source.",
                    "error": str(exc),
                })
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception:
                    pass
            with self.lock:
                self.status["running"] = False
                if self.status.get("message") == "Streaming and inspecting frames.":
                    self.status["message"] = "Stopped."


service = InspectionService()


# --------------------------------- web API ----------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html", context={"app_title": APP_TITLE})


@app.get("/api/status")
def get_status() -> JSONResponse:
    _, data, status, processing = service.get_snapshot()
    return JSONResponse({
        "status": status,
        "processing": processing,
        "data": data,
        "template": {
            "saved": bool(service.template),
            "name": service.template.get("name") if service.template else None,
            "created_at": service.template.get("created_at") if service.template else None,
            "line_count": len(service.template.get("edges", [])) if service.template else 0,
            "edge_count": len(service.template.get("edges", [])) if service.template else 0,
            "corner_count": len(service.template.get("corners", [])) if service.template else 0,
        },
        "source_config": service.source_config,
    })


@app.post("/api/start")
def start_source(payload: dict[str, Any]) -> JSONResponse:
    source_type = str(payload.get("source_type", "video")).lower()
    if source_type not in {"video", "webcam", "arena"}:
        raise HTTPException(status_code=400, detail="source_type must be video, webcam, or arena")
    clean_source = {
        "source_type": source_type,
        "video_path": str(payload.get("video_path", "")).strip(),
        "webcam_index": max(0, int(payload.get("webcam_index", 0))),
        "arena_serial": str(payload.get("arena_serial", "")).strip(),
        "loop_video": bool(payload.get("loop_video", True)),
    }
    service.start(clean_source)
    return JSONResponse({"ok": True, "message": "Source start requested.", "source": clean_source})


@app.post("/api/stop")
def stop_source() -> JSONResponse:
    service.stop()
    return JSONResponse({"ok": True, "message": "Capture stopped."})


@app.post("/api/settings")
def update_settings(payload: dict[str, Any]) -> JSONResponse:
    service.update_processing(payload)
    return JSONResponse({"ok": True, "processing": service.processing_config})


@app.post("/api/template/save")
def save_template(payload: dict[str, Any]) -> JSONResponse:
    name = str(payload.get("name", "Reference hardware"))
    try:
        template = service.save_template(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({
        "ok": True,
        "name": template["name"],
        "created_at": template["created_at"],
        "line_count": len(template["edges"]),
        "edge_count": len(template["edges"]),
        "corner_count": len(template["corners"]),
    })


@app.post("/api/template/clear")
def clear_template() -> JSONResponse:
    service.clear_template()
    return JSONResponse({"ok": True, "message": "Saved reference template cleared."})


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)) -> JSONResponse:
    filename = Path(file.filename or "video.mp4").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpeg", ".mpg", ".wmv"}:
        raise HTTPException(status_code=400, detail="Unsupported extension. Upload an MP4, AVI, MOV, MKV, or MPEG video.")
    # Store under a generated name to avoid collisions/path tricks.
    target = UPLOAD_DIR / f"{uuid.uuid4().hex[:10]}_{filename}"
    size = 0
    max_bytes = 1024 * 1024 * 1024  # 1 GiB safety cap
    try:
        with target.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail="Video upload exceeds the 1 GiB limit.")
                handle.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    # Check that OpenCV can at least open the uploaded container.
    capture = cv2.VideoCapture(str(target))
    readable = capture.isOpened()
    capture.release()
    if not readable:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="OpenCV could not open the uploaded video. Check the codec/container.")
    return JSONResponse({"ok": True, "filename": target.name, "video_path": str(target), "size_bytes": size})


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    def generate():
        last_jpeg: Optional[bytes] = None
        while True:
            jpeg, _, status, _ = service.get_snapshot()
            if jpeg != last_jpeg:
                last_jpeg = jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + jpeg + b"\r\n"
            elif not status.get("running"):
                # Keep the connection alive for the UI, without busy-spinning.
                time.sleep(0.12)
            else:
                time.sleep(0.01)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})


@app.on_event("shutdown")
def shutdown_event() -> None:
    service.stop()
