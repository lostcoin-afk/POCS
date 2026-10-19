"""FastAPI dimensional-measurement inspector.

Run from this directory:
    uvicorn app:app --host 0.0.0.0 --port 8000

Input can be a saved video, webcam, or an optional LUCID Arena SDK camera.
All uncalibrated dimensions are reported in pixels. Use the calibration panel to
convert pixels to millimetres using a known length measured in the same image plane.
"""
from __future__ import annotations

import threading
import time
import uuid
import json
import math
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from geometry import process_measurement

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
for directory in (TEMPLATES_DIR, DATA_DIR, UPLOAD_DIR):
    directory.mkdir(parents=True, exist_ok=True)

APP_TITLE = "Hardware Dimension Inspector"
DEFAULT_PROCESSING: dict[str, Any] = {
    "canny_low": 45,
    "canny_high": 130,
    "blur_kernel": 5,
    "morph_kernel": 5,
    # Rows above roi_top_pct are excluded; the ROI continues to the bottom/full width.
    "roi_top_pct": 0.0,
    "min_area_ratio": 0.015,
    "max_area_ratio": 0.88,
    "circle_min_radius": 6,
    "circle_max_radius": 140,
    "circle_min_distance": 58,
    "circle_param1": 110,
    "circle_param2": 34,
    "circle_min_contrast": 16.0,
    "circle_min_edge_support": 0.18,
    # Extended axis-aligned silhouette guides and dimensions
    "line_hough_threshold": 55,
    "line_min_length": 75,
    "line_max_gap": 20,
    "axis_tolerance_deg": 10.0,
    "line_merge_px": 13.0,
    "max_axis_lines": 8,
    "min_dimension_px": 18.0,
    "mm_per_px": None,
    # Optional planar homography written by calibrate_scale.py.
    "homography_image_to_mm": None,
    "calibration_image_size": None,
}
DEFAULT_SOURCE: dict[str, Any] = {
    "source_type": "video",  # video | webcam | arena
    "video_path": "",
    "webcam_index": 0,
    "arena_serial": "",
    "loop_video": True,
}

app = FastAPI(title=APP_TITLE, version="1.0.0")
app.mount("/templates", StaticFiles(directory=str(TEMPLATES_DIR)), name="static-templates")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _safe_data(data: dict[str, Any]) -> dict[str, Any]:
    """Convert NumPy scalars/containers to JSON-friendly Python values."""
    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    return convert(data)


class FrameSource:
    label = "image source"
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError
    def close(self) -> None:
        pass


class OpenCVSource(FrameSource):
    def __init__(self, config: dict[str, Any]):
        source_type = config.get("source_type", "video")
        if source_type == "webcam":
            index = int(config.get("webcam_index", 0))
            self.capture = cv2.VideoCapture(index)
            self.label = f"Webcam index {index}"
        else:
            video_path = str(config.get("video_path", "")).strip()
            if not video_path:
                raise RuntimeError("Enter a saved-video path or upload a video first.")
            path = Path(video_path).expanduser()
            if not path.is_absolute():
                path = (BASE_DIR / path).resolve()
            if not path.exists() or not path.is_file():
                raise RuntimeError(f"Video file not found: {path}")
            self.capture = cv2.VideoCapture(str(path))
            self.label = f"Video: {path.name}"
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open {self.label}. Check the source and codec support.")
        self.loop_video = bool(config.get("loop_video", True)) and source_type == "video"
        fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_interval = 1.0 / min(60.0, max(1.0, fps)) if 0.1 < fps < 1000 else 1.0 / 25.0
        self.next_read_at: Optional[float] = None

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
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
    """LUCID Arena SDK adapter; requires the official SDK and Python bindings."""
    _BAYER = {
        "BayerRG8": cv2.COLOR_BAYER_RG2BGR,
        "BayerGR8": cv2.COLOR_BAYER_GR2BGR,
        "BayerGB8": cv2.COLOR_BAYER_GB2BGR,
        "BayerBG8": cv2.COLOR_BAYER_BG2BGR,
    }

    def __init__(self, config: dict[str, Any]):
        try:
            from arena_api.system import system  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Arena SDK Python bindings are unavailable. Install LUCID's Arena SDK in this environment.") from exc
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
                    raise RuntimeError(f"Camera serial {serial} not found. Available serials: {available}")
                devices = system.create_device(device_infos=selected)
            else:
                devices = system.create_device()
            if not devices:
                raise RuntimeError("No LUCID Arena cameras found. Check camera power, Ethernet, and ArenaView.")
            self.device = devices[0]
            self.pixel_format = str(self.device.nodemap["PixelFormat"].value)
            allowed = {"Mono8", "BayerRG8", "BayerGR8", "BayerGB8", "BayerBG8", "RGB8", "BGR8", "RGBA8", "BGRA8"}
            if self.pixel_format not in allowed:
                try:
                    self.device.nodemap["PixelFormat"].value = "Mono8"
                    self.pixel_format = str(self.device.nodemap["PixelFormat"].value)
                except Exception as exc:
                    raise RuntimeError(f"Unsupported PixelFormat {self.pixel_format}; set Mono8 or an 8-bit color format in ArenaView.") from exc
            self.device.start_stream()
            self.streaming = True
            try:
                model = str(self.device.nodemap["DeviceModelName"].value)
            except Exception:
                model = "LUCID camera"
            self.label = f"Arena SDK: {model} ({serial or 'first available'})"
        except Exception:
            self.close()
            raise

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        assert self.device is not None
        buffer = self.device.get_buffer()
        try:
            height, width, bits = int(buffer.height), int(buffer.width), int(buffer.bits_per_pixel)
            if bits not in (8, 24, 32):
                raise RuntimeError(f"Unsupported image depth ({bits} bits). Configure an 8-bit camera pixel format.")
            channels = bits // 8
            raw = np.ctypeslib.as_array(buffer.pdata, shape=(height * width * channels,)).copy()
            image = raw.reshape((height, width) if channels == 1 else (height, width, channels))
            pixel_format = self.pixel_format
        finally:
            # Copy before requeue; Arena can reuse its buffer after it is requeued.
            self.device.requeue_buffer(buffer)
        if pixel_format in self._BAYER and image.ndim == 2:
            image = cv2.cvtColor(image, self._BAYER[pixel_format])
        elif pixel_format == "RGB8" and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif pixel_format == "RGBA8" and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif pixel_format == "BGRA8" and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
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
    if config.get("source_type") == "arena":
        return ArenaSource(config)
    if config.get("source_type") in {"video", "webcam"}:
        return OpenCVSource(config)
    raise RuntimeError("source_type must be video, webcam, or arena")


def _jpeg(image: np.ndarray, quality: int = 84) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return encoded.tobytes() if ok else b""


def _placeholder(message: str = "Choose a source and click Start") -> bytes:
    image = np.zeros((600, 960, 3), dtype=np.uint8)
    image[:] = (23, 29, 39)
    cv2.putText(image, "HARDWARE DIMENSION INSPECTOR", (35, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (238, 242, 248), 2, cv2.LINE_AA)
    cv2.putText(image, message[:80], (35, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (135, 190, 220), 1, cv2.LINE_AA)
    return _jpeg(image)


class MeasurementService:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.source_config = dict(DEFAULT_SOURCE)
        self.processing = dict(DEFAULT_PROCESSING)
        self.temporal_state: dict[str, Any] = {}
        self._load_calibration()
        self.latest_overlay = _placeholder()
        self.latest_mask = _jpeg(np.zeros((600, 960), np.uint8))
        self.latest_edges = _jpeg(np.zeros((600, 960), np.uint8))
        self.latest_data: dict[str, Any] = {
            "image_width": 960, "image_height": 600, "silhouette": {"found": False},
            "length_px": None, "width_px": None, "length": None, "width": None, "circles": [],
            "linear_measurements": [], "horizontal_guides": [], "vertical_guides": [],
            "unit": "px", "mm_per_px": None, "processing_ms": None,
        }
        self.status: dict[str, Any] = {
            "running": False, "source_label": "Not connected", "source_type": "video",
            "message": "Choose a source and click Start.", "error": None, "fps": 0.0,
            "last_frame_time": None,
        }

    def _load_calibration(self) -> None:
        """Load a scale or planar homography produced by calibrate_scale.py."""
        path = DATA_DIR / "calibration.json"
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            scale = float(payload.get("mm_per_px", 0.0))
            if math.isfinite(scale) and scale > 0:
                self.processing["mm_per_px"] = scale
            homography = payload.get("homography_image_to_mm")
            if isinstance(homography, list) and len(homography) == 3:
                self.processing["homography_image_to_mm"] = homography
                self.processing["calibration_image_size"] = payload.get("image_size")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # Invalid calibration files are ignored instead of preventing server startup.
            return

    def _save_calibration(self, payload: dict[str, Any]) -> None:
        path = DATA_DIR / "calibration.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)

    def start(self, source_config: dict[str, Any]) -> None:
        self.stop()
        with self.lock:
            self.temporal_state.clear()
            self.source_config = {**DEFAULT_SOURCE, **source_config}
            self.status.update({"running": True, "source_label": "Connecting…", "source_type": self.source_config.get("source_type"), "message": "Connecting to source…", "error": None})
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run, name="dimension-capture", daemon=True)
            self.thread.start()

    def stop(self) -> None:
        with self.lock:
            thread = self.thread
            self.stop_event.set()
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=4.0)
        with self.lock:
            if not thread or not thread.is_alive():
                self.thread = None
                self.status["running"] = False
                if not self.status.get("error"):
                    self.status["message"] = "Stopped."

    def update_processing(self, updates: dict[str, Any]) -> dict[str, Any]:
        int_keys = {"canny_low", "canny_high", "blur_kernel", "morph_kernel", "circle_min_radius", "circle_max_radius", "circle_min_distance", "circle_param1", "circle_param2", "line_hough_threshold", "line_min_length", "line_max_gap", "max_axis_lines"}
        float_keys = {"min_area_ratio", "max_area_ratio", "circle_min_contrast", "circle_min_edge_support", "axis_tolerance_deg", "line_merge_px", "min_dimension_px", "roi_top_pct"}
        with self.lock:
            for key, value in updates.items():
                if key not in DEFAULT_PROCESSING or key == "mm_per_px":
                    continue
                try:
                    self.processing[key] = int(value) if key in int_keys else float(value) if key in float_keys else self.processing[key]
                except (TypeError, ValueError):
                    continue
            p = self.processing
            p["canny_low"] = max(0, min(254, int(p["canny_low"])))
            p["canny_high"] = max(p["canny_low"] + 1, min(255, int(p["canny_high"])))
            p["blur_kernel"] = max(1, min(15, int(p["blur_kernel"])))
            if p["blur_kernel"] % 2 == 0: p["blur_kernel"] += 1
            p["morph_kernel"] = max(1, min(31, int(p["morph_kernel"])))
            if p["morph_kernel"] % 2 == 0: p["morph_kernel"] += 1
            p["min_area_ratio"] = max(0.001, min(0.5, float(p["min_area_ratio"])))
            p["max_area_ratio"] = max(0.1, min(0.95, float(p["max_area_ratio"])))
            p["roi_top_pct"] = max(0.0, min(80.0, float(p["roi_top_pct"])))
            p["circle_min_radius"] = max(2, min(500, int(p["circle_min_radius"])))
            p["circle_max_radius"] = max(p["circle_min_radius"] + 1, min(1000, int(p["circle_max_radius"])))
            p["circle_min_distance"] = max(5, min(1500, int(p["circle_min_distance"])))
            p["circle_param1"] = max(20, min(255, int(p["circle_param1"])))
            p["circle_param2"] = max(5, min(200, int(p["circle_param2"])))
            p["circle_min_contrast"] = max(0.0, min(100.0, float(p["circle_min_contrast"])))
            p["circle_min_edge_support"] = max(0.0, min(1.0, float(p["circle_min_edge_support"])))
            p["line_hough_threshold"] = max(10, min(200, int(p["line_hough_threshold"])))
            p["line_min_length"] = max(15, min(1000, int(p["line_min_length"])))
            p["line_max_gap"] = max(0, min(200, int(p["line_max_gap"])))
            p["axis_tolerance_deg"] = max(1.0, min(30.0, float(p["axis_tolerance_deg"])))
            p["line_merge_px"] = max(1.0, min(100.0, float(p["line_merge_px"])))
            p["max_axis_lines"] = max(2, min(16, int(p["max_axis_lines"])))
            p["min_dimension_px"] = max(1.0, min(1000.0, float(p["min_dimension_px"])))
            return dict(p)

    def calibrate(self, known_length_mm: float, known_length_px: float) -> dict[str, Any]:
        if known_length_mm <= 0 or known_length_px <= 0:
            raise ValueError("Both the known physical length and its measured pixel length must be greater than zero.")
        scale = float(known_length_mm / known_length_px)
        if not (0 < scale < 1000):
            raise ValueError("Calculated scale is outside the plausible range; check the entered values.")
        with self.lock:
            self.processing["mm_per_px"] = scale
            # Manual one-dimensional calibration supersedes any previous homography.
            self.processing["homography_image_to_mm"] = None
            self.processing["calibration_image_size"] = None
            self._save_calibration({
                "method": "single_known_dimension",
                "mm_per_px": scale,
                "known_length_mm": float(known_length_mm),
                "known_length_px": float(known_length_px),
                "note": "Scalar calibration only; reference and measured features must be coplanar.",
            })
            return {"mm_per_px": scale}

    def clear_calibration(self) -> None:
        with self.lock:
            self.processing["mm_per_px"] = None
            self.processing["homography_image_to_mm"] = None
            self.processing["calibration_image_size"] = None
            (DATA_DIR / "calibration.json").unlink(missing_ok=True)

    def get_snapshot(self) -> tuple[bytes, bytes, bytes, dict[str, Any], dict[str, Any], dict[str, Any]]:
        with self.lock:
            return self.latest_overlay, self.latest_mask, self.latest_edges, dict(self.latest_data), dict(self.status), dict(self.processing)

    def _run(self) -> None:
        source: Optional[FrameSource] = None
        prev_time: Optional[float] = None
        fps_samples: list[float] = []
        try:
            with self.lock:
                config = dict(self.source_config)
            source = create_source(config)
            with self.lock:
                self.status.update({"source_label": source.label, "message": "Streaming and measuring frames.", "error": None})
            while not self.stop_event.is_set():
                ok, frame = source.read()
                if not ok or frame is None:
                    if config.get("source_type") == "video" and not config.get("loop_video", True):
                        with self.lock: self.status["message"] = "Video reached its end."
                        break
                    time.sleep(0.01)
                    continue
                start = time.perf_counter()
                with self.lock:
                    cfg = dict(self.processing)
                    cfg["_temporal_state"] = self.temporal_state
                overlay, data, mask, edges = process_measurement(frame, cfg)
                data["processing_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
                now = time.time()
                if prev_time is not None and now > prev_time:
                    fps_samples.append(1.0 / (now - prev_time))
                    fps_samples = fps_samples[-20:]
                prev_time = now
                with self.lock:
                    self.latest_overlay = _jpeg(overlay, 88)
                    self.latest_mask = _jpeg(mask, 80)
                    self.latest_edges = _jpeg(edges, 80)
                    self.latest_data = _safe_data(data)
                    self.status.update({
                        "running": True, "source_label": source.label, "source_type": config.get("source_type"),
                        "message": "Streaming and measuring frames.", "error": None,
                        "fps": round(sum(fps_samples) / len(fps_samples), 1) if fps_samples else 0.0,
                        "last_frame_time": now,
                    })
        except Exception as exc:
            with self.lock:
                self.status.update({"running": False, "message": "Could not start or process the selected source.", "error": str(exc)})
        finally:
            if source is not None:
                try: source.close()
                except Exception: pass
            with self.lock:
                self.status["running"] = False
                if self.status.get("message") == "Streaming and measuring frames.":
                    self.status["message"] = "Stopped."


service = MeasurementService()


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html", context={"app_title": APP_TITLE})


@app.get("/api/status")
def get_status() -> JSONResponse:
    _, _, _, data, status, processing = service.get_snapshot()
    return JSONResponse({"status": status, "processing": processing, "data": data, "source_config": service.source_config})


@app.post("/api/start")
def start_source(payload: dict[str, Any]) -> JSONResponse:
    source_type = str(payload.get("source_type", "video")).lower()
    if source_type not in {"video", "webcam", "arena"}:
        raise HTTPException(status_code=400, detail="source_type must be video, webcam, or arena")
    clean = {
        "source_type": source_type,
        "video_path": str(payload.get("video_path", "")).strip(),
        "webcam_index": max(0, int(payload.get("webcam_index", 0))),
        "arena_serial": str(payload.get("arena_serial", "")).strip(),
        "loop_video": bool(payload.get("loop_video", True)),
    }
    service.start(clean)
    return JSONResponse({"ok": True, "message": "Source start requested.", "source": clean})


@app.post("/api/stop")
def stop_source() -> JSONResponse:
    service.stop()
    return JSONResponse({"ok": True, "message": "Capture stopped."})


@app.post("/api/settings")
def update_settings(payload: dict[str, Any]) -> JSONResponse:
    return JSONResponse({"ok": True, "processing": service.update_processing(payload)})


@app.post("/api/calibration")
def set_calibration(payload: dict[str, Any]) -> JSONResponse:
    try:
        result = service.calibrate(float(payload.get("known_length_mm", 0)), float(payload.get("known_length_px", 0)))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"ok": True, **result, "message": "Scale applied. Measurements are now displayed in millimetres."})


@app.delete("/api/calibration")
def clear_calibration() -> JSONResponse:
    service.clear_calibration()
    return JSONResponse({"ok": True, "message": "Calibration cleared. Measurements are now in pixels."})


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)) -> JSONResponse:
    filename = Path(file.filename or "video.mp4").name
    if Path(filename).suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpeg", ".mpg", ".wmv"}:
        raise HTTPException(status_code=400, detail="Unsupported video extension.")
    target = UPLOAD_DIR / f"{uuid.uuid4().hex[:10]}_{filename}"
    size = 0
    try:
        with target.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk: break
                size += len(chunk)
                if size > 1024 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Video upload exceeds the 1 GiB limit.")
                handle.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    capture = cv2.VideoCapture(str(target))
    readable = capture.isOpened()
    capture.release()
    if not readable:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="OpenCV could not open the uploaded video.")
    return JSONResponse({"ok": True, "filename": target.name, "video_path": str(target), "size_bytes": size})


def _stream(view: str) -> StreamingResponse:
    def generate():
        last: Optional[bytes] = None
        while True:
            overlay, mask, edges, _, status, _ = service.get_snapshot()
            jpeg = overlay if view == "overlay" else mask if view == "mask" else edges
            if jpeg != last:
                last = jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + jpeg + b"\r\n"
            elif not status.get("running"):
                time.sleep(0.12)
            else:
                time.sleep(0.02)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    return _stream("overlay")


@app.get("/mask_feed")
def mask_feed() -> StreamingResponse:
    return _stream("mask")


@app.get("/edge_feed")
def edge_feed() -> StreamingResponse:
    return _stream("edges")


@app.on_event("shutdown")
def shutdown_event() -> None:
    service.stop()
