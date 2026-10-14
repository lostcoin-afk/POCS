from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import torch
import paddle

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from paddleocr import PaddleOCR


# ============================================================
# POC CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
CAPTURE_DIR = BASE_DIR / "captures"
# CAPTURE_DIR = /home/tachyon/Anomaly-Detection-Module/data/test_data_ocr
print(BASE_DIR)
FALLBACK_VIDEO_PATH = Path(
    os.getenv("FALLBACK_VIDEO_PATH", str(BASE_DIR / "videos" / "baseline_0.avi"))
)

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# Change this to the required physical digit sequence.
# Example: "123456789", "0123456789", "1234567890"
DEFAULT_DIGIT_SEQUENCE = os.getenv(
    "DEFAULT_DIGIT_SEQUENCE", "123456789"
)

# OCR settings. PP-OCRv6 is the default in current PaddleOCR 3.x.
OCR_LANG = os.getenv("OCR_LANG", "en")

CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="OCR Capture POC")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(TEMPLATES_DIR)), name="static")


# ============================================================
# CAMERA / FALLBACK VIDEO
# ============================================================

class FrameSource:
    """
    One source for the frontend stream and the capture endpoint.

    Priority:
      1. Physical camera
      2. Fallback video, looped continuously

    The most recent frame is kept in memory so the Capture button
    captures the same current stream frame rather than reopening
    the camera every time.
    """

    def __init__(self, camera_index: int, fallback_video: Path):
        self.camera_index = camera_index
        self.fallback_video = fallback_video

        self.cap: Optional[cv2.VideoCapture] = None
        self.using_camera = False

        self.latest_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()

        self._stop = threading.Event()
        self.thread = threading.Thread(
            target=self._reader_loop,
            name="frame-source",
            daemon=True,
        )

    def start(self) -> None:
        self._open_camera_or_video()
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _open_camera_or_video(self) -> None:
        # Try physical camera first.
        # camera = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        # if not camera.isOpened():
        #     camera.release()
        #     camera = cv2.VideoCapture(self.camera_index)

        # if camera.isOpened():
        #     self.cap = camera
        #     self.using_camera = True
        #     return

        # Camera unavailable -> fallback video.

        if self.fallback_video.exists():
            video = cv2.VideoCapture(str(self.fallback_video))
            if video.isOpened():
                self.cap = video
                self.using_camera = False
                return

        self.cap = None
        self.using_camera = False

    def _reopen_fallback(self) -> None:
        if self.cap is not None:
            self.cap.release()

        if self.fallback_video.exists():
            self.cap = cv2.VideoCapture(str(self.fallback_video))
            self.using_camera = False
        else:
            self.cap = None

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            if self.cap is None:
                time.sleep(0.5)
                self._open_camera_or_video()
                continue

            ok, frame = self.cap.read()

            if ok:
                with self.lock:
                    self.latest_frame = frame.copy()
                time.sleep(0.01)
                continue

            # End of fallback video -> loop it.
            if not self.using_camera:
                self._reopen_fallback()
                time.sleep(0.05)
                continue

            # Camera failed. Switch to fallback if available.
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            self._open_camera_or_video()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    @property
    def source_name(self) -> str:
        if self.using_camera:
            return f"camera:{self.camera_index}"
        if self.fallback_video.exists():
            return f"fallback:{self.fallback_video.name}"
        return "none"

print(f"@@@@@@@@@{FALLBACK_VIDEO_PATH}")
frame_source = FrameSource(CAMERA_INDEX, FALLBACK_VIDEO_PATH)


# ============================================================
# OCR
# ============================================================

def select_ocr_device() -> str:
    try:
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu:0"
    except Exception:
        pass
    return "cpu"


OCR_DEVICE = select_ocr_device()

# PaddleOCR 3.x exposes PP-OCRv6 by default. We keep document-specific
# orientation/unwarping disabled because this POC is aimed at short
# industrial numeric strings rather than documents.
# ocr = PaddleOCR(
#     lang=OCR_LANG,
#     device=OCR_DEVICE,
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=True,
# )
ocr = PaddleOCR(
    lang=OCR_LANG,
    device=OCR_DEVICE,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    enable_mkldnn=False,
    enable_hpi = True,
)

# ============================================================
# OCR RESULT NORMALIZATION
# ============================================================

def _as_dict(result: Any) -> Dict[str, Any]:
    """
    PaddleOCR 3.x Result objects expose a .json property.
    Be defensive so the POC is tolerant of minor result-wrapper changes.
    """
    data = getattr(result, "json", None)

    if callable(data):
        data = data()

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return {}

    if isinstance(data, dict):
        return data

    # Some result wrappers may behave like dicts.
    try:
        if "res" in result:
            return dict(result)
    except Exception:
        pass

    return {}


def extract_ocr_items(results: Any) -> List[Dict[str, Any]]:
    """
    Convert PaddleOCR output to a simple internal representation:
      [{"text": "...", "score": 0.99, "box": [x1,y1,x2,y2]}, ...]
    """
    items: List[Dict[str, Any]] = []

    for result in results:
        payload = _as_dict(result)
        res = payload.get("res", payload)

        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        boxes = res.get("rec_boxes", None)
        polys = res.get("rec_polys", res.get("dt_polys", None))

        for i, text in enumerate(texts):
            text = str(text).strip()
            if not text:
                continue

            score = float(scores[i]) if i < len(scores) else 0.0

            box = None

            if boxes is not None and i < len(boxes):
                b = np.asarray(boxes[i]).reshape(-1).tolist()
                if len(b) >= 4:
                    box = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]

            if box is None and polys is not None and i < len(polys):
                p = np.asarray(polys[i]).reshape(-1, 2)
                if p.size:
                    box = [
                        float(np.min(p[:, 0])),
                        float(np.min(p[:, 1])),
                        float(np.max(p[:, 0])),
                        float(np.max(p[:, 1])),
                    ]

            items.append(
                {
                    "text": text,
                    "score": score,
                    "box": box,
                }
            )

    # Read order: top-to-bottom, then left-to-right.
    def sort_key(item: Dict[str, Any]) -> Tuple[float, float]:
        box = item.get("box") or [0, 0, 0, 0]
        return (box[1], box[0])

    items.sort(key=sort_key)
    return items


def run_ocr(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Run OCR on one frame.
    """
    # BGR numpy arrays are accepted by PaddleOCR as image input.
    results = ocr.predict(image)
    return extract_ocr_items(results)


# ============================================================
# PYTORCH 180-DEGREE AUGMENTATION
# ============================================================

def rotate_180_with_torch(image_bgr: np.ndarray) -> np.ndarray:
    """
    Uses PyTorch for the 180-degree POC augmentation.

    This is intentionally simple: PyTorch is used here for tensor-based
    image preprocessing; PaddleOCR remains the OCR inference engine.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb)
    rotated = torch.rot90(tensor, k=2, dims=(0, 1))
    rotated_np = rotated.cpu().numpy()
    return cv2.cvtColor(rotated_np, cv2.COLOR_RGB2BGR)


# ============================================================
# VALIDATION
# ============================================================

DIGIT_RE = re.compile(r"^\d+$")
TWO_GROUP_RE = re.compile(r"^(\d+)\s(\d+)$")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def build_text(items: List[Dict[str, Any]]) -> str:
    return normalize_space(" ".join(x["text"] for x in items if x["text"]))


def numeric_text(text: str) -> str:
    return re.sub(r"[^0-9]", "", text)


def validate_digit_sequence(
    detected: str,
    expected: str,
) -> Dict[str, Any]:
    detected = numeric_text(detected)
    expected = numeric_text(expected)

    if not expected:
        return {
            "correct": False,
            "reason": "No expected digit sequence configured.",
        }

    if detected == expected:
        return {
            "correct": True,
            "reason": "Digit sequence is correct.",
            "expected": expected,
            "detected": detected,
        }

    # Find the first break.
    first_break = None
    for i, exp_char in enumerate(expected):
        if i >= len(detected):
            first_break = {
                "position": i + 1,
                "expected": exp_char,
                "detected": "<missing>",
            }
            break

        if detected[i] != exp_char:
            first_break = {
                "position": i + 1,
                "expected": exp_char,
                "detected": detected[i],
            }
            break

    if first_break is None and len(detected) > len(expected):
        first_break = {
            "position": len(expected) + 1,
            "expected": "<end>",
            "detected": detected[len(expected)],
        }

    return {
        "correct": False,
        "reason": (
            f"Sequence break at position {first_break['position']}: "
            f"expected {first_break['expected']}, "
            f"got {first_break['detected']}."
        )
        if first_break
        else "Detected sequence does not match expected sequence.",
        "expected": expected,
        "detected": detected,
        "break": first_break,
    }


def normalize_product_code(text: str) -> str:
    text = normalize_space(text)

    # Keep only digits and whitespace, then collapse spaces.
    text = re.sub(r"[^0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def validate_product_code(
    detected: str,
    expected: str,
) -> Dict[str, Any]:
    detected_norm = normalize_product_code(detected)
    expected_norm = normalize_product_code(expected)

    valid_shape = bool(TWO_GROUP_RE.fullmatch(detected_norm))

    if not valid_shape:
        return {
            "correct": False,
            "reason": (
                "Detected text is not a two-group numeric product code "
                "with one space, e.g. '1234 5678'."
            ),
            "expected": expected_norm,
            "detected": detected_norm,
        }

    correct = detected_norm == expected_norm

    return {
        "correct": correct,
        "reason": (
            "Product code matches."
            if correct
            else "Product code does not match the user-provided value."
        ),
        "expected": expected_norm,
        "detected": detected_norm,
    }


def classify_and_validate(
    normal_items: List[Dict[str, Any]],
    rotated_items: List[Dict[str, Any]],
    expected_sequence: str,
    expected_product_code: str,
) -> Dict[str, Any]:
    """
    Determine one of the two current POC types.

    For single-digit sequence:
      - accept all numeric OCR tokens after removing spaces;
      - compare exact sequence;
      - also try the 180-degree OCR candidate.

    For product code:
      - recognize a shape of two numeric groups separated by one space;
      - compare exact normalized string.

    The better candidate is chosen using:
      1. exact expected match,
      2. valid format,
      3. mean OCR confidence.
    """

    normal_text = build_text(normal_items)
    rotated_text = build_text(rotated_items)

    candidates = [
        {
            "orientation": "normal",
            "text": normal_text,
            "items": normal_items,
            "mean_score": float(
                np.mean([x["score"] for x in normal_items])
                if normal_items
                else 0.0
            ),
        },
        {
            "orientation": "rotated_180",
            "text": rotated_text,
            "items": rotated_items,
            "mean_score": float(
                np.mean([x["score"] for x in rotated_items])
                if rotated_items
                else 0.0
            ),
        },
    ]

    # First preference: exact sequence match.
    seq_checks = []
    for candidate in candidates:
        cleaned = numeric_text(candidate["text"])
        check = validate_digit_sequence(cleaned, expected_sequence)
        seq_checks.append((candidate, check))

    exact_seq = [
        (candidate, check)
        for candidate, check in seq_checks
        if check["correct"]
    ]
    if exact_seq:
        candidate, check = max(
            exact_seq, key=lambda pair: pair[0]["mean_score"]
        )
        return {
            "type": "single_digit_sequence",
            "correct": True,
            "defective": False,
            "orientation": candidate["orientation"],
            "detected_text": numeric_text(candidate["text"]),
            "ocr_items": candidate["items"],
            "validation": check,
            "reason": check["reason"],
        }

    # Then preference: product-code exact match.
    product_checks = []
    for candidate in candidates:
        check = validate_product_code(
            candidate["text"], expected_product_code
        )
        product_checks.append((candidate, check))

    exact_product = [
        (candidate, check)
        for candidate, check in product_checks
        if check["correct"]
    ]
    if exact_product:
        candidate, check = max(
            exact_product, key=lambda pair: pair[0]["mean_score"]
        )
        return {
            "type": "product_code",
            "correct": True,
            "defective": False,
            "orientation": candidate["orientation"],
            "detected_text": check["detected"],
            "ocr_items": candidate["items"],
            "validation": check,
            "reason": check["reason"],
        }

    # Determine likely type from the strongest candidate.
    best = max(candidates, key=lambda x: x["mean_score"])
    best_text = best["text"]

    if TWO_GROUP_RE.fullmatch(normalize_product_code(best_text)):
        best_product = validate_product_code(
            best_text, expected_product_code
        )
        return {
            "type": "product_code",
            "correct": False,
            "defective": True,
            "orientation": best["orientation"],
            "detected_text": best_product["detected"],
            "ocr_items": best["items"],
            "validation": best_product,
            "reason": best_product["reason"],
        }

    # Default to digit-sequence validation.
    best_seq = validate_digit_sequence(
        numeric_text(best_text), expected_sequence
    )

    return {
        "type": "single_digit_sequence",
        "correct": False,
        "defective": True,
        "orientation": best["orientation"],
        "detected_text": numeric_text(best_text),
        "ocr_items": best["items"],
        "validation": best_seq,
        "reason": best_seq["reason"],
    }


# ============================================================
# DRAW OCR BOXES
# ============================================================

def draw_ocr_items(
    image: np.ndarray,
    items: List[Dict[str, Any]],
    result: Dict[str, Any],
) -> np.ndarray:
    output = image.copy()

    for item in items:
        box = item.get("box")
        if not box:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]

        # Green for recognized text; final status banner is drawn below.
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 220, 0),
            2,
        )

        label = f"{item['text']} ({item['score']:.2f})"

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    status = "CORRECT" if result.get("correct") else "DEFECTIVE / INCORRECT"
    status_color = (0, 200, 0) if result.get("correct") else (0, 0, 255)

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 48),
        status_color,
        -1,
    )
    cv2.putText(
        output,
        f"{status} | {result.get('type', 'unknown')}",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


# ============================================================
# CAPTURE + VALIDATION
# ============================================================

def process_current_frame(
    expected_sequence: str,
    expected_product_code: str,
) -> Dict[str, Any]:
    frame = frame_source.get_latest_frame()

    if frame is None:
        raise HTTPException(
            status_code=503,
            detail="No camera frame or fallback video frame is available.",
        )

    # OCR normal orientation.
    normal_items = run_ocr(frame)

    # OCR 180-degree orientation. This is especially useful for upside-down
    # digit strings. PaddleOCR also has a text-line orientation module, but
    # this second view provides an explicit fallback for this industrial POC.
    rotated = rotate_180_with_torch(frame)
    rotated_items = run_ocr(rotated)

    result = classify_and_validate(
        normal_items=normal_items,
        rotated_items=rotated_items,
        expected_sequence=expected_sequence or DEFAULT_DIGIT_SEQUENCE,
        expected_product_code=expected_product_code,
    )

    # Keep the image corresponding to the selected orientation for visualization.
    selected_image = (
        rotated
        if result["orientation"] == "rotated_180"
        else frame
    )
    annotated = draw_ocr_items(
        selected_image,
        result["ocr_items"],
        result,
    )

    filename = f"capture_{int(time.time() * 1000)}.jpg"
    path = CAPTURE_DIR / filename
    cv2.imwrite(str(path), annotated)

    result["capture_url"] = f"/captures/{filename}"
    result["source"] = frame_source.source_name
    result["timestamp"] = time.time()

    return result


# ============================================================
# FASTAPI ROUTES
# ============================================================

@app.on_event("startup")
def startup_event() -> None:
    frame_source.start()


@app.on_event("shutdown")
def shutdown_event() -> None:
    frame_source.stop()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "ocr_device": OCR_DEVICE,
            "source": frame_source.source_name,
            "default_sequence": DEFAULT_DIGIT_SEQUENCE,
        },
    )


def mjpeg_generator() -> Generator[bytes, None, None]:
    while True:
        frame = frame_source.get_latest_frame()

        if frame is None:
            time.sleep(0.1)
            continue

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
        )

        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )

        # Approximately 20 FPS for the browser feed.
        time.sleep(0.05)


@app.get("/feed")
async def feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/capture")
async def capture(
    expected_sequence: str = DEFAULT_DIGIT_SEQUENCE,
    expected_product_code: str = "",
):
    # import pdb
    # pdb.set_trace()
    return JSONResponse(
        process_current_frame(
            expected_sequence=expected_sequence,
            expected_product_code=expected_product_code,
        )
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ocr_device": OCR_DEVICE,
        "source": frame_source.source_name,
        "camera_index": CAMERA_INDEX,
        "fallback_video": str(FALLBACK_VIDEO_PATH),
        "fallback_exists": FALLBACK_VIDEO_PATH.exists(),
    }


@app.get("/captures/{filename}")
async def captured_image(filename: str):
    path = CAPTURE_DIR / filename

    if not path.exists():
        raise HTTPException(status_code=404, detail="Capture not found.")

    from fastapi.responses import FileResponse
    return FileResponse(path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
