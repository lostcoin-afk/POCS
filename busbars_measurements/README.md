# Hardware Dimension Inspector — Dynamic ROI, Stabilized Guides, and Planar Calibration

A standalone FastAPI + OpenCV prototype for silhouette-based dimensional measurements. It supports saved video, webcam, and LUCID Arena SDK input. This is independent of the angle-inspection project.

## New capabilities

- **Dynamic top ROI:** change the top cutoff while the feed is running using the slider or enable drag mode and drag the cutoff on the live image. Rows above the cutoff are excluded from segmentation, guide detection, circle detection, and measurement. The active ROI uses the full image width and extends down to the bottom of the frame. Lines are rendered only from the selected top boundary downward.
- **More stable detection:** small Gaussian blur, local contrast normalization for the edge preview and Hough circle proposal stage, silhouette-boundary-only line detection, stronger minimum line segment defaults, merged near-duplicate guide coordinates, median smoothing over a short temporal window, and persistence filtering to reject one-frame noise.
- **Circle duplicate suppression:** Hough candidates are scored by intensity contrast and circumference-edge support; overlapping candidates with similar radii are suppressed; multi-frame persistence further reduces flickering detections.
- **Bolder overlays:** thicker lime contour, red horizontal guides, blue vertical guides, green dimension arrows, and yellow circle overlays.
- **Reference rectangle calibration:** `calibrate_scale.py` lets you select the four corners of a known rectangular artefact in an image/video frame and writes `data/calibration.json`. The file contains a simple scalar scale and a planar image-to-mm homography. The application loads the calibration at startup.

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000`.

## Input sources

- **Saved video:** enter a path on the server or upload a video.
- **Webcam:** choose an OpenCV camera index.
- **LUCID Arena SDK camera:** install the official Arena SDK and Python bindings, then select the Arena source. Supported configured pixel formats include Mono8, 8-bit Bayer, RGB8/BGR8, and RGBA8/BGRA8.

## ROI operation

The ROI is a full-width horizontal crop. `roi_top_pct` ranges from 0 to 80 percent of the frame height. Set it so the green ROI line is at the required upper measurement limit. The bottom, left, and right of the image are not intentionally cropped by this ROI. The silhouette segment may touch the ROI top edge because that clipping is expected; touching the other full-frame boundaries can still cause the silhouette candidate to be rejected to avoid treating a frame edge as the part boundary.

## Detection tuning

Settings are available in the dashboard:

- **Canny thresholds / blur:** tune edge preview and circle proposals for noise and lighting.
- **Morphology kernel and minimum silhouette area:** tune component segmentation. Large closing kernels can bridge notches or fill small gaps, so keep the kernel small enough to preserve real geometry.
- **Circle minimum/maximum radius, center distance, Hough threshold and contrast:** tune the circular-feature detector. Increase center distance or Hough threshold to suppress duplicates/weak candidates; keep the minimum radius below the smallest real feature. Candidate ranking, overlapping-circle suppression, edge support and temporal persistence are applied in addition to the Hough transform.
- **Line Hough threshold, minimum segment length, maximum gap, axis tolerance, merge tolerance and maximum lines:** tune the red/blue guide set. Raising the Hough threshold and minimum segment length generally suppresses weak fragments. Lowering merge tolerance helps preserve nearby parallel physical edges.

The short temporal filter smooths line locations, circle centers/radii, and the silhouette bounding box using a rolling median. Confirm results after a few frames because new features may need several frames to pass the persistence gate. Temporal smoothing is intended to reduce noise, not replace calibrated measurement or geometric validation.

## Rectangular-reference calibration (recommended when physical units matter)

Capture an image with the reference rectangle held flat on the **same physical plane** as the measured component. Keep camera, lens, focus, zoom, working distance, lighting and source resolution fixed. The calibration utility prompts for four corners in this order: 1 top-left, 2 top-right, 3 bottom-right, 4 bottom-left. Enter the reference width and height in millimetres.

### Example using a saved frame

```bash
python calibrate_scale.py --image /path/to/reference_frame.png --width-mm 50 --height-mm 25
```

Or use a specific frame from a video:

```bash
python calibrate_scale.py --video /path/to/reference_video.mp4 --frame-index 100 --width-mm 50 --height-mm 25
```

The tool opens an interactive window. Click the four corners clockwise as instructed. Press `S` or `Enter` to save; press `R` to reset the points; press `Q` or `Esc` to cancel. By default the calibration JSON is written to this project's `data/calibration.json`; it also writes `calibration_preview.png`. Restart the inspector after calibrating so the app loads the file. The image resolution must match the runtime frame resolution for the homography to be applied.

The JSON stores `mm_per_px`, separate X/Y scale estimates, the disagreement between the axis estimates, the selected corner points, the source resolution, and a 3x3 planar image-to-mm homography. The homography maps line endpoints to millimetres and maps sampled points around a detected circle before fitting its physical-plane diameter. The UI's manual one-length calibration is a simpler scalar method; applying it replaces the homography, so do not use that button if you want to keep the rectangular-reference perspective correction.

### Is the same factor valid for circles and straight lines?

- **Single scalar `mm_per_px`:** one factor can be used for both straight-line distances and circle diameters when the reference and target lie on the same plane, the view is close to orthogonal, distortion is negligible, and scale is effectively constant over the field of view.
- **Planar homography:** under perspective, there is no single globally correct scalar. The same image-to-plane homography can be used for both measurements if they lie on the calibrated plane. The app maps line endpoints through it and transforms sampled circle-boundary points into the physical plane before fitting the diameter.
- **Different depth / non-coplanar features:** neither one scalar nor a plane homography can correct arbitrary depth differences. Use a reference at the same height/plane as the object, or use a calibrated 3D measurement method.

For high accuracy, first calibrate lens intrinsics/distortion with a checkerboard or suitable calibration target at the same focus/zoom, undistort images, and then use a large known rectangle to estimate the planar homography. Use a reference with sharply defined edges, click corners carefully, inspect `calibration_preview.png`, and validate the final scale with a second independent known artefact in the actual measurement region. Manual corner picking and Hough circle detection are not substitutes for a validated metrology system.

The camera working distance (39 cm) and image resolution (1900x1200) alone do not determine mm/px.

## API

- `GET /api/status`
- `POST /api/start`
- `POST /api/stop`
- `POST /api/settings` (includes `roi_top_pct` and circle/line parameters)
- `POST /api/calibration` with `{"known_length_mm": 25, "known_length_px": 150}` — applies scalar calibration and clears any homography
- `DELETE /api/calibration`
- `POST /api/video/upload`
- MJPEG feeds: `/video_feed`, `/mask_feed`, `/edge_feed`

## Tests

Install the development/test dependencies and run the synthetic validation suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover ROI cropping and coordinate reporting, stability under small intensity changes, one-circle duplicate suppression on a synthetic hole, rectangle-calibration homography, and invalid corner-order rejection. Real-camera/video validation is still required before relying on measurements.
