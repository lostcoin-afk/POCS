# Hardware Geometry Inspector — Silhouette / Corner Version

A FastAPI + OpenCV prototype for industrial part profile inspection. This version extracts the component silhouette, simplifies the external contour into boundary edges, measures the angle at each physical contour corner, and compares corner positions/angles/adjacent edge orientations with a saved reference template.

## Project layout

```text
hardware_angle_inspection/
├── app.py                 # FastAPI, capture sources, inspection service and API
├── geometry.py            # Silhouette segmentation, contour geometry and corner matching
├── build_template.py      # Interactive known-good reference builder
├── requirements.txt
├── data/
│   └── angle_template.json  # Written on template save
├── uploads/
└── templates/
    ├── index.html
    ├── style.css
    └── app.js
```

## Install and run the web inspector

Python 3.10+ is recommended.

```bash
cd hardware_angle_inspection
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000`.

## Input sources

In the web dashboard, select a saved video, uploaded video, webcam, or LUCID Arena SDK camera. For a local server-side video path, use the path as seen by the machine running FastAPI.

For a LUCID camera, install the official Arena SDK and its Python bindings in the same environment used to run Uvicorn. Confirm the camera streams in ArenaView first. The adapter is designed for common 8-bit Mono, Bayer, RGB/BGR and RGBA/BGRA formats; camera behavior should be validated with the actual model and SDK version.

## Build a known-good corner template

Use the interactive builder to check the silhouette and tune the polygon simplification before saving the reference.

```bash
# Still image
python build_template.py --image /path/to/known_good_part.png --name "Bracket A"

# Video: pause on a suitable frame with SPACE, then press S
python build_template.py --video /path/to/reference_video.mp4 --name "Bracket A"

# Webcam or LUCID camera
python build_template.py --webcam 0 --name "Bracket A"
python build_template.py --arena --arena-serial YOUR_CAMERA_SERIAL --name "Bracket A"
```

Builder keys: `S` saves the current frame, `SPACE` pauses/resumes video/camera acquisition, and `Q` or `ESC` exits. The preview windows show the annotated part, the Canny edge map, and the segmentation mask.

The main visual checkpoint is the **green outer contour**: it should follow the complete physical outline, especially the two short diagonal edges and endpoints of the central notch. If the contour includes shadows/background or omits part of the metal profile, do not save the template yet.

The default output is `data/angle_template.json`, the file loaded by `app.py`. When replacing an existing template, the builder asks for confirmation unless `--force` is passed. An alternate `--output` path is not loaded by the web app automatically.

The template stores:

- Normalized positions of silhouette corners.
- The included/opening angle at each corner (0–180°).
- The true polygon interior angle and whether the point is concave.
- The orientations and normalized lengths of the two adjacent edges.
- Normalized edge endpoints and lengths.
- The detection settings used when the template was created.

The web app loads the stored processing settings on startup. Restart Uvicorn after building a template outside the dashboard.

## What the processing pipeline does

1. **Preprocess:** Convert to grayscale and blur lightly.
2. **Segment silhouette:** Generate candidate masks with Otsu and adaptive thresholding; also evaluate a closed Canny edge ring. Choose a plausible external contour that does not touch the image frame. The selection is heuristic and must be visually verified.
3. **Fill component mask:** The external contour is filled, so internal holes, scratches and reflections are not treated as separate outline components.
4. **Polygon approximation:** `cv2.approxPolyDP` simplifies the contour. Its epsilon is a fraction of the contour perimeter.
5. **Boundary-edge fitting:** Each polygon side is fit independently to its contour chain. A first line estimate rejects off-line points, then refits to inliers; this is robust trimming, not a full randomized RANSAC implementation.
6. **Corner geometry:** Each polygon vertex becomes a corner. The reported opening angle is calculated between the rays from the corner toward the previous and next vertices. For a concave notch, `interior_angle_deg` is separately stored as the polygon interior angle, which can exceed 180°.
7. **Template matching:** Current corners are associated with template corners using normalized position, opening angle and adjacent-edge orientation. Per-frame `C01` IDs are labels only; they are not assumed to identify a permanent part feature.
8. **Inspection result:** Matched corner angles and adjacent edge orientations are compared with the saved reference. The dashboard lists measured/reference angle, deviation, adjacent-orientation difference and status.

## Tune the main parameters

- **Polygon approximation ratio:** Start around `0.004`. Reduce it if the central notch loses vertices; increase it if noisy boundaries produce too many corners. Typical experimental range: `0.002–0.010`, depending on resolution and contour quality.
- **Silhouette morphology kernel:** Start at `5`. Larger kernels can close boundary gaps but can erase narrow features or notch openings.
- **Minimum component area ratio:** The minimum area of a candidate contour divided by the total image area. Lower it if a small part is ignored; raise it to reject background fragments.
- **Canny thresholds:** Affect the visible edge overlay and the edge-ring segmentation fallback. They do not replace visual validation of the silhouette mask.
- **Corner angle tolerance:** Maximum allowed absolute difference from the reference. It should be set based on repeatability studies and the inspection specification, not chosen only to get a PASS.

## Important limitations

- **Check the silhouette mask with your real camera/video.** Global and adaptive thresholds can mistake shadows, background patterns, labels or a dark halo for part boundaries. Fixed, diffuse lighting and a non-textured background are strongly recommended. Backlighting may make external profiles much easier to segment.
- The first version assumes the component is in a repeatable pose. Normalized coordinates compensate for changes in translation/scale relative to the silhouette bounding box, but the application does not perform full rotation/pose alignment or perspective rectification. A fixed fixture is recommended.
- Corner IDs are assigned in contour order per frame. Template matching uses position and local geometry, not the literal ID string.
- The pass/fail rule is a prototype, not a certified metrology procedure. It fails when matched corner angles or adjacent orientations exceed tolerance, or when reference/current corners remain unmatched. Validate repeatability and false-accept/false-reject rates on representative known-good and defective parts.
- Angles are image-space measurements. They do not produce physical dimensions without calibration, and perspective/lens distortion can change apparent angles. The reported geometry should not be interpreted as micrometre-level measurement without a calibrated optical/metrology setup.
- Old templates based on Hough line pairs are intentionally not loaded. Build and save a new corner-based template after installing this version.

## API endpoints

- `GET /` — frontend
- `GET /video_feed` — processed MJPEG stream
- `GET /api/status` — current corners, edges, comparison metrics, source and settings
- `POST /api/start` — start a `video`, `webcam`, or `arena` source
- `POST /api/stop` — stop capture
- `POST /api/settings` — update segmentation/contour settings
- `POST /api/template/save` — save current silhouette geometry as reference
- `POST /api/template/clear` — delete the saved reference
- `POST /api/video/upload` — upload a video file

## Troubleshooting

- **Silhouette not found:** Reduce the minimum component area ratio, check focus and lighting, position the whole part inside the frame, and inspect the Canny/mask preview in `build_template.py`.
- **Silhouette includes a shadow or background:** Use a plain matte background, soften/diffuse reflections, and adjust blur/morphology. Consider backlighting for profile inspection.
- **Too many corners:** Increase contour approximation ratio gradually (for example 0.004 to 0.006).
- **The notch corners disappear:** Reduce approximation ratio (for example 0.004 to 0.003), and avoid an excessive morphology kernel.
- **Template matches poorly:** Use a fixed fixture, build the template from a clean known-good part, verify the silhouette every time, and compare multiple known-good samples before setting a production tolerance.
- **Arena cannot import:** Run Uvicorn/builder from the Python environment containing the official Arena SDK bindings.
