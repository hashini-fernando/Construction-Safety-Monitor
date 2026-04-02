"""
app.py
Construction Safety Monitor

Run:
    streamlit run app.py
"""

import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Construction Safety Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f1117;
        color: #e6edf3;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    .topbar {
        background: #161b22;
        border-bottom: 2px solid #f5a623;
        padding: 16px 24px;
        margin: -1rem -1rem 2rem -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .topbar-title {
        font-size: 20px;
        font-weight: 700;
        color: #f5a623;
    }
    .topbar-sub {
        font-size: 12px;
        color: #8b949e;
        margin-top: 4px;
    }
    .card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px;
        margin: 10px 0;
    }
    .safe-box {
        background: #0d2a1a;
        border-left: 5px solid #2ea043;
        padding: 14px;
        border-radius: 8px;
    }
    .unsafe-box {
        background: #2a0d0d;
        border-left: 5px solid #f85149;
        padding: 14px;
        border-radius: 8px;
    }
    .uncertain-box {
        background: #2a220d;
        border-left: 5px solid #d29922;
        padding: 14px;
        border-radius: 8px;
    }
    .neutral-box {
        background: #161b22;
        border-left: 5px solid #8b949e;
        padding: 14px;
        border-radius: 8px;
    }
    .metric-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin: 12px 0;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        min-width: 120px;
        text-align: center;
        flex: 1;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #f5a623;
    }
    .metric-label {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
    }
    .worker-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    .badge-ok {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        background: #0d2a1a;
        color: #3fb950;
        border: 1px solid #2ea043;
        font-size: 11px;
        margin-right: 6px;
    }
    .badge-miss {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        background: #2a0d0d;
        color: #f85149;
        border: 1px solid #da3633;
        font-size: 11px;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONFIG
# ============================================================

CLASS_NAMES = ["person", "hard_hat", "no_hard_hat", "safety_vest", "no_safety_vest"]

STATUS_BGR = {
    "SAFE": (50, 200, 80),
    "VIOLATION": (40, 60, 220),
    "UNCERTAIN": (30, 180, 255),
    "NEUTRAL": (120, 120, 120),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX

DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.45
DEFAULT_UNCERTAIN = 0.50
DEFAULT_VIDEO_FPS = 30


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class WorkerStatus:
    worker_id: int
    conf: float
    box: List[float]
    has_hardhat: bool
    has_vest: bool
    violations: List[str]
    uncertain: bool
    status_label: str
    compliant: bool


@dataclass
class InferenceResult:
    verdict: str
    workers_total: int
    workers_violated: int
    workers_compliant: int
    workers_uncertain: int
    inference_ms: float
    workers: List[WorkerStatus]


# ============================================================
# CORE LOGIC
# ============================================================

def iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 1e-6 else 0.0


def centroid_in(ppe_box, worker_box):
    cx = (ppe_box[0] + ppe_box[2]) / 2
    cy = (ppe_box[1] + ppe_box[3]) / 2
    return worker_box[0] <= cx <= worker_box[2] and worker_box[1] <= cy <= worker_box[3]


def run_inference(model, frame, conf_th=DEFAULT_CONF, iou_th=DEFAULT_IOU, uncertainty_th=DEFAULT_UNCERTAIN):
    start = time.perf_counter()
    pred = model.predict(frame, conf=conf_th, iou=iou_th, verbose=False)[0]
    inference_ms = (time.perf_counter() - start) * 1000

    detections = []
    for box in pred.boxes:
        cls_name = pred.names[int(box.cls)]
        if cls_name not in CLASS_NAMES:
            continue
        detections.append(
            {
                "cls": cls_name,
                "conf": float(box.conf),
                "box": box.xyxy[0].tolist(),
            }
        )

    workers = [d for d in detections if d["cls"] == "person"]
    ppe_dets = [d for d in detections if d["cls"] != "person"]

    worker_states = []
    for idx, worker in enumerate(workers):
        worker_states.append(
            {
                "id": idx + 1,
                "box": worker["box"],
                "conf": worker["conf"],
                "has_hardhat": False,
                "has_vest": False,
                "violations": [],
                "uncertain": worker["conf"] < uncertainty_th,
            }
        )

    for ppe in ppe_dets:
        best_idx = -1
        best_score = -1.0

        for idx, worker in enumerate(workers):
            score = iou(ppe["box"], worker["box"])
            if score < 0.05 and centroid_in(ppe["box"], worker["box"]):
                score = 0.01
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0:
            ws = worker_states[best_idx]

            if ppe["cls"] == "hard_hat":
                ws["has_hardhat"] = True
            if ppe["cls"] == "safety_vest":
                ws["has_vest"] = True
            if ppe["conf"] < uncertainty_th:
                ws["uncertain"] = True

    final_workers = []
    for ws in worker_states:
        suffix = " (uncertain)" if ws["uncertain"] else ""

        if not ws["has_hardhat"]:
            ws["violations"].append("Missing hard hat" + suffix)
        if not ws["has_vest"]:
            ws["violations"].append("Missing safety vest" + suffix)

        if not ws["violations"]:
            status_label = "SAFE"
        elif ws["uncertain"]:
            status_label = "UNCERTAIN"
        else:
            status_label = "VIOLATION"

        final_workers.append(
            WorkerStatus(
                worker_id=ws["id"],
                conf=ws["conf"],
                box=ws["box"],
                has_hardhat=ws["has_hardhat"],
                has_vest=ws["has_vest"],
                violations=ws["violations"],
                uncertain=ws["uncertain"],
                status_label=status_label,
                compliant=len(ws["violations"]) == 0,
            )
        )

    violated = any(not w.compliant for w in final_workers)
    uncertain = any(w.uncertain for w in final_workers)

    verdict = "UNSAFE" if violated else "SAFE"
    if uncertain and final_workers:
        verdict += "-UNCERTAIN"
    if not final_workers:
        verdict = "NO_WORKERS"

    return InferenceResult(
        verdict=verdict,
        workers_total=len(final_workers),
        workers_violated=sum(1 for w in final_workers if not w.compliant),
        workers_compliant=sum(1 for w in final_workers if w.compliant),
        workers_uncertain=sum(1 for w in final_workers if w.uncertain),
        inference_ms=round(inference_ms, 1),
        workers=final_workers,
    )


def annotate_frame(frame, result: InferenceResult):
    ann = frame.copy()
    h, w = ann.shape[:2]

    if result.verdict == "NO_WORKERS":
        top_color = STATUS_BGR["NEUTRAL"]
    elif result.verdict.startswith("SAFE"):
        top_color = STATUS_BGR["SAFE"]
    elif "UNCERTAIN" in result.verdict:
        top_color = STATUS_BGR["UNCERTAIN"]
    else:
        top_color = STATUS_BGR["VIOLATION"]

    cv2.rectangle(ann, (0, 0), (w, 32), top_color, -1)
    header = f"{result.verdict} | Workers: {result.workers_total} | Violations: {result.workers_violated}"
    cv2.putText(ann, header, (8, 22), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    for ws in result.workers:
        x1, y1, x2, y2 = map(int, ws.box)

        if ws.status_label == "SAFE":
            color = STATUS_BGR["SAFE"]
        elif ws.status_label == "UNCERTAIN":
            color = STATUS_BGR["UNCERTAIN"]
        else:
            color = STATUS_BGR["VIOLATION"]

        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
        cv2.putText(ann, f"W{ws.worker_id} {ws.status_label}", (x1, max(16, y1 - 4)), FONT, 0.5, color, 2, cv2.LINE_AA)

        for i, v in enumerate(ws.violations):
            yy = y2 + 18 + (i * 16)
            if yy < h - 4:
                cv2.putText(ann, f"! {v}", (x1, yy), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return ann


def make_report(result: InferenceResult, source="", frame_id=None, timestamp=None):
    return {
        "source": source,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "verdict": result.verdict,
        "workers_total": result.workers_total,
        "workers_compliant": result.workers_compliant,
        "workers_violated": result.workers_violated,
        "workers_uncertain": result.workers_uncertain,
        "inference_ms": result.inference_ms,
        "workers": [asdict(w) for w in result.workers],
    }


# ============================================================
# UTILITIES
# ============================================================

def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def verdict_class(verdict: str):
    if verdict == "NO_WORKERS":
        return "neutral-box"
    if "UNCERTAIN" in verdict:
        return "uncertain-box"
    if verdict.startswith("SAFE"):
        return "safe-box"
    return "unsafe-box"


def render_metrics(result: InferenceResult):
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card"><div class="metric-value">{result.workers_total}</div><div class="metric-label">Workers</div></div>
            <div class="metric-card"><div class="metric-value" style="color:#3fb950">{result.workers_compliant}</div><div class="metric-label">Safe</div></div>
            <div class="metric-card"><div class="metric-value" style="color:#f85149">{result.workers_violated}</div><div class="metric-label">Violations</div></div>
            <div class="metric-card"><div class="metric-value" style="color:#d29922">{result.workers_uncertain}</div><div class="metric-label">Uncertain</div></div>
            <div class="metric-card"><div class="metric-value">{result.inference_ms:.0f}ms</div><div class="metric-label">Speed</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_worker_cards(result: InferenceResult):
    if not result.workers:
        return

    st.subheader("Worker details")
    for worker in result.workers:
        hardhat = '<span class="badge-ok">hard hat present</span>' if worker.has_hardhat else '<span class="badge-miss">hard hat missing</span>'
        vest = '<span class="badge-ok">vest present</span>' if worker.has_vest else '<span class="badge-miss">vest missing</span>'

        violations_html = ""
        for v in worker.violations:
            violations_html += f"<div style='color:#f0883e;font-size:12px;margin-top:4px;'>{v}</div>"

        st.markdown(
            f"""
            <div class="worker-card">
                <div style="font-size:11px;color:#8b949e;">Worker #{worker.worker_id} | confidence {worker.conf:.2f}</div>
                <div style="font-weight:700;margin:6px 0;">{worker.status_label}</div>
                <div>{hardhat}{vest}</div>
                <div style="margin-top:6px;">{violations_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_result(frame, result: InferenceResult, source_name="result", show_json=False):
    ann = annotate_frame(frame, result)
    col1, col2 = st.columns([3, 2])

    with col1:
        st.image(bgr_to_rgb(ann), use_container_width=True)
        ok, buf = cv2.imencode(".jpg", ann)
        if ok:
            st.download_button(
                "Download annotated result",
                data=buf.tobytes(),
                file_name=f"annotated_{Path(source_name).name}.jpg",
                mime="image/jpeg",
            )

    with col2:
        st.markdown(
            f"""
            <div class="{verdict_class(result.verdict)}">
                <div style="font-size:20px;font-weight:700;">{result.verdict}</div>
                <div style="font-size:12px;color:#8b949e;margin-top:6px;">
                    {result.workers_total} workers |
                    {result.workers_violated} violations |
                    {result.inference_ms:.0f} ms
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_metrics(result)
        render_worker_cards(result)

        if show_json:
            report = make_report(result, source_name)
            st.code(json.dumps(report, indent=2), language="json")


def process_video(model, video_bytes, file_name, conf_th, iou_th, unc_th, target_fps, show_preview=True):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_in:
        temp_in.write(video_bytes)
        input_path = temp_in.name

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Could not open video.")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_every = max(1, int(round(src_fps / max(target_fps, 1))))

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_out:
        output_path = temp_out.name

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps,
        (width, height),
    )

    preview_slot = st.empty()
    progress_bar = st.progress(0)
    log_slot = st.empty()

    reports = []
    log_lines = []
    frame_id = 0
    last_annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % sample_every == 0:
            result = run_inference(model, frame, conf_th, iou_th, unc_th)
            last_annotated = annotate_frame(frame, result)
            reports.append(make_report(result, file_name, frame_id, frame_id / src_fps))

            log_lines.append(
                f"frame {frame_id} | {result.verdict} | workers={result.workers_total} | violations={result.workers_violated}"
            )
            if len(log_lines) > 12:
                log_lines = log_lines[-12:]

            if show_preview:
                preview_slot.image(bgr_to_rgb(last_annotated), use_container_width=True)
                log_slot.code("\n".join(log_lines), language="text")

        writer.write(last_annotated if last_annotated is not None else frame)

        if total_frames > 0:
            progress_bar.progress(min((frame_id + 1) / total_frames, 1.0))

        frame_id += 1

    cap.release()
    writer.release()

    with open(output_path, "rb") as f:
        st.download_button(
            "Download annotated video",
            data=f.read(),
            file_name=f"annotated_{file_name}",
            mime="video/mp4",
        )

    if reports:
        st.subheader("Video summary")
        unsafe_frames = sum(1 for r in reports if not str(r["verdict"]).startswith("SAFE"))
        avg_ms = sum(r["inference_ms"] for r in reports) / len(reports)
        st.write(
            f"Source FPS: {src_fps:.1f} | Target processing FPS: {target_fps} | "
            f"Processed every {sample_every} frame(s) | Sampled frames: {len(reports)} | "
            f"Unsafe sampled frames: {unsafe_frames} | Avg inference: {avg_ms:.1f} ms"
        )


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Model")
    weights_file = st.file_uploader("Upload model weights", type=["pt"])

    st.header("Thresholds")
    conf_th = st.slider("Confidence threshold", 0.10, 0.90, DEFAULT_CONF, 0.05)
    iou_th = st.slider("IoU threshold", 0.10, 0.90, DEFAULT_IOU, 0.05)
    unc_th = st.slider("Uncertainty threshold", 0.10, 0.90, DEFAULT_UNCERTAIN, 0.05)

    st.header("Options")
    show_json = st.checkbox("Show JSON report", value=False)

model = None
model_status = "No model loaded"

if weights_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_weights:
        temp_weights.write(weights_file.read())
        temp_model_path = temp_weights.name
    try:
        model = load_model(temp_model_path)
        model_status = "Model ready"
    except Exception as e:
        st.error(f"Failed to load model: {e}")
elif Path("model/best.pt").exists():
    model = load_model("model/best.pt")
    model_status = "Model ready"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="topbar">
        <div>
            <div class="topbar-title">Construction Safety Monitor</div>
            <div class="topbar-sub">Single-page PPE compliance application</div>
        </div>
        <div style="font-size:12px;color:#8b949e;">{model_status}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.warning("Upload best.pt from the sidebar, or place it at model/best.pt")


# ============================================================
# UNIFIED INPUT
# ============================================================

mode = st.selectbox(
    "Input type",
    [
        "Single image",
        "Batch images",
        "Video",
        "Camera snapshot",
        "Local live webcam",
    ],
)

# ============================================================
# SINGLE IMAGE
# ============================================================

if mode == "Single image":
    uploaded = st.file_uploader("Upload one image", type=["jpg", "jpeg", "png", "bmp"], key="single_image")
    if uploaded is not None and model is not None:
        frame = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        result = run_inference(model, frame, conf_th, iou_th, unc_th)
        render_result(frame, result, uploaded.name, show_json)

# ============================================================
# BATCH IMAGES
# ============================================================

elif mode == "Batch images":
    uploaded_imgs = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch_images",
    )

    if uploaded_imgs and model is not None:
        if st.button("Run batch inference"):
            progress_bar = st.progress(0)
            results = []

            for i, file in enumerate(uploaded_imgs):
                frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                result = run_inference(model, frame, conf_th, iou_th, unc_th)
                annotated = annotate_frame(frame, result)
                results.append((file.name, result, annotated))
                progress_bar.progress((i + 1) / len(uploaded_imgs))

            st.subheader("Batch summary")
            total = len(results)
            safe_count = sum(1 for _, r, _ in results if r.verdict.startswith("SAFE"))
            unsafe_count = total - safe_count
            total_workers = sum(r.workers_total for _, r, _ in results)
            total_violations = sum(r.workers_violated for _, r, _ in results)

            st.write(
                f"Images: {total} | Safe: {safe_count} | Unsafe: {unsafe_count} | "
                f"Workers: {total_workers} | Violations: {total_violations}"
            )

            cols = st.columns(3)
            for idx, (name, result, annotated) in enumerate(results):
                with cols[idx % 3]:
                    st.image(
                        bgr_to_rgb(annotated),
                        caption=f"{name} | {result.verdict}",
                        use_container_width=True,
                    )

# ============================================================
# VIDEO
# ============================================================

elif mode == "Video":
    vid_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="video_file")
    target_fps = st.slider("Processing FPS target", 1, 60, DEFAULT_VIDEO_FPS, 1)
    show_preview = st.checkbox("Show live preview while processing", value=True)

    if vid_file is not None and model is not None:
        if st.button("Run video inference"):
            process_video(
                model=model,
                video_bytes=vid_file.read(),
                file_name=vid_file.name,
                conf_th=conf_th,
                iou_th=iou_th,
                unc_th=unc_th,
                target_fps=target_fps,
                show_preview=show_preview,
            )

# ============================================================
# CAMERA SNAPSHOT
# ============================================================

elif mode == "Camera snapshot":
    st.info("Use the device camera to capture a single image.")
    camera_file = st.camera_input("Capture image")

    if camera_file is not None and model is not None:
        file_bytes = np.frombuffer(camera_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        result = run_inference(model, frame, conf_th, iou_th, unc_th)
        render_result(frame, result, "camera_snapshot", show_json)

# ============================================================
# LOCAL LIVE WEBCAM
# ============================================================

elif mode == "Local live webcam":
    st.warning("This mode works when Streamlit is running on your own computer. For browser-based capture, use Camera snapshot.")

    cam_id = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
    target_fps = st.slider("Live display FPS target", 1, 60, DEFAULT_VIDEO_FPS, 1, key="live_fps")
    max_frames = st.slider("Maximum frames per session", 30, 600, 240, 30)

    if st.button("Start live webcam") and model is not None:
        cap = cv2.VideoCapture(int(cam_id))
        if not cap.isOpened():
            st.error(f"Could not open camera {cam_id}")
        else:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            sample_every = max(1, int(round(src_fps / max(target_fps, 1))))

            frame_slot = st.empty()
            info_slot = st.empty()

            frame_id = 0
            while frame_id < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % sample_every == 0:
                    result = run_inference(model, frame, conf_th, iou_th, unc_th)
                    annotated = annotate_frame(frame, result)
                    frame_slot.image(bgr_to_rgb(annotated), use_container_width=True)
                    info_slot.write(
                        f"{result.verdict} | workers={result.workers_total} | "
                        f"violations={result.workers_violated} | inference={result.inference_ms:.0f} ms"
                    )

                frame_id += 1

            cap.release()
            st.success("Webcam session ended.")