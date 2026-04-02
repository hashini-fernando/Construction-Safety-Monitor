"""
Construction Safety Monitor — Streamlit App
============================================
Run:  streamlit run app.py
"""

import json
import time
import tempfile
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Construction Safety Monitor",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark industrial theme */
.stApp {
    background-color: #0f1117;
    color: #e8e6e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #2a2f3a;
}

/* Hide default streamlit header */
header[data-testid="stHeader"] { display: none; }

/* Custom top bar */
.topbar {
    background: #161b22;
    border-bottom: 2px solid #f5a623;
    padding: 16px 24px;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 16px;
}
.topbar-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 18px;
    font-weight: 600;
    color: #f5a623;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.topbar-sub {
    font-size: 12px;
    color: #6e7681;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.05em;
}

/* Verdict banners */
.verdict-safe {
    background: #0d2a1a;
    border: 1.5px solid #2ea043;
    border-left: 5px solid #2ea043;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 12px 0;
}
.verdict-unsafe {
    background: #2a0d0d;
    border: 1.5px solid #da3633;
    border-left: 5px solid #da3633;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 12px 0;
}
.verdict-uncertain {
    background: #1f1a0a;
    border: 1.5px solid #d29922;
    border-left: 5px solid #d29922;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 12px 0;
}
.verdict-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    letter-spacing: 0.1em;
}
.verdict-safe   .verdict-label { color: #3fb950; }
.verdict-unsafe .verdict-label { color: #f85149; }
.verdict-uncertain .verdict-label { color: #d29922; }
.verdict-sub {
    font-size: 12px;
    color: #8b949e;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 16px 0;
    flex-wrap: wrap;
}
.metric-card {
    background: #161b22;
    border: 1px solid #2a2f3a;
    border-radius: 8px;
    padding: 14px 18px;
    flex: 1;
    min-width: 100px;
    text-align: center;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: #f5a623;
    line-height: 1;
}
.metric-label {
    font-size: 11px;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
}

/* Worker cards */
.worker-card {
    background: #161b22;
    border: 1px solid #2a2f3a;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 8px 0;
}
.worker-card.violation { border-left: 4px solid #f85149; }
.worker-card.safe      { border-left: 4px solid #3fb950; }
.worker-card.uncertain { border-left: 4px solid #d29922; }
.worker-id {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.worker-status-safe      { color: #3fb950; font-weight: 600; font-size: 14px; }
.worker-status-violation { color: #f85149; font-weight: 600; font-size: 14px; }
.worker-status-uncertain { color: #d29922; font-weight: 600; font-size: 14px; }
.violation-item {
    font-size: 12px;
    color: #f0883e;
    padding: 3px 0;
    font-family: 'IBM Plex Mono', monospace;
}
.violation-item::before { content: "▸ "; }

/* PPE badge */
.ppe-badge {
    display: inline-block;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    font-family: 'IBM Plex Mono', monospace;
    margin: 2px;
}
.ppe-ok   { background: #0d2a1a; color: #3fb950; border: 1px solid #2ea043; }
.ppe-miss { background: #2a0d0d; color: #f85149; border: 1px solid #da3633; }

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6e7681;
    border-bottom: 1px solid #2a2f3a;
    padding-bottom: 6px;
    margin: 20px 0 12px 0;
}

/* JSON block */
.json-block {
    background: #0d1117;
    border: 1px solid #2a2f3a;
    border-radius: 6px;
    padding: 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #79c0ff;
    overflow-x: auto;
    white-space: pre;
    max-height: 300px;
    overflow-y: auto;
}

/* Inference log */
.log-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #8b949e;
    padding: 2px 0;
}
.log-line.safe     { color: #3fb950; }
.log-line.unsafe   { color: #f85149; }

/* Upload zone */
[data-testid="stFileUploader"] {
    background: #161b22;
    border: 2px dashed #2a2f3a;
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background: #f5a623;
    color: #0f1117;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    border: none;
    border-radius: 4px;
    font-size: 13px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 8px 20px;
}
.stButton > button:hover {
    background: #ffbb47;
    color: #0f1117;
}

/* Sliders and selects */
[data-testid="stSlider"] > div > div { background: #2a2f3a; }

/* Progress bar */
.stProgress > div > div { background: #f5a623; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Inference logic (inline — no external import needed)
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["person","hard_hat","no_hard_hat","safety_vest","no_safety_vest"]

COLORS_BGR = {
    "person":         (200,200,200),
    "hard_hat":       ( 50,200, 80),
    "safety_vest":    ( 50,200, 80),
    "no_hard_hat":    ( 40, 60,220),
    "no_safety_vest": ( 40, 60,220),
}
STATUS_BGR = {
    "SAFE":      ( 50,200, 80),
    "VIOLATION": ( 40, 60,220),
    "UNCERTAIN": ( 30,180,255),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    aa=(a[2]-a[0])*(a[3]-a[1]); ab=(b[2]-b[0])*(b[3]-b[1])
    return inter/(aa+ab-inter) if aa+ab-inter>1e-6 else 0.0

def centroid_in(ppe, w):
    cx=(ppe[0]+ppe[2])/2; cy=(ppe[1]+ppe[3])/2
    return w[0]<=cx<=w[2] and w[1]<=cy<=w[3]

def run_inference(model, frame, conf_th, iou_th, uncertainty_th):
    t0   = time.perf_counter()
    pred = model.predict(frame, conf=conf_th, iou=iou_th, verbose=False)[0]
    ms   = (time.perf_counter()-t0)*1000

    dets = []
    for box in pred.boxes:
        cls = pred.names[int(box.cls)]
        if cls not in CLASS_NAMES: continue
        dets.append({
            "cls":  cls,
            "conf": float(box.conf),
            "box":  box.xyxy[0].tolist(),
        })

    # Worker–PPE association
    workers  = [d for d in dets if d["cls"]=="person"]
    ppe_dets = [d for d in dets if d["cls"]!="person"]
    statuses = []
    for i,w in enumerate(workers):
        statuses.append({
            "id": i+1, "box": w["box"], "conf": w["conf"],
            "has_hardhat": False, "has_vest": False,
            "violations": [], "uncertain": w["conf"]<uncertainty_th, "ppe": [],
        })

    for ppe in ppe_dets:
        best_i,best_s=-1,-1.0
        for i,w in enumerate(workers):
            s=iou(ppe["box"],w["box"])
            if s<0.05 and centroid_in(ppe["box"],w["box"]): s=0.01
            if s>best_s: best_s,best_i=s,i
        if best_i>=0:
            ws=statuses[best_i]; ws["ppe"].append(ppe)
            if ppe["cls"]=="hard_hat":    ws["has_hardhat"]=True
            if ppe["cls"]=="safety_vest": ws["has_vest"]=True
            if ppe["conf"]<uncertainty_th: ws["uncertain"]=True

    for ws in statuses:
        sfx=" (uncertain)" if ws["uncertain"] else ""
        if not ws["has_hardhat"]:   ws["violations"].append("Missing hard hat"+sfx)
        if not ws["has_vest"]:      ws["violations"].append("Missing safety vest"+sfx)

    violated  = any(ws["violations"] for ws in statuses)
    uncertain = any(ws["uncertain"]  for ws in statuses)
    verdict   = ("UNSAFE" if violated else "SAFE")
    if uncertain: verdict+="-UNCERTAIN"
    if not statuses: verdict="NO_WORKERS"

    # Annotate frame
    ann = frame.copy()
    h,w_px = ann.shape[:2]
    bc = STATUS_BGR.get("SAFE" if verdict.startswith("SAFE") else
                        "UNCERTAIN" if "UNCERTAIN" in verdict else "VIOLATION")
    cv2.rectangle(ann,(0,0),(w_px,30),bc,-1)
    txt=(f"{verdict}  |  Workers: {len(statuses)}  |  "
         f"Violations: {sum(1 for ws in statuses if ws['violations'])}")
    cv2.putText(ann,txt,(8,21),FONT,0.55,(255,255,255),1,cv2.LINE_AA)

    for ws in statuses:
        col=STATUS_BGR.get(
            "UNCERTAIN" if ws["uncertain"] and ws["violations"] else
            "SAFE" if not ws["violations"] else "VIOLATION")
        b=ws["box"]
        cv2.rectangle(ann,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),col,2)
        lbl=("SAFE" if not ws["violations"] else
             "UNCERTAIN" if ws["uncertain"] else "VIOLATION")
        (tw,th),bl=cv2.getTextSize(f"W{ws['id']} {lbl}",FONT,0.45,1)
        cv2.rectangle(ann,(int(b[0]),int(b[1])-th-6),
                      (int(b[0])+tw+4,int(b[1])+bl),col,-1)
        cv2.putText(ann,f"W{ws['id']} {lbl}",
                    (int(b[0])+2,int(b[1])-2),FONT,0.45,(255,255,255),1,cv2.LINE_AA)
        for vi,v in enumerate(ws["violations"]):
            (tw2,th2),bl2=cv2.getTextSize(f"! {v}",FONT,0.38,1)
            yy=int(b[3])+18+vi*17
            cv2.rectangle(ann,(int(b[0]),yy-th2-3),(int(b[0])+tw2+4,yy+bl2),
                          (40,60,220),-1)
            cv2.putText(ann,f"! {v}",(int(b[0])+2,yy),FONT,0.38,
                        (255,255,255),1,cv2.LINE_AA)
        for ppe in ws["ppe"]:
            pc=COLORS_BGR.get(ppe["cls"],(180,180,180)); pb=ppe["box"]
            cv2.rectangle(ann,(int(pb[0]),int(pb[1])),(int(pb[2]),int(pb[3])),pc,1)
            cv2.putText(ann,f"{ppe['cls']} {ppe['conf']:.2f}",
                        (int(pb[0]),int(pb[1])-3),FONT,0.33,pc,1,cv2.LINE_AA)

    return {
        "verdict": verdict, "workers": statuses,
        "inference_ms": round(ms,1),
        "workers_total": len(statuses),
        "workers_violated": sum(1 for ws in statuses if ws["violations"]),
        "workers_compliant": sum(1 for ws in statuses if not ws["violations"]),
        "workers_uncertain": sum(1 for ws in statuses if ws["uncertain"]),
    }, ann


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def verdict_banner(report):
    v = report["verdict"]
    css = ("safe" if v.startswith("SAFE") else
           "uncertain" if "UNCERTAIN" in v else "unsafe")
    icons = {"safe":"✓","unsafe":"✗","uncertain":"⚠"}
    st.markdown(f"""
    <div class="verdict-{css}">
      <div class="verdict-label">{icons[css]} {v}</div>
      <div class="verdict-sub">
        {report['workers_total']} workers detected &nbsp;·&nbsp;
        {report['workers_violated']} violation(s) &nbsp;·&nbsp;
        {report['inference_ms']:.0f} ms inference
      </div>
    </div>""", unsafe_allow_html=True)


def metrics_row(report):
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-value">{report['workers_total']}</div>
        <div class="metric-label">Workers</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:#3fb950">
          {report['workers_compliant']}</div>
        <div class="metric-label">Compliant</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:#f85149">
          {report['workers_violated']}</div>
        <div class="metric-label">Violations</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:#d29922">
          {report['workers_uncertain']}</div>
        <div class="metric-label">Uncertain</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:#8b949e">
          {report['inference_ms']:.0f}ms</div>
        <div class="metric-label">Speed</div>
      </div>
    </div>""", unsafe_allow_html=True)


def worker_cards(report):
    st.markdown('<div class="section-header">Worker details</div>',
                unsafe_allow_html=True)
    for ws in report["workers"]:
        css  = ("safe" if not ws["violations"] else
                "uncertain" if ws["uncertain"] else "violation")
        stat = ("SAFE" if not ws["violations"] else
                "UNCERTAIN" if ws["uncertain"] else "VIOLATION")
        hat_b  = '<span class="ppe-badge ppe-ok">hard hat ✓</span>'  if ws["has_hardhat"] \
            else '<span class="ppe-badge ppe-miss">hard hat ✗</span>'
        vest_b = '<span class="ppe-badge ppe-ok">vest ✓</span>'      if ws["has_vest"] \
            else '<span class="ppe-badge ppe-miss">vest ✗</span>'
        viols  = "".join(
            f'<div class="violation-item">{v}</div>'
            for v in ws["violations"]
        )
        st.markdown(f"""
        <div class="worker-card {css}">
          <div class="worker-id">Worker #{ws['id']}
            &nbsp;·&nbsp; conf {ws['conf']:.2f}
          </div>
          <div class="worker-status-{css.replace('-','')}" style="margin:4px 0">
            {stat}</div>
          <div>{hat_b}{vest_b}</div>
          {viols}
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(path):
    return YOLO(path)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙ Configuration")

    weights_file = st.file_uploader("Upload best.pt", type=["pt"],
                                     help="Your trained YOLOv8 weights")
    st.divider()

    conf_th  = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
    iou_th   = st.slider("NMS IoU threshold",    0.10, 0.90, 0.45, 0.05)
    unc_th   = st.slider("Uncertainty flag below",0.10,0.90, 0.50, 0.05,
                         help="Detections below this confidence are flagged as uncertain")
    st.divider()

    show_json = st.checkbox("Show raw JSON report", value=False)
    st.divider()

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                color:#444d56;line-height:1.8">
    CLASSES<br>
    · person<br>
    · hard_hat ✓<br>
    · no_hard_hat ✗<br>
    · safety_vest ✓<br>
    · no_safety_vest ✗
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Top bar
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="topbar">
  <div>
    <div class="topbar-title">🦺 Construction Safety Monitor</div>
    <div class="topbar-sub">PPE compliance detection · YOLOv8</div>
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

model = None
if weights_file:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        f.write(weights_file.read())
        tmp_pt = f.name
    try:
        model = load_model(tmp_pt)
        st.success(f"Model loaded  ·  classes: {list(model.names.values())}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
else:
    # Try loading default best.pt if it exists locally
    if Path("best.pt").exists():
        model = load_model("best.pt")
        st.info("Using local `best.pt`  — or upload a different one in the sidebar.")
    else:
        st.warning("Upload `best.pt` in the sidebar to get started.")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_img, tab_folder, tab_video, tab_webcam = st.tabs([
    "📷  Image", "📁  Folder", "🎬  Video", "📹  Webcam"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single image
# ══════════════════════════════════════════════════════════════════════════════
with tab_img:
    uploaded = st.file_uploader("Upload an image",
                                type=["jpg","jpeg","png","bmp"],
                                key="img_upload")
    if uploaded and model:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Running inference ..."):
            report, ann = run_inference(model, frame, conf_th, iou_th, unc_th)

        col_img, col_info = st.columns([3, 2])

        with col_img:
            st.markdown('<div class="section-header">Annotated result</div>',
                        unsafe_allow_html=True)
            rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            st.image(rgb, use_container_width=True)

            # Download annotated image
            _, buf = cv2.imencode(".jpg", ann)
            st.download_button("⬇ Download annotated image",
                               data=buf.tobytes(),
                               file_name=f"annotated_{uploaded.name}",
                               mime="image/jpeg")

        with col_info:
            verdict_banner(report)
            metrics_row(report)
            worker_cards(report)

            if show_json:
                st.markdown('<div class="section-header">JSON report</div>',
                            unsafe_allow_html=True)
                clean = {k: v for k, v in report.items() if k != "workers"}
                clean["workers"] = [
                    {k2: v2 for k2, v2 in w.items() if k2 != "ppe"}
                    for w in report["workers"]
                ]
                st.markdown(
                    f'<div class="json-block">{json.dumps(clean, indent=2)}</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Folder / batch
# ══════════════════════════════════════════════════════════════════════════════
with tab_folder:
    uploaded_imgs = st.file_uploader(
        "Upload multiple images",
        type=["jpg","jpeg","png","bmp"],
        accept_multiple_files=True,
        key="folder_upload")

    if uploaded_imgs and model:
        if st.button("Run batch inference", key="run_batch"):
            progress  = st.progress(0, text="Starting ...")
            log_area  = st.empty()
            results   = []
            log_lines = []

            for i, f in enumerate(uploaded_imgs):
                file_bytes = np.frombuffer(f.read(), np.uint8)
                frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                report, ann = run_inference(model, frame, conf_th, iou_th, unc_th)
                results.append((f.name, report, ann))

                icon  = "✓" if report["verdict"].startswith("SAFE") else "✗"
                css   = "safe" if report["verdict"].startswith("SAFE") else "unsafe"
                log_lines.append(
                    f'<div class="log-line {css}">'
                    f'{icon} {f.name:<40} {report["verdict"]:<22} '
                    f'workers={report["workers_total"]}  '
                    f'violations={report["workers_violated"]}  '
                    f'{report["inference_ms"]:.0f}ms</div>'
                )
                log_area.markdown(
                    '<div style="background:#0d1117;border:1px solid #2a2f3a;'
                    'border-radius:6px;padding:12px;max-height:220px;overflow-y:auto">'
                    + "".join(log_lines) + "</div>",
                    unsafe_allow_html=True)
                progress.progress((i+1)/len(uploaded_imgs),
                                  text=f"Processing {i+1}/{len(uploaded_imgs)} ...")

            progress.empty()

            # Summary
            st.markdown('<div class="section-header">Batch summary</div>',
                        unsafe_allow_html=True)
            n        = len(results)
            unsafe_n = sum(1 for _,r,_ in results if not r["verdict"].startswith("SAFE"))
            total_w  = sum(r["workers_total"]   for _,r,_ in results)
            total_v  = sum(r["workers_violated"] for _,r,_ in results)
            avg_ms   = sum(r["inference_ms"]     for _,r,_ in results)/n

            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-card">
                <div class="metric-value">{n}</div>
                <div class="metric-label">Images</div>
              </div>
              <div class="metric-card">
                <div class="metric-value" style="color:#f85149">{unsafe_n}</div>
                <div class="metric-label">Unsafe</div>
              </div>
              <div class="metric-card">
                <div class="metric-value" style="color:#3fb950">{n-unsafe_n}</div>
                <div class="metric-label">Safe</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">{total_w}</div>
                <div class="metric-label">Workers</div>
              </div>
              <div class="metric-card">
                <div class="metric-value" style="color:#f85149">{total_v}</div>
                <div class="metric-label">Violations</div>
              </div>
              <div class="metric-card">
                <div class="metric-value" style="color:#8b949e">{avg_ms:.0f}ms</div>
                <div class="metric-label">Avg speed</div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Gallery
            st.markdown('<div class="section-header">Results gallery</div>',
                        unsafe_allow_html=True)
            cols = st.columns(3)
            for idx, (fname, rep, ann) in enumerate(results):
                with cols[idx % 3]:
                    rgb   = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                    color = "#f85149" if not rep["verdict"].startswith("SAFE") else "#3fb950"
                    st.image(rgb, caption=fname, use_container_width=True)
                    st.markdown(
                        f'<div style="text-align:center;font-family:IBM Plex Mono,monospace;'
                        f'font-size:11px;color:{color};margin-top:-8px">'
                        f'{rep["verdict"]}</div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Video
# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    vid_file = st.file_uploader("Upload a video",
                                type=["mp4","avi","mov","mkv"],
                                key="vid_upload")
    sample_every = st.slider("Process every N frames", 1, 10, 1,
                             help="Higher = faster but less coverage")

    if vid_file and model:
        if st.button("Run video inference", key="run_video"):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(vid_file.read()); tmp_vid = f.name

            cap   = cv2.VideoCapture(tmp_vid)
            fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
            fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_tmp  = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            out_path = out_tmp.name; out_tmp.close()
            writer   = cv2.VideoWriter(out_path,
                                       cv2.VideoWriter_fourcc(*"mp4v"),
                                       fps, (fw, fh))

            prog      = st.progress(0, text="Processing video ...")
            preview   = st.empty()
            log_area2 = st.empty()
            reports   = []; log_lines2 = []; fid = 0

            while True:
                ret, frame = cap.read()
                if not ret: break
                if fid % sample_every == 0:
                    report, ann = run_inference(model, frame, conf_th, iou_th, unc_th)
                    last_ann = ann
                else:
                    ann = last_ann if fid > 0 else frame

                writer.write(ann)
                if fid % sample_every == 0:
                    reports.append(report)
                    icon = "✓" if report["verdict"].startswith("SAFE") else "✗"
                    css  = "safe" if report["verdict"].startswith("SAFE") else "unsafe"
                    log_lines2.append(
                        f'<div class="log-line {css}">'
                        f'{icon} frame {fid:>5}  {report["verdict"]:<22} '
                        f'workers={report["workers_total"]}  '
                        f'{report["inference_ms"]:.0f}ms</div>'
                    )
                    if fid % (sample_every*10) == 0:
                        rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                        preview.image(rgb, caption=f"Frame {fid}", use_container_width=True)
                        log_area2.markdown(
                            '<div style="background:#0d1117;border:1px solid #2a2f3a;'
                            'border-radius:6px;padding:12px;max-height:180px;overflow-y:auto">'
                            + "".join(log_lines2[-20:]) + "</div>",
                            unsafe_allow_html=True)

                if total > 0:
                    prog.progress(min(fid/total, 1.0), text=f"Frame {fid}/{total}")
                fid += 1

            cap.release(); writer.release(); prog.empty()

            with open(out_path,"rb") as f:
                st.download_button("⬇ Download annotated video",
                                   data=f.read(),
                                   file_name=f"annotated_{vid_file.name}",
                                   mime="video/mp4")

            if reports:
                n2      = len(reports)
                unsafe2 = sum(1 for r in reports if not r["verdict"].startswith("SAFE"))
                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-card">
                    <div class="metric-value">{fid}</div>
                    <div class="metric-label">Frames</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-value" style="color:#f85149">{unsafe2}</div>
                    <div class="metric-label">Unsafe frames</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-value">{sum(r["workers_total"] for r in reports)}</div>
                    <div class="metric-label">Workers detected</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-value" style="color:#f85149">
                      {sum(r["workers_violated"] for r in reports)}</div>
                    <div class="metric-label">Violations</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-value" style="color:#8b949e">
                      {sum(r["inference_ms"] for r in reports)/n2:.0f}ms</div>
                    <div class="metric-label">Avg speed</div>
                  </div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Webcam
# ══════════════════════════════════════════════════════════════════════════════
with tab_webcam:
    st.markdown("""
    <div style="background:#161b22;border:1px solid #2a2f3a;border-radius:8px;
                padding:20px;text-align:center;margin:20px 0">
      <div style="font-family:IBM Plex Mono,monospace;color:#f5a623;
                  font-size:16px;margin-bottom:8px">Live webcam mode</div>
      <div style="color:#8b949e;font-size:13px;line-height:1.7">
        Streamlit cannot stream directly from webcam in the browser.<br>
        Use the command line script instead for live detection:
      </div>
      <div style="background:#0d1117;border:1px solid #2a2f3a;border-radius:6px;
                  padding:14px;margin-top:16px;font-family:IBM Plex Mono,monospace;
                  font-size:13px;color:#79c0ff;text-align:left;display:inline-block">
        python inference.py --source 0 --show
      </div>
      <div style="color:#6e7681;font-size:11px;margin-top:12px;
                  font-family:IBM Plex Mono,monospace">
        Press S to save a snapshot &nbsp;·&nbsp; Press Q to quit
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Or test with a snapshot image</div>',
                unsafe_allow_html=True)
    snap_file = st.file_uploader("Upload a webcam screenshot",
                                  type=["jpg","jpeg","png"],
                                  key="snap_upload")
    if snap_file and model:
        file_bytes = np.frombuffer(snap_file.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        report, ann = run_inference(model, frame, conf_th, iou_th, unc_th)
        rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
        st.image(rgb, use_container_width=True)
        verdict_banner(report)
        metrics_row(report)
        worker_cards(report)