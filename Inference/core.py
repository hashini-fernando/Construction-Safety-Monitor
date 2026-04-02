"""
inference/core.py
=================
Detection, worker-PPE association, and compliance rule engine.

"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "person",
    "hard_hat",
    "no_hard_hat",
    "safety_vest",
    "no_safety_vest",
]

CONF_THRESH    = 0.35
IOU_THRESH     = 0.45
UNCERTAINTY_TH = 0.50   # predictions below this are flagged as uncertain


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single raw detection from the YOLO model."""
    cls_name: str
    conf:     float
    box:      List[float]   # [x1, y1, x2, y2] absolute pixels

    @property
    def cx(self): return (self.box[0] + self.box[2]) / 2
    @property
    def cy(self): return (self.box[1] + self.box[3]) / 2
    @property
    def area(self):
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


@dataclass
class WorkerStatus:
    """Per-worker compliance state after PPE association."""
    worker_id:   int
    box:         List[float]   # [x1, y1, x2, y2]
    conf:        float
    has_hardhat: bool          = False
    has_vest:    bool          = False
    violations:  List[str]     = field(default_factory=list)
    uncertain:   bool          = False
    ppe_items:   List[Detection] = field(default_factory=list)

    @property
    def compliant(self) -> bool:
        return len(self.violations) == 0

    @property
    def status_label(self) -> str:
        if self.uncertain and not self.compliant:
            return "UNCERTAIN"
        return "SAFE" if self.compliant else "VIOLATION"


@dataclass
class InferenceResult:
    """Complete result for one frame."""
    verdict:           str              # SAFE / UNSAFE / *-UNCERTAIN / NO_WORKERS
    workers:           List[WorkerStatus]
    inference_ms:      float
    workers_total:     int
    workers_compliant: int
    workers_violated:  int
    workers_uncertain: int


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a: List[float], b: List[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def _centroid_inside(ppe: List[float], worker: List[float]) -> bool:
    """Check whether the centroid of ppe box falls inside worker box."""
    cx = (ppe[0] + ppe[2]) / 2
    cy = (ppe[1] + ppe[3]) / 2
    return worker[0] <= cx <= worker[2] and worker[1] <= cy <= worker[3]


# ─────────────────────────────────────────────────────────────────────────────
# Association
# ─────────────────────────────────────────────────────────────────────────────

def associate_ppe_to_workers(
    detections: List[Detection],
    uncertainty_th: float = UNCERTAINTY_TH,
) -> List[WorkerStatus]:
    """
    Match PPE detections to workers using IoU overlap + centroid fallback.
    Returns one WorkerStatus per detected person.
    """
    persons  = [d for d in detections if d.cls_name == "person"]
    ppe_dets = [d for d in detections if d.cls_name != "person"]

    statuses = [
        WorkerStatus(
            worker_id = i + 1,
            box       = w.box,
            conf      = w.conf,
            uncertain = w.conf < uncertainty_th,
        )
        for i, w in enumerate(persons)
    ]

    for ppe in ppe_dets:
        best_idx   = -1
        best_score = -1.0

        for i, person in enumerate(persons):
            score = _iou(ppe.box, person.box)
            # IoU too low — try centroid fallback
            if score < 0.05 and _centroid_inside(ppe.box, person.box):
                score = 0.01
            if score > best_score:
                best_score = score
                best_idx   = i

        if best_idx >= 0:
            ws = statuses[best_idx]
            ws.ppe_items.append(ppe)
            if ppe.cls_name == "hard_hat":    ws.has_hardhat = True
            if ppe.cls_name == "safety_vest": ws.has_vest    = True
            if ppe.conf < uncertainty_th:     ws.uncertain   = True

    return statuses


# ─────────────────────────────────────────────────────────────────────────────
# Compliance rule engine
# ─────────────────────────────────────────────────────────────────────────────

def apply_compliance_rules(
    statuses: List[WorkerStatus],
) -> List[WorkerStatus]:
    """
    Apply per-worker PPE compliance rules.
    Mutates each WorkerStatus in place, returns the same list.

    Rules:
      Rule 1 — Hard hat required at all times.
      Rule 2 — Safety vest required in active work zone.
      Rule 3 — Uncertain detections are flagged, not silently classified.
    """
    for ws in statuses:
        sfx = " (uncertain — manual review)" if ws.uncertain else ""
        if not ws.has_hardhat:
            ws.violations.append(f"Missing hard hat{sfx}")
        if not ws.has_vest:
            ws.violations.append(f"Missing safety vest{sfx}")
    return statuses


# ─────────────────────────────────────────────────────────────────────────────
# Verdict
# ─────────────────────────────────────────────────────────────────────────────

def compute_verdict(statuses: List[WorkerStatus]) -> str:
    """
    Compute scene-level verdict from per-worker statuses.

    Returns:
      NO_WORKERS          — no persons detected
      SAFE                — all workers compliant
      SAFE-UNCERTAIN      — all compliant but some predictions uncertain
      UNSAFE              — one or more violations
      UNSAFE-UNCERTAIN    — violations present and some predictions uncertain
    """
    if not statuses:
        return "NO_WORKERS"
    violated  = any(not ws.compliant for ws in statuses)
    uncertain = any(ws.uncertain     for ws in statuses)
    verdict   = "UNSAFE" if violated else "SAFE"
    if uncertain:
        verdict += "-UNCERTAIN"
    return verdict


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model,
    frame:          np.ndarray,
    conf_th:        float = CONF_THRESH,
    iou_th:         float = IOU_THRESH,
    uncertainty_th: float = UNCERTAINTY_TH,
) -> InferenceResult:
    """
    Run the full inference pipeline on a single BGR frame.

    Steps:
      1. YOLO detection
      2. Worker-PPE association
      3. Compliance rule engine
      4. Verdict computation

    Args:
        model:          Loaded Ultralytics YOLO model
        frame:          BGR numpy array (OpenCV image)
        conf_th:        Confidence threshold for YOLO detections
        iou_th:         NMS IoU threshold for YOLO
        uncertainty_th: Detections below this confidence are flagged uncertain

    Returns:
        InferenceResult dataclass
    """
    t0   = time.perf_counter()
    pred = model.predict(frame, conf=conf_th, iou=iou_th, verbose=False)[0]
    ms   = (time.perf_counter() - t0) * 1000

    # Parse raw detections
    detections: List[Detection] = []
    for box in pred.boxes:
        cls_name = pred.names[int(box.cls)]
        if cls_name not in CLASS_NAMES:
            continue
        detections.append(Detection(
            cls_name = cls_name,
            conf     = float(box.conf),
            box      = box.xyxy[0].tolist(),
        ))

    # Association + compliance
    statuses = associate_ppe_to_workers(detections, uncertainty_th)
    statuses = apply_compliance_rules(statuses)
    verdict  = compute_verdict(statuses)

    return InferenceResult(
        verdict           = verdict,
        workers           = statuses,
        inference_ms      = round(ms, 1),
        workers_total     = len(statuses),
        workers_compliant = sum(1 for ws in statuses if ws.compliant),
        workers_violated  = sum(1 for ws in statuses if not ws.compliant),
        workers_uncertain = sum(1 for ws in statuses if ws.uncertain),
    )