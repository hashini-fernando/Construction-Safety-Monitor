"""
inference/annotate.py
=====================
All OpenCV frame annotation functions.
"""

import cv2
import numpy as np
from typing import List, Optional

from .core import InferenceResult, WorkerStatus

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (BGR)
# ─────────────────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "person":          (200, 200, 200),
    "hard_hat":        ( 50, 200,  80),
    "safety_vest":     ( 50, 200,  80),
    "no_hard_hat":     ( 40,  60, 220),
    "no_safety_vest":  ( 40,  60, 220),
}

STATUS_COLORS = {
    "SAFE":      ( 50, 200,  80),
    "VIOLATION": ( 40,  60, 220),
    "UNCERTAIN": ( 30, 180, 255),
}

FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.45
FONT_THICKNESS = 1
BOX_THICKNESS  = 2


# ─────────────────────────────────────────────────────────────────────────────
# Low-level drawing primitives
# ─────────────────────────────────────────────────────────────────────────────

def _label_bg(
    img:   np.ndarray,
    text:  str,
    x:     int,
    y:     int,
    color: tuple,
    scale: float = FONT_SCALE,
) -> None:
    """Draw a filled rectangle behind text for readability."""
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, FONT_THICKNESS)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + bl), color, -1)
    cv2.putText(
        img, text, (x + 2, y),
        FONT, scale, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA
    )


def _get_status_color(worker: WorkerStatus) -> tuple:
    """Return the BGR colour for a worker based on their compliance status."""
    return STATUS_COLORS.get(worker.status_label, (200, 200, 200))


# ─────────────────────────────────────────────────────────────────────────────
# Per-element drawing functions
# ─────────────────────────────────────────────────────────────────────────────

def draw_verdict_banner(img: np.ndarray, result: InferenceResult) -> None:
    """Draw a colour-coded verdict banner at the top of the frame."""
    h, w = img.shape[:2]
    verdict = result.verdict
    color   = STATUS_COLORS.get(
        "SAFE"      if verdict.startswith("SAFE") else
        "UNCERTAIN" if "UNCERTAIN" in verdict     else "VIOLATION"
    )
    cv2.rectangle(img, (0, 0), (w, 30), color, -1)
    text = (
        f"{verdict}  |  Workers: {result.workers_total}  |  "
        f"Violations: {result.workers_violated}"
    )
    cv2.putText(img, text, (8, 21), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_worker_box(img: np.ndarray, worker: WorkerStatus) -> None:
    """Draw bounding box + status label for a single worker."""
    color = _get_status_color(worker)
    b     = worker.box
    cv2.rectangle(
        img,
        (int(b[0]), int(b[1])),
        (int(b[2]), int(b[3])),
        color, BOX_THICKNESS
    )
    header = f"W{worker.worker_id} {worker.status_label} {worker.conf:.2f}"
    _label_bg(img, header, int(b[0]), int(b[1]) - 4, color)


def draw_violations(img: np.ndarray, worker: WorkerStatus) -> None:
    """Draw violation text lines below the worker bounding box."""
    b = worker.box
    for vi, viol in enumerate(worker.violations):
        _label_bg(
            img,
            f"! {viol}",
            int(b[0]),
            int(b[3]) + 18 + vi * 17,
            (40, 60, 220),
            scale=0.38,
        )


def draw_ppe_boxes(img: np.ndarray, worker: WorkerStatus) -> None:
    """Draw smaller PPE detection boxes for each item associated with a worker."""
    for ppe in worker.ppe_items:
        color = CLASS_COLORS.get(ppe.cls_name, (180, 180, 180))
        pb    = ppe.box
        cv2.rectangle(
            img,
            (int(pb[0]), int(pb[1])),
            (int(pb[2]), int(pb[3])),
            color, 1
        )
        cv2.putText(
            img,
            f"{ppe.cls_name} {ppe.conf:.2f}",
            (int(pb[0]), int(pb[1]) - 3),
            FONT, 0.33, color, 1, cv2.LINE_AA
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main annotation function
# ─────────────────────────────────────────────────────────────────────────────

def annotate_frame(
    frame:         np.ndarray,
    result:        InferenceResult,
    show_ppe_boxes: bool = True,
) -> np.ndarray:
    """
    Draw full annotation on a copy of frame.

    Args:
        frame:          BGR numpy array (original, not modified)
        result:         InferenceResult from core.run_inference()
        show_ppe_boxes: Whether to draw individual PPE detection boxes

    Returns:
        Annotated BGR numpy array
    """
    out = frame.copy()

    draw_verdict_banner(out, result)

    for worker in result.workers:
        draw_worker_box(out, worker)
        draw_violations(out, worker)
        if show_ppe_boxes:
            draw_ppe_boxes(out, worker)

    return out