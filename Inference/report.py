"""
inference/report.py
===================
Structured report generation — JSON per frame, CSV log, summary.
"""

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

from .core import InferenceResult, WorkerStatus


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame report
# ─────────────────────────────────────────────────────────────────────────────

def worker_to_dict(ws: WorkerStatus) -> dict:
    """Serialise a WorkerStatus to a plain dict (JSON-safe)."""
    return {
        "id":          ws.worker_id,
        "status":      ws.status_label,
        "compliant":   ws.compliant,
        "has_hardhat": ws.has_hardhat,
        "has_vest":    ws.has_vest,
        "violations":  ws.violations,
        "confidence":  round(ws.conf, 3),
        "uncertain":   ws.uncertain,
        "box_pixels":  [int(v) for v in ws.box],
    }


def make_report(
    result:    InferenceResult,
    source:    str  = "",
    frame_id:  int  = 0,
    ts_sec:    float = 0.0,
) -> dict:
    """
    Build a JSON-serialisable report dict from an InferenceResult.

    Args:
        result:   Output of core.run_inference()
        source:   Image/video path or source identifier
        frame_id: Frame number (0 for single images)
        ts_sec:   Timestamp in seconds (for video)

    Returns:
        Plain dict ready for json.dumps()
    """
    return {
        "source":            source,
        "frame_id":          frame_id,
        "timestamp_sec":     round(ts_sec, 3),
        "verdict":           result.verdict,
        "workers_total":     result.workers_total,
        "workers_compliant": result.workers_compliant,
        "workers_violated":  result.workers_violated,
        "workers_uncertain": result.workers_uncertain,
        "inference_ms":      result.inference_ms,
        "workers":           [worker_to_dict(ws) for ws in result.workers],
    }


def save_report(report: dict, path: Path) -> None:
    """Write a single frame report to a JSON file."""
    path.write_text(json.dumps(report, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate summary
# ─────────────────────────────────────────────────────────────────────────────

def make_summary(reports: List[dict], source: str = "") -> dict:
    """
    Compute aggregate statistics across a list of frame reports.

    Args:
        reports: List of dicts produced by make_report()
        source:  Original source path / label

    Returns:
        Summary dict
    """
    if not reports:
        return {}

    n         = len(reports)
    unsafe    = sum(1 for r in reports if not r["verdict"].startswith("SAFE"))
    viol_cnt: dict = defaultdict(int)

    for r in reports:
        for w in r["workers"]:
            for v in w["violations"]:
                viol_cnt[v] += 1

    return {
        "generated_at":           datetime.now().isoformat(),
        "source":                 source,
        "frames_processed":       n,
        "unsafe_frames":          unsafe,
        "safe_frames":            n - unsafe,
        "unsafe_rate_pct":        round(unsafe / n * 100, 1),
        "total_workers_detected": sum(r["workers_total"]   for r in reports),
        "total_violations":       sum(r["workers_violated"] for r in reports),
        "total_uncertain":        sum(r["workers_uncertain"] for r in reports),
        "avg_inference_ms":       round(sum(r["inference_ms"] for r in reports) / n, 1),
        "violation_breakdown":    dict(viol_cnt),
    }


def save_summary(reports: List[dict], out_dir: Path, source: str = "") -> dict:
    """
    Write summary.json and violations_log.csv to out_dir.

    Args:
        reports: List of frame report dicts
        out_dir: Directory to write output files
        source:  Source label for the summary

    Returns:
        Summary dict
    """
    if not reports:
        return {}

    summary = make_summary(reports, source)
    out_dir.mkdir(parents=True, exist_ok=True)

    # summary.json
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # violations_log.csv
    csv_path = out_dir / "violations_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source", "frame_id", "timestamp_sec", "verdict",
            "worker_id", "status", "violations",
            "has_hardhat", "has_vest", "confidence", "uncertain",
        ])
        for r in reports:
            for w in r["workers"]:
                writer.writerow([
                    r["source"], r["frame_id"], r["timestamp_sec"], r["verdict"],
                    w["id"], w["status"],
                    "; ".join(w["violations"]),
                    w["has_hardhat"], w["has_vest"],
                    w["confidence"], w["uncertain"],
                ])

    # Print to console
    print(f"\n{'='*52}")
    print(f"  SUMMARY")
    print(f"{'='*52}")
    print(f"  Frames processed : {summary['frames_processed']}")
    print(f"  Unsafe frames    : {summary['unsafe_frames']}  ({summary['unsafe_rate_pct']}%)")
    print(f"  Workers detected : {summary['total_workers_detected']}")
    print(f"  Total violations : {summary['total_violations']}")
    print(f"  Avg inference    : {summary['avg_inference_ms']} ms/frame")
    print(f"  Output           : {out_dir}")
    print(f"{'='*52}\n")

    return summary