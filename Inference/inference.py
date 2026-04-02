#!/usr/bin/env python3
"""
Construction Safety Monitor — Local Inference

Usage examples:
    python inference.py --weights best.pt --source image.jpg --show
    python inference.py --weights best.pt --source video.mp4 --save
    python inference.py --weights best.pt --source 0 --show
    python inference.py --weights best.pt --source images_folder --save-dir outputs
"""

import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


CLASS_NAMES = ["person", "hard_hat", "no_hard_hat", "safety_vest", "no_safety_vest"]

COLORS_BGR = {
    "person": (200, 200, 200),
    "hard_hat": (50, 200, 80),
    "safety_vest": (50, 200, 80),
    "no_hard_hat": (40, 60, 220),
    "no_safety_vest": (40, 60, 220),
}

STATUS_BGR = {
    "SAFE": (50, 200, 80),
    "VIOLATION": (40, 60, 220),
    "UNCERTAIN": (30, 180, 255),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def iou(box_a, box_b):
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    denom = area_a + area_b - inter
    return inter / denom if denom > 1e-6 else 0.0


def centroid_in(ppe_box, worker_box):
    cx = (ppe_box[0] + ppe_box[2]) / 2
    cy = (ppe_box[1] + ppe_box[3]) / 2
    return worker_box[0] <= cx <= worker_box[2] and worker_box[1] <= cy <= worker_box[3]


def draw_text_bg(image, text, x, y, bg_color, font_scale=0.45, thickness=1):
    (tw, th), bl = cv2.getTextSize(text, FONT, font_scale, thickness)
    cv2.rectangle(image, (x, y - th - 6), (x + tw + 4, y + bl), bg_color, -1)
    cv2.putText(image, text, (x + 2, y - 2), FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def run_inference(model, frame, conf_th=0.35, iou_th=0.45, uncertainty_th=0.50):
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

    worker_statuses = []
    for idx, worker in enumerate(workers):
        worker_statuses.append(
            {
                "id": idx + 1,
                "box": worker["box"],
                "conf": worker["conf"],
                "has_hardhat": False,
                "has_vest": False,
                "violations": [],
                "uncertain": worker["conf"] < uncertainty_th,
                "ppe": [],
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
            ws = worker_statuses[best_idx]
            ws["ppe"].append(ppe)

            if ppe["cls"] == "hard_hat":
                ws["has_hardhat"] = True
            if ppe["cls"] == "safety_vest":
                ws["has_vest"] = True
            if ppe["conf"] < uncertainty_th:
                ws["uncertain"] = True

    for ws in worker_statuses:
        suffix = " (uncertain)" if ws["uncertain"] else ""
        if not ws["has_hardhat"]:
            ws["violations"].append("Missing hard hat" + suffix)
        if not ws["has_vest"]:
            ws["violations"].append("Missing safety vest" + suffix)

    violated = any(ws["violations"] for ws in worker_statuses)
    uncertain = any(ws["uncertain"] for ws in worker_statuses)

    verdict = "UNSAFE" if violated else "SAFE"
    if uncertain:
        verdict += "-UNCERTAIN"
    if not worker_statuses:
        verdict = "NO_WORKERS"

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    top_color = STATUS_BGR.get(
        "SAFE" if verdict.startswith("SAFE")
        else "UNCERTAIN" if "UNCERTAIN" in verdict
        else "VIOLATION"
    )

    cv2.rectangle(annotated, (0, 0), (w, 30), top_color, -1)
    header = (
        f"{verdict} | Workers: {len(worker_statuses)} | "
        f"Violations: {sum(1 for ws in worker_statuses if ws['violations'])}"
    )
    cv2.putText(annotated, header, (8, 21), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    for ws in worker_statuses:
        status_color = STATUS_BGR.get(
            "UNCERTAIN" if ws["uncertain"] and ws["violations"]
            else "SAFE" if not ws["violations"]
            else "VIOLATION"
        )

        box = ws["box"]
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), status_color, 2)

        if not ws["violations"]:
            label = "SAFE"
        elif ws["uncertain"]:
            label = "UNCERTAIN"
        else:
            label = "VIOLATION"

        draw_text_bg(annotated, f"W{ws['id']} {label}", x1, y1, status_color)

        for i, violation in enumerate(ws["violations"]):
            yy = y2 + 18 + i * 17
            draw_text_bg(annotated, f"! {violation}", x1, yy, (40, 60, 220), font_scale=0.38)

        for ppe in ws["ppe"]:
            ppe_color = COLORS_BGR.get(ppe["cls"], (180, 180, 180))
            pb = list(map(int, ppe["box"]))
            cv2.rectangle(annotated, (pb[0], pb[1]), (pb[2], pb[3]), ppe_color, 1)
            cv2.putText(
                annotated,
                f"{ppe['cls']} {ppe['conf']:.2f}",
                (pb[0], max(12, pb[1] - 3)),
                FONT,
                0.33,
                ppe_color,
                1,
                cv2.LINE_AA,
            )

    report = {
        "verdict": verdict,
        "workers": worker_statuses,
        "inference_ms": round(inference_ms, 1),
        "workers_total": len(worker_statuses),
        "workers_violated": sum(1 for ws in worker_statuses if ws["violations"]),
        "workers_compliant": sum(1 for ws in worker_statuses if not ws["violations"]),
        "workers_uncertain": sum(1 for ws in worker_statuses if ws["uncertain"]),
    }

    return report, annotated


def print_report(report):
    print("\n" + "=" * 60)
    print(f"VERDICT: {report['verdict']}")
    print(f"Workers detected   : {report['workers_total']}")
    print(f"Workers compliant  : {report['workers_compliant']}")
    print(f"Workers violated   : {report['workers_violated']}")
    print(f"Workers uncertain  : {report['workers_uncertain']}")
    print(f"Inference time (ms): {report['inference_ms']}")
    print("-" * 60)

    for ws in report["workers"]:
        if not ws["violations"]:
            status = "SAFE"
        elif ws["uncertain"]:
            status = "UNCERTAIN"
        else:
            status = "VIOLATION"

        print(f"Worker #{ws['id']}  | conf={ws['conf']:.2f} | {status}")
        print(f"  Hard hat : {'YES' if ws['has_hardhat'] else 'NO'}")
        print(f"  Vest     : {'YES' if ws['has_vest'] else 'NO'}")
        if ws["violations"]:
            for v in ws["violations"]:
                print(f"  - {v}")
    print("=" * 60 + "\n")


def save_json_report(report, path):
    clean = dict(report)
    clean["workers"] = []
    for worker in report["workers"]:
        w = dict(worker)
        w.pop("ppe", None)
        clean["workers"].append(w)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)


def process_image(model, image_path, args):
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    report, annotated = run_inference(
        model,
        frame,
        conf_th=args.conf,
        iou_th=args.iou,
        uncertainty_th=args.uncertainty,
    )

    print(f"[IMAGE] {image_path.name}")
    print_report(report)

    if args.save:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        out_img = args.save_dir / f"{image_path.stem}_annotated.jpg"
        out_json = args.save_dir / f"{image_path.stem}_report.json"
        cv2.imwrite(str(out_img), annotated)
        save_json_report(report, out_json)
        print(f"[SAVED] {out_img}")
        print(f"[SAVED] {out_json}")

    if args.show:
        cv2.imshow("Construction Safety Monitor", annotated)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("s"):
            args.save_dir.mkdir(parents=True, exist_ok=True)
            snap_path = args.save_dir / f"{image_path.stem}_snapshot.jpg"
            cv2.imwrite(str(snap_path), annotated)
            print(f"[SAVED] {snap_path}")
        cv2.destroyAllWindows()


def process_folder(model, folder_path, args):
    image_files = sorted(
        [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )

    if not image_files:
        print(f"[ERROR] No images found in folder: {folder_path}")
        return

    print(f"[INFO] Found {len(image_files)} image(s) in {folder_path}")

    args.save_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_files:
        process_image(model, image_path, args)


def process_video(model, video_path, args):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    out_video = None
    if args.save:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        out_video = args.save_dir / f"{video_path.stem}_annotated.mp4"
        writer = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    last_annotated = None
    frame_id = 0

    print(f"[INFO] Processing video: {video_path.name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % args.sample_every == 0:
            report, annotated = run_inference(
                model,
                frame,
                conf_th=args.conf,
                iou_th=args.iou,
                uncertainty_th=args.uncertainty,
            )
            last_annotated = annotated

            print(
                f"[FRAME {frame_id:05d}] {report['verdict']} | "
                f"workers={report['workers_total']} | "
                f"violations={report['workers_violated']} | "
                f"{report['inference_ms']:.1f} ms"
            )
        else:
            annotated = last_annotated if last_annotated is not None else frame

        if writer is not None:
            writer.write(annotated)

        if args.show:
            cv2.imshow("Construction Safety Monitor", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                args.save_dir.mkdir(parents=True, exist_ok=True)
                snap_path = args.save_dir / f"{video_path.stem}_frame_{frame_id}.jpg"
                cv2.imwrite(str(snap_path), annotated)
                print(f"[SAVED] {snap_path}")

        frame_id += 1
        if total_frames > 0 and frame_id % 20 == 0:
            print(f"[INFO] Progress: {frame_id}/{total_frames} frames")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if out_video:
        print(f"[SAVED] {out_video}")


def process_webcam(model, cam_index, args):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam: {cam_index}")
        return

    print("[INFO] Webcam started. Press 'q' to quit, 's' to save snapshot.")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read webcam frame.")
            break

        report, annotated = run_inference(
            model,
            frame,
            conf_th=args.conf,
            iou_th=args.iou,
            uncertainty_th=args.uncertainty,
        )

        if frame_id % 10 == 0:
            print(
                f"[WEBCAM {frame_id:05d}] {report['verdict']} | "
                f"workers={report['workers_total']} | "
                f"violations={report['workers_violated']} | "
                f"{report['inference_ms']:.1f} ms"
            )

        if args.show:
            cv2.imshow("Construction Safety Monitor", annotated)

        if args.save and args.record_webcam:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            if not hasattr(process_webcam, "writer"):
                h, w = annotated.shape[:2]
                out_path = args.save_dir / "webcam_recording.mp4"
                process_webcam.writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20.0,
                    (w, h),
                )
                print(f"[SAVED TO] {out_path}")
            process_webcam.writer.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            args.save_dir.mkdir(parents=True, exist_ok=True)
            snap_path = args.save_dir / f"webcam_snapshot_{frame_id}.jpg"
            cv2.imwrite(str(snap_path), annotated)
            print(f"[SAVED] {snap_path}")

        frame_id += 1

    cap.release()
    if hasattr(process_webcam, "writer"):
        process_webcam.writer.release()
        del process_webcam.writer
    cv2.destroyAllWindows()


def parse_source(source_str):
    if source_str.isdigit():
        return "webcam", int(source_str)

    path = Path(source_str)
    if path.is_dir():
        return "folder", path
    if path.is_file():
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            return "image", path
        if ext in VIDEO_EXTS:
            return "video", path

    return None, source_str


def build_parser():
    parser = argparse.ArgumentParser(description="Construction Safety Monitor — Local Inference")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Image, video, folder, or webcam index")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--uncertainty", type=float, default=0.50, help="Threshold below which detections are flagged uncertain")
    parser.add_argument("--show", action="store_true", help="Display output window")
    parser.add_argument("--save", action="store_true", help="Save annotated outputs")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs"), help="Directory to save outputs")
    parser.add_argument("--sample-every", type=int, default=1, help="Process every N frames for video")
    parser.add_argument("--record-webcam", action="store_true", help="Save webcam stream when --save is used")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERROR] Weights file not found: {weights_path}")
        return

    print(f"[INFO] Loading model from: {weights_path}")
    model = YOLO(str(weights_path))

    source_type, source_obj = parse_source(args.source)
    if source_type is None:
        print(f"[ERROR] Unsupported source: {args.source}")
        print("Use an image, video, folder path, or webcam index like 0")
        return

    if source_type == "image":
        process_image(model, source_obj, args)
    elif source_type == "folder":
        process_folder(model, source_obj, args)
    elif source_type == "video":
        process_video(model, source_obj, args)
    elif source_type == "webcam":
        process_webcam(model, source_obj, args)


if __name__ == "__main__":
    main()