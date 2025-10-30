#!/usr/bin/env python3
"""Run RT-DETR inference on a video with TensorRT or PyTorch models.
Draws bounding boxes, saves annotated output, and logs detections to CSV.

This version supports:
- TensorRT engines (.engine files) for maximum performance
- PyTorch models (.pt files) with FP16/FP32 precision
- Automatic detection of model format
- Video processing with frame-by-frame annotation

Examples:
  # TensorRT engine (fastest)
  python video_inference.py --source video.mp4 --model rtdetr-l.engine --conf 0.5 --verbose

  # PyTorch model with FP32
  python video_inference.py --source video.mp4 --model rtdetr-l.pt --conf 0.5 --no-fp16 --verbose

  # PyTorch model with FP16
  python video_inference.py --source video.mp4 --model rtdetr-l.pt --conf 0.5 --verbose
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# Define bird-related class names for reference
BIRD_CLASSES = {'bird', 'kite', 'eagle', 'owl', 'hawk', 'seabird', 'waterfowl'}

try:
    from ultralytics import YOLO, RTDETR
    import torch
except Exception:
    print("Failed to import 'ultralytics' or 'torch'. Please install dependencies: pip install -r requirements.txt")
    raise

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="RT-DETR video inference with FP16/TensorRT support")
    p.add_argument("--source", "-s", required=True, help="Path to input video file or camera index (0,1,...)")
    p.add_argument("--output", "-o", default="output_video.mp4", help="Path to save annotated output video")
    p.add_argument("--model", "-m", default="rtdetr-l.engine", help="Model path: .pt (PyTorch) or .engine (TensorRT)")
    p.add_argument("--model-type", "-t", choices=("auto", "yolo", "rtdetr"), default="rtdetr",
                   help="Model type hint: 'rtdetr' (default for RT-DETR-L), 'yolo', or 'auto'")
    p.add_argument("--conf", type=float, default=0.7, help="Confidence threshold for detection (default: 0.7)")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (RT-DETR-L optimal: 640)")
    p.add_argument("--device", "-d", default="", help="Device to run on, e.g. cpu or 0/1 for CUDA. Leave empty for auto")
    p.add_argument("--show", action="store_true", help="Show annotated frames in a window while processing")
    p.add_argument("--max-frames", type=int, default=0, help="If >0, stops after this many frames (useful for quick tests)")
    p.add_argument("--csv", default="detections.csv", help="CSV file to write detections to")
    p.add_argument("--verbose", action="store_true", help="Print verbose startup/progress information")
    p.add_argument("--no-fp16", action="store_true", help="Disable FP16 inference for PyTorch models (use FP32 instead)")
    return p.parse_args()


def get_video_fps(source: str) -> float:
    """Return FPS for video path or default 30 for camera streams."""
    try:
        int(source)
        return 30.0
    except Exception:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return 30.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        return float(fps)


def _to_numpy(x):
    """Convert a tensor-like object to numpy safely (best-effort)."""
    try:
        return x.cpu().numpy()
    except Exception:
        try:
            return x.numpy()
        except Exception:
            try:
                return list(x)
            except Exception:
                return None


def main() -> None:
    args = parse_args()

    if args.verbose:
        print(f"Loading RT-DETR model {args.model}...")

    # Detect model format
    model_path = Path(args.model)
    is_tensorrt = model_path.suffix == '.engine'

    if is_tensorrt:
        if args.verbose:
            print(f"Detected TensorRT engine file: {args.model}")
        use_fp16 = True
        # Try to load TensorRT engine, fallback to PyTorch if it fails
        try:
            # Use RTDETR class for .engine files (same as Rter-singlle.py)
            model = RTDETR(args.model)
            if args.verbose:
                print("TensorRT engine loaded (precision is pre-compiled)")
        except Exception as e:
            error_msg = str(e)
            print(f"\n⚠️  Warning: Failed to load TensorRT engine: {error_msg}")
            print("Falling back to PyTorch model...")
            
            # Try to find corresponding .pt file
            pt_model_path = model_path.with_suffix('.pt')
            if pt_model_path.exists():
                print(f"Using PyTorch model: {pt_model_path}")
                model = RTDETR(str(pt_model_path))
                is_tensorrt = False
            else:
                # Try default rtdetr-l.pt
                default_pt = Path("rtdetr-l.pt")
                if default_pt.exists():
                    print(f"Using default PyTorch model: {default_pt}")
                    model = RTDETR(str(default_pt))
                    is_tensorrt = False
                else:
                    raise RuntimeError(
                        f"TensorRT engine failed to load and no PyTorch fallback found.\n"
                        f"Please ensure you have a .pt file (e.g., rtdetr-l.pt) in the current directory.\n"
                        f"Original error: {error_msg}"
                    )
    else:
        # Regular PyTorch model - use RTDETR for RT-DETR models
        if args.model_type == "rtdetr" or "rtdetr" in args.model.lower():
            model = RTDETR(args.model)
        else:
            model = YOLO(args.model)
        use_fp16 = not args.no_fp16 and torch.cuda.is_available()

        if args.verbose:
            if use_fp16:
                print("FP16 (half precision) enabled for inference")
            elif args.no_fp16:
                print("FP16 disabled by user, using FP32 precision")
            else:
                print("CUDA not available, using FP32 precision on CPU")

    # For RT-DETR-L specific optimizations
    model_type = args.model_type
    if model_type == "auto":
        model_type = "rtdetr"

    if args.verbose:
        print(f"Model type: {model_type}")
        print("is_tensorrt:", is_tensorrt)
        if is_tensorrt:
            print(f"Using TensorRT engine (precision: FP32)")
        else:
            print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")

    src = args.source
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fps = get_video_fps(src)

    writer: Optional[cv2.VideoWriter] = None
    processed_frames = 0
    t0 = time.time()

    # Prepare CSV writer
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "time_s", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])

    counts = Counter()
    bird_counts = Counter()
    confidence_stats = defaultdict(list)
    frame_idx = 0
    inference_times = []

    try:
        if args.verbose:
            print(f"Source: {src}, output: {out_path}, csv: {csv_path}, fps: {fps}")

        # Prepare prediction kwargs
        predict_kwargs = {
            'source': src,
            'stream': True,
            'imgsz': args.imgsz,
            'conf': args.conf,
            'iou': args.iou,
            'device': (args.device or None),
            'verbose': False  # Disable verbose to avoid class ID errors
        }

        # Add half precision flag if FP16 is enabled and NOT using TensorRT
        if use_fp16 and not is_tensorrt:
            predict_kwargs['half'] = True

        for result in model.predict(**predict_kwargs):
            # Track inference time
            if hasattr(result, 'speed') and result.speed is not None:
                inference_times.append(result.speed.get('inference', 0))

            # Plot with the same confidence threshold as detection
            annotated = result.plot(conf=args.conf)

            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, max(0.1, fps), (w, h))

            writer.write(annotated)

            if args.show:
                cv2.imshow("annotated", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Extract detections
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                xyxy = getattr(boxes, "xyxy", None)
                confs = getattr(boxes, "conf", None)
                cls_ids = getattr(boxes, "cls", None)

                xyxy_np = _to_numpy(xyxy) if xyxy is not None else None
                confs_np = _to_numpy(confs) if confs is not None else None
                cls_np = _to_numpy(cls_ids) if cls_ids is not None else None

                # Fallback to .tolist() when needed
                if xyxy_np is None and xyxy is not None:
                    try:
                        xyxy_np = xyxy.tolist()
                    except Exception:
                        xyxy_np = None

                if confs_np is None and confs is not None:
                    try:
                        confs_np = confs.tolist()
                    except Exception:
                        confs_np = None

                if cls_np is None and cls_ids is not None:
                    try:
                        cls_np = cls_ids.tolist()
                    except Exception:
                        cls_np = None

                if xyxy_np is not None:
                    n = len(xyxy_np)
                    for i in range(n):
                        box = xyxy_np[i]
                        conf = float(confs_np[i]) if confs_np is not None and len(confs_np) > i else None
                        cls_id = int(cls_np[i]) if cls_np is not None and len(cls_np) > i else None

                        # Get class name
                        try:
                            names = None
                            names = getattr(result, "names", None)
                            if not names:
                                names = getattr(model, "names", None) or getattr(getattr(model, "model", None), "names", None)
                            if not names:
                                names = {}
                            class_name = names.get(int(cls_id), str(cls_id)) if cls_id is not None else ""
                        except Exception:
                            class_name = str(cls_id)

                        x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                        time_s = (frame_idx / fps) if fps and fps > 0 else ""

                        # Write class name to CSV (no mapping - kite and bird are separate)
                        csv_writer.writerow([frame_idx, time_s, cls_id if cls_id is not None else "",
                                          class_name,
                                          conf if conf is not None else "", x1, y1, x2, y2])

                        # Track all detections with their actual class names
                        counts[class_name] += 1
                        confidence_stats[class_name].append(conf if conf is not None else 0.0)

            processed_frames += 1
            frame_idx += 1
            if args.max_frames and processed_frames >= args.max_frames:
                break
            if args.verbose and processed_frames % 100 == 0:
                print(f"Processed frames: {processed_frames}")

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during inference: {e}")
        raise
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        csv_file.close()

    dt = time.time() - t0
    actual_fps = (processed_frames/dt) if dt>0 else 0
    print(f"Processed {processed_frames} frames in {dt:.2f}s ({actual_fps:.2f} FPS). Saved to: {out_path}")

    # Print inference time statistics
    if inference_times:
        avg_inference_time = sum(inference_times) / len(inference_times)
        print(f"Average inference time: {avg_inference_time:.2f}ms per frame")

    print("\nDetections Summary:")
    total = sum(counts.values())
    print(f"Total detections: {total}")

    print("\nClass detections:")
    for cls_name, cnt in counts.most_common():
        if cnt > 0:  # Only show classes that were actually detected
            avg_conf = sum(confidence_stats[cls_name]) / len(confidence_stats[cls_name])
            print(f"  {cls_name}: {cnt} detections (avg conf: {avg_conf:.2f})")

    if is_tensorrt:
        print(f"\nInference backend: TensorRT (FP32)")
    else:
        print(f"\nInference precision: {'FP16 (half)' if use_fp16 else 'FP32 (float)'}")


if __name__ == "__main__":
    main()
