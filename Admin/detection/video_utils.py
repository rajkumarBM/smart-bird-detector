"""Utility functions for video bird detection inference."""
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import os

# Add navy_birds-main to path to import inference code
# Calculate path: Admin/detection/video_utils.py -> Admin -> .. -> navy_birds-main
current_file = Path(__file__).resolve()
admin_dir = current_file.parent.parent  # Admin directory
project_root = admin_dir.parent  # Project root
navy_birds_dir = project_root / 'navy_birds-main'
if navy_birds_dir.exists():
    sys.path.insert(0, str(navy_birds_dir))

try:
    from ultralytics import YOLO, RTDETR
    import torch
except ImportError:
    raise ImportError("Failed to import 'ultralytics' or 'torch'. Please install dependencies.")

import cv2


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


def run_video_inference(
    source_path: str,
    output_path: str,
    device: str = "cpu",
    model_path: str = "rtdetr-l.pt",
    conf_threshold: float = 0.7,
    verbose: bool = False,
    save_frames: bool = True,
    frames_output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run video inference for bird detection.
    
    Args:
        source_path: Path to input video file
        output_path: Path to save annotated output video
        device: Device to use ('cpu' or 'cuda'/'gpu' or '0'/'1' for specific GPU)
        model_path: Path to model file (.pt or .engine)
        conf_threshold: Confidence threshold for detection
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with detection results and statistics
    """
    # Set device based on selection
    if device.lower() == "cpu":
        device_str = "cpu"
        use_fp16 = False
    elif device.lower() in ["gpu", "cuda"]:
        if torch.cuda.is_available():
            device_str = "0"  # Use first GPU (or can use "cuda:0")
            use_fp16 = True
        else:
            device_str = "cpu"
            use_fp16 = False
            if verbose:
                print("GPU not available, falling back to CPU")
    else:
        # Allow specific device IDs like "0", "1", etc.
        device_str = device if device else None
        if device_str and device_str != "cpu":
            try:
                # Check if it's a valid CUDA device
                device_int = int(device_str)
                if torch.cuda.is_available() and device_int < torch.cuda.device_count():
                    device_str = str(device_int)
                    use_fp16 = True
                else:
                    device_str = "cpu"
                    use_fp16 = False
            except ValueError:
                device_str = "cpu"
                use_fp16 = False
        else:
            device_str = "cpu" if device_str == "cpu" else None
            use_fp16 = torch.cuda.is_available() if device_str and device_str != "cpu" else False
    
    # Ensure output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_path_obj.with_suffix('.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Loading model from {model_path}...")
    
    # Detect model format
    model_path_obj = Path(model_path)
    is_tensorrt = model_path_obj.suffix == '.engine'
    
    # Find model file - check multiple locations
    if not model_path_obj.exists():
        # Try in navy_birds-main directory
        current_file = Path(__file__).resolve()
        admin_dir = current_file.parent.parent
        project_root = admin_dir.parent
        navy_birds_dir = project_root / 'navy_birds-main'
        alt_model_path = navy_birds_dir / model_path_obj.name
        if alt_model_path.exists():
            model_path = str(alt_model_path)
            model_path_obj = alt_model_path
    
    if is_tensorrt:
        if verbose:
            print(f"Detected TensorRT engine file: {model_path}")
        use_fp16 = True
        try:
            model = RTDETR(model_path)
            if verbose:
                print("TensorRT engine loaded")
        except Exception as e:
            if verbose:
                print(f"Failed to load TensorRT engine: {e}, trying PyTorch fallback")
            # Try .pt file
            pt_model_path = model_path_obj.with_suffix('.pt')
            current_file = Path(__file__).resolve()
            admin_dir = current_file.parent.parent
            project_root = admin_dir.parent
            navy_birds_dir = project_root / 'navy_birds-main'
            if not pt_model_path.exists() and (navy_birds_dir / pt_model_path.name).exists():
                pt_model_path = navy_birds_dir / pt_model_path.name
            if pt_model_path.exists():
                model = RTDETR(str(pt_model_path))
                is_tensorrt = False
            else:
                raise RuntimeError(f"Could not load model: {model_path}")
    else:
        # Regular PyTorch model
        if "rtdetr" in model_path.lower():
            model = RTDETR(model_path)
        else:
            model = YOLO(model_path)
        
        if device_str and device_str != "cpu" and torch.cuda.is_available():
            use_fp16 = True
        else:
            use_fp16 = False
    
    fps = get_video_fps(source_path)
    
    writer: Optional[cv2.VideoWriter] = None
    processed_frames = 0
    t0 = time.time()
    
    # Prepare CSV writer
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "time_s", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
    
    counts = Counter()
    confidence_stats = defaultdict(list)
    frame_idx = 0
    inference_times = []
    frame_images_data = []  # Store frame image data for detections
    
    # Setup frames output directory
    if save_frames and frames_output_dir:
        frames_dir = Path(frames_output_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_frames = False
    
    try:
        if verbose:
            print(f"Processing video: {source_path}")
        
        # Prepare prediction kwargs
        predict_kwargs = {
            'source': source_path,
            'stream': True,
            'imgsz': 640,
            'conf': conf_threshold,
            'iou': 0.7,
            'device': device_str,
            'verbose': False
        }
        
        # Add half precision flag if FP16 is enabled and NOT using TensorRT
        if use_fp16 and not is_tensorrt:
            predict_kwargs['half'] = True
        
        for result in model.predict(**predict_kwargs):
            # Track inference time
            if hasattr(result, 'speed') and result.speed is not None:
                inference_times.append(result.speed.get('inference', 0))
            
            # Get original frame from result object
            # Ultralytics returns annotated image via plot(), we need the original
            # The result object may have orig_img or we can extract from annotated image
            original_frame = None
            if save_frames:
                # Try multiple ways to get original frame
                if hasattr(result, 'orig_img') and result.orig_img is not None:
                    orig = result.orig_img
                    # If it's a numpy array
                    if hasattr(orig, 'shape'):
                        if len(orig.shape) == 3:
                            # Check color channels - ultralytics might return RGB
                            original_frame = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR) if orig.shape[2] == 3 else orig
                        else:
                            original_frame = orig
                    else:
                        original_frame = orig
                elif hasattr(result, 'imgs') and result.imgs:
                    # Sometimes stored in imgs list
                    orig = result.imgs[0] if isinstance(result.imgs, list) else result.imgs
                    if hasattr(orig, 'shape'):
                        original_frame = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR) if len(orig.shape) == 3 else orig
                    else:
                        original_frame = orig
                # Fallback: Use the annotated image (will have boxes but better than nothing)
                if original_frame is None:
                    annotated = result.plot(conf=conf_threshold)
                    original_frame = annotated.copy()
            
            # Plot with the same confidence threshold as detection
            annotated = result.plot(conf=conf_threshold)
            
            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, max(0.1, fps), (w, h))
            
            writer.write(annotated)
            
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
                        
                        csv_writer.writerow([
                            frame_idx, time_s, 
                            cls_id if cls_id is not None else "",
                            class_name,
                            conf if conf is not None else "", 
                            x1, y1, x2, y2
                        ])
                        
                        counts[class_name] += 1
                        confidence_stats[class_name].append(conf if conf is not None else 0.0)
                        
                        # Save frame image if detection found and original frame is available
                        if save_frames and original_frame is not None:
                            # Create cropped frame with padding around detection
                            h, w = original_frame.shape[:2]
                            pad = 20  # Padding around detection
                            x1_crop = max(0, int(x1) - pad)
                            y1_crop = max(0, int(y1) - pad)
                            x2_crop = min(w, int(x2) + pad)
                            y2_crop = min(h, int(y2) + pad)
                            
                            # Save full frame or cropped detection (using full frame for context)
                            frame_filename = f"frame_{frame_idx}_{class_name}_{int(conf*100)}_{len(frame_images_data)}.jpg"
                            frame_path = frames_dir / frame_filename
                            cv2.imwrite(str(frame_path), original_frame)
                            
                            frame_images_data.append({
                                'frame_number': frame_idx,
                                'class_name': class_name,
                                'confidence': conf,
                                'bbox': (x1, y1, x2, y2),
                                'image_path': str(frame_path),
                                'time_s': time_s
                            })
            
            processed_frames += 1
            frame_idx += 1
            if verbose and processed_frames % 100 == 0:
                print(f"Processed frames: {processed_frames}")
    
    except KeyboardInterrupt:
        if verbose:
            print("Interrupted by user")
        raise
    except Exception as e:
        if verbose:
            print(f"Error during inference: {e}")
        raise
    finally:
        if writer is not None:
            writer.release()
        csv_file.close()
    
    dt = time.time() - t0
    actual_fps = (processed_frames / dt) if dt > 0 else 0
    
    # Build results summary
    results = {
        'output_video': output_path,
        'csv_file': str(csv_path),
        'processed_frames': processed_frames,
        'total_time': dt,
        'actual_fps': actual_fps,
        'device_used': device_str,
        'precision': 'TensorRT' if is_tensorrt else ('FP16' if use_fp16 else 'FP32'),
        'detections': dict(counts),
        'total_detections': sum(counts.values()),
        'frame_images': frame_images_data,  # List of frame images saved
    }
    
    # Add average confidence per class
    avg_confidences = {}
    for cls_name, confs in confidence_stats.items():
        if confs:
            avg_confidences[cls_name] = sum(confs) / len(confs)
    results['avg_confidences'] = avg_confidences
    
    if inference_times:
        results['avg_inference_time_ms'] = sum(inference_times) / len(inference_times)
    
    return results

