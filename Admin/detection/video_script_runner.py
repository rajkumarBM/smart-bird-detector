"""Run video_inference.py script via subprocess."""
import subprocess
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, List
from django.utils import timezone
import cv2

def run_video_inference_script(
    source_path: str,
    output_path: str = None,
    device: str = "cpu",
    model_path: str = "rtdetr-l.pt",
    conf_threshold: float = 0.7,
    csv_output_path: str = None,
    frames_output_dir: str = None,
    verbose: bool = True,
    save_only_detection_frames: bool = True
) -> Dict[str, Any]:
    """
    Run video_inference.py script via subprocess.
    
    IMPORTANT: When save_only_detection_frames=True:
    - Video output is skipped (written to /dev/null)
    - ONLY frames where objects are detected are saved as images
    - Detection CSV is generated with all detection details
    
    Args:
        source_path: Path to input video file
        output_path: Path to save annotated output video (ignored if save_only_detection_frames=True)
        device: Device to use ('cpu' or 'gpu'/'cuda')
        model_path: Path to model file (.pt or .engine)
        conf_threshold: Confidence threshold for detection
        csv_output_path: Path to CSV file for detections (auto-generated if None)
        frames_output_dir: Path to directory to save detection frame images
        verbose: Whether to show verbose output
        save_only_detection_frames: If True, skip video output and ONLY save frames where objects are detected
        
    Returns:
        Dictionary with detection results and statistics
    """
    
    if save_only_detection_frames and not frames_output_dir:
        raise ValueError("frames_output_dir is required when save_only_detection_frames=True")
    # Setup paths
    current_file = Path(__file__).resolve()
    admin_dir = current_file.parent.parent
    project_root = admin_dir.parent
    navy_birds_dir = project_root / 'navy_birds-main'
    
    # Try video_inference.py first (as per user's command), then video_inference_cpu.py
    script_path = navy_birds_dir / 'video_inference.py'
    if not script_path.exists():
        script_path = navy_birds_dir / 'video_inference_cpu.py'
    
    # Generate CSV path if not provided
    if csv_output_path is None:
        if output_path:
            csv_output_path = str(Path(output_path).with_suffix('.csv'))
        else:
            # Use source path with .csv extension
            csv_output_path = str(Path(source_path).with_suffix('.csv'))
    
    # Check if script exists
    if not script_path.exists():
        available_scripts = list(navy_birds_dir.glob('video*.py'))
        raise FileNotFoundError(
            f"Video inference script not found: {script_path}\n"
            f"Available video scripts: {[str(f) for f in available_scripts]}\n"
            f"Please ensure video_inference.py or video_inference_cpu.py exists in {navy_birds_dir}"
        )
    
    # Verify source video exists
    source_path_obj = Path(source_path)
    if not source_path_obj.exists():
        raise FileNotFoundError(f"Source video not found: {source_path}")
    
    # Verify model exists
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Prepare command
    venv_python = navy_birds_dir / 'ai-env' / 'bin' / 'python'
    
    # Use venv python if available, otherwise use system python
    if venv_python.exists():
        python_cmd = str(venv_python)
    else:
        python_cmd = 'python3'
    
    # Build command arguments - use absolute paths
    # If save_only_detection_frames is True, still provide a valid MP4 path to satisfy OpenCV VideoWriter
    if save_only_detection_frames:
        # Create temporary output video path inside navy_birds-main/output-video
        out_dir = navy_birds_dir / 'output-video'
        out_dir.mkdir(parents=True, exist_ok=True)
        temp_output = str(out_dir / f"tmp_{int(time.time())}.mp4")
        if verbose:
            print("Mode: Detection frames only (no video output will be used)", flush=True)
            print(f"Providing temporary video path to avoid OpenCV errors: {temp_output}", flush=True)
    else:
        temp_output = output_path if output_path else str((navy_birds_dir / 'output-video' / f"tmp_{int(time.time())}.mp4"))
    
    cmd = [
        python_cmd,
        str(script_path.resolve()),  # Use absolute path
        '--source', str(Path(source_path).resolve()),  # Absolute path for source
        '--output', temp_output,  # Use temp_output (may be /dev/null to skip video)
        '--model', str(Path(model_path).resolve()),  # Absolute path for model
        '--conf', str(conf_threshold),
        '--csv', str(Path(csv_output_path).resolve()),  # Absolute path for CSV
        '--imgsz', '640',
    ]
    
    # Add device argument
    if device and device.lower() in ['cpu', 'gpu', 'cuda']:
        if device.lower() == 'cpu':
            cmd.extend(['--device', 'cpu'])
        elif device.lower() in ['gpu', 'cuda']:
            cmd.extend(['--device', '0'])  # Use first GPU
    
    # Add verbose flag
    if verbose:
        cmd.append('--verbose')
    
    # Change to navy_birds directory to run script
    original_cwd = os.getcwd()
    
    try:
        os.chdir(str(navy_birds_dir))
        
        start_time = time.time()
        
        # Always print start message (not just if verbose)
        print(f"\n{'='*60}", flush=True)
        print(f"VIDEO INFERENCE SCRIPT RUNNER", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Command: {' '.join(cmd[:3])} ... {cmd[-1]}", flush=True)  # Show first 3 args and last
        print(f"Working directory: {navy_birds_dir}", flush=True)
        print(f"Script: {script_path.name}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        if verbose:
            print(f"Full command: {' '.join(cmd)}", flush=True)
            print(f"Video processing started. This may take several minutes depending on video length and device.", flush=True)
        
        # Run the script with proper output handling
        output_lines = []
        error_lines = []
        output_lock = threading.Lock()
        
        def read_output(pipe, is_stderr=False):
            """Read output from subprocess pipe."""
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        line_clean = line.rstrip()
                        with output_lock:
                            if is_stderr:
                                error_lines.append(line)
                            else:
                                output_lines.append(line)
                        
                        # Always show important progress messages (not just verbose)
                        if not is_stderr and line_clean:
                            # Show progress if it contains frame counts or completion messages
                            if any(keyword in line_clean.lower() for keyword in ['processed', 'frame', 'detection', 'saved', 'fps']):
                                print(f"  → {line_clean}", flush=True)
                            elif verbose:
                                print(f"  {line_clean}", flush=True)
                        
                        if is_stderr and verbose:
                            print(f"  ⚠ STDERR: {line_clean}", flush=True)
            except Exception as e:
                if verbose:
                    print(f"Error reading output: {e}", flush=True)
            finally:
                try:
                    pipe.close()
                except:
                    pass
        
        # Print command details (always, not just verbose)
        print(f"\n{'='*60}", flush=True)
        print(f"EXECUTING VIDEO DETECTION", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Python: {python_cmd}", flush=True)
        print(f"Script: {script_path}", flush=True)
        print(f"Script exists: {script_path.exists()}", flush=True)
        print(f"Video source: {source_path}", flush=True)
        print(f"Video exists: {Path(source_path).exists()}", flush=True)
        print(f"Model: {model_path}", flush=True)
        print(f"Model exists: {Path(model_path).exists()}", flush=True)
        print(f"Device: {device}", flush=True)
        print(f"Confidence: {conf_threshold}", flush=True)
        print(f"CSV output: {csv_output_path}", flush=True)
        if verbose:
            print(f"\nFull command: {' '.join(cmd)}", flush=True)
        print(f"{'='*60}\n", flush=True)
        print(f"Starting process...", flush=True)
        
        # Create environment with venv paths
        env = os.environ.copy()
        if venv_python.exists():
            # Add venv bin to PATH
            venv_bin = str(venv_python.parent)
            env['PATH'] = f"{venv_bin}:{env.get('PATH', '')}"
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(navy_birds_dir),
            env=env
        )
        
        # Immediately flush and show process started
        print(f"✓ Process started with PID: {process.pid}", flush=True)
        if verbose:
            print("Reading output in background...", flush=True)
        
        # Start threads to read stdout and stderr non-blocking
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, False))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, True))
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Give threads a moment to start and process to initialize
        time.sleep(1.0)
        
        # Check if process is still running (hasn't died immediately)
        if process.poll() is not None:
            # Process ended quickly - likely an error
            error_msg = "Process ended immediately. "
            with output_lock:
                if error_lines:
                    error_msg += "Errors: " + '\n'.join(error_lines[-10:])
                elif output_lines:
                    error_msg += "Output: " + '\n'.join(output_lines[-10:])
            raise RuntimeError(error_msg)
        
        print(f"✓ Process is running (PID {process.pid}). Detection has started!", flush=True)
        if verbose:
            print("Monitoring output and waiting for completion...", flush=True)
        
        # Wait for process to complete with progress updates
        print("Waiting for process to complete...", flush=True)
        
        # Wait for process with periodic status updates
        last_status_time = time.time()
        status_interval = 30.0  # Print status every 30 seconds
        max_wait_time = 3600 * 2  # Maximum 2 hours (adjust as needed)
        wait_start = time.time()
        
        # Prepare live CSV tailing and incremental frame saving (optional)
        live_cap = None
        frames_dir = None
        frames_saved_live = set()
        last_csv_rows_count = 0
        if save_only_detection_frames and frames_output_dir:
            try:
                frames_dir = Path(frames_output_dir)
                frames_dir.mkdir(parents=True, exist_ok=True)
                live_cap = cv2.VideoCapture(str(Path(source_path).resolve()))
                if not live_cap.isOpened():
                    print("⚠️  Live preview: failed to open video for incremental frame extraction", flush=True)
                    live_cap = None
            except Exception as _e:
                print(f"⚠️  Live preview init error: {_e}", flush=True)

        while True:
            return_code = process.poll()
            
            if return_code is not None:
                # Process finished - check if it crashed
                if return_code != 0:
                    # Get error output
                    with output_lock:
                        recent_errors = error_lines[-20:] if error_lines else []
                        recent_output = output_lines[-20:] if output_lines else []
                    error_msg = "Process exited with error.\n"
                    if recent_errors:
                        error_msg += "STDERR:\n" + '\n'.join(recent_errors)
                    if recent_output:
                        error_msg += "\nSTDOUT:\n" + '\n'.join(recent_output)
                    raise RuntimeError(f"Video inference failed (exit code {return_code}):\n{error_msg}")
                # Process finished successfully
                break
            
            # Check for timeout
            elapsed = time.time() - wait_start
            if elapsed > max_wait_time:
                print(f"\n⚠️  Process exceeded maximum time ({max_wait_time/60:.1f} minutes), terminating...", flush=True)
                process.terminate()
                time.sleep(5)
                if process.poll() is None:
                    process.kill()
                raise RuntimeError(f"Process timeout after {max_wait_time/60:.1f} minutes")
            
            # Print periodic status updates
            current_time = time.time()
            if current_time - last_status_time >= status_interval:
                elapsed_minutes = elapsed / 60
                # Get latest output lines for context
                with output_lock:
                    last_lines = output_lines[-3:] if output_lines else []
                    last_line = last_lines[-1].strip() if last_lines else "Processing..."
                    output_count = len(output_lines)
                
                # Also check if any errors occurred
                with output_lock:
                    error_count = len(error_lines)
                    if error_lines:
                        last_error = error_lines[-1].strip()[:60]
                        print(f"⏳ [{elapsed_minutes:.1f} min] Processing... ({output_count} output lines, {error_count} errors) Latest: {last_line[:60]}...", flush=True)
                        if "error" in last_error.lower() or "exception" in last_error.lower():
                            print(f"  ⚠️  Warning: {last_error}", flush=True)
                    else:
                        print(f"⏳ [{elapsed_minutes:.1f} min] Processing... ({output_count} lines) Latest: {last_line[:60]}...", flush=True)
                last_status_time = current_time
            
            # Incremental: try to read new CSV rows and save frames live
            try:
                if csv_output_path and live_cap is not None and os.path.exists(csv_output_path):
                    with open(csv_output_path, 'r') as _f:
                        _reader = csv.DictReader(_f)
                        _rows = list(_reader)
                    if last_csv_rows_count < len(_rows):
                        new_rows = _rows[last_csv_rows_count:]
                        last_csv_rows_count = len(_rows)
                        for row in new_rows:
                            try:
                                class_name = (row.get('class_name') or 'unknown').strip() or 'unknown'
                                frame_num = int(float(row.get('frame', 0)))
                                if frame_num in frames_saved_live:
                                    continue
                                # Save one image per frame (first time we see that frame)
                                live_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                                _ret, _frame = live_cap.read()
                                if not _ret:
                                    continue
                                safe_class_name = class_name.replace(' ', '_').replace('/', '_').replace('\\', '_').lower()
                                safe_class_name = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in safe_class_name)
                                conf_val = float(row.get('confidence', 0.0))
                                frame_filename = f"live_{frame_num:06d}_{safe_class_name}_{int(conf_val*100)}.jpg"
                                frame_path = frames_dir / frame_filename
                                if cv2.imwrite(str(frame_path), _frame):
                                    frames_saved_live.add(frame_num)
                            except Exception as _inner_e:
                                # Skip faulty row
                                pass
            except Exception as _e:
                # Non-fatal; continue
                pass

            # Sleep briefly to avoid busy-waiting
            time.sleep(2)
        
        # Process completed, wait for threads to finish reading
        stdout_thread.join(timeout=10)
        stderr_thread.join(timeout=10)
        
        elapsed_time = time.time() - start_time
        
        # Always print completion message (not just verbose)
        print(f"\n{'='*60}", flush=True)
        if return_code == 0:
            print(f"✓ Process completed successfully! (Exit code: {return_code})", flush=True)
        else:
            print(f"⚠️  Process completed with exit code: {return_code}", flush=True)
        print(f"Total processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)", flush=True)
        
        # Show summary of output
        with output_lock:
            if output_lines:
                print(f"Captured {len(output_lines)} output lines", flush=True)
            if error_lines:
                print(f"Captured {len(error_lines)} error/warning lines", flush=True)
                if verbose:
                    print("Last few errors/warnings:", flush=True)
                    for err in error_lines[-5:]:
                        print(f"  ⚠️  {err.rstrip()}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        # Parse CSV to get detection results - these are frames where objects were detected
        detections = []
        detection_counts = {}
        
        if not os.path.exists(csv_output_path):
            # Give more detailed error message
            error_details = f"CSV file not found: {csv_output_path}\n"
            error_details += f"Working directory: {os.getcwd()}\n"
            error_details += f"Script executed: {script_path}\n"
            with output_lock:
                if error_lines:
                    error_details += f"\nError output:\n{chr(10).join(error_lines[-10:])}\n"
                if output_lines:
                    error_details += f"\nLast output:\n{chr(10).join(output_lines[-10:])}"
            raise RuntimeError(f"CSV file not created. Detection script may have failed.\n{error_details}")
        
        print(f"\nReading detections from CSV: {csv_output_path}", flush=True)
        with open(csv_output_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_name = row.get('class_name', 'unknown').strip()
                if not class_name or class_name == '':
                    continue  # Skip empty detections
                    
                confidence = float(row.get('confidence', 0.0))
                frame_num = int(row.get('frame', 0))
                
                detections.append({
                    'frame': frame_num,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [
                        float(row.get('x1', 0)),
                        float(row.get('y1', 0)),
                        float(row.get('x2', 0)),
                        float(row.get('y2', 0))
                    ]
                })
                
                # Count detections
                if class_name not in detection_counts:
                    detection_counts[class_name] = 0
                detection_counts[class_name] += 1
        
        print(f"✓ Found {len(detections)} detections in {len(set(d['frame'] for d in detections))} unique frames", flush=True)
        
        # Get video info
        cap = cv2.VideoCapture(source_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        
        # Calculate processing stats
        total_detections = sum(detection_counts.values())
        
        # Extract frame images from detections (group by frame)
        frames_with_detections = {}
        for det in detections:
            frame_num = det['frame']
            if frame_num not in frames_with_detections:
                frames_with_detections[frame_num] = []
            frames_with_detections[frame_num].append(det)
        
        # Extract and save ONLY frames with detections
        frame_images_list = []
        if detections and frames_output_dir:
            frames_dir = Path(frames_output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'='*60}", flush=True)
            print(f"EXTRACTING DETECTION FRAMES", flush=True)
            print(f"Total detections: {total_detections}", flush=True)
            print(f"Unique frames with detections: {len(frames_with_detections)}", flush=True)
            print(f"Saving frames to: {frames_dir}", flush=True)
            print(f"{'='*60}\n", flush=True)
            
            cap = cv2.VideoCapture(str(Path(source_path).resolve()))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {source_path}")
            
            saved_count = 0
            
            # Get unique frame numbers (one image per frame with detections)
            unique_frames = sorted(set(frames_with_detections.keys()))
            
            for idx, frame_num in enumerate(unique_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    # Get all detections in this frame
                    frame_dets = frames_with_detections[frame_num]
                    
                    # Get all class names in this frame (there might be multiple detections)
                    class_names_in_frame = list(set(det['class_name'] for det in frame_dets))
                    
                    # Use the detection with highest confidence for primary label
                    best_det = max(frame_dets, key=lambda x: x['confidence'])
                    primary_class = best_det['class_name']
                    
                    # Create filename with label name: frame_000123_bird_85.jpg 
                    # Format: frame_{framenum}_{label_name}_{confidence}.jpg
                    safe_class_name = primary_class.replace(' ', '_').replace('/', '_').replace('\\', '_').lower()
                    # Remove any special characters that might cause issues in filenames
                    safe_class_name = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in safe_class_name)
                    frame_filename = f"frame_{frame_num:06d}_{safe_class_name}_{int(best_det['confidence']*100)}.jpg"
                    frame_path = frames_dir / frame_filename
                    
                    # Save the original frame (not annotated - just the detection frame)
                    success = cv2.imwrite(str(frame_path), frame)
                    if success:
                        saved_count += 1
                        # If multiple classes detected in same frame, combine labels
                        if len(class_names_in_frame) > 1:
                            combined_label = '+'.join(sorted(class_names_in_frame))
                        else:
                            combined_label = primary_class
                        
                        frame_images_list.append({
                            'frame_number': frame_num,
                            'class_name': combined_label,  # Use combined label if multiple detections
                            'confidence': best_det['confidence'],
                            'bbox': best_det['bbox'],
                            'time_s': f"{frame_num / fps:.2f}" if fps > 0 else "",
                            'image_path': str(frame_path),
                            'all_classes': class_names_in_frame  # Store all classes for reference
                        })
                        
                        if verbose:
                            label_str = combined_label if len(class_names_in_frame) > 1 else primary_class
                            print(f"  ✓ Saved frame {frame_num:06d} -> {frame_filename} (Label: {label_str}, Confidence: {best_det['confidence']:.2f})", flush=True)
                        elif saved_count % 10 == 0 or saved_count == len(unique_frames):
                            print(f"  ✓ Saved {saved_count}/{len(unique_frames)} detection frames...", flush=True)
                    else:
                        print(f"Warning: Failed to save frame {frame_num} to {frame_path}", flush=True)
                else:
                    print(f"Warning: Could not read frame {frame_num} from video", flush=True)
            
            cap.release()
            print(f"\n{'='*60}", flush=True)
            print(f"✓ SUCCESS: Saved {saved_count} detection frame images", flush=True)
            print(f"  All frames saved with label names in filename (e.g., frame_000123_bird_85.jpg)", flush=True)
            print(f"  Only frames with detected objects were saved (no full video output)", flush=True)
            print(f"  Location: {frames_dir}", flush=True)
            
            # Show summary of labels saved
            if frame_images_list:
                labels_saved = {}
                for img in frame_images_list:
                    label = img['class_name']
                    labels_saved[label] = labels_saved.get(label, 0) + 1
                print(f"  Labels detected and saved:", flush=True)
                for label, count in sorted(labels_saved.items()):
                    print(f"    - {label}: {count} frame(s)", flush=True)
            
            print(f"{'='*60}\n", flush=True)
        elif not detections:
            print("\n⚠️  No detections found in video. No frame images to save.", flush=True)
        elif not frames_output_dir:
            print("⚠️  Warning: frames_output_dir not provided. Detection frames will not be saved.", flush=True)
        
        # Calculate average confidences per class
        avg_confidences = {}
        for det in detections:
            class_name = det['class_name']
            conf = det['confidence']
            if class_name not in avg_confidences:
                avg_confidences[class_name] = []
            avg_confidences[class_name].append(conf)
        
        # Convert to averages
        for class_name, confs in avg_confidences.items():
            avg_confidences[class_name] = sum(confs) / len(confs) if confs else 0.0
        
        # Calculate total time
        total_time = time.time() - start_time
        
        results = {
            'output_video': output_path if not save_only_detection_frames else None,  # No video output if frames-only mode
            'csv_file': csv_output_path,
            'processed_frames': frame_count,
            'total_time': total_time,
            'actual_fps': fps,
            'device_used': device,
            'precision': 'FP32',  # Default, script handles this
            'detections': detection_counts,
            'total_detections': total_detections,
            'avg_confidences': avg_confidences,
            'frame_images': frame_images_list,
            'detection_list': detections,
            'save_only_frames': save_only_detection_frames,
        }
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Error running video inference script: {str(e)}")
    
    finally:
        os.chdir(original_cwd)

