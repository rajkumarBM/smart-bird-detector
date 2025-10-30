from django.shortcuts import render, redirect
from django.http import HttpResponse, Http404
from django.http import HttpResponse, Http404
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from django.conf import settings
from datetime import timedelta
from django.db.models import Count
import json
from pathlib import Path
from .models import BirdDetection
from .video_utils import run_video_inference
from .video_info import get_video_info
from .video_script_runner import run_video_inference_script
import subprocess
import csv as csv_module
import sys

def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password')
    
    return render(request, 'detection/login.html')

@login_required
def dashboard(request):
    # Date and count chart data
    today = timezone.now().date()
    last_30_days = []
    counts_by_date = []
    
    for i in range(30):
        date = today - timedelta(days=i)
        count = BirdDetection.objects.filter(
            detection_time__date=date
        ).count()
        last_30_days.append(date.strftime('%Y-%m-%d'))
        counts_by_date.append(count)
    
    last_30_days.reverse()
    counts_by_date.reverse()
    
    # Time and category chart data
    last_24_hours = BirdDetection.objects.filter(
        detection_time__gte=timezone.now() - timedelta(hours=24)
    ).values('category').annotate(count=Count('id')).order_by('-count')
    
    categories = [item['category'] for item in last_24_hours]
    category_counts = [item['count'] for item in last_24_hours]
    
    # Get time slots for last 24 hours
    time_slots = []
    time_counts = []
    
    for i in range(24):
        hour_start = timezone.now() - timedelta(hours=24-i)
        hour_end = hour_start + timedelta(hours=1)
        count = BirdDetection.objects.filter(
            detection_time__gte=hour_start,
            detection_time__lt=hour_end
        ).count()
        time_slots.append(hour_start.strftime('%H:00'))
        time_counts.append(count)
    
    # Total counts
    total_detections = BirdDetection.objects.count()
    today_count = BirdDetection.objects.filter(
        detection_time__date=today
    ).count()
    
    context = {
        'date_labels': json.dumps(last_30_days),
        'date_counts': json.dumps(counts_by_date),
        'categories': json.dumps(categories),
        'category_counts': json.dumps(category_counts),
        'time_slots': json.dumps(time_slots),
        'time_counts': json.dumps(time_counts),
        'total_detections': total_detections,
        'today_count': today_count,
    }
    
    return render(request, 'detection/dashboard.html', context)


def get_available_videos():
    """Get list of available videos from sample-video directory."""
    current_file = Path(__file__).resolve()
    admin_dir = current_file.parent.parent
    project_root = admin_dir.parent
    navy_birds_dir = project_root / 'navy_birds-main'
    sample_video_dir = navy_birds_dir / 'sample-video'
    
    videos = []
    if sample_video_dir.exists():
        for video_file in sample_video_dir.glob('*.mp4'):
            videos.append({
                'name': video_file.name,
                'path': str(video_file.resolve()),
                'size': video_file.stat().st_size / (1024 * 1024)  # Size in MB
            })
    return videos

def video_detection(request):
    """View for uploading videos or selecting existing videos and running bird detection."""
    if request.method == 'POST':
        try:
            video_source_type = request.POST.get('video_source_type', 'upload')  # 'upload' or 'existing'
            device = request.POST.get('device', 'cpu')
            conf_threshold = float(request.POST.get('conf_threshold', 0.7))
            
            # Handle either upload or existing video selection
            if video_source_type == 'existing':
                # Use existing video file
                existing_video_path = request.POST.get('existing_video_path', '')
                if not existing_video_path or not Path(existing_video_path).exists():
                    messages.error(request, 'Selected video file not found')
                    context = {'available_videos': get_available_videos()}
                    return render(request, 'detection/video_detection.html', context)
                
                input_path = Path(existing_video_path)
                input_filename = input_path.name
                
            else:
                # Handle upload
                if 'video_file' not in request.FILES:
                    messages.error(request, 'No video file provided')
                    context = {'available_videos': get_available_videos()}
                    return render(request, 'detection/video_detection.html', context)
                
                video_file = request.FILES['video_file']
                
                # Validate file extension
                allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
                file_ext = Path(video_file.name).suffix.lower()
                if file_ext not in allowed_extensions:
                    messages.error(request, f'Invalid file type. Allowed: {", ".join(allowed_extensions)}')
                    context = {'available_videos': get_available_videos()}
                    return render(request, 'detection/video_detection.html', context)
                
                # Create media directory if it doesn't exist
                media_root = Path(settings.MEDIA_ROOT)
                media_root.mkdir(parents=True, exist_ok=True)
                
                # Save uploaded video
                input_filename = f"{timezone.now().strftime('%Y%m%d_%H%M%S')}_{video_file.name}"
                input_path = media_root / 'uploads' / input_filename
                input_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(input_path, 'wb+') as destination:
                    for chunk in video_file.chunks():
                        destination.write(chunk)
            
            # Get current timestamp when video is uploaded/selected
            upload_datetime = timezone.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Get video metadata with upload date based on selection/upload time
            video_info = get_video_info(str(input_path), upload_date=upload_datetime)
            
            # Create media directory if it doesn't exist (needed for outputs even with existing videos)
            media_root = Path(settings.MEDIA_ROOT)
            media_root.mkdir(parents=True, exist_ok=True)
            
            # Generate output paths
            output_filename = f"output_{input_filename}"
            output_path = media_root / 'outputs' / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Setup frames directory - where ONLY detection frames will be saved
            frames_dir = media_root / 'detections' / 'frames' / f"{timezone.now().strftime('%Y%m%d_%H%M%S')}"
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Detection frames will be saved to: {frames_dir}", flush=True)
            job_id = frames_dir.name
            
            # Find model file
            current_file = Path(__file__).resolve()
            admin_dir = current_file.parent.parent
            project_root = admin_dir.parent
            navy_birds_dir = project_root / 'navy_birds-main'
            model_path = navy_birds_dir / 'rtdetr-l.pt'
            if not model_path.exists():
                # Try .engine file
                model_path = navy_birds_dir / 'rtdetr-l.engine'
            if not model_path.exists():
                messages.error(request, 'Model file not found. Please ensure rtdetr-l.pt or rtdetr-l.engine exists in navy_birds-main directory.')
                return render(request, 'detection/video_detection.html')
            
            # Calculate estimated time based on video info
            video_duration_min = video_info.get('duration', 0) / 60
            estimated_time = ""
            if video_duration_min > 0:
                # Rough estimate: 1 minute of video takes ~2-5 minutes to process on CPU
                # Faster on GPU (1-2 minutes per minute of video)
                if device.lower() == 'cpu':
                    est_min = int(video_duration_min * 3)  # ~3x real-time on CPU
                else:
                    est_min = int(video_duration_min * 1.5)  # ~1.5x real-time on GPU
                estimated_time = f" (Estimated: ~{est_min} minutes)"
            
            messages.info(request, f'Starting detection with {device.upper()}...{estimated_time} Mode: Only frames with detected objects will be saved (no video output).')
            
            # Setup CSV path
            csv_path = media_root / 'outputs' / Path(output_path).with_suffix('.csv').name
            
            # Log start time for user feedback
            import time as time_module
            processing_start_time = time_module.time()
            
            # Run inference using video_inference_cpu.py script
            try:
                print(f"\n{'='*60}")
                print(f"DETECTION STARTED - {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Video: {input_filename}")
                print(f"Device: {device.upper()}")
                print(f"Confidence Threshold: {conf_threshold}")
                print(f"Video Duration: {video_info.get('duration_formatted', 'Unknown')}")
                print(f"Input Path: {input_path}")
                print(f"Output Path: {output_path}")
                print(f"{'='*60}\n")
                print("Calling video inference script...")
                sys.stdout.flush()  # Ensure output is visible immediately
                
                save_video = request.POST.get('save_video') == 'on'
                results = run_video_inference_script(
                    source_path=str(input_path),
                    output_path=str(output_path),  # Will be ignored if save_only_detection_frames=True
                    device=device,
                    model_path=str(model_path),
                    conf_threshold=conf_threshold,
                    csv_output_path=str(csv_path),
                    frames_output_dir=str(frames_dir),
                    verbose=True,
                    save_only_detection_frames=not save_video  # If not saving video, only frames
                )
                
                # Save detections to database with frame images
                from django.core.files import File
                from django.utils.dateparse import parse_datetime
                import os
                
                # Helper function to get video upload datetime
                def get_video_upload_datetime():
                    if video_info.get('upload_date'):
                        try:
                            dt = parse_datetime(video_info['upload_date'])
                            return dt if dt else timezone.now()
                        except:
                            return timezone.now()
                    return timezone.now()
                
                video_upload_datetime = get_video_upload_datetime()
                
                frame_images_list = results.get('frame_images', [])
                detection_list = results.get('detection_list', [])  # All detections from CSV
                
                print(f"\n{'='*60}", flush=True)
                print(f"SAVING DETECTIONS TO DATABASE", flush=True)
                print(f"Frame images to save: {len(frame_images_list)}", flush=True)
                print(f"Total detections from CSV: {len(detection_list)}", flush=True)
                print(f"Video filename: {input_filename}", flush=True)
                print(f"{'='*60}\n", flush=True)
                
                # ALWAYS ensure we save detections from CSV, even if images aren't available
                if not detection_list:
                    print("  ⚠️  WARNING: No detection_list from CSV! This should not happen.", flush=True)
                    print(f"  Results keys: {list(results.keys())}", flush=True)
                
                saved_to_db = 0
                skipped_no_image = 0
                errors = 0
                
                # Save each detection with its frame image (if available)
                if frame_images_list:
                    for frame_data in frame_images_list:
                        image_path = frame_data.get('image_path')
                        if image_path and os.path.exists(image_path):
                            try:
                                bbox = frame_data.get('bbox', [0, 0, 0, 0])
                                
                                detection = BirdDetection(
                                    category=frame_data.get('class_name', 'unknown'),
                                    confidence=float(frame_data.get('confidence', 0.0)),
                                    frame_number=int(frame_data.get('frame_number', 0)),
                                    video_file=input_filename,
                                    video_upload_date=video_upload_datetime,
                                    bbox_x1=float(bbox[0]) if len(bbox) > 0 else 0.0,
                                    bbox_y1=float(bbox[1]) if len(bbox) > 1 else 0.0,
                                    bbox_x2=float(bbox[2]) if len(bbox) > 2 else 0.0,
                                    bbox_y2=float(bbox[3]) if len(bbox) > 3 else 0.0,
                                )
                                # Save frame image
                                frame_filename = os.path.basename(image_path)
                                with open(image_path, 'rb') as img_file:
                                    detection.frame_image.save(frame_filename, File(img_file), save=False)
                                detection.save()
                                saved_to_db += 1
                                if saved_to_db % 10 == 0:
                                    print(f"  ✓ Saved {saved_to_db} detections to database...", flush=True)
                            except Exception as e:
                                # Log error but continue with other detections
                                import traceback
                                errors += 1
                                print(f"  ⚠️  Error saving detection for frame {frame_data.get('frame_number', 0)}: {e}", flush=True)
                                traceback.print_exc()
                                continue
                        else:
                            skipped_no_image += 1
                
                # If we have CSV detections but didn't save enough (or no images), save them from CSV
                # This ensures CSV data is always saved to DB even if image extraction fails
                if detection_list and saved_to_db < len(detection_list):
                    remaining = len(detection_list) - saved_to_db
                    print(f"  Saving remaining {remaining} detections from CSV data...", flush=True)
                    
                    # Create a set of already saved frames to avoid duplicates
                    saved_frames = set()
                    if frame_images_list:
                        saved_frames = {d.get('frame_number', -1) for d in frame_images_list}
                    
                    for det in detection_list:
                        # Skip if we already saved this frame from images
                        if det.get('frame', -1) in saved_frames:
                            continue
                        
                        try:
                            BirdDetection.objects.create(
                                category=det.get('class_name', 'unknown'),
                                confidence=float(det.get('confidence', 0.0)),
                                frame_number=int(det.get('frame', 0)),
                                video_file=input_filename,
                                video_upload_date=video_upload_datetime,
                                bbox_x1=float(det.get('bbox', [0])[0]) if len(det.get('bbox', [])) > 0 else 0.0,
                                bbox_y1=float(det.get('bbox', [0])[1]) if len(det.get('bbox', [])) > 1 else 0.0,
                                bbox_x2=float(det.get('bbox', [0])[2]) if len(det.get('bbox', [])) > 2 else 0.0,
                                bbox_y2=float(det.get('bbox', [0])[3]) if len(det.get('bbox', [])) > 3 else 0.0,
                            )
                            saved_to_db += 1
                            if saved_to_db % 50 == 0:
                                print(f"  ✓ Saved {saved_to_db} detections from CSV...", flush=True)
                        except Exception as e:
                            errors += 1
                            print(f"  ⚠️  Error saving detection from CSV (frame {det.get('frame', 0)}): {e}", flush=True)
                            import traceback
                            traceback.print_exc()
                            continue
                
                # Fallback: Save summary detections (for backward compatibility)
                if saved_to_db == 0 and not detection_list:
                    print(f"  Saving summary detections from results...", flush=True)
                    for class_name, count in results.get('detections', {}).items():
                        # Limit entries to avoid overwhelming the database
                        for _ in range(min(count, 100)):
                            try:
                                BirdDetection.objects.create(
                                    category=class_name,
                                    confidence=results.get('avg_confidences', {}).get(class_name, 0.0),
                                    video_file=input_filename,
                                    video_upload_date=video_upload_datetime
                                )
                                saved_to_db += 1
                            except Exception as e:
                                errors += 1
                                print(f"  ⚠️  Error saving summary detection: {e}", flush=True)
                                continue
                
                print(f"\n{'='*60}", flush=True)
                print(f"✓ DATABASE SAVE COMPLETE", flush=True)
                print(f"  Successfully saved: {saved_to_db} detections", flush=True)
                if skipped_no_image > 0:
                    print(f"  Skipped (no image): {skipped_no_image}", flush=True)
                if errors > 0:
                    print(f"  Errors: {errors}", flush=True)
                print(f"{'='*60}\n", flush=True)
                
                # Prepare response with results
                # Only include output video URL if video was actually created
                output_url = None
                if results.get('output_video') and Path(results['output_video']).exists():
                    output_url = f"/{settings.MEDIA_URL}outputs/{output_filename}"
                
                csv_filename = Path(output_path).with_suffix('.csv').name
                csv_url = f"/{settings.MEDIA_URL}outputs/{csv_filename}"
                
                # Get stored detections for display
                stored_detections = BirdDetection.objects.filter(video_file=input_filename).order_by('-detection_time')[:50]
                
                context = {
                    'success': True,
                    'output_video_url': output_url,
                    'csv_url': csv_url,
                    'results': results,
                    'device_used': results.get('device_used', device),
                    'precision': results.get('precision', 'Unknown'),
                    'stored_detections': stored_detections,
                    'input_video_url': f"/{settings.MEDIA_URL}uploads/{input_filename}",
                    'video_info': video_info,
                    'input_filename': input_filename,
                    'live_job_id': job_id,
                }
                
                processing_time = time_module.time() - processing_start_time
                processing_min = int(processing_time // 60)
                processing_sec = int(processing_time % 60)
                
                messages.success(request, f'Detection completed successfully! Processed {results["processed_frames"]} frames in {processing_min}m {processing_sec}s.')
                
            except FileNotFoundError as e:
                error_msg = f"Required file not found: {str(e)}"
                print(f"ERROR: {error_msg}", flush=True)
                import traceback
                traceback.print_exc()
                messages.error(request, error_msg)
                context = {'success': False, 'error': error_msg}
                return render(request, 'detection/video_detection.html', context)
            except RuntimeError as e:
                error_msg = f"Detection failed: {str(e)}"
                print(f"ERROR: {error_msg}", flush=True)
                import traceback
                traceback.print_exc()
                messages.error(request, error_msg)
                context = {'success': False, 'error': error_msg}
                return render(request, 'detection/video_detection.html', context)
            except Exception as e:
                error_msg = f'Error during detection: {str(e)}'
                print(f"ERROR: {error_msg}", flush=True)
                import traceback
                traceback.print_exc()
                messages.error(request, error_msg)
                context = {'success': False, 'error': str(e)}
            
            return render(request, 'detection/video_detection.html', context)
            
        except Exception as e:
            error_msg = f'Error: {str(e)}'
            print(f"ERROR: {error_msg}", flush=True)
            import traceback
            traceback.print_exc()
            messages.error(request, error_msg)
            return render(request, 'detection/video_detection.html', {'success': False, 'error': str(e), 'available_videos': get_available_videos()})
    
    # GET request - show form with available videos
    context = {
        'available_videos': get_available_videos()
    }
    return render(request, 'detection/video_detection.html', context)


@login_required
def detection_list(request):
    """View for listing all detections with filtering options."""
    from django.db.models import Q
    from datetime import datetime, timedelta
    
    # Get filter parameters
    category_filter = request.GET.get('category', '')
    video_filter = request.GET.get('video', '')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    min_confidence = request.GET.get('min_confidence', '')
    search_query = request.GET.get('search', '')
    
    # Start with all detections
    detections = BirdDetection.objects.all()
    
    # Apply filters
    if category_filter:
        detections = detections.filter(category__icontains=category_filter)
    
    if video_filter:
        detections = detections.filter(video_file__icontains=video_filter)
    
    if date_from:
        try:
            date_from_obj = datetime.strptime(date_from, '%Y-%m-%d')
            detections = detections.filter(detection_time__gte=date_from_obj)
        except ValueError:
            pass
    
    if date_to:
        try:
            date_to_obj = datetime.strptime(date_to, '%Y-%m-%d') + timedelta(days=1)
            detections = detections.filter(detection_time__lt=date_to_obj)
        except ValueError:
            pass
    
    if min_confidence:
        try:
            detections = detections.filter(confidence__gte=float(min_confidence))
        except ValueError:
            pass
    
    if search_query:
        detections = detections.filter(
            Q(category__icontains=search_query) |
            Q(video_file__icontains=search_query)
        )
    
    # Order by detection time (newest first)
    detections = detections.order_by('-detection_time')
    
    # Get unique categories and videos for filter dropdowns
    all_categories = BirdDetection.objects.values_list('category', flat=True).distinct()
    all_videos = BirdDetection.objects.values_list('video_file', flat=True).distinct().exclude(video_file__isnull=True).exclude(video_file='')
    
    # Pagination
    from django.core.paginator import Paginator
    paginator = Paginator(detections, 24)  # 24 items per page
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
    
    # Statistics (get total before filtering for display)
    total_all = BirdDetection.objects.count()
    filtered_count = page_obj.paginator.count
    
    context = {
        'detections': page_obj,
        'total_count': total_all,
        'filtered_count': filtered_count,
        'all_categories': sorted(set(all_categories)),
        'all_videos': sorted(set(all_videos)),
        'category_filter': category_filter,
        'video_filter': video_filter,
        'date_from': date_from,
        'date_to': date_to,
        'min_confidence': min_confidence,
        'search_query': search_query,
    }
    
    return render(request, 'detection/detection_list.html', context)


def live_latest_frame(request, job_id: str):
    """Serve the latest saved detection frame for a given job id as JPEG.
    job_id corresponds to a folder name under MEDIA_ROOT/detections/frames/<job_id>.
    """
    from pathlib import Path as _Path
    import glob as _glob
    base_dir = _Path(settings.MEDIA_ROOT) / 'detections' / 'frames' / job_id
    if not base_dir.exists():
        raise Http404("Job not found")
    files = sorted(_glob.glob(str(base_dir / '*.jpg')))
    if not files:
        return HttpResponse(status=204)
    latest = files[-1]
    try:
        with open(latest, 'rb') as f:
            data = f.read()
        resp = HttpResponse(data, content_type='image/jpeg')
        resp['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp['Pragma'] = 'no-cache'
        resp['Expires'] = '0'
        return resp
    except FileNotFoundError:
        raise Http404("Frame not found")
