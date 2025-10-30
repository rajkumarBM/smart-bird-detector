"""Utility to extract video metadata."""
import cv2
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

def get_video_info(video_path: str, upload_date: Optional[str] = None) -> Dict[str, any]:
    """
    Extract video metadata including date, duration, size, etc.
    
    Args:
        video_path: Path to video file
        upload_date: Custom upload/selection date (YYYY-MM-DD HH:MM:SS format)
                     If not provided, uses current time when video is uploaded
        
    Returns:
        Dictionary with video metadata
    """
    info = {
        'file_name': os.path.basename(video_path),
        'file_size': 0,
        'file_size_mb': 0,
        'duration': 0,
        'duration_formatted': '0:00',
        'fps': 0,
        'width': 0,
        'height': 0,
        'frame_count': 0,
        'file_date': None,
        'upload_date': None,
        'codec': None,
    }
    
    try:
        # Get file info
        if os.path.exists(video_path):
            stat = os.stat(video_path)
            info['file_size'] = stat.st_size
            info['file_size_mb'] = round(stat.st_size / (1024 * 1024), 2)
            
            # Use upload_date (when video was selected/uploaded) as the primary date
            if upload_date:
                info['upload_date'] = upload_date
                info['file_date'] = upload_date  # Also set file_date for backward compatibility
            else:
                # Use current time as upload date
                upload_datetime = datetime.now()
                info['upload_date'] = upload_datetime.strftime('%Y-%m-%d %H:%M:%S')
                info['file_date'] = info['upload_date']
        
        # Get video properties using OpenCV
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            info['fps'] = cap.get(cv2.CAP_PROP_FPS) or 0
            info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            info['frame_count'] = frame_count
            
            # Calculate duration
            if info['fps'] > 0:
                duration_seconds = frame_count / info['fps']
                info['duration'] = duration_seconds
                minutes = int(duration_seconds // 60)
                seconds = int(duration_seconds % 60)
                info['duration_formatted'] = f"{minutes}:{seconds:02d}"
            
            # Try to get codec
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            if fourcc != 0:
                codec_bytes = fourcc.to_bytes(4, 'little')
                info['codec'] = codec_bytes.decode('ascii', errors='ignore')
            
            cap.release()
    except Exception as e:
        print(f"Error getting video info: {e}")
    
    return info

