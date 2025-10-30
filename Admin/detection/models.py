from django.db import models
from django.contrib.auth.models import User

class BirdDetection(models.Model):
    category = models.CharField(max_length=100)
    detection_time = models.DateTimeField(auto_now_add=True)
    confidence = models.FloatField(default=0.0)
    frame_image = models.ImageField(upload_to='detections/frames/', blank=True, null=True)
    frame_number = models.IntegerField(default=0)
    video_file = models.CharField(max_length=500, blank=True, null=True)
    video_upload_date = models.DateTimeField(null=True, blank=True, help_text="Date when video was uploaded/selected")
    bbox_x1 = models.FloatField(default=0.0)
    bbox_y1 = models.FloatField(default=0.0)
    bbox_x2 = models.FloatField(default=0.0)
    bbox_y2 = models.FloatField(default=0.0)
    
    class Meta:
        ordering = ['-detection_time']
        
    def __str__(self):
        return f"{self.category} - Frame {self.frame_number} - {self.detection_time}"
