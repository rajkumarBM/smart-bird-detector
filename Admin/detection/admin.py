from django.contrib import admin
from .models import BirdDetection

@admin.register(BirdDetection)
class BirdDetectionAdmin(admin.ModelAdmin):
    list_display = ['category', 'detection_time', 'confidence']
    list_filter = ['category', 'detection_time']
    search_fields = ['category']
