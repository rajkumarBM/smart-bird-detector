from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('video-detection/', views.video_detection, name='video_detection'),
    path('detections/', views.detection_list, name='detection_list'),
    path('live/<str:job_id>/latest/', views.live_latest_frame, name='live_latest_frame'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
]

