from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
import random
from detection.models import BirdDetection

class Command(BaseCommand):
    help = 'Create sample bird detection data'

    def handle(self, *args, **options):
        categories = ['Eagle', 'Sparrow', 'Robin', 'Hawk', 'Owl', 'Crow', 'Pigeon', 'Dove']
        
        # Create detections for last 30 days
        for day in range(30):
            date = timezone.now() - timedelta(days=day)
            # Random number of detections per day (0-10)
            num_detections = random.randint(0, 10)
            
            for _ in range(num_detections):
                # Random time within the day
                hour = random.randint(0, 23)
                minute = random.randint(0, 59)
                detection_time = date.replace(hour=hour, minute=minute, second=random.randint(0, 59))
                
                BirdDetection.objects.create(
                    category=random.choice(categories),
                    detection_time=detection_time,
                    confidence=round(random.uniform(0.5, 1.0), 2)
                )
        
        self.stdout.write(self.style.SUCCESS(f'Successfully created sample bird detection data'))

