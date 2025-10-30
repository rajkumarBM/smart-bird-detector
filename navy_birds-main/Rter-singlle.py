from ultralytics import RTDETR
import cv2

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.engine")
# Run inference on the image
results = model.predict("Pasted image.png", save=True)

# Alternative: Get the annotated image and save it manually
for result in results:
    # Get the annotated image (with bounding boxes drawn)
    annotated_img = result.plot()

    # Save the annotated image
    cv2.imwrite("output_annotated.jpg", annotated_img)

    print(f"Image saved as: output_annotated.jpg")
    print(f"Detections: {len(result.boxes)} objects found")
