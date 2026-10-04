from ultralytics import YOLO
import cv2

# Load our trained model
MODEL_PATH = "runs/detect/runs/improved_640/weights/best.pt"

model = YOLO(MODEL_PATH)


def detect_defects(image_path, confidence=0.25):

    # Read image using OpenCV
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read the image.")

    # Run YOLO prediction
    results = model.predict(
        source=image,
        conf=confidence,
        imgsz=640,
        verbose=False
    )

    # Get first result
    result = results[0]

    # Draw bounding boxes
    annotated_image = result.plot()

    # Convert BGR → RGB
    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    return annotated_image, result