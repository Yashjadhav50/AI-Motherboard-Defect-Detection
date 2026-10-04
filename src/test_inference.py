import cv2
from inference import detect_defects

# Change this to any motherboard image
IMAGE_PATH = "dataset/test/images/"

# Get first image from test folder
import os

images = os.listdir(IMAGE_PATH)

if len(images) == 0:
    raise ValueError("No images found.")

image_file = os.path.join(
    IMAGE_PATH,
    images[0]
)

# Run detection
annotated_image, result = detect_defects(
    image_file,
    confidence=0.25
)

# Convert RGB → BGR for OpenCV display
annotated_image = cv2.cvtColor(
    annotated_image,
    cv2.COLOR_RGB2BGR
)

# Save result
output_path = "runs/inference_result.jpg"

cv2.imwrite(
    output_path,
    annotated_image
)

print("Detection completed!")
print(f"Saved result to: {output_path}")

# Print detected objects
if result.boxes is not None:

    print("\nDetected objects:")

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = result.names[class_id]

        print(
            f"{class_name}: "
            f"{confidence:.2f}"
        )