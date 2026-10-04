import cv2
import os
import random
import matplotlib.pyplot as plt

# Dataset paths
image_folder = "dataset/train/images"
label_folder = "dataset/train/labels"

# Class names from data.yaml
class_names = [
    "CPU_FAN_NO_Screws",
    "CPU_FAN_Screw_loose",
    "CPU_FAN_Screws",
    "CPU_fan",
    "CPU_fan_port",
    "CPU_fan_port_detached",
    "Incorrect_Screws",
    "Loose_Screws",
    "No_Screws",
    "Scratch",
    "Screws"
]

# Get all images
images = os.listdir(image_folder)

# Select a random image
image_name = random.choice(images)

image_path = os.path.join(image_folder, image_name)

# Corresponding label file
label_name = os.path.splitext(image_name)[0] + ".txt"
label_path = os.path.join(label_folder, label_name)

# Read image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get image dimensions
height, width, _ = image.shape

# Read annotations
with open(label_path, "r") as file:

    for line in file:

        values = line.strip().split()

        class_id = int(values[0])

        x_center = float(values[1])
        y_center = float(values[2])
        box_width = float(values[3])
        box_height = float(values[4])

        # Convert YOLO coordinates to pixel coordinates
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # Class name
        label = class_names[class_id]

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

# Display
plt.figure(figsize=(12, 8))
plt.imshow(image)
plt.axis("off")
plt.title("Motherboard Defect Annotations")
plt.show()