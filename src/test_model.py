from ultralytics import YOLO

# Load improved model
model = YOLO(
    "runs/detect/runs/improved_640/weights/best.pt"
)

# Predict on validation images
results = model.predict(
    source="dataset/valid/images",
    conf=0.25,
    save=True,
    project="runs",
    name="improved_predictions"
)

print("Improved model prediction completed!")