from ultralytics import YOLO

# Load pretrained YOLO11 Nano
model = YOLO("yolo11n.pt")

# Improved training
results = model.train(
    data="dataset/data.yaml",
    epochs=20,
    imgsz=640,
    batch=8,
    workers=0,
    cache=True,
    project="runs",
    name="improved_640"
)