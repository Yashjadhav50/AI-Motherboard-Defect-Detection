from ultralytics import YOLO

# Load our final selected model
model = YOLO(
    "runs/detect/runs/improved_640/weights/best.pt"
)

# Evaluate on the TEST dataset
results = model.val(
    data="dataset/data.yaml",
    split="test",
    imgsz=640,
    plots=True
)

print("\n===== FINAL TEST RESULTS =====")
print(f"Precision : {results.box.mp:.4f}")
print(f"Recall    : {results.box.mr:.4f}")
print(f"mAP50     : {results.box.map50:.4f}")
print(f"mAP50-95  : {results.box.map:.4f}")