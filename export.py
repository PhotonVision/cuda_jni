from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best.pt")

# Export the model to ONNX format
export_path = model.export(format="onnx")

print(f"Model exported to {export_path}")