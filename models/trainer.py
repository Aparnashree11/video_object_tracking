from ultralytics import YOLO
import os
import yaml

def train_on_mot(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use YOLOv8n (nano) for faster training
    model = YOLO("yolov8n.pt")

    model.train(
        data=config_path,
        epochs=30,
        imgsz=416,
        batch=16,
        name="yolov8n-mot17",
        project="runs/train",
        workers=4,
        exist_ok=True,
        amp=True,      # Enable mixed precision
        device=0       # Use GPU if available
    )

    trained_model_path = os.path.join("runs/train/yolov8n-mot17", "weights", "best.pt")
    print(f"Training completed. Trained weights: {trained_model_path}")
