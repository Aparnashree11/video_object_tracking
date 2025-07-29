import argparse
import os
from ultralytics import YOLO
import cv2

def track_video(model_path, video_path, output_path, conf_thres=0.3, iou_thres=0.5):
    # Load trained model
    model = YOLO(model_path)

    # Run tracking
    print(f"Tracking objects in: {video_path}")
    results = model.track(
        source=video_path,
        conf=conf_thres,
        iou=iou_thres,
        imgsz=640,
        save=True,
        save_txt=False,
        save_conf=False,
        tracker="bytetrack.yaml",  # Default tracker config (can use custom one)
        project="outputs",
        name="custom_video_track",
        exist_ok=True,
        device="cpu"  # or "cuda:0" if using GPU
    )

    # Move result video to output_path
    result_dir = "outputs"
    result_files = [f for f in os.listdir(result_dir) if f.endswith(".mp4")]
    if result_files:
        output_file = os.path.join(result_dir, result_files[0])
        os.rename(output_file, output_path)
        print(f"[INFO] Output saved to {output_path}")
    else:
        print("[ERROR] Tracking failed — no output video generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runs/train/yolov8n-mot17/weights/best.pt", help="Path to trained YOLOv8 model")
    parser.add_argument("--video", type=str, default="data/pedestrian.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="outputs/tracked_output.mp4", help="Path to save tracked output video")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for tracking")
    args = parser.parse_args()

    track_video(args.model, args.video, args.output, args.conf, args.iou)