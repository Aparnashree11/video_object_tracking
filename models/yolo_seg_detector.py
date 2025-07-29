import torch
import numpy as np
from ultralytics import YOLO


class YOLOSegDetector:
    def __init__(self, model_weights="yolov8s.pt", conf_threshold=0.3):
        """
        Initialize YOLOv8 model for object detection or segmentation.
        :param model_weights: Path to YOLOv8 model weights (e.g., .pt file)
        :param conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_weights)
        self.conf_threshold = conf_threshold

    def detect_and_segment(self, frame):
        """
        Perform detection and segmentation on a single frame.
        :param frame: Input image (BGR format)
        :return: detections (for DeepSORT), masks (optional segmentation masks)
        """
        results = self.model.predict(frame, verbose=False)
        result = results[0]  # get first result

        detections = []
        masks = []

        if result.boxes is not None:
            for box in result.boxes:
                if box.conf[0] < self.conf_threshold:
                    continue

                # Convert box to xywh for DeepSORT
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                xywh = np.array([x1 + w / 2, y1 + h / 2, w, h])

                conf = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append((xywh, conf, cls))

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()

        return detections, masks