# 🧠 Video Object Tracking with YOLOv8 + DeepSORT

This project implements a **real-time video object tracking system** using [YOLOv8](https://github.com/ultralytics/ultralytics) for detection and [DeepSORT](https://github.com/nwojke/deep_sort) for tracking. It supports training on MOT datasets, evaluating performance, and running on custom videos to track and identify pedestrians (or other objects).

---

## 📂 Project Structure
```bash
video_object_tracking/
├── data/
│ ├── MOT_YOLO/ # Preprocessed MOT dataset (images + labels)
│ ├── pedestrian.mp4 # Example input video
├── evaluation/
│ └── evaluate.py # Evaluation script
├── models/
│ ├── deep_sort_tracker.py
│ ├── trainer.py
│ └── yolo_seg_detector.py
├── outputs/ # Results: tracked videos, .txt outputs
├── runs/ # YOLO training + detection outputs
├── utils/
│ └── main.py # Utility functions
├── track_predict.py # Run tracking on any video
├── train.py # Train YOLOv8 on MOT dataset
├── requirements.txt
└── README.md
```

---

## 🚀 Features

- 🔍 **YOLOv8** for real-time object detection and segmentation
- 👣 **DeepSORT** for multi-object tracking
- 📊 Train on custom MOT17-style datasets
- 🎥 Track pedestrians (or other classes) in videos
- 🧪 Output results in MOT format for evaluation

---

## 🔧 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/video_object_tracking.git
cd video_object_tracking
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Download YOLOv8 weights:**

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

---

## 🏋️ Train on MOT Dataset
Ensure the dataset is structured like this:
```bash
data/MOT_YOLO/
├── images/
│   ├── train/
│   └── test/
├── labels/
│   ├── train/
│   └── test/
└── config.yaml
```
To start training:

```bash
python train.py
```

Update model path (yolov8n.pt, yolov8s.pt) and training parameters in train.py.

---

## 🧪 Evaluate Trained Model

``` bash
python evaluation/evaluate.py
```

Generates bounding box predictions in MOT format and evaluates them (optionally).

---

## 📹 Track Pedestrians in Video
Use your trained YOLOv8 model with DeepSORT on a video:

```bash
python track_predict.py \
  --video data/pedestrian.mp4 \
  --output_video outputs/output.mp4 \
  --output_txt outputs/output_mot.txt \
  --weights runs/train/yolov8n-mot17/weights/best.pt
```

This saves:

Annotated video with tracked objects

MOT-format .txt file with per-frame object positions

---

## 📌 Output Format (MOT)
Each row in output_mot.txt:

```
frame_id, track_id, x, y, w, h, 1, -1, -1, -1
```

---

## 🧠 Model Architecture
YOLOv8: For fast object detection or segmentation.

YOLOSegDetector: Wrapper over YOLOv8 for detection.

DeepSORT: Tracks objects across frames using motion + appearance.

DeepSortTracker: Wrapper class that handles integration with detections.

---

## 🛠 TODO
 Add web UI for upload + tracking

 Add detection class filters

 ---

## 🤝 Contributing
Open to suggestions, bug fixes, and feature contributions. PRs welcome!


