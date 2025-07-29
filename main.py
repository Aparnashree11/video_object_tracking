from models.yolo_seg_detector import YOLOSegDetector
from models.deep_sort_tracker import DeepSortTracker
from utils.video_utils import read_video, write_video

import os, cv2

def main(video_path, save_path):
    detector = YOLOSegDetector()
    tracker = DeepSortTracker()

    cap, width, height, fps = read_video(video_path)
    writer = write_video(save_path, width, height, fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections, masks = detector.detect_and_segment(frame)
        tracks = tracker.update_tracks(detections, frame)

        for track in tracks:
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f'ID: {track.track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[✓] Saved tracked video to {save_path}")

if __name__ == "__main__":
    main("data/test_video.mp4", "outputs/result.mp4")
