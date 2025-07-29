import cv2
import numpy as np

def draw_boxes(frame, tracks):
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def draw_masks(frame, masks, alpha=0.5):
    for mask in masks:
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        mask_color = np.zeros_like(frame, dtype=np.uint8)
        mask_color[mask.astype(bool)] = color
        frame = cv2.addWeighted(frame, 1, mask_color, alpha, 0)
    return frame
