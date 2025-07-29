import os
import cv2

def convert_mot_to_yolo(mot_path, yolo_path, class_id=0):
    os.makedirs(yolo_path, exist_ok=True)

    for seq in os.listdir(mot_path):
        seq_path = os.path.join(mot_path, seq)
        if not os.path.isdir(seq_path): continue

        label_out = os.path.join(yolo_path, "labels", seq)
        image_out = os.path.join(yolo_path, "images", seq)
        os.makedirs(label_out, exist_ok=True)
        os.makedirs(image_out, exist_ok=True)

        gt_file = os.path.join(seq_path, "gt", "gt.txt")
        img_dir = os.path.join(seq_path, "img1")
        annotations = {}

        with open(gt_file) as f:
            for line in f:
                parts = list(map(float, line.strip().split(',')))
                if len(parts) < 7:
                    continue  # skip malformed lines
                frame, track_id, x, y, w, h = parts[:6]
                frame = int(frame)
                if frame not in annotations:
                    annotations[frame] = []
                annotations[frame].append((track_id, x, y, w, h))


        for img_name in sorted(os.listdir(img_dir)):
            frame = int(img_name.split('.')[0])
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            h_img, w_img = img.shape[:2]

            frame_labels = annotations.get(frame, [])
            yolo_lines = []
            for _, x, y, w, h in frame_labels:
                xc = (x + w / 2) / w_img
                yc = (y + h / 2) / h_img
                ww = w / w_img
                hh = h / h_img
                yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

            label_file = os.path.join(label_out, img_name.replace(".jpg", ".txt"))
            with open(label_file, "w") as f_out:
                f_out.write("\n".join(yolo_lines))

            # Copy image
            cv2.imwrite(os.path.join(image_out, img_name), img)


    print(f"MOT17 converted to YOLOv8 format at: {yolo_path}")

def main():
    mot_path = "data/MOT/MOT17/train"          
    yolo_path = "data/MOT_YOLO"          

    convert_mot_to_yolo(mot_path, yolo_path, class_id=0)

if __name__ == "__main__":
    main()