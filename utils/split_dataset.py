import os
import shutil
from sklearn.model_selection import train_test_split

# Root paths
images_root = "data/MOT_YOLO/images"
labels_root = "data/MOT_YOLO/labels"

# Ignore folders like 'train' and 'val' during listing
EXCLUDE = {"train", "val"}

# Get all sequence directories (e.g., MOT17-02-DPM)
all_sequences = [d for d in os.listdir(images_root)
                 if os.path.isdir(os.path.join(images_root, d)) and d not in EXCLUDE]

# Split 80% train, 20% val
train_seqs, val_seqs = train_test_split(all_sequences, test_size=0.2, random_state=42)

# Destination directories
image_train_dir = os.path.join(images_root, "train")
image_val_dir = os.path.join(images_root, "val")
label_train_dir = os.path.join(labels_root, "train")
label_val_dir = os.path.join(labels_root, "val")

# Create target dirs
os.makedirs(image_train_dir, exist_ok=True)
os.makedirs(image_val_dir, exist_ok=True)
os.makedirs(label_train_dir, exist_ok=True)
os.makedirs(label_val_dir, exist_ok=True)

# Move image and label sequence folders
def move_sequence(seq_name, split_type):
    img_src = os.path.join(images_root, seq_name)
    lbl_src = os.path.join(labels_root, seq_name)

    img_dst = os.path.join(image_train_dir if split_type == "train" else image_val_dir, seq_name)
    lbl_dst = os.path.join(label_train_dir if split_type == "train" else label_val_dir, seq_name)

    if os.path.exists(img_src):
        shutil.move(img_src, img_dst)
    if os.path.exists(lbl_src):
        shutil.move(lbl_src, lbl_dst)

# Move folders
for seq in train_seqs:
    move_sequence(seq, "train")

for seq in val_seqs:
    move_sequence(seq, "val")

print(f"Split completed: {len(train_seqs)} train sequences, {len(val_seqs)} val sequences.")
