import os
import cv2
import random
import shutil
import hashlib
from PIL import Image
from tqdm import tqdm
from collections import Counter

# =========================
# CONFIG
# =========================
DATASET_PATH = r"D:\Assesments\safety-system\dataset"
DATASET_PATH_TRAIN = os.path.join(DATASET_PATH, "train")
OUTPUT_PATH = r"D:\Assesments\safety-system\cleaned-construction-safety-dataset"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
TARGET_SIZE = (640, 640)

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

random.seed(42)

# =========================
# HELPERS
# =========================
def is_image(file):
    return os.path.splitext(file)[1] in IMG_EXTS

def get_label(img_file):
    return os.path.splitext(img_file)[0] + ".txt"

def verify_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def resize_image(path):
    img = cv2.imread(path)
    if img is None:
        return
    resized = cv2.resize(img, TARGET_SIZE)
    cv2.imwrite(path, resized)

def clean_label(label_path):
    if not os.path.exists(label_path):
        return 0

    with open(label_path, "r") as f:
        lines = f.readlines()

    cleaned = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            x = min(max(float(parts[1]), 0.0), 1.0)
            y = min(max(float(parts[2]), 0.0), 1.0)
            w = min(max(float(parts[3]), 0.0), 1.0)
            h = min(max(float(parts[4]), 0.0), 1.0)

            if w <= 0 or h <= 0:
                continue

            cleaned.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        except Exception:
            continue

    with open(label_path, "w") as f:
        f.writelines(cleaned)

    return len(cleaned)

def get_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# =========================
# CLEANING
# =========================
def clean_dataset():
    image_dir = os.path.join(DATASET_PATH_TRAIN, "images")
    label_dir = os.path.join(DATASET_PATH_TRAIN, "labels")

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image folder not found: {image_dir}")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"Label folder not found: {label_dir}")

    cleaned_data = []
    seen_hashes = set()
    stats = Counter()

    for img_file in tqdm(os.listdir(image_dir), desc="Cleaning"):
        if not is_image(img_file):
            continue

        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, get_label(img_file))

        if not verify_image(img_path):
            stats["corrupt_removed"] += 1
            continue

        h = get_hash(img_path)
        if h in seen_hashes:
            stats["duplicate_removed"] += 1
            continue
        seen_hashes.add(h)

        if not os.path.exists(label_path):
            stats["no_label_removed"] += 1
            continue

        valid_boxes = clean_label(label_path)
        if valid_boxes == 0:
            stats["empty_label_removed"] += 1
            continue

        resize_image(img_path)

        cleaned_data.append((img_path, label_path))
        stats["kept"] += 1

    print("\nCleaning Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return cleaned_data

# =========================
# SPLITTING
# =========================
def split_dataset(cleaned_data):
    random.shuffle(cleaned_data)
    total = len(cleaned_data)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": cleaned_data[:train_end],
        "valid": cleaned_data[train_end:val_end],
        "test": cleaned_data[val_end:],
    }

    for split_name, items in splits.items():
        img_out = os.path.join(OUTPUT_PATH, split_name, "images")
        lbl_out = os.path.join(OUTPUT_PATH, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img_path, lbl_path in items:
            shutil.copy(img_path, os.path.join(img_out, os.path.basename(img_path)))
            shutil.copy(lbl_path, os.path.join(lbl_out, os.path.basename(lbl_path)))

    print("\nDataset split:")
    for split_name, items in splits.items():
        print(f"  {split_name}: {len(items)} images")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    data = clean_dataset()
    split_dataset(data)

    DATASET_PATH_CLEAN = OUTPUT_PATH
    print(f"\nCleaned dataset ready at: {DATASET_PATH_CLEAN}")