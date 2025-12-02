import os
import shutil
import random

# ===== CONFIG =====
# Path to the folder with all images organized by class
images_dir = r"C:\Users\manid\OneDrive\Desktop\codes\machinelearning\food_prediction\data\food-101\images"

# Where to put train/val folders
train_dir = r"C:\Users\manid\OneDrive\Desktop\codes\machinelearning\food_prediction\data\train"
val_dir = r"C:\Users\manid\OneDrive\Desktop\codes\machinelearning\food_prediction\data\val"

# Train split ratio
train_ratio = 0.8
# ==================

# Create train/val folders
for folder in [train_dir, val_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Iterate classes
classes = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]

for class_name in classes:
    class_path = os.path.join(images_dir, class_name)

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    imgs = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
    random.shuffle(imgs)

    split_idx = int(len(imgs) * train_ratio)
    train_imgs = imgs[:split_idx]
    val_imgs = imgs[split_idx:]

    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

print("Train/Val split completed!")
print(f"Train samples: {sum(len(os.listdir(os.path.join(train_dir,c))) for c in classes)}")
print(f"Val samples: {sum(len(os.listdir(os.path.join(val_dir,c))) for c in classes)}")
