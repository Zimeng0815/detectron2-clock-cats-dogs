import os
import random
import shutil

source_folder = "/Users/zhuzimeng/Desktop/clock_dataset"

output_folder = "/Users/zhuzimeng/Desktop/clock_yolo"

images_train = os.path.join(output_folder, "images", "train")
images_val = os.path.join(output_folder, "images", "val")
labels_train = os.path.join(output_folder, "labels", "train")
labels_val = os.path.join(output_folder, "labels", "val")

os.makedirs(images_train, exist_ok=True)
os.makedirs(images_val, exist_ok=True)
os.makedirs(labels_train, exist_ok=True)
os.makedirs(labels_val, exist_ok=True)

images = [
    f for f in os.listdir(source_folder)
    if f.endswith(".jpeg")
]

random.seed(42)
random.shuffle(images)

train_images = images[:160]
val_images = images[160:]

for image_name in train_images:
    base_name = os.path.splitext(image_name)[0]

    shutil.copy(
        os.path.join(source_folder, image_name),
        os.path.join(images_train, image_name)
    )

    shutil.copy(
        os.path.join(source_folder, base_name + ".txt"),
        os.path.join(labels_train, base_name + ".txt")
    )

for image_name in val_images:
    base_name = os.path.splitext(image_name)[0]

    shutil.copy(
        os.path.join(source_folder, image_name),
        os.path.join(images_val, image_name)
    )

    shutil.copy(
        os.path.join(source_folder, base_name + ".txt"),
        os.path.join(labels_val, base_name + ".txt")
    )

print("划分完成！")
print("train:", len(train_images))
print("val:", len(val_images))