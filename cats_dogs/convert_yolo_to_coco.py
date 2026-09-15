import json
from pathlib import Path

from PIL import Image


YOLO_DIR = Path("/Users/zhuzimeng/Desktop/ultralytics/datasets/cats_dogs")
OUTPUT_DIR = Path("/Users/zhuzimeng/Desktop/cats_dogs_detectron2/annotations")

CATEGORIES = [
    {"id": 1, "name": "cat", "supercategory": "animal"},
    {"id": 2, "name": "dog", "supercategory": "animal"},
]


def convert_split(split):
    image_dir = YOLO_DIR / "images" / split
    label_dir = YOLO_DIR / "labels" / split
    image_paths = sorted(image_dir.glob("*.jpg"))

    coco = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            width, height = image.size

        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"找不到标注：{label_path}")

        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue

            class_id, x_center, y_center, box_width, box_height = map(
                float, line.split()
            )
            box_width *= width
            box_height *= height
            xmin = x_center * width - box_width / 2
            ymin = y_center * height - box_height / 2

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id) + 1,
                    "bbox": [xmin, ymin, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"cats_dogs_{split}.json"
    output_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2))

    print(f"{split} 转换完成")
    print(f"图片数量：{len(coco['images'])}")
    print(f"标注框数量：{len(coco['annotations'])}")
    print(f"输出文件：{output_path}")


convert_split("train")
convert_split("val")
print("猫狗数据全部转换完成！")
