import json
from pathlib import Path


# LabelMe 图片和 JSON 所在目录
LABELME_DIR = Path("/Users/zhuzimeng/Desktop/clock_dataset")

# 已经划分好的训练集和验证集
YOLO_DIR = Path("/Users/zhuzimeng/Desktop/clock_yolo")

# COCO 标注输出目录
OUTPUT_DIR = Path(
    "/Users/zhuzimeng/Desktop/clock_detectron2/annotations"
)

# COCO 的类别编号从 1 开始
CATEGORIES = [
    {
        "id": 1,
        "name": "clock",
        "supercategory": "object",
    }
]


def convert_split(split):
    """
    把一个数据集划分转换成 COCO 格式。

    split 可以是：
    - train
    - val
    """

    image_dir = YOLO_DIR / "images" / split

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }

    image_id = 1
    annotation_id = 1

    image_paths = sorted(image_dir.glob("*"))

    for image_path in image_paths:
        if image_path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
        }:
            continue

        json_path = LABELME_DIR / f"{image_path.stem}.json"

        if not json_path.exists():
            print(f"缺少 JSON：{json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as file:
            labelme_data = json.load(file)

        image_width = labelme_data["imageWidth"]
        image_height = labelme_data["imageHeight"]

        # COCO 中每张图片的信息
        coco_data["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": image_width,
                "height": image_height,
            }
        )

        for shape in labelme_data["shapes"]:
            if shape["label"] != "clock":
                print(
                    f"跳过未知标签："
                    f"{json_path.name} -> {shape['label']}"
                )
                continue

            if shape["shape_type"] != "rectangle":
                print(
                    f"跳过非矩形标注："
                    f"{json_path.name} -> {shape['shape_type']}"
                )
                continue

            points = shape["points"]

            x1, y1 = points[0]
            x2, y2 = points[1]

            xmin = min(x1, x2)
            ymin = min(y1, y2)
            xmax = max(x1, x2)
            ymax = max(y1, y2)

            # 防止标注框超出图片边界
            xmin = max(0, min(xmin, image_width))
            ymin = max(0, min(ymin, image_height))
            xmax = max(0, min(xmax, image_width))
            ymax = max(0, min(ymax, image_height))

            box_width = xmax - xmin
            box_height = ymax - ymin

            if box_width <= 0 or box_height <= 0:
                print(f"跳过无效标注框：{json_path.name}")
                continue

            # COCO 的 bbox 格式是：
            # [左上角x, 左上角y, 宽度, 高度]
            coco_data["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [
                        xmin,
                        ymin,
                        box_width,
                        box_height,
                    ],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )

            annotation_id += 1

        image_id += 1

    output_path = OUTPUT_DIR / f"clock_{split}.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            coco_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"{split} 转换完成")
    print(f"图片数量：{len(coco_data['images'])}")
    print(f"标注框数量：{len(coco_data['annotations'])}")
    print(f"输出文件：{output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    convert_split("train")
    convert_split("val")

    print("全部转换完成！")


if __name__ == "__main__":
    main()
