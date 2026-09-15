from pathlib import Path

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


PROJECT_DIR = Path("/Users/zhuzimeng/Desktop/cats_dogs_detectron2")
VAL_DIR = Path("/Users/zhuzimeng/Desktop/ultralytics/datasets/cats_dogs/images/val")
MODEL_PATH = PROJECT_DIR / "output" / "model_final.pth"
OUTPUT_DIR = PROJECT_DIR / "prediction"

test_images = [
    sorted(VAL_DIR.glob("cat.*.jpg"))[0],
    sorted(VAL_DIR.glob("dog.*.jpg"))[0],
]

cfg = get_cfg()
config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
cfg.MODEL.WEIGHTS = str(MODEL_PATH)

metadata = MetadataCatalog.get("cats_dogs_prediction")
metadata.thing_classes = ["cat", "dog"]

predictor = DefaultPredictor(cfg)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for image_path in test_images:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"图片无法读取：{image_path}")

    instances = predictor(image)["instances"].to("cpu")
    visualizer = Visualizer(
        image[:, :, ::-1],
        metadata=metadata,
        scale=1.0,
        instance_mode=ColorMode.IMAGE,
    )
    result = visualizer.draw_instance_predictions(instances)
    output_path = OUTPUT_DIR / f"predicted_{image_path.name}"
    cv2.imwrite(str(output_path), result.get_image()[:, :, ::-1])

    classes = instances.pred_classes.tolist()
    scores = instances.scores.tolist()
    predictions = [
        f"{metadata.thing_classes[class_id]} {score:.2%}"
        for class_id, score in zip(classes, scores)
    ]
    print(f"{image_path.name}：{predictions}")
    print(f"结果已保存：{output_path}")

print("猫狗预测完成！")
