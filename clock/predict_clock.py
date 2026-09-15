from pathlib import Path

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


PROJECT_DIR = Path("/Users/zhuzimeng/Desktop/clock_detectron2")
INPUT_IMAGE = Path("/Users/zhuzimeng/Desktop/new_clock.png")
MODEL_PATH = PROJECT_DIR / "output" / "model_final.pth"
OUTPUT_DIR = PROJECT_DIR / "prediction"
OUTPUT_IMAGE = OUTPUT_DIR / "new_clock_detectron2.jpg"


if not INPUT_IMAGE.exists():
    raise FileNotFoundError(f"找不到测试图片：{INPUT_IMAGE}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"找不到训练好的模型：{MODEL_PATH}")


cfg = get_cfg()
config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = str(MODEL_PATH)


image = cv2.imread(str(INPUT_IMAGE))
if image is None:
    raise RuntimeError(f"图片无法读取：{INPUT_IMAGE}")

predictor = DefaultPredictor(cfg)
outputs = predictor(image)
instances = outputs["instances"].to("cpu")

metadata = MetadataCatalog.get("clock_prediction")
metadata.thing_classes = ["clock"]

visualizer = Visualizer(
    image[:, :, ::-1],
    metadata=metadata,
    scale=1.0,
    instance_mode=ColorMode.IMAGE,
)
result = visualizer.draw_instance_predictions(instances)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT_IMAGE), result.get_image()[:, :, ::-1])

scores = instances.scores.tolist() if instances.has("scores") else []
print(f"检测到的时钟数量：{len(instances)}")
print("置信度：", [round(score, 4) for score in scores])
print(f"预测图片已保存：{OUTPUT_IMAGE}")
