from pathlib import Path

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


PROJECT_DIR = Path("/Users/zhuzimeng/Desktop/cats_dogs_detectron2")
DATASET_DIR = Path("/Users/zhuzimeng/Desktop/ultralytics/datasets/cats_dogs")

register_coco_instances(
    "cats_dogs_val",
    {},
    str(PROJECT_DIR / "annotations" / "cats_dogs_val.json"),
    str(DATASET_DIR / "images" / "val"),
)

cfg = get_cfg()
config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.WEIGHTS = str(PROJECT_DIR / "output" / "model_final.pth")
cfg.DATASETS.TEST = ("cats_dogs_val",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.INPUT.MIN_SIZE_TEST = 640
cfg.INPUT.MAX_SIZE_TEST = 1024
cfg.OUTPUT_DIR = str(PROJECT_DIR / "evaluation")
Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

model = DefaultTrainer.build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

evaluator = COCOEvaluator(
    "cats_dogs_val",
    output_dir=cfg.OUTPUT_DIR,
)
data_loader = build_detection_test_loader(cfg, "cats_dogs_val")
results = inference_on_dataset(model, data_loader, evaluator)

print("猫狗验证结果：")
print(results)
