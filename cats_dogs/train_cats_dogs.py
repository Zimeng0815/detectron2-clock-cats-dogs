from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer


PROJECT_DIR = Path("/Users/zhuzimeng/Desktop/cats_dogs_detectron2")
DATASET_DIR = Path("/Users/zhuzimeng/Desktop/ultralytics/datasets/cats_dogs")

register_coco_instances(
    "cats_dogs_train",
    {},
    str(PROJECT_DIR / "annotations" / "cats_dogs_train.json"),
    str(DATASET_DIR / "images" / "train"),
)
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
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

cfg.DATASETS.TRAIN = ("cats_dogs_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0

cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300
cfg.SOLVER.STEPS = []
cfg.SOLVER.CHECKPOINT_PERIOD = 50

cfg.INPUT.MIN_SIZE_TRAIN = (640,)
cfg.INPUT.MAX_SIZE_TRAIN = 1024
cfg.INPUT.MIN_SIZE_TEST = 640
cfg.INPUT.MAX_SIZE_TEST = 1024

cfg.OUTPUT_DIR = str(PROJECT_DIR / "output")
Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("开始训练猫狗 Detectron2 对照模型……")
print("训练图片：880张；类别：cat、dog；迭代次数：300")

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print(f"训练完成，模型保存在：{cfg.OUTPUT_DIR}/model_final.pth")
