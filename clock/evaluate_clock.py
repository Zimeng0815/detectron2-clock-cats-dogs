from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


# 注册验证集
register_coco_instances(
    "clock_val",
    {},
    "/Users/zhuzimeng/Desktop/clock_detectron2/annotations/clock_val.json",
    "/Users/zhuzimeng/Desktop/clock_yolo/images/val",
)


# 创建与训练时相同的模型配置
cfg = get_cfg()
config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(config_file))

cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = (
    "/Users/zhuzimeng/Desktop/"
    "clock_detectron2/output/model_final.pth"
)

cfg.DATASETS.TEST = ("clock_val",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.INPUT.MIN_SIZE_TEST = 640
cfg.INPUT.MAX_SIZE_TEST = 1024


# 创建模型并加载训练好的权重
model = DefaultTrainer.build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)


# 创建 COCO 验证器和验证数据加载器
evaluator = COCOEvaluator(
    "clock_val",
    output_dir=(
        "/Users/zhuzimeng/Desktop/"
        "clock_detectron2/evaluation"
    ),
)

val_loader = build_detection_test_loader(
    cfg,
    "clock_val",
)


# 在验证集上计算指标
results = inference_on_dataset(
    model,
    val_loader,
    evaluator,
)

print("验证结果：")
print(results)
