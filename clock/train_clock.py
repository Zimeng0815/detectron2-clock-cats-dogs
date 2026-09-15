import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer


# 1. 注册自己的 COCO 数据集
register_coco_instances(
    "clock_train",
    {},
    "/Users/zhuzimeng/Desktop/clock_detectron2/annotations/clock_train.json",
    "/Users/zhuzimeng/Desktop/clock_yolo/images/train",
)

register_coco_instances(
    "clock_val",
    {},
    "/Users/zhuzimeng/Desktop/clock_detectron2/annotations/clock_val.json",
    "/Users/zhuzimeng/Desktop/clock_yolo/images/val",
)


# 2. 创建 Detectron2 配置
cfg = get_cfg()

# 使用 Faster R-CNN 目标检测模型
config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(config_file))


# 3. 设置训练集和验证集
cfg.DATASETS.TRAIN = ("clock_train",)
cfg.DATASETS.TEST = ("clock_val",)

# Mac 上设为 0，避免多进程读取问题
cfg.DATALOADER.NUM_WORKERS = 0


# 4. 加载官方预训练权重
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)


# 5. 设置训练参数
# Mac 使用 CPU
cfg.MODEL.DEVICE = "cpu"

# 每次训练一张图片
cfg.SOLVER.IMS_PER_BATCH = 1

# 学习率
cfg.SOLVER.BASE_LR = 0.00025

# 总训练迭代次数
cfg.SOLVER.MAX_ITER = 300

# 不在训练中途改变学习率
cfg.SOLVER.STEPS = []

# 每隔 50 次迭代保存一次模型
cfg.SOLVER.CHECKPOINT_PERIOD = 50

# 每隔 50 次迭代验证一次
cfg.TEST.EVAL_PERIOD = 50

# 每张图片抽取的候选框数量
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

# 只有一个类别：clock
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

# 降低输入尺寸，减轻 CPU 训练负担
cfg.INPUT.MIN_SIZE_TRAIN = (640,)
cfg.INPUT.MAX_SIZE_TRAIN = 1024
cfg.INPUT.MIN_SIZE_TEST = 640
cfg.INPUT.MAX_SIZE_TEST = 1024


# 6. 设置输出目录
cfg.OUTPUT_DIR = "/Users/zhuzimeng/Desktop/clock_detectron2/output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# 7. 创建并开始训练
trainer = DefaultTrainer(cfg)

# False 表示从预训练模型开始，而不是继续旧训练
trainer.resume_or_load(resume=False)
trainer.train()
