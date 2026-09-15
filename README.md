# Clock Detection and Cat–Dog Control Experiments

本项目先使用 Ultralytics YOLO 训练时钟检测模型，随后使用 Detectron2 的 Faster R-CNN 完成时钟目标检测，并使用猫狗数据集进行对照实验，以帮助判断检测问题主要来自代码流程还是数据。

## 项目内容

本仓库包含三个部分：

1. `ultralytics`：Ultralytics YOLO 时钟检测实验
2. `clock`：Detectron2 时钟检测实验
3. `cats_dogs`：Detectron2 猫狗对照实验

## 1. Ultralytics 时钟实验

首先使用 LabelMe 对200张时钟图片进行标注，然后：

1. 使用 `convert_labelme_to_yolo.py` 将 LabelMe 标注转换为 YOLO 格式。
2. 使用 `split_clock_dataset.py` 将数据划分为训练集和验证集。
3. 使用 `clock.yaml` 配置图片路径和类别名称。
4. 使用 `train_clock.py` 加载 YOLOv8n 预训练模型并训练30个 epoch。
5. 使用训练后的模型预测新时钟图片。

仓库中保留了训练曲线、混淆矩阵、训练参数、数值记录和预测图片。

## 2. Detectron2 时钟实验

时钟数据划分为：

- 训练集：160张
- 验证集：40张
- 类别：clock

LabelMe 标注被转换为 Detectron2 使用的 COCO JSON 格式，随后使用 Faster R-CNN R50-FPN 和官方预训练权重进行迁移学习，共训练300次迭代。

## 3. Detectron2 猫狗对照实验

为了检查同一套 Detectron2 训练流程能否在另一套数据上正常工作，使用猫狗数据进行对照实验：

- 训练集：880张
- 验证集：220张
- 类别：cat、dog

猫狗数据原本采用 YOLO 标注格式，因此先转换为 COCO JSON，再进行训练、验证和预测。

## Detectron2 验证结果

| 实验 | AP | AP50 | AP75 |
| --- | ---: | ---: | ---: |
| 时钟 | 75.74 | 100.00 | 95.79 |
| 猫狗 | 55.34 | 92.30 | 57.27 |

时钟模型能够检测测试图片中的实物时钟及其镜中倒影。猫狗对照实验也可以正常完成训练、验证和预测，说明没有发现明显的整体代码流程故障。

综合判断：时钟倒影被识别为时钟，更可能与时钟数据量较少、训练集中缺少倒影和负样本，以及目标检测模型只根据视觉特征识别物体有关。

## 文件结构

```text
ultralytics/
├── convert_labelme_to_yolo.py
├── split_clock_dataset.py
├── train_clock.py
├── clock.yaml
├── training_results/
└── prediction/

clock/
├── convert_labelme_to_coco.py
├── train_clock.py
├── evaluate_clock.py
├── predict_clock.py
├── annotations/
├── evaluation/
└── prediction/

cats_dogs/
├── convert_yolo_to_coco.py
├── train_cats_dogs.py
├── evaluate_cats_dogs.py
├── predict_cats_dogs.py
├── annotations/
├── evaluation/
└── prediction/
```

## 使用环境

- Python 3.11
- PyTorch
- Detectron2
- Ultralytics
- OpenCV
- pycocotools

Detectron2 脚本运行前激活环境：

```bash
conda activate detectron2_env
```

## Detectron2 运行顺序

时钟实验：

```bash
python convert_labelme_to_coco.py
python train_clock.py
python evaluate_clock.py
python predict_clock.py
```

猫狗对照实验：

```bash
python convert_yolo_to_coco.py
python train_cats_dogs.py
python evaluate_cats_dogs.py
python predict_cats_dogs.py
```

## 说明

大型模型权重（`.pt`、`.pth`）和训练过程中的中间检查点未上传。仓库保留了实验代码、配置、COCO 标注、验证结果和代表性预测图片。代码中的数据路径为本次实验所用的本地路径，在其他电脑运行时需要修改相应路径。
