# Detectron2 Clock Detection and Cat–Dog Control Experiment

本项目使用 Detectron2 的 Faster R-CNN 完成时钟目标检测，并使用猫狗数据集进行对照实验，以判断检测问题主要来自代码流程还是数据。

## 项目内容

本仓库包含两个实验：

1. `clock`：时钟目标检测实验
2. `cats_dogs`：猫狗目标检测对照实验

两个实验都包括：

- 数据格式转换
- 模型训练
- 模型验证
- 新图片预测
- 代表性预测结果

## 实验流程

### 1. 时钟实验

首先使用 LabelMe 对 200 张时钟图片进行标注，然后将数据划分为：

- 训练集：160 张
- 验证集：40 张

LabelMe 标注被转换成 Detectron2 使用的 COCO JSON 格式，随后使用 Faster R-CNN R50-FPN 进行训练。

### 2. 猫狗对照实验

为了判断时钟预测问题来自代码还是数据，使用猫狗数据进行了相同的训练、验证和预测流程：

- 训练集：880 张
- 验证集：220 张
- 类别：cat、dog

### 3. 实验结果

| 实验 | AP | AP50 | AP75 |
| --- | ---: | ---: | ---: |
| 时钟 | 75.74 | 100.00 | 95.79 |
| 猫狗 | 55.34 | 92.30 | 57.27 |

时钟模型能够检测测试图片中的实物时钟及其镜中倒影。猫狗对照实验也可以正常完成训练、验证和预测，说明整体代码流程可以正常运行。

综合判断：时钟倒影被识别为时钟，主要与时钟数据量较少，以及训练集中缺少倒影和负样本有关，而不是整体代码流程无法运行。

## 文件结构

```text
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