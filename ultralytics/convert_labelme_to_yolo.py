import json
import os

# 你的 200 张图片和 LabelMe JSON 所在的文件夹
folder = "/Users/zhuzimeng/Desktop/clock_dataset"

# 遍历文件夹里的所有文件
for filename in os.listdir(folder):

    # 只处理 LabelMe 生成的 JSON 文件
    if filename.endswith(".json"):

        json_path = os.path.join(folder, filename)

        # 打开 JSON
        with open(json_path, "r") as f:
            data = json.load(f)

        # 获取图片宽和高
        image_width = data["imageWidth"]
        image_height = data["imageHeight"]

        # 用来保存这一张图片的 YOLO 标注
        yolo_lines = []

        # 读取这一张图片里的所有标注框
        for shape in data["shapes"]:

            # 我们现在只识别 clock
            if shape["label"] == "clock":

                points = shape["points"]

                # LabelMe 矩形的两个角
                x1, y1 = points[0]
                x2, y2 = points[1]

                # 防止两个角的顺序相反
                xmin = min(x1, x2)
                xmax = max(x1, x2)
                ymin = min(y1, y2)
                ymax = max(y1, y2)

                # 转换成 YOLO 需要的格式
                x_center = ((xmin + xmax) / 2) / image_width
                y_center = ((ymin + ymax) / 2) / image_height
                width = (xmax - xmin) / image_width
                height = (ymax - ymin) / image_height

                # clock 是唯一类别，所以类别编号是 0
                yolo_line = f"0 {x_center} {y_center} {width} {height}"

                yolo_lines.append(yolo_line)

        # 生成对应的 txt 文件名
        txt_name = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(folder, txt_name)

        # 写入 YOLO txt 文件
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

print("转换完成！")
