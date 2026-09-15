from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="/Users/zhuzimeng/Desktop/clock_yolo/clock.yaml",
    epochs=30,
    imgsz=640,
    device="mps"
)
