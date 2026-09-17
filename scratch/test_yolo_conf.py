from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, "runs", "detect", "models", "yolov8_custom", "weights", "best.pt")
image_path = os.path.join(BASE_DIR, "data", "raw", "dataset", "train", "images", "mug_photo_1_1789587375569_jpg.rf.563e7c3092ff6b034db9d111ef8db12a.jpg")

print(f"Loading YOLOv8 weights from {YOLO_WEIGHTS_PATH}...")
model = YOLO(YOLO_WEIGHTS_PATH)

print(f"Running inference on {image_path}...")
results = model.predict(source=image_path, conf=0.01)

for r in results:
    boxes = r.boxes
    if len(boxes) > 0:
        for box in boxes:
            print(f"Detección: Confianza = {float(box.conf[0]):.4f}, Clase = {int(box.cls[0])}")
    else:
        print("No se detectó nada con conf=0.01")
