from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import os
import shutil
from ultralytics import YOLO
from src.models.rag_pipeline import get_answer

app = FastAPI(title="Asistente de Inventario y Manuales")

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, "runs", "detect", "models", "yolov8_custom", "weights", "best.pt")
TEMP_IMAGE_PATH = os.path.join(BASE_DIR, "scratch", "temp_upload.jpg")

# Cargar YOLOv8 globalmente
print(f"Loading YOLOv8 weights from {YOLO_WEIGHTS_PATH}...")
try:
    model = YOLO(YOLO_WEIGHTS_PATH)
    yolo_loaded = True
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    yolo_loaded = False


class AnalyzeResponse(BaseModel):
    object_detected: bool
    confidence: float
    answer: str


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_product(file: UploadFile = File(...), question: str = Form(...)):
    # 1. Guardar la imagen de forma temporal
    with open(TEMP_IMAGE_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    object_detected = False
    confidence = 0.0
    answer = "No se detectó el objeto."

    if not yolo_loaded:
        return AnalyzeResponse(
            object_detected=False,
            confidence=0.0,
            answer="El modelo YOLOv8 no pudo ser cargado."
        )

    # 2. Inferencia con YOLOv8 (Umbral bajo para el MVP debido a las pocas epochs de entrenamiento)
    results = model.predict(source=TEMP_IMAGE_PATH, conf=0.01)
    
    # 3. Analizar predicciones
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            object_detected = True
            # Tomar la primera detección con mayor confianza
            confidence = float(boxes[0].conf[0])
            break

    # 4. Si se detectó el objeto (taza), lanzar la consulta al RAG
    if object_detected:
        try:
            answer = get_answer(question)
        except Exception as e:
            answer = f"Error al procesar la pregunta en el RAG: {e}"
    else:
        answer = "No se detectó una taza en la imagen proporcionada. Por lo tanto, no se consultó el manual."

    # --- 5. Logging para Monitoreo de Drift ---
    import csv
    from datetime import datetime
    
    logs_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "inference_logs.csv")
    
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question_length", "yolo_confidence", "answer_length"])
        
        writer.writerow([
            datetime.now().isoformat(),
            len(question),
            round(confidence, 4),
            len(answer)
        ])
    # ------------------------------------------

    return AnalyzeResponse(
        object_detected=object_detected,
        confidence=confidence,
        answer=answer
    )

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Asistente de Inventario API en funcionamiento."}
