from ultralytics import YOLO
import os

def train_model():
    # Cargar el modelo base YOLOv8 nano
    print("Cargando modelo YOLOv8n base...")
    model = YOLO('yolov8n.pt') 

    # Ruta absoluta al archivo data.yaml descargado de Roboflow
    data_yaml_path = os.path.abspath('data/raw/dataset/data.yaml')
    
    if not os.path.exists(data_yaml_path):
        print(f"Error: No se encontró el archivo de configuración del dataset en {data_yaml_path}")
        print("Asegúrate de que el dataset se haya descargado y descomprimido correctamente.")
        return
        
    print(f"Iniciando entrenamiento con 5 epochs usando: {data_yaml_path}")
    print("Nota: 5 epochs es solo para fines de demostración rápida.")
    
    # Entrenar el modelo
    results = model.train(
        data=data_yaml_path,
        epochs=5,          # 5 epochs para que sea rápido
        imgsz=512,         # Tamaño que definimos en el preprocesamiento de Roboflow
        project='models',  # Carpeta donde se guardarán los resultados
        name='yolov8_custom',
        exist_ok=True
    )
    
    print("\n✅ Entrenamiento completado.")
    print("El mejor modelo entrenado (pesos) se encuentra en: models/yolov8_custom/weights/best.pt")

if __name__ == '__main__':
    train_model()
