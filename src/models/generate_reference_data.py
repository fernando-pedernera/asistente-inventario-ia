import csv
import os
import random
from datetime import datetime, timedelta

def generate_reference_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(logs_dir, exist_ok=True)
    
    reference_file = os.path.join(logs_dir, "reference_logs.csv")
    
    # Parámetros para la simulación
    num_samples = 50
    start_time = datetime.now() - timedelta(days=30)
    
    print(f"Generando {num_samples} registros de referencia en {reference_file}...")
    
    with open(reference_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "question_length", "yolo_confidence", "answer_length"])
        
        for i in range(num_samples):
            # Simular marcas de tiempo a lo largo del último mes
            ts = (start_time + timedelta(hours=i*14)).isoformat()
            
            # Asumimos que los usuarios suelen hacer preguntas cortas y directas (30-80 caracteres)
            q_len = int(random.normalvariate(55, 10))
            if q_len < 10: q_len = 10
            
            # Asumimos que YOLO funciona muy bien (0.80 - 0.95 de confianza)
            yolo_conf = min(0.99, max(0.5, random.normalvariate(0.85, 0.05)))
            
            # Las respuestas del LLM suelen ser de 150-400 caracteres
            ans_len = int(random.normalvariate(250, 50))
            
            writer.writerow([ts, q_len, round(yolo_conf, 4), ans_len])

    print("Archivo de referencia generado con éxito.")

if __name__ == "__main__":
    generate_reference_data()
