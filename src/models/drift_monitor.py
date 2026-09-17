import os
import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset

def monitor_drift():
    print("Iniciando monitoreo de Data Drift...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, "data", "processed")
    
    ref_path = os.path.join(logs_dir, "reference_logs.csv")
    cur_path = os.path.join(logs_dir, "inference_logs.csv")
    report_path = os.path.join(base_dir, "drift_report.html")
    
    if not os.path.exists(ref_path):
        print(f"Error: No se encontró el dataset de referencia en {ref_path}")
        return
        
    if not os.path.exists(cur_path):
        print(f"Error: No se encontraron logs de inferencia en {cur_path}. Asegúrate de hacer llamadas a la API primero.")
        return

    # Cargar datos
    reference_data = pd.read_csv(ref_path)
    current_data = pd.read_csv(cur_path)
    
    # Nos aseguramos de parsear las fechas para el reporte si es necesario, 
    # pero Evidently por defecto maneja las columnas numéricas para detectar drift de distribución.
    # Vamos a dropear el timestamp temporalmente o dejar que Evidently lo trate como categoría/texto.
    reference_data = reference_data.drop(columns=['timestamp'])
    current_data = current_data.drop(columns=['timestamp'])

    print(f"Comparando {len(current_data)} logs recientes contra {len(reference_data)} logs de referencia...")

    # Generar el reporte de Drift
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])

    data_drift_report.run(reference_data=reference_data, current_data=current_data)
    
    # Guardar en HTML
    data_drift_report.save_html(report_path)
    print(f"¡Reporte generado con éxito! Puedes abrirlo en tu navegador: {report_path}")

if __name__ == "__main__":
    monitor_drift()
