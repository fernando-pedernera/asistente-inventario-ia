import os
from roboflow import Roboflow
from dotenv import load_dotenv

# Cargar variables de entorno (por ejemplo, desde un archivo .env)
load_dotenv()

def download_roboflow_dataset():
    """
    Descarga un dataset desde Roboflow usando la API.
    Asegúrate de configurar la variable de entorno ROBOFLOW_API_KEY.
    """
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("No se encontró ROBOFLOW_API_KEY en las variables de entorno.")

    rf = Roboflow(api_key=api_key)
    
    # TODO: Reemplaza "TU_ESPACIO_DE_TRABAJO", "TU_PROYECTO" y la version con tus datos de Roboflow
    # Puedes obtener este fragmento de código directamente desde la interfaz web de Roboflow
    # en la sección de "Export" -> "Format: YOLOv8" -> "show download code".
    project = rf.workspace("fernando-pedernera").project("inventario-productos")
    
    # Especificar la ruta de descarga dentro de nuestro proyecto
    download_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
    os.makedirs(download_path, exist_ok=True)
    
    print(f"Descargando dataset en {download_path}...")
    dataset = project.version(1).download("yolov8", location=download_path)
    print("Descarga completada!")

if __name__ == "__main__":
    download_roboflow_dataset()
