import requests
import os

url = "http://127.0.0.1:8000/analyze"

# Using absolute path just to be safe in the test script
image_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "dataset", "train", "images", "mug_photo_1_1789587375569_jpg.rf.563e7c3092ff6b034db9d111ef8db12a.jpg")

try:
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        data = {"question": "¿Qué precauciones debo tener al lavar la taza de viaje (travel)?"}
        
        print("Enviando solicitud a la API...")
        response = requests.post(url, files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        print("Respuesta:")
        print(response.json())
except FileNotFoundError:
    print(f"Error: La imagen no se encuentra en la ruta {image_path}. Por favor ajusta la ruta en el script.")
except Exception as e:
    print(f"Error: {e}")
