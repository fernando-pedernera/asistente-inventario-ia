from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Asistente de Inventario API en funcionamiento."}

# Nota: No probamos /analyze en el CI/CD básico porque requeriría 
# cargar el modelo YOLOv8 y conectarse a Gemini API, lo cual no siempre
# es ideal en un entorno de integración continua gratuito sin mocks profundos.
