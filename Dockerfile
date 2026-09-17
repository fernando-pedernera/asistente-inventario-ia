# Dockerfile optimizado para la API de FastAPI (Visión Computacional + RAG)
FROM python:3.10-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar las dependencias del sistema necesarias para OpenCV y Ultralytics (YOLO)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de dependencias
COPY requirements.txt .

# Instalar las dependencias de Python
# Usamos --no-cache-dir para mantener la imagen liviana
RUN pip install --no-cache-dir -r requirements.txt

# Instalar httpx para que los tests pasen (por si no está en requirements.txt)
RUN pip install --no-cache-dir httpx

# Copiar todo el código, modelos y datos necesarios al contenedor
COPY src/ src/
COPY models/ models/
COPY runs/ runs/
COPY data/ data/
COPY scratch/ scratch/

# Exponer el puerto de la API
EXPOSE 8000

# Comando para iniciar la aplicación
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
