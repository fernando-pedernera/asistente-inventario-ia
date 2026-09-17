# 🤖 Asistente Inteligente de Inventario y Manuales (MVP)

Este proyecto es un **Producto Mínimo Viable (MVP)** diseñado para demostrar un flujo *End-to-End* de Machine Learning y Operaciones de IA (MLOps). Combina Visión por Computadora (CV) y Procesamiento de Lenguaje Natural (NLP) a través de una arquitectura RAG (Retrieval-Augmented Generation).

---

## 🏗️ Arquitectura del Sistema

La solución está construida sobre una API en **FastAPI** que actúa como orquestador entre el usuario, el modelo de visión y el modelo de lenguaje.

```mermaid
graph TD
    A[Usuario / App] -->|Sube Imagen + Pregunta| B(FastAPI Endpoint /analyze)
    
    subgraph Computer Vision
    B -->|Imagen| C(YOLOv8)
    C -->|Detección de Objeto| D{¿Detecta Taza?}
    end
    
    subgraph NLP & RAG
    D -->|Sí| E[LangChain RAG Pipeline]
    E -->|Pregunta| F[(FAISS Vector Store)]
    F -->|Documentos Relevantes| G(Google Gemini 1.5 Flash)
    end
    
    D -->|No| Z[Retorna: 'No se detectó el objeto']
    G -->|Respuesta Generada| H[Retorna: JSON]
    
    subgraph MLOps & Monitoreo
    B -.->|Guarda métricas| L[(inference_logs.csv)]
    L -.->|Reporte de Desviación| M[EvidentlyAI Drift Monitor]
    end
```

## ✨ Características Principales

1. **Visión Computacional:** Un modelo `YOLOv8` entrenado a medida utilizando **Ultralytics** y datos sintéticos aumentados desde **Roboflow**. Identifica la presencia de productos de inventario (en este caso, tazas) con alta precisión.
2. **Q&A sobre Manuales (RAG):** Integración de **LangChain** y **Google Gemini API** para que el sistema responda preguntas basándose exclusivamente en manuales técnicos vectorizados localmente mediante **FAISS** y embeddings de HuggingFace.
3. **Monitoreo de Data Drift:** Implementación de **EvidentlyAI** para generar dashboards automáticos que alertan cuando los datos de producción varían significativamente respecto a los datos de referencia (entrenamiento).
4. **Despliegue y DevOps:** Contenedores construidos con **Docker**, orquestación local con `docker-compose`, y automatización de pruebas CI/CD mediante **GitHub Actions**.

---

## 🚀 Cómo ejecutar el proyecto

Existen dos formas principales de arrancar la aplicación:

### Opción 1: Con Docker (Recomendado)
Asegúrate de tener instalado Docker y Docker Compose, además de proveer tu `GEMINI_API_KEY`.

```bash
# 1. Clonar el repositorio
git clone https://github.com/fernando-pedernera/asistente-inventario-ia.git
cd asistente-inventario-ia

# 2. Configurar variables de entorno
# Renombra .env.example a .env e inserta tu clave de Google Gemini API
cp .env.example .env

# 3. Levantar los contenedores
docker-compose up --build
```
La API estará expuesta en `http://localhost:8000`. Puedes probar el endpoint interactivo en `http://localhost:8000/docs`.

### Opción 2: Ejecución Local en Python (Modo Desarrollo)
Asegúrate de tener Python 3.10+ y un entorno virtual configurado.

```bash
# Instalar requerimientos y librerías del sistema para OpenCV
pip install -r requirements.txt

# Iniciar servidor Uvicorn
uvicorn src.api.main:app --reload
```

---

## 📂 Estructura del Repositorio

```text
├── .github/workflows/   # Pipelines de CI/CD (GitHub Actions)
├── data/
│   ├── raw/             # Datos crudos (Imágenes descargadas, PDFs originales)
│   └── processed/       # Datos procesados y Logs de monitoreo (Evidently)
├── models/
│   └── faiss_index/     # Base de datos vectorial persistente
├── runs/                # Salidas y pesos (.pt) de entrenamiento YOLOv8
├── src/                 # Código fuente principal
│   ├── api/             # FastAPI y endpoints (main.py, test_api.py)
│   ├── data/            # Scripts de ingesta (Roboflow)
│   └── models/          # Scripts de RAG, entrenamiento YOLO y Monitoreo Drift
├── tests/               # Pruebas unitarias para CI/CD
├── .dockerignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## 📈 Pruebas y Monitoreo (Evidently)

El proyecto incluye un script para evaluar si los datos de entrada están sufriendo *Data Drift*. Para generar un reporte estadístico visual (HTML):

```bash
python src/models/drift_monitor.py
```
El reporte se guardará en la raíz como `drift_report.html`, mostrándote gráficas detalladas sobre las distribuciones estadísticas de las métricas registradas (longitud de preguntas, confianza de visión, etc).
