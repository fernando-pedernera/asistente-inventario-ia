# Plan de Proyecto End-to-End: Asistente Inteligente de Inventario y Manuales

Este documento detalla un proyecto mínimo, sin costos, diseñado específicamente para demostrar tus habilidades en respuesta a los requerimientos del puesto.

## 🎯 Objetivo del Proyecto
Crear una API REST (FastAPI) containerizada con Docker, que integre dos funcionalidades clave:
1. **Detección de Productos (Visión Computacional)**: Utilizando un dataset sintético gestionado con Roboflow.
2. **Q&A sobre Manuales (RAG)**: Un sistema de Generación Aumentada por Recuperación para responder preguntas sobre los productos.

Todo esto envuelto en un pipeline MLOps con monitoreo de drift y CI/CD.

---

## 🏗️ Arquitectura y Tecnologías (Costo $0)
- **Lenguaje**: Python 3.10+
- **Framework API**: FastAPI
- **Visión / Dataset**: Roboflow (Free tier) + YOLOv8 (Ultralytics) o modelo pre-entrenado.
- **RAG**: LangChain / LlamaIndex + FAISS/ChromaDB (Vector store local) + LLM gratuito (Gemini API free tier o HuggingFace local).
- **MLOps & Tracking**: MLflow (local) para tracking de experimentos, DVC para versionado de datos (junto con Git).
- **Monitoreo (Drift)**: EvidentlyAI (Open Source).
- **Despliegue**: Docker & Docker Compose.
- **CI/CD**: GitHub Actions.

---

## 📝 Respuestas a los Puntos de la Entrevista (Integrados en el Proyecto)

### 4) Método para armar un pipeline de ML end-to-end
El proyecto demostrará esta estructura:
1. **Preparación de Datos**: Scripts de extracción y limpieza (`src/data/`). Integración con la API de Roboflow para descargar imágenes sintéticas aumentadas y procesamiento de PDFs para el RAG.
2. **Entrenamiento (Training)**: Scripts modulares (`src/models/train.py`). Uso de MLflow para registrar hiperparámetros y métricas. 
3. **Despliegue (Deploy)**: Empaquetado del modelo y vector store dentro de una imagen Docker que expone endpoints a través de FastAPI.
4. **Monitoreo**: Un script asíncrono o endpoint `/metrics` que recolecta logs de predicción y utiliza **EvidentlyAI** para generar reportes HTML de data drift.

### 5) Versionado, Automatización de Deploys y Protocolo de Fallos
- **Versionado**: Git para código fuente; DVC (Data Version Control) para los datasets y binarios de los modelos, almacenando los archivos pesados en Google Drive o AWS S3 (Free Tier).
- **CI/CD**: Un workflow de GitHub Actions que:
  1. Ejecuta tests unitarios (pytest).
  2. Construye la imagen Docker (`docker build`).
  3. (Opcional) Publica la imagen en Docker Hub.
- **Protocolo de Acción ante Fallos de Contenedores**:
  1. **Prevención**: Uso de *Healthchecks* en Docker y endpoints `/health` en FastAPI. Configuración de `restart: always` (o `on-failure`) en `docker-compose.yml`.
  2. **Acción**: Si el contenedor falla, revisar logs vía `docker logs`. Implementar un fallback en la API (ej. devolver una respuesta cacheada o genérica si el modelo LLM no responde). Alertas configuradas vía webhooks hacia Slack/Email si el healthcheck falla repetidamente.

### 6) Arquitectura RAG y Uso de Roboflow
- **Base RAG (Arquitectura y Evaluación)**:
  - *Ingesta*: PyPDFLoader para extraer texto de manuales de usuario.
  - *Chunking*: Separación semántica de textos.
  - *Embeddings*: Uso de HuggingFace embeddings (ej. `all-MiniLM-L6-v2`) almacenados en FAISS o ChromaDB local.
  - *Generación*: Conexión a un LLM (API de Gemini).
  - *Evaluación*: Implementación de RAGAS (RAG Assessment) u otra métrica automatizada básica (Context Precision/Recall) comparando respuestas generadas contra un set de pruebas (Ground Truth).
- **Roboflow (Datasets Sintéticos)**:
  - Explicar/Demostrar cómo se usa Roboflow para aplicar *Data Augmentation* (rotaciones, brillo, ruido) a unas pocas fotos de un producto para multiplicar el tamaño del dataset de entrenamiento, simulando un entorno de producción con variaciones de iluminación.

### 7) Exposición como API y Detección de Drift
- **Exposición**: FastAPI por su alto rendimiento, soporte nativo asíncrono y auto-generación de documentación (Swagger/OpenAPI).
- **Detección de Drift (EvidentlyAI)**:
  - Guardar las entradas del usuario (texto para RAG y características extraídas de imágenes) junto con las predicciones en una base de datos local SQLite o archivo CSV/JSONL.
  - Un job programado (o un endpoint específico `/report/drift`) que toma un "dataset de referencia" (training data) y lo compara con el "dataset de producción" actual utilizando Evidently.
  - Si el drift detectado supera cierto umbral (ej. p-value < 0.05 en pruebas de distribución), se marca una alerta o se inhabilita temporalmente la predicción confiable.

---

## 🛠️ Plan de Ejecución (Paso a Paso)

### Fase 1: Setup y Versionado (Día 1)
- [ ] Inicializar repositorio Git.
- [ ] Configurar entorno virtual (`venv` o `conda`) y `requirements.txt`.
- [ ] Inicializar DVC para versionamiento de datos (`dvc init`).
- [ ] Crear la estructura de carpetas (ej. Cookiecutter Data Science: `data/`, `notebooks/`, `src/`, `models/`).

### Fase 2: Pipeline de Datos y Roboflow (Día 1-2)
- [ ] Crear cuenta gratuita en Roboflow, subir 10-20 fotos de objetos de escritorio (ej. tazas, ratones).
- [ ] Aplicar aumentos en Roboflow para generar un dataset sintético grande.
- [ ] Escribir script en Python que descargue el dataset usando la API de Roboflow.
- [ ] Recolectar 2 o 3 PDFs (ej. manuales de usuario de los objetos anteriores) y guardarlos en `data/raw`.

### Fase 3: RAG y Modelado (Día 2-3)
- [ ] Entrenar un modelo pequeño (ej. YOLOv8 nano) con los datos de Roboflow y guardar los pesos.
- [ ] Crear el pipeline RAG usando LangChain/LlamaIndex: Ingesta de los PDFs, creación del vector store y función de consulta.
- [ ] Crear tests de evaluación (RAGAS / métricas básicas).

### Fase 4: API con FastAPI (Día 3)
- [ ] Crear `app.py`.
- [ ] Endpoint `POST /predict/image`: Recibe imagen, devuelve clasificación/bounding boxes.
- [ ] Endpoint `POST /chat/rag`: Recibe pregunta, busca en FAISS, devuelve respuesta del LLM.
- [ ] Endpoint `GET /health`.
- [ ] Sistema de guardado de logs de inferencia (para el drift).

### Fase 5: Docker y CI/CD (Día 4)
- [ ] Crear el `Dockerfile` optimizado (multi-stage preferentemente, cuidando tamaños de librerías como PyTorch).
- [ ] Crear `docker-compose.yml`.
- [ ] Crear pipeline de GitHub Actions (`.github/workflows/main.yml`) que haga linting (flake8/black) y build.

### Fase 6: Monitoreo de Drift (Día 5)
- [ ] Escribir un notebook o script que consuma los logs guardados por la API.
- [ ] Generar un reporte de EvidentlyAI comparando los textos ingresados recientemente vs los textos con los que se pensó el sistema originalmente.

## 🚀 Próximos pasos
Si apruebas este plan, podemos empezar a crear la estructura base del proyecto y el entorno de desarrollo aquí mismo. ¡Solo indícamelo!
