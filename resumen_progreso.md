# Resumen de Progreso: Asistente Inteligente de Inventario y Manuales

Este documento registra los avances realizados en el proyecto hasta la fecha, siguiendo el plan de ejecución establecido. Al finalizar todas las fases, se incluirán aquí las respuestas formales a las preguntas de la entrevista.

---

## ✅ Fase 1: Setup y Versionado Completado
Durante esta fase se estableció la base de la arquitectura del proyecto y el entorno de desarrollo:

- **Estructura del Proyecto:** Se crearon los directorios clave siguiendo las mejores prácticas para ciencia de datos (`data/`, `models/`, `notebooks/`, `src/`).
- **Entorno Virtual:** Se configuró el entorno aislado `venv` y se definieron las dependencias en `requirements.txt` (FastAPI, Langchain, Ultralytics, Roboflow, etc.).
- **Variables de Entorno:** Se creó una plantilla `.env.example` y el archivo `.env` local para manejar secretos de forma segura, como la `ROBOFLOW_API_KEY`.
- **Control de Versiones:** Se configuró `.gitignore` para prevenir la subida de archivos pesados de datos y entornos virtuales a Git, preparando el terreno para DVC.

---

## ✅ Fase 2: Pipeline de Datos y Roboflow Completado
En esta fase se prepararon y procesaron los datos crudos que alimentarán a los modelos de Visión Computacional (YOLOv8) y NLP (RAG):

- **Recolección de Datos Visuales:** Se generaron 5 imágenes base de una taza en diferentes perspectivas y condiciones de iluminación.
- **Anotación (Roboflow):** Las 5 imágenes fueron subidas al proyecto "inventario-productos" y etiquetadas manualmente mediante bounding boxes (clase `taza`).
- **Data Augmentation:** Se aplicaron técnicas de aumento de datos (Rotación entre -15° y +15°, y variaciones de Brillo) para multiplicar el dataset original, generando una versión robusta de 15 imágenes (3x).
- **Descarga Automatizada:** Se desarrolló el script `src/data/download_dataset.py` el cual se conecta a la API de Roboflow, descarga el dataset aumentado en formato YOLOv8 y lo aloja automáticamente en `data/raw`.
- **Preparación RAG (Documentos):** Se generaron sintéticamente dos manuales en formato PDF (`manual_taza_premium.pdf` y `manual_taza_travel.pdf`) y se ubicaron en `data/raw/` listos para ser vectorizados en la siguiente fase.

---

## ✅ Fase 3: Entrenamiento RAG y Modelado Completado
En esta fase se entrenó el modelo de visión y se construyó el sistema para el Chatbot de los manuales:

- **Entrenamiento YOLOv8:** Se entrenó el modelo de detección de objetos (YOLOv8) con el dataset generado (clase `taza`) durante 5 epochs. Los pesos del modelo (`best.pt`) quedaron almacenados en `models/yolov8_custom/weights/`.
- **Pipeline RAG (Ingesta):** Se creó `src/models/rag_pipeline.py` para cargar los PDFs (`PyPDFLoader`), dividir el texto en fragmentos (Chunking con `RecursiveCharacterTextSplitter`) y almacenarlos en una base de datos vectorial con FAISS.
- **Embeddings Locales:** Se integró HuggingFace (`all-MiniLM-L6-v2`) para la creación de embeddings sin depender de conexión a internet o cuotas de API de pago.
- **Integración de LLM:** Se configuró el modelo de Google Gemini (`gemini-3.6-flash`) usando LCEL (LangChain Expression Language) para responder consultas precisas recuperando contexto relevante de los manuales. El pipeline RAG probó su funcionamiento respondiendo consultas de mantenimiento de los manuales en la terminal.

---

## ✅ Fase 4: API FastAPI y Endpoint Unificado Completado
En esta fase unimos los dos mundos (Visión por Computadora y NLP) en un solo servicio backend:

- **Servidor Web:** Desarrollamos una aplicación web usando `FastAPI` en `src/api/main.py`.
- **Endpoint Unificado:** Creamos el endpoint `POST /analyze` que recibe una imagen (multipart) y una pregunta (texto).
- **Flujo de Ejecución Conjunta:** 
  - El servidor guarda la imagen temporalmente.
  - Ejecuta el modelo `YOLOv8` cargando los pesos entrenados (`best.pt`) para verificar si existe una "taza".
  - Si la taza es detectada, el servidor toma la pregunta del usuario y ejecuta el pipeline de LangChain (RAG) para buscar la respuesta en la base de datos de manuales (FAISS) utilizando el modelo `gemini-3.6-flash`.
- **Resultados:** Las pruebas locales confirman que al enviar una foto y preguntar por el lavado de la taza *Travel*, el sistema detecta la taza en la imagen y responde exitosamente con las advertencias correctas (no lavar en lavavajillas).

---

## ✅ Fase 5: Docker y CI/CD Completados
En esta fase preparamos el proyecto para su despliegue en cualquier entorno de producción de forma confiable:

- **Contenedores (Docker):** 
  - Se creó un `Dockerfile` utilizando la imagen `python:3.10-slim`, instalando dependencias del sistema operativo (para OpenCV y YOLOv8) y empaquetando el código.
  - Se definió un archivo `.dockerignore` para excluir archivos innecesarios como cachés locales y acelerar la creación de la imagen.
  - Se configuró `docker-compose.yml` para levantar la API de forma rápida y sencilla mapeando puertos y manejando las variables de entorno.
- **Integración Continua (GitHub Actions):** 
  - Se creó un pipeline de CI/CD automatizado en `.github/workflows/main.yml`. 
  - Este pipeline instala las dependencias y ejecuta pruebas unitarias (`pytest`) cada vez que haya un *push* o *pull request* en la rama `main`, garantizando que código defectuoso no rompa el sistema.
- **Tests Unitarios:** Se creó un script en `tests/test_api_ci.py` que permite a GitHub Actions certificar que la aplicación base responde correctamente (`HTTP 200 OK`) antes de autorizar cualquier despliegue.

---

## ✅ Fase 6: Monitoreo de Data Drift (Evidently) Completado
En esta fase final completamos el ciclo MLOps asegurándonos de que el modelo pueda ser monitoreado en producción:

- **Sistema de Logging:** Modificamos la API (`src/api/main.py`) para registrar las métricas de inferencia de cada petición en el archivo `data/processed/inference_logs.csv` (incluyendo la confianza del modelo YOLOv8 y las longitudes de las preguntas/respuestas del RAG).
- **Dataset de Referencia:** Creamos un script `src/models/generate_reference_data.py` que simula un mes de tráfico "sano" e ideal (con altas tasas de confianza) y lo guardamos como baseline en `reference_logs.csv`.
- **Generación de Reportes Automáticos:** Construimos `src/models/drift_monitor.py` utilizando la librería `evidently`. Este script compara los logs en producción contra los logs de referencia y exporta un panel interactivo (`drift_report.html`) detallando matemáticamente si las distribuciones de los datos han cambiado y si los modelos necesitan ser reentrenados.

---

## 🎉 Proyecto Completado
¡Todas las fases han sido ejecutadas exitosamente! El MVP (Producto Mínimo Viable) es capaz de identificar objetos visualmente y responder consultas a través de manuales vectorizados con IA Generativa.
