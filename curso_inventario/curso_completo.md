# Curso Académico: Desarrollo MLOps End-to-End
## Asistente Inteligente de Inventario y Manuales

Este documento es una guía de estudio profundo diseñada para entender teóricamente cada fase del proyecto. Explica el **"por qué"** detrás de cada decisión técnica.

---

### Fase 1: Arquitectura y Versionado
**Concepto Clave:** MLOps (Machine Learning Operations).
MLOps es un conjunto de prácticas que busca unificar el desarrollo de sistemas de Machine Learning (ML) y las operaciones de software (Ops). Su objetivo es desplegar y mantener modelos en producción de forma confiable y eficiente.

- **Entornos Virtuales (`venv`):** En Python, un entorno virtual aísla las librerías instaladas para este proyecto del resto del sistema operativo. Esto garantiza que la versión de `FastAPI` o `YOLO` que usamos aquí no rompa otros proyectos en nuestra máquina.
- **`.gitignore`:** Archivo que indica a `git` qué archivos no deben subirse a la nube. En ML, nunca subimos los pesos de los modelos (archivos `.pt` de gran tamaño) ni los conjuntos de datos masivos.

---

### Fase 2: Pipeline de Datos y Visión Artificial (CV)
**Concepto Clave:** Visión por Computadora (Computer Vision) y Data Augmentation.

- **Anotación (Bounding Boxes):** Para que un modelo supervised-learning entienda qué es una "taza", debemos encerrar manualmente el objeto en un rectángulo (Bounding Box) indicando sus coordenadas relativas (x, y, ancho, alto).
- **Data Augmentation (Aumento de Datos):** Los modelos Deep Learning requieren miles de imágenes para generalizar correctamente. Como solo teníamos 5, aplicamos rotaciones y cambios de brillo. Esto engaña al modelo enseñándole que un objeto girado sigue siendo el mismo objeto, previniendo el **Overfitting** (sobreajuste).
- **Roboflow:** Herramienta utilizada para gestionar, versionar y pre-procesar datasets visuales antes de pasarlos a algoritmos de redes neuronales.

---

### Fase 3: Procesamiento de Lenguaje Natural (NLP) y Arquitectura RAG
**Concepto Clave:** Retrieval-Augmented Generation (Generación Aumentada por Recuperación).

El problema principal de los LLMs (como ChatGPT o Gemini) es que sufren de **Alucinaciones** y **Falta de Conocimiento Privado** (no conocen tus PDFs). La solución es **RAG**.

#### Componentes de RAG:
1. **Document Loader (Cargador):** Lee nuestro PDF.
2. **Text Splitter (Chunking):** Los LLMs tienen un límite de tokens (palabras) que pueden procesar a la vez. Dividimos el PDF en pequeños párrafos (Chunks).
3. **Embeddings:** Convertimos cada fragmento de texto en un **Vector Matemático** (una lista de números). Si dos textos hablan de cosas similares, sus vectores estarán cerca en el espacio multidimensional. Usamos `HuggingFace`.
4. **Vector Store (FAISS):** Una base de datos especializada en almacenar y buscar vectores.
5. **Generador (LLM):** Cuando el usuario pregunta algo, la pregunta se convierte a vector, FAISS busca el párrafo más similar y se lo entrega a `Gemini`. Gemini solo debe reformular ese párrafo.

**YOLOv8 (Entrenamiento):** Entrenamos la arquitectura YOLO (You Only Look Once), que es el estándar de la industria por ser capaz de detectar objetos en tiempo real (una sola pasada por la red neuronal) frente a modelos antiguos como R-CNN.

---

### Fase 4: Integración API Backend (FastAPI)
**Concepto Clave:** Microservicios y APIs RESTful.

- **FastAPI:** Framework moderno de Python. Es rápido porque soporta ejecución asíncrona (`async/await`), ideal para tareas que tardan mucho, como invocar a una API externa (Google Gemini).
- **Flujo Lógico:**
  1. Recibimos imagen y texto vía `POST /analyze`.
  2. Ejecutamos YOLOv8 (Inferencia).
  3. Si la confianza > 1%, activamos la capa NLP.
  4. Ejecutamos LangChain y devolvemos la respuesta unificada en un archivo `JSON`.

---

### Fase 5: Contenedores y CI/CD
**Conceptos Claves:** Dockerización y Continuous Integration (CI).

- **Docker:** Resuelve el clásico problema *"En mi máquina sí funciona"*. Empaquetamos todo el código, la versión exacta de Python y del sistema operativo (Debian/Ubuntu) dentro de una caja sellada (Contenedor). Donde sea que corra Docker, el sistema correrá igual.
- **Docker Compose:** Orquesta múltiples contenedores (aunque aquí es uno solo, permite escalar fácilmente a futuro añadiendo bases de datos como PostgreSQL).
- **GitHub Actions (CI):** Un robot en la nube que vigila nuestro código. Si un desarrollador sube código que rompe la aplicación (por ejemplo, con un error de sintaxis), las pruebas automatizadas (`pytest`) fallarán e impedirán que esa versión dañada llegue a producción.

---

### Fase 6: Monitoreo MLOps (Data Drift)
**Concepto Clave:** Data Drift (Desviación de Datos) y Model Decay.

Cuando un modelo de IA se despliega, el mundo exterior cambia pero el modelo no. 
- **Data Drift:** Ocurre cuando la distribución estadística de los datos de entrada en producción difiere de los datos de entrenamiento. (Ejemplo: Entrenamos con fotos claras de tazas, pero en producción los usuarios envían fotos oscuras o borrosas).
- **EvidentlyAI:** Realiza pruebas estadísticas clásicas (como Kolmogorov-Smirnov) comparando la tabla de logs actual vs el *Baseline* (histórico ideal). Si la distancia estadística es muy alta, genera una alerta para que los Ingenieros de Datos vuelvan a la "Fase 2" y re-entrenen al modelo con los nuevos datos.
