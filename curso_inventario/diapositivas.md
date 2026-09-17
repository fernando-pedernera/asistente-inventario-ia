---
marp: true
theme: default
paginate: true
---

# 🤖 Proyecto End-to-End MLOps
## Asistente Inteligente de Inventario y Manuales
**Conceptos, Fases y Arquitectura**

---

# 🎯 El Objetivo (MVP)
Crear un sistema integral que demuestre conocimientos avanzados combinando:
1. **Visión por Computadora (YOLOv8)** para detección de objetos.
2. **Procesamiento de Lenguaje Natural (RAG)** para lectura de manuales técnicos.
3. **Ingeniería de Software** (APIs con FastAPI).
4. **DevOps & Monitoreo** (Docker, GitHub Actions, Data Drift).

---

# ⚙️ Fase 1 y 2: Preparación y Pipeline Visual
**Concepto Clave: Data Augmentation & Bounding Boxes**

- Las Redes Neuronales Profundas (Deep Learning) necesitan MUCHOS datos.
- ¿La solución? **Aumento de Datos**. Partimos de 5 fotos y aplicamos matemáticas (rotación, brillo) para multiplicarlas.
- Usamos **Roboflow** para dibujar *Bounding Boxes* y etiquetar dónde está exactamente el objeto (Taza), facilitando el aprendizaje supervisado.

---

# 🧠 Fase 3: RAG (Retrieval-Augmented Generation)
**El problema:** Los modelos grandes como Gemini inventan información (Alucinan) o no conocen los manuales privados de nuestra empresa.
**La solución (RAG):**
1. **Chunking:** Partimos nuestro manual en trozos pequeños.
2. **Embeddings:** Convertimos cada trozo en un Vector (números). Textos similares se agrupan matemáticamente cerca en el espacio de n-dimensiones.
3. **Vector Store:** Guardamos los vectores en FAISS.
4. Cuando el usuario pregunta, le pasamos al LLM **sólo** el párrafo matemáticamente más relevante.

---

# 👁️ Fase 3 (cont.): Visión Artificial (YOLO)
**Concepto Clave: You Only Look Once**

- Entrenamos la arquitectura **YOLOv8**. 
- A diferencia de métodos antiguos que escanean la imagen por pedazos, YOLO procesa toda la matriz de píxeles de una sola vez, logrando velocidad y precisión en tiempo real.
- Exportamos los pesos del entrenamiento a un archivo `.pt` (PyTorch).

---

# 🔌 Fase 4: API con FastAPI
**Concepto Clave: Orquestación y Microservicios**

- Necesitamos un "puente" para conectar el FrontEnd (Aplicación o Usuario) con los pesados modelos de IA.
- **FastAPI** recibe una imagen y texto simultáneamente.
- **Flujo:** Primero corre YOLO. ¿Hay taza? Sí -> Corre el flujo RAG. No -> Cancela operación. Todo empaquetado en un archivo `JSON` limpio.

---

# 🐳 Fase 5: Docker y CI/CD
**Concepto Clave: Escalabilidad y Despliegues Confiables**

- **Docker:** Construye una "caja" que incluye Linux + Python + Librerías + Nuestro Código. Así funciona idéntico en mi PC y en un Servidor AWS.
- **CI (Integración Continua):** Usamos GitHub Actions. Un robot ejecuta scripts de prueba (`pytest`) cada vez que alguien intenta subir código nuevo a la rama principal, evitando colapsos en producción.

---

# 📊 Fase 6: Monitoreo (Data Drift)
**Concepto Clave: El mundo cambia, el modelo no.**

- **Data Drift:** Cuando los datos de los usuarios en producción son estadísticamente diferentes a los datos con los que entrenamos.
- Usamos **EvidentlyAI** para comparar un histórico "Sano" (Baseline) con los logs actuales ("Anómalos").
- Si detecta anomalías (ej: la confianza de YOLO bajó del 80% al 10%), dispara alertas para alertar al equipo de Data Science que es hora de re-entrenar.

---

# 🎉 Arquitectura Final (Resumen)

1. **Usuario** -> (Imagen + Texto) -> **FastAPI**
2. **FastAPI** -> Llama a **YOLOv8** -> *(Detecta Objeto)*
3. **FastAPI** -> Busca en **FAISS** -> Le habla a **Google Gemini**
4. **Respuesta** -> Vuelve al **Usuario**
5. **Background** -> Guarda métricas -> **EvidentlyAI** detecta Drift.

*Gracias por su atención.*
