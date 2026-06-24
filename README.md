# Triaje Clínico - Sistema Multiagente IA

Sistema de triaje clínico basado en LangGraph + Ollama desarrollado por
**Pepi, Javier y Antonio** como proyecto del curso.

---

## Arquitectura

```
mensaje del usuario (entrada)
        │
        ▼
┌─────────────────────┐
│  Clasificador       │  classify_chat_node
│  ¿Síntomas o charla?│  → categoria (SINTOMAS / GENERAL)
└────────┬────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌─────────────────────┐
│Recep.  │  │  find_causes_node   │
│Saluda  │  │  RAG sobre manuales │  → causes
└────────┘  └────────┬────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ evaluate_specialist  │
          │ Asigna especialista  │  → respuesta final
          └─────────────────────┘
                     │
                     ▼
              respuesta al usuario
```

### Ficheros principales

| Fichero | Responsable | Función |
|---|---|---|
| `main.py` | Todos | Grafo LangGraph + API FastAPI |
| `schemas.py` | Antonio | Modelos Pydantic |
| `sessions.py` | Antonio | Gestión de sesiones en memoria |
| `upload_manual.py` | Antonio | Indexar PDFs en ChromaDB |
| `static/index.html` | Javier | Interfaz web del chat |
| `Dockerfile` | Pepi | Imagen Docker de la app |
| `.dockerignore` | Pepi | Exclusiones del contexto Docker |
| `docker-compose.yml` | Pepi | Orquestación de contenedores |

---

## Instalación sin Docker

### 1. Requisitos previos
- Python 3.11+
- [Ollama](https://ollama.com) instalado y corriendo localmente

### 2. Clonar e instalar dependencias

```bash
git clone <url-del-repo>
cd MedMindAgents

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Descargar los modelos

```bash
ollama pull nomic-embed-text
ollama pull gemma3:4b
```

### 4. (Opcional) Indexar documentos médicos en ChromaDB

Coloca tus PDFs médicos en la carpeta `./docs/` y ejecuta:

```bash
python upload_manual.py
```

Esto crea la carpeta `./chroma_db/` con los vectores indexados.
Sin este paso, el sistema responde sin contexto de manuales.

### 5. Ejecutar

En una terminal arranca Ollama:
```bash
ollama serve
```

En otra terminal, con el entorno activado:
```bash
uvicorn main:app --reload
```

Abre el navegador en **http://localhost:8000**

---

## 6. Instalación con Docker

### Requisitos previos
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo

### Ficheros necesarios en la raíz del proyecto

**`Dockerfile`**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`.dockerignore`**

**`docker-compose.yml`**

### Cambios necesarios en `main.py`

Añadir `import os` al bloque de imports y agregar la variable `OLLAMA_HOST`
justo después de las constantes de modelo:

```python
import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
```

Modificar la inicialización de `llm` y `embeddings` para que usen esa variable:

```python
llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE, base_url=OLLAMA_HOST)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
```

### Pasos

**0. Abrir docker desktop**


**1. Construir y arrancar los contenedores**
```bash
docker compose up --build
```

**2. Descargar los modelos dentro del contenedor de Ollama**

Con los contenedores corriendo, en otra terminal:
```bash
docker exec -it ollama ollama pull nomic-embed-text
docker exec -it ollama ollama pull gemma3:4b
```

> `gemma3:4b` pesa ~5 GB. La descarga puede tardar varios minutos.

**3. Abrir en el navegador**

Cuando en los logs aparezca:
```
medmindagents | INFO:     Application startup complete.
```

Abre **http://localhost:8000**

**4. (Opcional) Indexar documentos médicos**

Con los contenedores corriendo:
```bash
docker exec -it medmindagents python upload_manual.py
```

**5. Parar los contenedores**
```bash
docker compose down
```

> Los modelos quedan guardados en el volumen `ollama_data` y no hay que volver a descargarlos.

---
