# Triaje Clínico Orientativo — Sistema Multiagente IA

Sistema de triaje clínico basado en LangGraph + Ollama desarrollado por
**Pepi, Javier y Antonio** como proyecto del curso.

---

## Arquitectura

```
raw_chat_history (entrada)
        │
        ▼
┌─────────────────────┐
│  Javier             │  agent_javier.py
│  Extractor síntomas │  → extracted_symptoms
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Antonio            │  agent_antonio.py
│  Consultor RAG      │  → medical_context
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Pepi               │  agent_pepi.py
│  Evaluador urgencia │  → urgency_level
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Supervisor         │  supervisor.py
│  Informe SBAR       │  → final_report
└─────────────────────┘
         │
         ▼
    traces.jsonl  (log de ejecuciones)
```

## Niveles de urgencia (Manchester adaptado)

| Nivel | Color | Tiempo máximo |
|---|---|---|
| INMEDIATA | 🔴 | < 0 min |
| MUY URGENTE | 🟠 | < 10 min |
| URGENTE | 🟡 | < 30 min |
| NORMAL | 🟢 | < 120 min |
| NO URGENTE | 🔵 | < 240 min |

---

## Instalación

### 1. Requisitos previos
- Python 3.11+
- [Ollama](https://ollama.com) instalado y corriendo localmente

### 2. Clonar e instalar dependencias

```bash
git clone <url-del-repo>
cd triage-clinico

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Descargar el modelo

```bash
ollama pull llama3
```

### 4. (Opcional) Indexar documentos médicos en ChromaDB

Coloca tus PDFs médicos en la carpeta `./docs/` y ejecuta:

```bash
python agent_antonio.py
```

Esto crea la carpeta `./chroma_db/` con los vectores indexados.
Sin este paso, el sistema funciona en **modo simulado**.

### 5. Ejecutar

```bash
python main.py
```

---

## Estructura de archivos

```
triage-clinico/
├── state.py            # TriageState — objeto compartido entre agentes
├── agent_javier.py     # Agente 1: extractor de síntomas
├── agent_antonio.py    # Agente 2: consultor RAG (ChromaDB)
├── agent_pepi.py       # Agente 3: evaluador de urgencia
├── supervisor.py       # Supervisor: informe SBAR final (todos)
├── main.py             # Punto de entrada + construcción del grafo
├── requirements.txt    # Dependencias
├── traces.jsonl        # Log automático de ejecuciones
├── docs/               # (crear) PDFs médicos para indexar
└── chroma_db/          # (auto) Base vectorial generada por Antonio
```

---

## Quién hace qué

| Fichero | Responsable |
|---|---|
| `agent_javier.py` | Javier |
| `agent_antonio.py` | Antonio |
| `agent_pepi.py` | Pepi |
| `supervisor.py` | Pepi + Javier + Antonio |
| `state.py`, `main.py` | Todos |
