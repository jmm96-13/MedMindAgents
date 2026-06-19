# MedMindAgents
## Arquitectura
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

## Estructura de archivos
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
