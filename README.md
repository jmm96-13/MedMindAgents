# MedMindAgents
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
