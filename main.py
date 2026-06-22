import time
import logging
from typing import TypedDict, Literal, List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import config
from . import config, indexing, sessions
from .graph import construir_grafo
from .schemas import ChatRequest, ChatResponse, FinalizarResponse


LLM_MODEL = "qwen3:8b"
EMBEDDING_MODEL = "nomic-embed-text"
TEMPERATURE = 0.5

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="Triaje Médico - Chatbot")

app.mount("/static", StaticFiles(directory=str("./static")), name="static")

@app.get("/")
def index():
    return FileResponse(str("./static/index.html"))

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Procesa un mensaje del usuario.

    Esta función es síncrona a propósito: ChatOllama.invoke bloquea, así que
    FastAPI la ejecuta en su threadpool sin congelar el event loop.
    """
    # 1. Crear sesión si es el primer mensaje.
    session_id = req.session_id or sessions.crear_sesion()
    sesion = sessions.get_sesion(session_id)
    if sesion is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    if not sesion["activa"]:
        raise HTTPException(status_code=409, detail="La sesión ya está finalizada")

    # 2. Feedback de insatisfacción sobre la respuesta anterior → escalar.
    if req.feedback == "insatisfecho":
        sessions.escalar(session_id, "feedback")

    # 3. Ejecutar el grafo con el nivel actual de la sesión.
    estado_inicial = {
        "pregunta": req.mensaje,
        "nivel": sesion["nivel"],
        "es_python": False,
        "contexto": [],
        "respuesta": "",
        "escalado": False,
        "motivo": sesion.get("motivo_escalado", ""),
    }
    resultado = RECURSOS["grafo"].invoke(estado_inicial)

    # 4. Si el grafo escaló (auto), persistir el nivel 2 en la sesión.
    if resultado.get("escalado") and sesion["nivel"] != 2:
        sessions.escalar(session_id, resultado.get("motivo", "auto"))

    # 5. Registrar el turno y responder.
    sessions.registrar_turno(
        session_id, req.mensaje, resultado["respuesta"], resultado["nivel"]
    )
    return ChatResponse(
        session_id=session_id,
        nivel=resultado["nivel"],
        respuesta=resultado["respuesta"],
        escalado=resultado.get("escalado", False),
        motivo=resultado.get("motivo", ""),
    )


logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("assitant")

llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)


# Crear / cargar el vector store persistente
vector_store = Chroma(
    collection_name="manuales",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# Parámetro de recuperación: valor recomendado de `k`
# Recomendación: para síntomas con muchas posibles causas, usar k=6..12.
# k más alto recupera más contexto pero puede introducir ruido; k=8 es
# un buen valor por defecto equilibrado.
RETRIEVE_K = 8

# Crear el retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": RETRIEVE_K}  # recupera los chunks más relevantes
)

# ----------------------------------------------------------
# 3. Estado compartido del grafo
# ----------------------------------------------------------
class SupportState(TypedDict):
    # `symptoms` puede contener uno o varios síntomas (lista de strings)
    symptoms: List[str]
    context: List[str]   # fragmentos recuperados por RAG
    causes: str

# ----------------------------------------------------------
# 4. Prompts (LangChain)
# ----------------------------------------------------------
# Prompt específico para extraer posibles causas de síntomas desde el contexto
causes_prompt = ChatPromptTemplate.from_template(
    """Eres un asistente que extrae POSIBLES CAUSAS de síntomas a partir
    ÚNICAMENTE del contexto proporcionado (fuente: libro de diagnóstico).
    No inventes información fuera del contexto. Responde en formato de lista
    enumerada con cada posible causa seguida de una breve razón y la
    referencia de la fuente entre corchetes.

    Contexto:
    {context}

    Síntomas (separados por comas o espacios): {symptoms}

    Respuesta:"""
)

# ----------------------------------------------------------
# 5. Nodos del grafo (LangGraph)
# ----------------------------------------------------------
def find_causes_node(state: SupportState) -> SupportState:
    """Buscar en Chroma los fragmentos relevantes y pedir al LLM que
    extraiga posibles causas de los síntomas."""
    symptoms_text = ", ".join(state.get("symptoms", []))
    # Recuperar documentos relevantes (usar texto de síntomas como query)
    docs = retriever.invoke(symptoms_text)
    context = [
        f"[{doc.metadata.get('source','?')} p.{doc.metadata.get('page','?')}]\n{doc.page_content}"
        for doc in docs
    ]
    logger.info(f"Recuperados {len(context)} fragmentos de ChromaDB para síntomas")

    context_text = "\n\n".join(context)
    chain = causes_prompt | llm
    causes_text = chain.invoke({"context": context_text, "symptoms": symptoms_text}).content
    return {"context": context, "causes": causes_text}


def _parse_causes_from_text(text: str) -> List[str]:
    lines = [l.strip() for l in text.splitlines()]
    causes = []
    for l in lines:
        if not l:
            continue
        # remover prefijos tipo '1.' '- ' '• '
        for p in ("- ", "• ", "* "):
            if l.startswith(p):
                l = l[len(p):].strip()
        # remover numeración
        if l[0:3].strip().rstrip('.').isdigit():
            # caso '1. Causa'
            parts = l.split('.', 1)
            if len(parts) > 1:
                l = parts[1].strip()
        # evitar líneas que no parecen causas (muy cortas)
        if len(l) >= 3:
            causes.append(l)
    return causes



SPECIALIST_KEYWORDS = {
    "cardiología": ["infarto", "angina", "dolor toracico", "palpitaciones"],
    "neumología": ["dificultad respiratoria", "disnea", "tos", "sibilancias"],
    "neurología": ["mareo", "convuls", "pérdida de consciencia", "cefalea", "dolor de cabeza"],
    "ginecología": ["sangrado vaginal", "dolor pélvico"],
    "digestivo": ["dolor abdominal", "náuseas", "vómito", "diarrea"],
    "dermatología": ["erupción", "rash", "urticaria", "prurito"],
    "urología": ["dolor lumbar", "hematouria", "disuria"],
}


SEVERE_KEYWORDS = ["parada", "paro", "hemorragia", "desmayo", "pérdida de consciencia", "sangrado profuso"]


def _assign_specialist_for_cause(cause: str) -> str:
    low = cause.lower()
    for severe in SEVERE_KEYWORDS:
        if severe in low:
            return "urgencias"
    for specialist, keys in SPECIALIST_KEYWORDS.items():
        for k in keys:
            if k in low:
                return specialist
    # si no se encuentra, recomendar médico de familia
    return "médico de familia"


def evaluate_specialist_node(state: SupportState) -> SupportState:
    """Evalúa a qué especialista acudir para cada causa encontrada y
    muestra la lista formateada. Si no se puede asignar, recomendar
    'médico de familia'; si la causa parece grave, recomendar 'urgencias'.
    """
    causes_text = state.get("causes", "")
    causes_list = _parse_causes_from_text(causes_text)

    results = []
    for c in causes_list:
        specialist = _assign_specialist_for_cause(c)
        needs_more_info = (specialist == "médico de familia")
        results.append({"cause": c, "specialist": specialist, "needs_more_info": needs_more_info})

    # Imprimir en formato legible: Causa -> Especialista (nota si necesita más info)
    print("\nPosibles causas y especialista recomendado:\n")
    for r in results:
        print(f"- Causa: {r['cause']}")
        print(f"  Especialista recomendado: {r['specialist']}")
        if r["needs_more_info"]:
            print("  Nota: asignación genérica; obtener más contexto si es posible.")
        print()

    # Si no se encontraron causas específicas, recomendar médico de familia
    if not results:
        print("No se identificaron causas concretas en el contexto. Recomendar: médico de familia o urgencias según la gravedad.")

    return {"evaluated": True}

# (Ya no se usa en el flujo simplificado)

# ----------------------------------------------------------
# 7. Construir el grafo simple (síntomas -> buscar causas -> imprimir)
# ----------------------------------------------------------
builder = StateGraph(SupportState)

builder.add_node("find_causes", find_causes_node)
builder.add_node("evaluate_specialist", evaluate_specialist_node)

builder.add_edge(START, "find_causes")
builder.add_edge("find_causes", "evaluate_specialist")
builder.add_edge("evaluate_specialist", END)

graph = builder.compile()

# ----------------------------------------------------------
# 8. Invocar el grafo desde __main__ (ejemplo sencillo)
# ----------------------------------------------------------
if __name__ == "__main__":
    # Probar el flujo: pasar síntomas y mostrar causas encontradas
    symptoms = ["dolor de garganta", "cefalea"]
    result = graph.invoke({"symptoms": symptoms})
    # El nodo evaluate_specialist ya imprime las causas y especialista; mostrar metadatos opcionales
    context_used = result.get("context", [])
    if context_used:
        print(f"Fragmentos recuperados: {len(context_used)}")
