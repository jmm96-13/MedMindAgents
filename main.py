import time
import logging
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel
from typing import TypedDict, List, Optional
from schemas import FinalizarResponse
import sessions
from contextlib import asynccontextmanager


LLM_MODEL = "qwen3:8b"
EMBEDDING_MODEL = "nomic-embed-text"
TEMPERATURE = 0.5

BASE_DIR = Path(__file__).resolve().parent


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    mensaje: str


class ChatResponse(BaseModel):
    session_id: str
    respuesta: str


class SessionDict(TypedDict):
    activa: bool
    nivel: int
    motivo_escalado: str


SESSIONS: dict[str, SessionDict] = {}


def crear_sesion() -> str:
    session_id = str(int(time.time() * 1000))
    SESSIONS[session_id] = {"activa": True, "nivel": 1, "motivo_escalado": "", "historial": []}
    return session_id


def get_sesion(session_id: str) -> Optional[SessionDict]:
    return SESSIONS.get(session_id)


def registrar_turno(session_id: str, mensaje: str, respuesta: str, nivel: int = 1) -> None:
    if session_id in SESSIONS:
        SESSIONS[session_id]["nivel"] = nivel
        historial = SESSIONS[session_id].setdefault("historial", [])
        historial.append({"pregunta": mensaje, "respuesta": respuesta, "nivel": nivel})


RECURSOS: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Compilando el grafo de triaje...")
    RECURSOS["grafo"] = build_graph()
    logger.info("Listo. Servidor preparado.")
    yield
    RECURSOS.clear()


app = FastAPI(title="Triaje Médico - Chatbot", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or crear_sesion()
    sesion = get_sesion(session_id)
    if sesion is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    if not sesion["activa"]:
        raise HTTPException(status_code=409, detail="La sesión ya está finalizada")

    estado_inicial = {
        "pregunta": req.mensaje,
        "respuesta": "",
        "symptoms": [],
        "context": [],
        "causes": "",
        "categoria": "",
        "recommended_specialists": [],
    }
    resultado = RECURSOS["grafo"].invoke(estado_inicial)

    registrar_turno(session_id, req.mensaje, resultado["respuesta"])
    return ChatResponse(
        session_id=session_id,
        respuesta=resultado["respuesta"],
    )


@app.post("/session/{session_id}/finalizar", response_model=FinalizarResponse)
def finalizar(session_id: str):
    sesion = sessions.finalizar(session_id)
    if sesion is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    num_turnos = len(sesion.get("historial", []))
    mensaje_despedida = (
        f"¡Muchas gracias por utilizar el triaje médico!\n\n"
        f"Hemos registrado tu consulta. Recuerda:\n"
        f"• Seguir las recomendaciones de los especialistas\n"
        f"• Acudir a urgencias si experimentas síntomas graves\n"
        f"• Mantener un registro de tu historial médico\n\n"
        f"Te deseamos una pronta recuperación. 💚\n\n"
        f"(Sesión completada - {num_turnos} turno(s) registrado(s))"
    )

    return FinalizarResponse(
        session_id=session_id,
        mensaje=mensaje_despedida,
        turnos=num_turnos,
    )


@app.get("/session/{session_id}")
def estado_sesion(session_id: str):
    sesion = sessions.get_sesion(session_id)
    if sesion is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    return sesion


# ----------------------------------------------------------
# Logging y modelos
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("triaje")

llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name="manuales",
    embedding_function=embeddings,
    persist_directory=str(BASE_DIR / "chroma_db"),
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8},
)


# ----------------------------------------------------------
# Estado del grafo
# ----------------------------------------------------------
class SupportState(TypedDict, total=False):
    pregunta: str
    respuesta: str
    symptoms: List[str]
    context: List[str]
    causes: str
    categoria: str
    recommended_specialists: List[str]


# ----------------------------------------------------------
# Prompts
# ----------------------------------------------------------
prompt_clasificador = ChatPromptTemplate.from_template(
    """Clasifica el mensaje del usuario en UNA de estas categorías:

- SINTOMAS: el usuario describe síntomas físicos o psicológicos y quiere orientación médica.
- GENERAL: saludos, presentaciones, charla, agradecimientos, despedidas o cualquier cosa
  que NO sea una descripción de síntomas médicos.

Responde EXACTAMENTE con una sola palabra: SINTOMAS o GENERAL. Nada más.

Mensaje: {pregunta}

Categoría:"""
)

prompt_recepcion = ChatPromptTemplate.from_template(
    """Eres el agente de recepción de un servicio de triaje médico orientativo.
Tu trabajo es atender el inicio de la conversación: saluda con amabilidad,
responde a la charla cordial (presentaciones, agradecimientos, despedidas) e
invita al usuario a describir sus síntomas.

Si el usuario no ha descrito síntomas todavía, anímale a hacerlo.
Sé breve, cercano y natural.

Mensaje del usuario: {pregunta}

Respuesta:"""
)

prompt_extraer_sintomas = ChatPromptTemplate.from_template(
    """Extrae una lista de SÍNTOMAS del siguiente mensaje médico.
Devuelve los síntomas separados por comas, en una sola línea, sin explicaciones adicionales.

Mensaje: {pregunta}

Síntomas:"""
)

prompt_causas = ChatPromptTemplate.from_template(
    """Eres un asistente de triaje médico. Tu tarea es identificar POSIBLES CAUSAS
de los síntomas descritos, basándote ÚNICAMENTE en el contexto proporcionado
(fuente: manuales de diagnóstico médico).

No inventes información fuera del contexto.
Responde en formato de lista numerada: cada causa con una breve justificación
y la referencia de la fuente entre corchetes.

Contexto:
{context}

Síntomas del paciente: {symptoms}

Posibles causas:"""
)


# ----------------------------------------------------------
# Diccionarios de especialistas (ampliados)
# ----------------------------------------------------------
SPECIALIST_KEYWORDS = {
    "cardiología": [
        "infarto", "angina", "dolor torácico", "dolor en el pecho", "presión en el pecho",
        "opresión torácica", "dolor pectoral", "palpitaciones", "taquicardia", "bradicardia",
        "arritmia", "fibrilación", "latidos irregulares", "corazón acelerado", "corazón lento",
        "insuficiencia cardíaca", "edema en piernas", "hinchazón de tobillos", "cianosis",
        "labios morados", "dedos morados", "soplo cardíaco", "dolor irradiado al brazo",
        "dolor irradiado a la mandíbula",
    ],
    "neumología": [
        "dificultad respiratoria", "dificultad para respirar", "disnea", "ahogo",
        "falta de aire", "sensación de asfixia", "respiración entrecortada",
        "tos", "tos seca", "tos con flemas", "tos crónica", "tos nocturna",
        "tos con sangre", "expectoración", "esputo", "sibilancias", "pitos al respirar",
        "ronquidos", "estridor", "asma", "bronquitis", "neumonía", "enfisema",
        "apnea del sueño", "dolor al respirar", "pleuresía",
    ],
    "neurología": [
        "cefalea", "dolor de cabeza", "migraña", "jaqueca", "mareo", "vértigo",
        "desequilibrio", "inestabilidad al caminar", "pérdida de consciencia", "desmayo",
        "síncope", "convulsión", "epilepsia", "confusión", "desorientación",
        "pérdida de memoria", "amnesia", "temblor", "rigidez muscular", "espasmos",
        "parálisis", "paresia", "debilidad en brazo", "debilidad en pierna",
        "hormigueo", "entumecimiento", "visión doble", "diplopía",
        "dificultad para hablar", "disartria", "afasia", "ictus", "derrame cerebral",
        "esclerosis", "neuropatía",
    ],
    "traumatología": [
        "fractura", "hueso roto", "dolor óseo", "dolor articular", "artralgia",
        "artritis", "artrosis", "luxación", "esguince", "distensión",
        "dolor de rodilla", "dolor de cadera", "dolor de hombro", "dolor de codo",
        "dolor de muñeca", "dolor de tobillo", "dolor de columna", "escoliosis",
        "hernia discal", "ciática", "dolor lumbar", "lumbago", "contractura",
        "tendinitis", "rotura de ligamento", "menisco", "bursitis",
    ],
    "gastroenterología": [
        "dolor abdominal", "dolor de estómago", "dolor de barriga", "cólico",
        "dolor epigástrico", "ardor de estómago", "acidez", "náuseas", "vómitos",
        "vómito", "arcadas", "regurgitación", "diarrea", "estreñimiento",
        "heces blandas", "heces duras", "sangre en heces", "heces negras", "melena",
        "hinchazón abdominal", "distensión", "gases", "flatulencia", "eructos",
        "pérdida de apetito", "disfagia", "dificultad para tragar",
        "ictericia", "color amarillo piel", "hepatitis", "úlcera",
    ],
    "ginecología": [
        "sangrado vaginal", "hemorragia vaginal", "menstruación abundante",
        "reglas irregulares", "amenorrea", "falta de menstruación",
        "sangrado fuera de ciclo", "dolor pélvico", "dolor menstrual", "dismenorrea",
        "dolor ovárico", "dolor vaginal", "dolor durante relaciones", "dispareunia",
        "flujo vaginal", "picor vaginal", "ardor vaginal", "vulvitis",
        "bulto en mama", "dolor de mama", "mastitis", "pezón hundido",
        "secreción del pezón",
    ],
    "urología": [
        "dolor al orinar", "escozor al orinar", "disuria", "ardor al orinar",
        "sangre en orina", "hematuria", "orina oscura", "orina turbia",
        "frecuencia urinaria", "necesidad urgente de orinar", "incontinencia",
        "retención de orina", "dificultad para orinar", "dolor renal",
        "cólico renal", "cólico nefrítico", "piedras en el riñón", "cálculos renales",
        "dolor de próstata", "prostatitis", "chorro de orina débil",
    ],
    "dermatología": [
        "erupción", "erupción cutánea", "rash", "sarpullido", "manchas en la piel",
        "lesiones cutáneas", "ampollas", "vesículas", "pústulas", "costras",
        "urticaria", "prurito", "picor en la piel", "escozor cutáneo",
        "piel seca", "descamación", "psoriasis", "eczema", "dermatitis",
        "rojez", "eritema", "enrojecimiento", "lunar", "nevus", "cambio en lunar",
        "bulto en la piel", "quiste", "acné", "forúnculo",
        "caída del cabello", "alopecia", "uñas frágiles",
    ],
    "oftalmología": [
        "pérdida de visión", "visión borrosa", "visión doble", "diplopía",
        "moscas volantes", "destellos", "fotopsias", "visión reducida",
        "ojo rojo", "conjuntivitis", "dolor ocular", "ardor en ojos",
        "picor en ojos", "lagrimeo excesivo", "ojo seco",
        "párpado caído", "ptosis", "orzuelo", "chalazión",
        "glaucoma", "cataratas", "presión ocular",
    ],
    "otorrinolaringología": [
        "dolor de oído", "otalgia", "otitis", "pérdida de audición", "sordera",
        "pitidos en los oídos", "tinnitus", "acúfenos", "tapón de cera",
        "congestión nasal", "moco", "rinitis", "sinusitis", "dolor facial",
        "sangrado nasal", "epistaxis", "pérdida del olfato", "anosmia",
        "dolor de garganta", "faringitis", "amigdalitis", "ronquera",
        "disfonía", "nódulos en cuello",
    ],
    "endocrinología": [
        "bocio", "nódulo tiroideo", "hipotiroidismo", "hipertiroidismo",
        "intolerancia al frío", "intolerancia al calor",
        "sed excesiva", "polidipsia", "orina frecuente", "poliuria",
        "hambre excesiva", "hipoglucemia", "glucosa alta",
        "aumento de peso inexplicable", "pérdida de peso inexplicable",
        "fatiga crónica", "cansancio extremo", "sudoración excesiva", "hiperhidrosis",
    ],
    "reumatología": [
        "artritis reumatoide", "articulaciones inflamadas", "rigidez matutina",
        "dolor en varias articulaciones", "poliartralgia", "lupus",
        "fibromialgia", "dolor muscular generalizado", "mialgia difusa",
        "gota", "dolor en dedo gordo del pie",
    ],
    "psiquiatría / psicología": [
        "depresión", "tristeza persistente", "falta de motivación",
        "ansiedad", "nerviosismo excesivo", "ataques de pánico", "pánico",
        "insomnio", "dificultad para dormir", "hipersomnia",
        "cambios de humor", "irritabilidad extrema", "agresividad",
        "pensamientos obsesivos", "alucinaciones", "paranoia", "delirios",
        "pensamientos de hacerse daño", "ideación suicida",
        "trastorno alimentario", "anorexia", "bulimia", "fobia",
    ],
    "hematología": [
        "anemia", "palidez", "sangrado fácil", "moratones sin causa",
        "hematomas espontáneos", "ganglios inflamados", "linfoma", "leucemia",
        "plaquetas bajas", "trombosis", "tromboflebitis",
    ],
    "infectología": [
        "fiebre persistente", "escalofríos", "sudores nocturnos",
        "infección generalizada", "sepsis", "picadura de garrapata",
        "VIH", "sida", "infecciones de repetición",
    ],
}

SEVERE_KEYWORDS = [
    "parada cardíaca", "paro cardíaco", "paro respiratorio", "parada respiratoria",
    "colapso", "pérdida de consciencia", "inconsciencia",
    "hemorragia", "sangrado profuso", "sangrado abundante", "sangrado incontrolable",
    "ictus", "derrame cerebral", "convulsiones repetidas", "parálisis súbita",
    "dificultad para hablar de repente", "confusión repentina",
    "asfixia", "no puede respirar", "labios morados", "cianosis",
    "dolor torácico intenso", "dolor de pecho muy fuerte",
    "reacción alérgica grave", "anafilaxia",
    "quemadura grave", "traumatismo grave", "fractura expuesta",
]


# ----------------------------------------------------------
# Nodos del grafo
# ----------------------------------------------------------
def classify_chat_node(state: SupportState) -> SupportState:
    pregunta = state.get("pregunta", "")
    chain = prompt_clasificador | llm
    # qwen3 a veces añade <think>…</think> antes de responder; nos quedamos solo con
    # la última palabra en mayúsculas para ser robustos ante ese formato.
    raw = chain.invoke({"pregunta": pregunta}).content.strip()
    # Buscar la primera ocurrencia de SINTOMAS o GENERAL en el texto
    upper = raw.upper()
    if "SINTOMAS" in upper or "SÍNTOMAS" in upper:
        categoria = "SINTOMAS"
    else:
        categoria = "GENERAL"

    logger.info(f"Clasificación: {categoria}")
    state["categoria"] = categoria

    if categoria == "SINTOMAS":
        chain2 = prompt_extraer_sintomas | llm
        sintomas_text = chain2.invoke({"pregunta": pregunta}).content.strip()
        sintomas = [s.strip() for s in sintomas_text.split(",") if s.strip()]
        state["symptoms"] = sintomas
        logger.info(f"Síntomas extraídos: {sintomas}")

    return state


def reception_node(state: SupportState) -> SupportState:
    chain = prompt_recepcion | llm
    respuesta = chain.invoke({"pregunta": state.get("pregunta", "")}).content.strip()
    state["respuesta"] = respuesta
    return state


def find_causes_node(state: SupportState) -> SupportState:
    """RAG: recupera fragmentos de Chroma y pide al LLM posibles causas.

    IMPORTANTE: mutamos el estado en lugar de devolver un dict nuevo,
    para no perder los campos 'symptoms' y 'pregunta' entre nodos.
    """
    symptoms_text = ", ".join(state.get("symptoms", []))
    docs = retriever.invoke(symptoms_text)
    context = [
        f"[{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    ]
    logger.info(f"Recuperados {len(context)} fragmentos de ChromaDB")

    context_text = "\n\n".join(context) if context else "(sin contexto en la base de datos)"
    chain = prompt_causas | llm
    causes_text = chain.invoke({"context": context_text, "symptoms": symptoms_text}).content

    # ✅ Mutamos el estado existente (no devolvemos dict nuevo)
    state["context"] = context
    state["causes"] = causes_text
    return state


def evaluate_specialist_node(state: SupportState) -> SupportState:
    causes_text = state.get("causes", "")
    causes_lower = causes_text.lower()

    recommended_specialists = set()

    # Primero: urgencias si hay síntoma grave
    if any(kw.lower() in causes_lower for kw in SEVERE_KEYWORDS):
        recommended_specialists.add("🚨 urgencias (acudir de inmediato)")

    # Luego: especialistas por palabras clave
    for specialist, keywords in SPECIALIST_KEYWORDS.items():
        if any(kw.lower() in causes_lower for kw in keywords):
            recommended_specialists.add(specialist)

    # También buscar en los síntomas directamente (por si las causas son breves)
    symptoms_text = ", ".join(state.get("symptoms", [])).lower()
    for specialist, keywords in SPECIALIST_KEYWORDS.items():
        if any(kw.lower() in symptoms_text for kw in keywords):
            recommended_specialists.add(specialist)

    if not recommended_specialists:
        recommended_specialists.add("medicina interna / médico de familia")

    state["recommended_specialists"] = list(recommended_specialists)

    symptoms_str = ", ".join(state.get("symptoms", []))
    specialists_str = "\n".join(f"  • {s}" for s in sorted(state["recommended_specialists"]))

    state["respuesta"] = (
        f"Basándome en tus síntomas ({symptoms_str}):\n\n"
        f"**Posibles causas:**\n{causes_text}\n\n"
        f"**Especialistas recomendados:**\n{specialists_str}\n\n"
        f"Por favor, acude a un centro de salud para una evaluación profesional."
    )
    return state


# ----------------------------------------------------------
# Construcción del grafo
# ----------------------------------------------------------
def build_graph():
    builder = StateGraph(SupportState)

    builder.add_node("classify_chat", classify_chat_node)
    builder.add_node("reception", reception_node)
    builder.add_node("find_causes", find_causes_node)
    builder.add_node("evaluate_specialist", evaluate_specialist_node)

    builder.add_edge(START, "classify_chat")
    builder.add_conditional_edges(
        "classify_chat",
        lambda s: s.get("categoria", "GENERAL"),
        path_map={"GENERAL": "reception", "SINTOMAS": "find_causes"},
    )
    builder.add_edge("reception", END)
    builder.add_edge("find_causes", "evaluate_specialist")
    builder.add_edge("evaluate_specialist", END)

    return builder.compile()