import time
import logging
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
from pydantic import BaseModel
from typing import TypedDict, Literal, List, Optional
from schemas import FinalizarResponse
import sessions
from contextlib import asynccontextmanager
import os



LLM_MODEL = "gemma3:4b"
EMBEDDING_MODEL = "nomic-embed-text"
TEMPERATURE = 0.5

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


BASE_DIR = Path(__file__).resolve().parent.parent

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

# Recursos compartidos, inicializados en el lifespan.
RECURSOS: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Compilando el grafo de triaje...")
    RECURSOS["grafo"] = build_graph()
    logger.info("Listo. Servidor preparado.")
    yield
    RECURSOS.clear()


app = FastAPI(title="Triaje Médico - Chatbot" , lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str("./static")), name="static")

@app.get("/")
def index():
    return FileResponse(str("./static/index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}



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

    # 3. Ejecutar el grafo con el nivel actual de la sesión.
    estado_inicial = {
        "pregunta": req.mensaje,
        "es_python": False,
        "contexto": [],
        "respuesta": "",
    }
    resultado = RECURSOS["grafo"].invoke(estado_inicial)

    # 5. Registrar el turno y responder.
    sessions.registrar_turno(session_id, req.mensaje, resultado["respuesta"])
    return ChatResponse(
        session_id=session_id,
        respuesta=resultado["respuesta"],
    )

@app.post("/session/{session_id}/finalizar", response_model=FinalizarResponse)
def finalizar(session_id: str):
    """Cierra la sesión cuando el usuario ha recibido las respuestas del triaje."""
    sesion = sessions.finalizar(session_id)
    if sesion is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    # Mensaje de despedida personalizado
    num_turnos = len(sesion["historial"])
    mensaje_despedida = f"""¡Muchas gracias por utilizar el triaje médico!

Hemos registrado tu consulta. Recuerda:
• Seguir las recomendaciones de los especialistas
• Acudir a urgencias si experimentas síntomas graves
• Mantener un registro de tu historial médico

Te deseamos una pronta recuperación. 💚

(Sesión completada - {num_turnos} turno(s) registrado(s))"""
    
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


logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("assitant")

llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE, base_url=OLLAMA_HOST)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)

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
class SupportState(TypedDict, total=False):
    # Campos de entrada y salida
    pregunta: str           # el mensaje del usuario
    respuesta: str          # la respuesta del bot
    # Campos del flujo de síntomas
    symptoms: List[str]     # síntomas extraídos
    context: List[str]      # fragmentos recuperados por RAG
    causes: str             # posibles causas identificadas
    categoria: str          # SINTOMAS o GENERAL
    # Campos opcionales
    es_python: bool
    contexto: List[str]
    evaluated: bool
    recommended_specialists: List[str]

# Recepción: clasifica si el mensaje es una consulta sobre Python o charla general.
# Devuelve una sola etiqueta para que el enrutado sea barato y predecible 
prompt_clasificador = ChatPromptTemplate.from_template(
    """Clasifica el mensaje del usuario en UNA de estas categorías:

- SINTOMAS: el usuario aporta síntomas físicos o psicológicos y quiere saber posibles causas.
- GENERAL: saludos, presentaciones, charla, agradecimientos, despedidas o
  cualquier cosa que NO sea una consulta técnica sobre Python.

Responde EXACTAMENTE con una sola palabra: SINTOMAS o GENERAL. Nada más.

Mensaje: {pregunta}

Categoría:"""
)

# Recepción (nivel 0): da la bienvenida y conduce hacia una pregunta sobre sintomas.
# No responde dudas técnicas, solo charla cordial y amable.
prompt_recepcion = ChatPromptTemplate.from_template(
    """Eres el agente de recepción de un servicio de diagnóstico.
Tu trabajo es atender el inicio de la conversación: saluda con amabilidad,
responde a la charla cordial (presentaciones, agradecimientos, despedidas) e
invita al usuario a contarte su duda sobre síntomas.

No respondas preguntas técnicas: si el usuario todavía no ha preguntado nada
sobre síntomas, anímale a hacerlo. Sé breve, cercano y natural.

Mensaje del usuario: {pregunta}

Respuesta:"""
)

# Prompt para extraer una lista de síntomas desde el texto del usuario.
prompt_extraer_sintomas = ChatPromptTemplate.from_template(
    """Extrae una lista de SINTOMAS del siguiente mensaje. Devuelve los
    síntomas separados por comas, en una sola línea, sin explicaciones.

    Mensaje: {pregunta}

    Síntomas:"""
)


# ----------------------------------------------------------
# 4. Prompts (LangChain)
# ----------------------------------------------------------
# Prompt específico para extraer posibles causas de síntomas desde el contexto

prompt_causas = ChatPromptTemplate.from_template(
    """Eres un asistente que extrae POSIBLES CAUSAS de síntomas obtenidos por el prompt de recepción. 
        Tu tarea es analizar la lista de síntomas,
      a partir de ÚNICAMENTE el contexto proporcionado (fuente: manuales de diagnostico).
      No inventes información fuera del contexto.
        Responde en formato de lista enumerada con cada posible causa seguida
        de una breve razón y la referencia de la fuente entre corchetes. 
        
    Contexto:
    {context}

    Causas (separadas por comas o espacios): {causes}
        
    Respuesta:"""
)
    
prompt_especialista = ChatPromptTemplate.from_template(
    """Eres un asistente que evalúa a qué especialista acudir para cada causa encontrada y
    muestra la lista formateada. Si no se puede asignar, recomendar 'médico de familia'; 
    si la causa parece grave, recomendar 'urgencias'.

    Causas encontradas: {causes}

    Respuesta:"""
)

# ----------------------------------------------------------
# 5. Nodos del grafo (LangGraph)
# ----------------------------------------------------------

def classify_chat_node(state: SupportState) -> SupportState:
    """Clasifica el mensaje del usuario en SINTOMAS o GENERAL."""
    pregunta = state.get("pregunta", "")
    if not pregunta:
        raise ValueError("No hay pregunta en el estado")

    # Invocar el prompt de clasificación
    chain = prompt_clasificador | llm
    categoria = chain.invoke({"pregunta": pregunta}).content.strip().upper()
    logger.info(f"Clasificación del mensaje: {categoria}")
    state["categoria"] = categoria

    # Si es categoría SINTOMAS, extraer lista de síntomas y guardarlos en el estado
    if categoria == "SINTOMAS":
        chain2 = prompt_extraer_sintomas | llm
        sintomas_text = chain2.invoke({"pregunta": pregunta}).content.strip()
        # convertir a lista separando por comas y limpiando espacios
        sintomas = [s.strip() for s in sintomas_text.split(",") if s.strip()]
        state["symptoms"] = sintomas

    return state

def decide_next_node(state: SupportState) -> str:
    """Decide el siguiente nodo según la categoría del mensaje."""
    categoria = state.get("categoria", "")
    if categoria == "GENERAL":
        return "reception"
    elif categoria == "SINTOMAS":
        return "find_causes"
    else:
        raise ValueError(f"Categoría desconocida: {categoria}")


def reception_node(state: SupportState) -> SupportState:
    """Nodo de recepción: saluda y conduce hacia la pregunta sobre síntomas."""
    pregunta = state.get("pregunta", "")
    if not pregunta:
        raise ValueError("No hay pregunta en el estado")

    # Invocar el prompt de recepción
    chain = prompt_recepcion | llm
    respuesta = chain.invoke({"pregunta": pregunta}).content.strip()
    logger.info(f"Respuesta de recepción: {respuesta}")
    state["respuesta"] = respuesta
    return state


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
    chain = prompt_causas | llm
    causes_text = chain.invoke({"context": context_text, "causes": symptoms_text}).content
    return {"context": context, "causes": causes_text}



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
def evaluate_specialist_node(state: SupportState) -> SupportState:
    """Evalúa a qué especialista acudir según las causas encontradas."""
    causes_text = state.get("causes", "")
    if not causes_text:
        raise ValueError("No hay causas en el estado")

    # Evaluar especialista según palabras clave
    recommended_specialists = set()
    for specialist, keywords in SPECIALIST_KEYWORDS.items():
        if any(keyword.lower() in causes_text.lower() for keyword in keywords):
            recommended_specialists.add(specialist)

    # Si no se encuentra especialista, recomendar médico de familia
    if not recommended_specialists:
        recommended_specialists.add("médico de familia")

    # Evaluar si alguna causa parece grave
    is_severe = any(keyword.lower() in causes_text.lower() for keyword in SEVERE_KEYWORDS)
    if is_severe:
        recommended_specialists.add("urgencias")

    state["recommended_specialists"] = list(recommended_specialists)
    
    # Generar respuesta formateada para el usuario
    symptoms_str = ", ".join(state.get("symptoms", []))
    specialists_str = ", ".join(sorted(state["recommended_specialists"]))
    respuesta_text = f"""Basándome en tus síntomas ({symptoms_str}):

**Posibles causas:**
{causes_text}

**Especialistas recomendados:**
{specialists_str}

Por favor, acude a un centro de salud para una evaluación profesional."""
    
    state["respuesta"] = respuesta_text
    return state



# ----------------------------------------------------------
# 7. Construir el grafo simple (síntomas -> buscar causas -> imprimir)
# ----------------------------------------------------------
def build_graph() -> StateGraph:
    builder = StateGraph(SupportState)

    builder.add_node("classify_chat", classify_chat_node)
    builder.add_node("reception", reception_node)
    builder.add_node("find_causes", find_causes_node)
    builder.add_node("evaluate_specialist", evaluate_specialist_node)

    builder.add_edge(START, "classify_chat")
    # Aristas condicionales desde classify_chat según la categoría
    builder.add_conditional_edges(
        "classify_chat",
        lambda s: s.get("categoria", "GENERAL"),
        path_map={"GENERAL": "reception", "SINTOMAS": "find_causes"},
    )
    builder.add_edge("reception", END)
    builder.add_edge("find_causes", "evaluate_specialist")
    builder.add_edge("evaluate_specialist", END)

    return builder.compile()

