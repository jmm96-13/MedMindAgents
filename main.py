import time
import logging
from typing import TypedDict, Literal, List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END

LLM_MODEL = "qwen3:8b"
EMBEDDING_MODEL = "nomic-embed-text"
TEMPERATURE = 0.5

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("assitant")

llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)


"""knowledge_base = [       AQUI VA EL INDEXADOaaa
    Document(
        page_content=(
            "Para reiniciar el servidor VPN corporativo, ejecuta el comando "
            "'sudo systemctl restart openvpn' en la terminal. Si el problema "
            "persiste, verifica que el puerto 1194 UDP esté abierto en el firewall."
        ),
        metadata={"source": "manual_vpn", "topic": "redes"},
    ),
    Document(
        page_content=(
            "Cuando la aplicación se cierra inesperadamente al abrir, suele deberse "
            "a un caché corrupto. Solución: borra la carpeta de caché en "
            "%AppData%/MiApp/cache y reinicia. Como alternativa, reinstala la app."
        ),
        metadata={"source": "manual_app", "topic": "software"},
    ),
    Document(
        page_content=(
            "El error 'Connection timeout' al conectarse a la base de datos indica "
            "que el host no es alcanzable. Revisa la cadena de conexión, las "
            "credenciales y que el servicio PostgreSQL esté activo en el puerto 5432."
        ),
        metadata={"source": "manual_db", "topic": "base_de_datos"},
    ),
    Document(
        page_content=(
            "Para configurar el correo en Outlook usa el servidor IMAP "
            "imap.miempresa.com puerto 993 con SSL, y SMTP smtp.miempresa.com "
            "puerto 587 con TLS."
        ),
        metadata={"source": "manual_correo", "topic": "correo"},
    ),
]"""

# Dividir documentos en chunks (importante para documentos largos)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(knowledge_base)

# Crear / cargar el vector store persistente
vector_store = Chroma(
    collection_name="soporte_tecnico",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # persiste en disco
)

# Indexar (en producción harías esto una sola vez, no en cada arranque)
# Usar ids deterministas para poder reindexar sin duplicar
ids = [f"{chunk.metadata['source']}_{i}" for i, chunk in enumerate(chunks)]
vector_store.add_documents(chunks, ids=ids)
logger.info(f"Indexados {len(chunks)} chunks en ChromaDB")

# Crear el retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # recupera los 2 chunks más relevantes
)

# ----------------------------------------------------------
# 3. Estado compartido del grafo
# ----------------------------------------------------------
class SupportState(TypedDict):
    question: str
    category: Literal["tecnica", "facturacion", "general"]
    context: List[str]   # fragmentos recuperados por RAG
    answer: str

# ----------------------------------------------------------
# 4. Prompts (LangChain)
# ----------------------------------------------------------
classify_prompt = ChatPromptTemplate.from_template(
    """Clasifica la consulta en UNA categoría: tecnica, facturacion o general.
    Responde SOLO con la palabra.

    Consulta: {question}"""
)

# Prompt RAG: usa el contexto recuperado
rag_prompt = ChatPromptTemplate.from_template(
    """Eres un experto en soporte técnico. Responde la pregunta usando
    ÚNICAMENTE el siguiente contexto. Si el contexto no contiene la respuesta,
    indícalo honestamente.

    Contexto:
    {context}

    Pregunta: {question}

    Respuesta:"""
)

facturacion_prompt = ChatPromptTemplate.from_template(
    "Eres un agente de facturación. Atiende amablemente: {question}"
)

general_prompt = ChatPromptTemplate.from_template(
    "Responde de forma educada y breve: {question}"
)

# ----------------------------------------------------------
# 5. Nodos del grafo (LangGraph)
# ----------------------------------------------------------
def classify_node(state: SupportState) -> SupportState:
    chain = classify_prompt | llm
    result = chain.invoke({"question": state["question"]})
    category = result.content.strip().lower()
    if category not in ("tecnica", "facturacion", "general"):
        category = "general"
    logger.info(f"Clasificado como: {category}")
    return {"category": category}

def retrieve_node(state: SupportState) -> SupportState:
    """NODO RAG: recupera contexto relevante desde ChromaDB."""
    docs = retriever.invoke(state["question"])
    context = [doc.page_content for doc in docs]
    logger.info(f"Recuperados {len(context)} fragmentos de ChromaDB")
    return {"context": context}

def generate_rag_node(state: SupportState) -> SupportState:
    """NODO RAG: genera la respuesta usando el contexto recuperado."""
    context_text = "\n\n".join(state["context"])
    chain = rag_prompt | llm
    answer = chain.invoke({
        "context": context_text,
        "question": state["question"]
    }).content
    return {"answer": answer}

def facturacion_node(state: SupportState) -> SupportState:
    chain = facturacion_prompt | llm
    answer = chain.invoke({"question": state["question"]}).content
    return {"answer": answer}

def general_node(state: SupportState) -> SupportState:
    chain = general_prompt | llm
    answer = chain.invoke({"question": state["question"]}).content
    return {"answer": answer}

# ----------------------------------------------------------
# 6. Enrutamiento condicional
# ----------------------------------------------------------
def route_by_category(state: SupportState) -> str:
    return state["category"]

# ----------------------------------------------------------
# 7. Construir el grafo (LangGraph)
# ----------------------------------------------------------
builder = StateGraph(SupportState)

builder.add_node("classify", classify_node)
builder.add_node("retrieve", retrieve_node)          # paso RAG 1
builder.add_node("generate_rag", generate_rag_node)  # paso RAG 2
builder.add_node("facturacion", facturacion_node)
builder.add_node("general", general_node)

builder.add_edge(START, "classify")

# Enrutamiento: técnica entra al pipeline RAG
builder.add_conditional_edges(
    "classify",
    route_by_category,
    {
        "tecnica": "retrieve",        # ► va al flujo RAG
        "facturacion": "facturacion",
        "general": "general",
    }
)

# Pipeline RAG en cadena: retrieve → generate_rag → END
builder.add_edge("retrieve", "generate_rag")
builder.add_edge("generate_rag", END)
builder.add_edge("facturacion", END)
builder.add_edge("general", END)

graph = builder.compile()

# ----------------------------------------------------------
# 8. HARNESS: motor de pruebas con métricas y validación
# ----------------------------------------------------------
class TestHarness:
    def __init__(self, compiled_graph):
        self.graph = compiled_graph
        self.results = []

    def run_case(self, question: str, expected_category: str = None,
                 must_contain: str = None):
        start = time.perf_counter()
        try:
            output = self.graph.invoke({"question": question})
            latency = time.perf_counter() - start

            answer = output.get("answer", "")
            category = output.get("category")

            # Validaciones automáticas
            answer_ok = bool(answer.strip())
            category_ok = (expected_category is None
                           or category == expected_category)
            # Verificar que la respuesta RAG contenga info esperada
            content_ok = (must_contain is None
                          or must_contain.lower() in answer.lower())

            record = {
                "question": question,
                "category": category,
                "expected_category": expected_category,
                "context_used": output.get("context", []),
                "answer": answer,
                "latency_s": round(latency, 3),
                "passed": answer_ok and category_ok and content_ok,
                "checks": {
                    "answer_ok": answer_ok,
                    "category_ok": category_ok,
                    "content_ok": content_ok,
                },
            }
        except Exception as e:
            record = {"question": question, "error": str(e), "passed": False}
            logger.error(f"Fallo en '{question}': {e}")

        self.results.append(record)
        return record

    def report(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get("passed"))
        avg_latency = (
            sum(r.get("latency_s", 0) for r in self.results) / total
            if total else 0
        )

        print("\n" + "=" * 60)
        print("REPORTE DEL HARNESS (RAG + Ollama + Chroma)")
        print("=" * 60)
        for r in self.results:
            status = "✅ PASS" if r.get("passed") else "❌ FAIL"
            print(f"\n{status} | {r['question']}")
            if "error" in r:
                print(f"  Error: {r['error']}")
            else:
                print(f"  Categoría: {r['category']} "
                      f"(esperada: {r['expected_category']})")
                print(f"  Latencia: {r['latency_s']}s")
                # Mostrar el contexto recuperado por RAG (si lo hubo)
                if r.get("context_used"):
                    print(f"  Fragmentos RAG recuperados: "
                          f"{len(r['context_used'])}")
                print(f"  Checks: {r['checks']}")
                print(f"  Respuesta: {r['answer'][:120]}...")
        print("\n" + "-" * 60)
        print(f"Total: {total} | Pasaron: {passed} | "
              f"Fallaron: {total - passed}")
        print(f"Tasa de éxito: {passed/total*100:.1f}%")
        print(f"Latencia promedio: {avg_latency:.3f}s")
        print("=" * 60)


# ----------------------------------------------------------
# 9. Ejecutar el harness con casos de prueba
# ----------------------------------------------------------
if __name__ == "__main__":
    harness = TestHarness(graph)

    casos = [
        # (pregunta, categoría_esperada, texto_que_debe_aparecer)
        ("¿Cómo reinicio el servidor VPN?", "tecnica", "openvpn"),
        ("La aplicación se cierra al abrir, ¿qué hago?", "tecnica", "caché"),
        ("¿Por qué me cobraron dos veces este mes?", "facturacion", None),
        ("¿Cuáles son sus horarios de atención?", "general", None),
        ("Tengo un error de Connection timeout en la base de datos",
         "tecnica", "5432"),
    ]

    for pregunta, categoria, contenido in casos:
        harness.run_case(pregunta, categoria, must_contain=contenido)

    harness.report()
