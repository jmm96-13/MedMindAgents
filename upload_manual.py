from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import json
import sys

embeddings = OllamaEmbeddings(model="nomic-embed-text")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# Conectar (o crear) un store persistente
store = Chroma(
    collection_name="manuales",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# Archivo que guarda la lista de PDFs ya indexados (evita reindexar)
indexed_file = Path("indexed_files.json")
if indexed_file.exists():
    try:
        indexed = set(json.loads(indexed_file.read_text(encoding="utf-8")))
    except Exception:
        indexed = set()
else:
    indexed = set()

# Indexar varios PDFs con metadatos enriquecidos (solo indexado)
for pdf in Path("./docs").glob("*.pdf"):
    if pdf.name in indexed:
        print(f"Saltando {pdf.name}: ya indexado")
        continue

    print(f"Indexando {pdf}")
    try:
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        chunks = splitter.split_documents(pages)

        # Añadir metadatos útiles y número de página si no existe
        print(f"Añadiendo metadatos a {pdf}")
        for i, c in enumerate(chunks):
            c.metadata["source"] = pdf.name
            c.metadata["categoria"] = "medicina"
            if "page" not in c.metadata:
                c.metadata["page"] = i + 1

        # Usar ids deterministas para poder reindexar sin duplicar
        ids = [f"{pdf.name}_{i}" for i in range(len(chunks))]
        store.add_documents(chunks, ids=ids)

        # Marcar como indexado y persistir la lista
        indexed.add(pdf.name)
        indexed_file.write_text(json.dumps(list(indexed), ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Indexado completado: {pdf.name} ({len(chunks)} chunks)")
    except Exception as e:
        print(f"Error indexando {pdf.name}: {e}", file=sys.stderr)

print("Indexado finalizado. Archivos indexados:", len(indexed))