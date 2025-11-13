# TEST_EMBEDDING.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# Cargar CSV limpio
df = pd.read_csv("papers_clean.csv")
for col in ["summary", "keywords"]:
    if col not in df.columns:
        df[col] = ""
df.fillna("", inplace=True)
df["text"] = df["title"] + ". " + df["summary"] + " " + df["keywords"]

texts = df["text"].tolist()
metadatas = df[["title", "authors", "year", "keywords"]].to_dict(orient="records")

# Generar embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embs = model.encode(texts, show_progress_bar=len(texts)>50, convert_to_numpy=True, batch_size=32)

# Crear cliente persistente (ya guarda automáticamente)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="papers_color")

# Insertar documentos
collection.add(
    documents=texts,
    embeddings=embs.tolist(),
    metadatas=metadatas,
    ids=[f"doc_{i}" for i in range(len(texts))]
)

print("Base vectorial creada en ./chroma_db")

# Prueba de consulta
query = "efecto del color azul en la actividad cerebral"
q_emb = model.encode([query])[0].tolist()
res = collection.query(query_embeddings=[q_emb], n_results=3, include=["documents", "metadatas"])

print("\nResultados de prueba:")
for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
    print("Título:", meta["title"])
    print("Autores:", meta.get("authors"))
    print("Año:", meta.get("year"))
    print("Texto (inicio):", doc[:200], "...\n")
