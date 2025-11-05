import pandas as pd
from sentence_transformers import SentenceTransformer

# 1) Cargar el modelo de embeddings local (no usa internet, no cobra nada)
print("Cargando modelo local...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2) Leer tu CSV
print("Cargando papers.csv...")
df = pd.read_csv("papers_clean.csv")

# Comprobamos que la columna summary exista
if "summary" not in df.columns:
    raise ValueError(" ERROR: Tu CSV debe tener una columna llamada 'summary'.")

# 3) Función para generar embeddings
def create_embedding(text):
    try:
        return model.encode(str(text)).tolist()
    except:
        return None

# 4) Crear los embeddings
df["embedding"] = df["summary"].apply(create_embedding)

# 5) Guardar resultado
df.to_csv("papers_with_embeddings.csv", index=False, encoding="utf-8")
print("Embeddings guardados en: papers_with_embeddings.csv")
