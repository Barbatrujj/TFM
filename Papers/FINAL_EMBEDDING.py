import os
import pandas as pd
from openai import OpenAI

# API Key
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 1. Cargar el CSV
df = pd.read_csv("papers_clean.csv")

# 2. Función para generar un embedding de un texto
def embed_text(text):
    if pd.isna(text) or text.strip() == "":
        return None
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# 3. Aplicar embeddings a la columna summary
df["embedding"] = df["summary"].apply(embed_text)

# 4. Guardar nuevo CSV con embeddings
df.to_csv("papers_with_embeddings.csv", index=False)

print("✅ Embeddings creados y guardados en papers_with_embeddings.csv")
