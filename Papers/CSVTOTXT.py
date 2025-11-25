# CSVTOTXT_FIXED.py
import pandas as pd
import os

df = pd.read_csv("papers_clean.csv")

# Crear carpeta para los txt si no existe
output_dir = "papers_txt"
os.makedirs(output_dir, exist_ok=True)

for idx, row in df.iterrows():
    # Nombre seguro para archivo: solo letras, números, guiones bajos
    title_clean = "".join(c for c in row["title"] if c.isalnum() or c in (" ", "_")).replace(" ", "_")
    if not title_clean:
        title_clean = f"paper_{idx}"

    filename = os.path.join(output_dir, f"{title_clean}.txt")

    # Texto que vamos a guardar: título + resumen + keywords
    text_content = row["title"] + "\n\n" + row["summary"]
    if "keywords" in row and pd.notna(row["keywords"]):
        text_content += "\n\nKeywords: " + row["keywords"]

    with open(filename, "w", encoding="utf-8") as f:
        f.write(text_content)

print(f"Archivos .txt generados en la carpeta '{output_dir}'")
