import pandas as pd
import re

# Load your CSV
df = pd.read_csv("Papers_Sumary.csv")

def clean_text(text):
    if pd.isna(text):
        return ""
    # Remove citation markers like [1], (2019), etc.
    text = re.sub(r'\[\d+\]|\(\d{4}\)', '', text)
    # Replace multiple spaces/newlines with one space
    text = re.sub(r'\s+', ' ', text)
    # Remove unwanted characters
    text = re.sub(r'[“”‘’"•·–—…]', '', text)
    # Optional: lowercase
    text = text.strip().lower()
    return text

# Clean relevant columns
df["title"] = df["title"].apply(clean_text)
df["summary"] = df["summary"].apply(clean_text)
df["keywords"] = df["keywords"].apply(clean_text)

# Save cleaned version
df.to_csv("papers_clean.csv", index=False)
print("Clean CSV saved as papers_clean.csv")
