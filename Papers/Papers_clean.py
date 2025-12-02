import pandas as pd
import re

df = pd.read_csv("Papers_Sumary.csv")

def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = re.sub(r'\[\d+\]|\(\d{4}\)', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'[“”‘’"•·–—…]', '', text)
    
    text = text.strip().lower()
    return text


df["title"] = df["title"].apply(clean_text)
df["summary"] = df["summary"].apply(clean_text)
df["keywords"] = df["keywords"].apply(clean_text)

df.to_csv("papers_clean.csv", index=False)
print("Clean CSV saved as papers_clean.csv")
