from pathlib import Path
import pandas as pd
import re

MIN_CHARS = 200
IN_PATH = Path("data/processed/unique_texts.parquet")
OUT_PATH = Path("data/processed/clean_texts.parquet")
df = pd.read_parquet(IN_PATH)

print(f"Dataset carregado: {IN_PATH}")
print(f"Carregados: {len(df)} textos únicos")
print(f"Colunas: {df.columns.tolist()}")
print(f"Textos vazios: {(df['texto'].str.len() == 0).sum()}")

# Removendo espaços múltiplos e espaços no início/fim
antes = df["texto"].str.contains(r"\s{2,}", regex=True).sum()
df["texto"] = df["texto"].str.replace(r"\s+", " ", regex=True).str.strip()
depois = df["texto"].str.contains(r"\s{2,}", regex=True).sum()
print(f"Textos com espaços múltiplos: {antes} -> {depois}")

# Removendo textos curtos
curtos = df["texto"].str.len() < MIN_CHARS
removidos = df[curtos]
print(f"Removidos por tamanho (< {MIN_CHARS} caracteres): {len(removidos)}")
print(f"Maior texto removido: {removidos['texto'].str.len().max()} caracteres")
df = df[~curtos].reset_index(drop=True)
print(f"Restam: {len(df)} textos")

# limpando títulos
df["formato"] = "texto"                                                  
df.loc[df["titulo"].str.startswith("Vídeo,"), "formato"] = "video"       
df.loc[df["titulo"].str.startswith("Áudio,"), "formato"] = "audio"      
print(f"\nFormato das páginas:\n{df['formato'].value_counts()}")        

# remocão de prefixos e sufixos nos títulos
df["titulo"] = df["titulo"].str.replace(r"Duration,\s*[\d,.:]+\s*$", "", regex=True)
df["titulo"] = df["titulo"].str.replace(r"^(?:Vídeo|Áudio),\s*(?:Em áudio\s*\|\s*)?", "", regex=True).str.strip()

print(f"\nTítulos ainda com 'Duration': {df['titulo'].str.contains('Duration').sum()}")
print(f"Títulos ainda com prefixo: {df['titulo'].str.match(r'^(Vídeo|Áudio),').sum()}")
print(f"Títulos vazios: {(df['titulo'].str.len() == 0).sum()}")
print(f"'Duration' dentro dos textos: {df['texto'].str.contains('Duration').sum()}")
print("Exemplos de títulos limpos:")
for x in df.loc[df["formato"] != "texto", "titulo"].head(5):
    print("  ", repr(x))

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PATH, index=False)

print(f"\nSalvo em: {OUT_PATH}")
print(f"Linhas: {len(df)} | Colunas: {df.columns.tolist()}")
print(f"\nNotícias por número de categorias:\n{df['n_categorias'].value_counts().sort_index()}")
print(f"\nCategoria única (treino/val/teste): {(df['n_categorias'] == 1).sum()}")
print(f"Ambíguas (avaliação à parte): {(df['n_categorias'] > 1).sum()}")