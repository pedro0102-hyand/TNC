from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/bbc_news_ptbr.parquet")
OUT_PATH = Path("data/processed/unique_texts.parquet")
df = pd.read_parquet(RAW_PATH)

# Agrupando por texto e agregando informações
unique = (df.groupby("texto", sort=False).agg(titulo=("titulo", "first"),categorias=("categoria", lambda s: sorted(set(s))),data=("data", "first"),link=("link", "first")).reset_index())
unique["n_categorias"] = unique["categorias"].map(len)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
unique.to_parquet(OUT_PATH, index=False)

# Informações sobre os textos únicos
print(f"Textos únicos: {len(unique)}")
print("\nTextos por número de categorias:")
print(unique["n_categorias"].value_counts().sort_index())
single = unique[unique["n_categorias"] == 1]
print(f"\nTextos com categoria única: {len(single)}")
print(single["categorias"].str[0].value_counts())