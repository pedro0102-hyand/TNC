import pandas as pd

DATASET_PATH = "data/raw/bbc_news.parquet"
data = pd.read_parquet(DATASET_PATH)

# verificando caracateristicas do titulo e do texto
data["titulo_chars"] = data["titulo"].str.len()
data["texto_chars"] = data["texto"].str.len()

data["titulo_words"] = data["titulo"].str.split().str.len()
data["texto_words"] = data["texto"].str.split().str.len()

# tamanho do titulo
print("\n=== TAMANHO DO TÍTULO (CARACTERES) ===")
print(data["titulo_chars"].describe())

print("\n=== TAMANHO DO TÍTULO (PALAVRAS) ===")
print(data["titulo_words"].describe())

# tamanho do texto
print("\n=== TAMANHO DO TEXTO (CARACTERES) ===")
print(data["texto_chars"].describe())

print("\n=== TAMANHO DO TEXTO (PALAVRAS) ===")
print(data["texto_words"].describe())

# tamanho do texto por categoria
print("\n=== TAMANHO DO TEXTO POR CATEGORIA (CARACTERES) ===")
print(data.groupby("categoria")["texto_chars"].agg(["count", "mean", "median", "min", "max"]).sort_values("mean", ascending=False))