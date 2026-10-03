from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/processed/clean_texts.parquet")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)
print(f"Dataset carregado: {DATA_PATH}")
print(f"Shape do dataset: {df.shape}")
print(f"Colunas do dataset: {df.columns.tolist()}")

# Informações sobre as categorias
categorias = df.explode("categorias")
contagem_categorias = categorias["categorias"].value_counts()
print(f"Contagem de categorias: {contagem_categorias.shape[0]}")
print(f"Categorias: {contagem_categorias.tolist()}")

# textos exclusivos por categoria: só aparecem como uma categoria e não como combinação
categorias_exclusivas = df[df["categorias"].map(len) == 1].copy()
contagem_exclusiva = categorias_exclusivas.explode("categorias")["categorias"].value_counts()
print("\nTextos classificados exclusivamente como cada categoria:")
print(contagem_exclusiva)

# distribuição de categorias
distribuicao_categorias = pd.DataFrame({
    "textos": contagem_categorias,
    "percentual": contagem_categorias / len(df) * 100,
    "exclusivas": contagem_exclusiva.reindex(contagem_categorias.index, fill_value=0),
}).round(1)
print("\nDistribuição de categorias:")
print(distribuicao_categorias)

# quantidade de categorias por texto
quantidade_por_texto = df["categorias"].map(len)
contagem_por_quantidade = quantidade_por_texto.value_counts().sort_index()
distribuicao_por_quantidade = pd.DataFrame({
    "textos": contagem_por_quantidade,
    "percentual": contagem_por_quantidade / len(df) * 100,
}).round(1)
print("\nQuantidade de categorias por texto:")
print(distribuicao_por_quantidade)

# gráfico: textos exclusivos por categoria
plt.figure(figsize=(10, 6))
contagem_exclusiva.sort_values().plot(kind="barh", color="#2E8B57")
plt.title("Textos classificados exclusivamente por categoria")
plt.xlabel("Quantidade de textos")
plt.ylabel("Categoria")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda3_exclusivas_por_categoria.png", dpi=150)
plt.close()
print("\nGráfico salvo em: reports/figures/eda3_exclusivas_por_categoria.png")
