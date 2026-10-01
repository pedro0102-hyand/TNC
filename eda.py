from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/raw/bbc_news_ptbr.parquet")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Carregar o dataset
df = pd.read_parquet(DATA_PATH)
print(f"Dataset carregado: {DATA_PATH}")

# Informações básicas
print(f"Shape do dataset: {df.shape}")
print(f"Colunas do dataset: {df.columns.tolist()}")
print(f"Tipos de dados das colunas:\n{df.dtypes}")
print(f"Valores nulos por coluna:\n{df.isnull().sum()}")

# Primeiros registros
print("Exemplo de registros do dataset:")
print(df.head(5))

# Distribuição das classes
print("Distribuição das classes:")
counts = df["categoria"].value_counts()
dist = pd.DataFrame({"quantidade": counts, "percentual": counts / len(df) * 100}).round(1)
print(dist)

# Gráfico: quantidade por classe
dist.plot(kind="bar", y="quantidade", legend=False)
plt.title("Distribuição das classes")
plt.xlabel("Categoria")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "classes_quantidade.png", dpi=150)
plt.close()

# Gráfico: percentual por classe
dist.plot(kind="bar", y="percentual", legend=False)
plt.title("Distribuição das classes (percentual)")
plt.xlabel("Categoria")
plt.ylabel("Percentual (%)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "classes_percentual.png", dpi=150)
plt.close()

# Exemplo de registro
print("\nExemplo:")
row = df.iloc[0]
print(f"Texto: {row['texto'][:300]}...")
print(f"Categoria: {row['categoria']}")
print(f"Titulo: {row['titulo']}")
