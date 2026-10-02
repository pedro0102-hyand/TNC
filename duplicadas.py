import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/bbc_news_ptbr.parquet")
df = pd.read_parquet(DATA_PATH)

# todas as linhas envolvidas em duplicacao
dups = df[df["texto"].duplicated(keep=False)]
print(f"Linhas envolvidas em duplicação: {len(dups)} ({len(dups) / len(df) * 100:.1f}%)")
print(f"Textos distintos nesses grupos: {dups['texto'].nunique()}")

# quantidade de categorias diferentes nos grupos de duplicados
category_counts = dups.groupby("texto")["categoria"].nunique()
print("\nQuantidade de categorias diferentes nos grupos de duplicados:")
print(category_counts.value_counts().sort_index())

# combinacoes de categorias em conflito
conflicting_groups = dups.groupby("texto")["categoria"].apply(lambda s : tuple(sorted(set(s))))
conflicting_groups = conflicting_groups[conflicting_groups.map(len) > 1]
print("\nCombinações de categorias mais frequentes (conflitos):")
print(conflicting_groups.value_counts().head(10))

# Repetições com a mesma categoria
iguais = df.duplicated(subset=["texto", "categoria"], keep="first").sum()
print(f"\nLinhas repetidas com texto E categoria iguais: {iguais}")

# Tamanho após remover duplicados por texto (apenas exploratório)
dedup = df.drop_duplicates(subset="texto")
print(f"\nApós dedup por texto: {len(dedup)} linhas")
print(dedup["categoria"].value_counts())