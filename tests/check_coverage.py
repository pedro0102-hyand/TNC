from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/unique_texts.parquet")  # depois: clean_texts.parquet

df = pd.read_parquet(DATA_PATH)

# Notícias que citam cada categoria (inclui as ambíguas)
todas = df["categorias"].explode().value_counts()

# Notícias em que a categoria é a única
unica = df.loc[df["n_categorias"] == 1, "categorias"].map(lambda c: c[0]).value_counts()

tabela = pd.DataFrame({"total": todas, "so_categoria_unica": unica}).fillna(0).astype(int)
tabela["ambiguas"] = tabela["total"] - tabela["so_categoria_unica"]
tabela["retencao_%"] = (tabela["so_categoria_unica"] / tabela["total"] * 100).round(1)
tabela["teste_esperado"] = (tabela["so_categoria_unica"] * 0.15).round().astype(int)

print(tabela.sort_values("so_categoria_unica", ascending=False))

# Teste: toda categoria do dataset tem notícias de categoria única?
faltando = set(todas.index) - set(unica.index)
assert not faltando, f"Categorias sem nenhuma notícia de categoria única: {faltando}"
print(f"\nOK: as {len(todas)} categorias aparecem entre as notícias de categoria única.")
print(f"Razão entre a maior e a menor classe: {unica.max() / unica.min():.1f}")

Path("reports").mkdir(exist_ok=True)
tabela.to_csv("reports/cobertura_categorias.csv", index=False)