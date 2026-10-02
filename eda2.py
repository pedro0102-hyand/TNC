from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/processed/unique_texts.parquet")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)
print(f"Dataset carregado: {DATA_PATH}")

# Informações básicas
print(f"Shape do dataset: {df.shape}")
print(f"Colunas do dataset: {df.columns.tolist()}")
print(f"Tipos de dados das colunas:\n{df.dtypes}")

# valores nulos por coluna
print(f"Valores nulos por coluna:\n{df.isnull().sum()}")

# campos vazios
print("\nCampos vazios:")
for coluna in ("texto", "titulo", "link"):
    vazios = df[coluna].fillna("").str.strip().eq("").sum()
    print(f"{coluna}: {vazios}")

# duplicados e categorias vazias
print(f"Categorias vazias: {df['categorias'].map(len).eq(0).sum()}")
print(f"Textos duplicados: {df['texto'].duplicated().sum()}")
print(f"Links duplicados: {df['link'].duplicated().sum()}")

# Informações sobre as categorias
categorias_explodidas = df.explode("categorias")
contagem_categorias = categorias_explodidas["categorias"].value_counts()

# distribuição de categorias
distribuicao_categorias = pd.DataFrame({
    "textos": contagem_categorias,
    "percentual": contagem_categorias / len(df) * 100,
}).round(1)

print("\nTextos associados a cada categoria:")
print(distribuicao_categorias)
print("Um texto pode pertencer a várias categorias, então os percentuais podem somar mais de 100%.")

# quantidade de categorias por texto
quantidade_por_texto = df["categorias"].map(len)
contagem_por_quantidade = quantidade_por_texto.value_counts().sort_index()
distribuicao_por_quantidade = pd.DataFrame({
    "textos": contagem_por_quantidade,
    "percentual": contagem_por_quantidade / len(df) * 100,
}).round(1)

print("\nQuantidade de categorias por texto:")
print(distribuicao_por_quantidade)

if "n_categorias" in df.columns:
    divergencias = df["n_categorias"].ne(quantidade_por_texto).sum()
    print(f"Registros em que n_categorias diverge da lista: {divergencias}")

# análise das combinações de categorias
combinacoes = df["categorias"].map(lambda categorias: tuple(sorted(categorias)))
contagem_combinacoes = combinacoes.value_counts().head(15)

distribuicao_combinacoes = pd.DataFrame({
    "textos": contagem_combinacoes,
    "percentual": contagem_combinacoes / len(df) * 100,
}).round(1)

print("\n15 combinações de categorias mais frequentes:")
print(distribuicao_combinacoes)

df["n_caracteres_texto"] = df["texto"].str.len()
df["n_palavras_texto"] = df["texto"].str.split().str.len()
df["n_caracteres_titulo"] = df["titulo"].str.len()

percentis = [0.5, 0.75, 0.9, 0.95, 0.99]

print("\nCaracteres por texto:")
print(df["n_caracteres_texto"].describe(percentiles=percentis).round(1))
print("\nPalavras por texto:")
print(df["n_palavras_texto"].describe(percentiles=percentis).round(1))
print("\nCaracteres por título:")
print(df["n_caracteres_titulo"].describe(percentiles=percentis).round(1))

datas = pd.to_datetime(df["data"], errors="coerce")
datas_validas = datas.dropna()
print(f"\nDatas inválidas: {datas.isna().sum()}")

if not datas_validas.empty:

    print(f"Período coberto: {datas_validas.min().date()} "f"a {datas_validas.max().date()}")
    textos_por_ano = datas_validas.dt.year.value_counts().sort_index()
    print("\nTextos por ano:")
    print(textos_por_ano)

# Gráfico: distribuição de textos por categoria
contagem_categorias.sort_values().plot(kind="barh",figsize=(9, 6),)
plt.title("Textos associados a cada categoria")
plt.xlabel("Quantidade de textos")
plt.ylabel("Categoria")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda2_categorias.png", dpi=150)
plt.close()

# Gráfico de quantidade de categorias por texto
contagem_por_quantidade.plot(kind="bar",figsize=(8, 5),)
plt.title("Quantidade de categorias por texto")
plt.xlabel("Número de categorias")
plt.ylabel("Quantidade de textos")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda2_numero_categorias.png", dpi=150)
plt.close()

# Histograma do tamanho dos textos (em caracteres)
plt.figure(figsize=(9, 5))
plt.hist(df["n_caracteres_texto"], bins=50)
plt.axvline(df["n_caracteres_texto"].median(),color="orange",linestyle="--",label="Mediana")
plt.title("Distribuição do tamanho dos textos")
plt.xlabel("Caracteres")
plt.ylabel("Quantidade de textos")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda2_tamanho_textos.png", dpi=150)
plt.close()

# Gráfico de textos por ano
if not datas_validas.empty:
    textos_por_ano.plot(kind="bar",figsize=(8, 5),)
    plt.title("Quantidade de textos por ano")
    plt.xlabel("Ano")
    plt.ylabel("Quantidade de textos")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda2_textos_por_ano.png", dpi=150)
    plt.close()