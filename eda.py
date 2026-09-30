import pandas as pd

DATASET_PATH = "data/raw/bbc_news.parquet"

# carregar o dataset
data = pd.read_parquet(DATASET_PATH)

# verificando dimensoes do dataset
print("\n=== DIMENSÕES DO DATASET ===")
print(f"Linhas: {data.shape[0]}")
print(f"Colunas: {data.shape[1]}")

# verificando os primeiros registros do dataset
print("\n=== PRIMEIROS REGISTROS ===")
print(data.head())

# verificando os ultimos registros do dataset
print("\n=== ÚLTIMOS REGISTROS ===")
print(data.tail())

# verificando as colunas do dataset
print("\n=== COLUNAS DO DATASET ===")
print(data.columns.tolist())

# informacoes gerais sobre o dataset
print("\n=== INFORMAÇÕES GERAIS ===")
print(data.info())

# tipos de variáveis no dataset
print("\n=== TIPOS DE VARIÁVEIS ===")
print(data.dtypes)

# verificando valores nulos no dataset
print("\n=== VALORES NULOS ===")
print(data.isnull().sum())
print("=== VALORES NULOS (%) ===")
print((data.isnull().sum() / len(data)) * 100)

# dados duplicados no dataset
print("\n=== DADOS DUPLICADOS ===")
duplicated_rows = data.duplicated().sum()
print(f"Quantidade de registros duplicados: {duplicated_rows}")

# duplicidade nas colunas do dataset
print("\n=== DUPLICIDADE NAS COLUNAS ===")
print(f"Títulos duplicados: {data['titulo'].duplicated().sum()}")
print(f"Textos duplicados: {data['texto'].duplicated().sum()}")

# analisando as variaveis alvo do dataset
print("\n=== ANÁLISE DAS VARIÁVEIS ALVO ===")
print("\n=== CATEGORIAS ===")
print(data["categoria"].unique())
print("\n=== QUANTIDADE DE CATEGORIAS ===")
print(data["categoria"].nunique())

# distribuição das categorias no dataset
print("\n=== DISTRIBUIÇÃO DAS CATEGORIAS ===")
category_distribution = data["categoria"].value_counts()
print(category_distribution)

# proporção das categorias no dataset
print("\n=== PROPORÇÃO DAS CATEGORIAS ===")
category_proportion = data["categoria"].value_counts(normalize=True) * 100
print(category_proportion)