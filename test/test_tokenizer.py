import pandas as pd
from transformers import BertTokenizer

DATASET_PATH = "data/raw/bbc_news.parquet"
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

data = pd.read_parquet(DATASET_PATH)

print("\nCarregando tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer carregado com sucesso!")

# concatenando título e texto do primeiro registro
texto = data.iloc[0]["titulo"] + " " + data.iloc[0]["texto"]

# tokenizando o texto
tokens = tokenizer.tokenize(texto)

print("\n=== TEXTO ORIGINAL ===")
print(texto)

print("\n=== TOKENS ===")
print(tokens)

print("\n=== QUANTIDADE DE TOKENS ===")
print(len(tokens))

# analisando a quantidade de tokens do dataset
token_counts = []

# iterando sobre cada registro do dataset
for _, row in data.iterrows():

    texto = row["titulo"] + " " + row["texto"]
    tokens = tokenizer.tokenize(texto)
    token_counts.append(len(tokens))

data["token_count"] = token_counts

print("\n=== QUANTIDADE DE TOKENS POR REGISTRO ===")
print(data["token_count"].describe())

# verificando truncamento em 512 tokens
data["tokens_512"] = data["token_count"].apply(lambda x : min(x, 512))
data["foi_truncado"] = data["token_count"] > 512

print("\n=== TRUNCAMENTO EM 512 TOKENS ===")
print(f"Textos que ultrapassam 512 tokens: {data['foi_truncado'].sum()}")
print(f"Textos que cabem em 512 tokens: {(~data['foi_truncado']).sum()}")

print("\n=== PERCENTUAIS ===")
print(f"Truncados: {data['foi_truncado'].mean() * 100:.2f}%")
print(f"Não truncados: {(~data['foi_truncado']).mean() * 100:.2f}%")

# Analisar quantidade de tokens descartados

data["tokens_descartados"] = (data["token_count"] - data["tokens_512"])
data["percentual_descartado"] = (data["tokens_descartados"] / data["token_count"]) * 100

print("\n=== TOKENS DESCARTADOS ===")

print(
    f"Total de tokens descartados: "
    f"{data['tokens_descartados'].sum():,.0f}"
)

print(
    f"Média de tokens descartados por texto: "
    f"{data['tokens_descartados'].mean():.2f}"
)

print(
    f"Percentual médio descartado: "
    f"{data['percentual_descartado'].mean():.2f}%"
)

print("\n=== DISTRIBUIÇÃO DO PERCENTUAL DESCARTADO ===")
print(data["percentual_descartado"].describe())
