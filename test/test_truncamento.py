import pandas as pd
from transformers import BertTokenizer

df = pd.read_parquet("data/raw/bbc_news.parquet")

# Carregar tokenizer
print("Carregando tokenizer...")
tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
print("Tokenizer carregado com sucesso!")

# Selecionar uma matéria
registro = df.iloc[0]
titulo = registro["titulo"]
texto = registro["texto"]
entrada = titulo + " " + texto

# Tokenizar sem truncamento
tokens_completos = tokenizer(entrada,truncation=False)

# Tokenizar com limite de 512 tokens
tokens_truncados = tokenizer(entrada,truncation=True,max_length=512)

quantidade_completa = len(tokens_completos["input_ids"])
quantidade_truncada = len(tokens_truncados["input_ids"])

print("\n=== ANÁLISE DA MATÉRIA ===")
print(f"Título: {titulo}")
print(f"Tokens completos: {quantidade_completa}")
print(f"Tokens utilizados: {quantidade_truncada}")
print(
    f"Tokens descartados: "
    f"{quantidade_completa - quantidade_truncada}"
)