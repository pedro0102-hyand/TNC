import pandas as pd
from transformers import BertTokenizer

#Carregar dataset
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

# Tokenizar a matéria completa
tokens_completos = tokenizer(entrada,truncation=False)["input_ids"]

# Tokenizar com limite de 512 tokens
tokens_truncados = tokenizer(entrada,truncation=True,max_length=512)["input_ids"]

# Converter IDs dos tokens para os tokens originais
tokens_completos = tokenizer.convert_ids_to_tokens(tokens_completos)
tokens_truncados = tokenizer.convert_ids_to_tokens(tokens_truncados)

# Mostrar início da sequência completa
print("\n=== INÍCIO DA MATÉRIA ===")
print(tokens_completos[:50])

# Mostrar final da sequência completa
print("\n=== FINAL DA MATÉRIA ===")
print(tokens_completos[-50:])

# Mostrar final da sequência que será enviada ao BERT
print("\n=== FINAL DOS 512 TOKENS ===")
print(tokens_truncados[-50:])

# Mostrar resumo
print("\n=== RESUMO ===")
print(f"Tokens completos: {len(tokens_completos)}")
print(f"Tokens utilizados: {len(tokens_truncados)}")
print(
    f"Tokens descartados: "
    f"{len(tokens_completos) - len(tokens_truncados)}"
)