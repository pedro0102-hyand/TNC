import pandas as pd
from transformers import BertTokenizer

#Carregar tokenizer
print("Carregando tokenizer...")
tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
print("Tokenizer carregado com sucesso!")

# Carregar dataset
DATA_PATH = "data/raw/bbc_news.parquet"
data = pd.read_parquet(DATA_PATH)

# Selecionar uma matéria
materia = data.iloc[0]
titulo = materia["titulo"]
texto = materia["texto"]
texto_completo = titulo + " " + texto

# Tokenizar sem truncamento
tokens = tokenizer.encode(texto_completo,add_special_tokens=False)

# Definir tamanho máximo de cada chunk
MAX_TOKENS = 510

# Dividir os tokens em chunks
chunks = [
    tokens[i:i + MAX_TOKENS]
    for i in range(0, len(tokens), MAX_TOKENS)
]

print("\n=== ANÁLISE DE CHUNKING ===")
print(f"Título: {titulo}")
print(f"Tokens totais: {len(tokens)}")
print(f"Quantidade de chunks: {len(chunks)}")
print("\n=== TAMANHO DOS CHUNKS ===")

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}: {len(chunk)} tokens")

# Converter o primeiro e último chunk novamente para texto
print("\n=== PRIMEIRO CHUNK ===")
primeiro_chunk = tokenizer.decode(chunks[0],skip_special_tokens=True)
print(primeiro_chunk)
print("\n=== ÚLTIMO CHUNK ===")
ultimo_chunk = tokenizer.decode(chunks[-1],skip_special_tokens=True)
print(ultimo_chunk)