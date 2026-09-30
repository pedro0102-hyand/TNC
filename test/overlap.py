import pandas as pd
from transformers import BertTokenizer

#Carregar tokenizer
print("Carregando tokenizer...")
tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
print("Tokenizer carregado com sucesso!")

# Carregar dataset
df = pd.read_parquet("data/raw/bbc_news.parquet")

# Selecionar uma matéria
materia = df.iloc[0]
titulo = materia["titulo"]
texto = materia["texto"]
texto_completo = titulo + " " + texto

# Tokenizar sem truncamento
tokens = tokenizer.encode(texto_completo,add_special_tokens=False)
print("\n=== MATÉRIA ===")
print(f"Título: {titulo}")
print(f"Tokens originais: {len(tokens)}")

MAX_TOKENS = 510
chunks_sem_overlap = [
    tokens[i:i + MAX_TOKENS]
    for i in range(0, len(tokens), MAX_TOKENS)
]
tokens_processados_sem_overlap = sum(len(chunk) for chunk in chunks_sem_overlap)

print("\n=== SEM OVERLAP ===")
print(f"Quantidade de chunks: {len(chunks_sem_overlap)}")
print(f"Tokens processados: {tokens_processados_sem_overlap}")
print(f"Tokens repetidos: 0")

OVERLAP = 50
passo = MAX_TOKENS - OVERLAP
chunks_com_overlap = [
    tokens[i:i + MAX_TOKENS]
    for i in range(0, len(tokens), passo)
]

tokens_processados_com_overlap = sum(len(chunk) for chunk in chunks_com_overlap)
tokens_repetidos = (tokens_processados_com_overlap - len(tokens))

print("\n=== COM OVERLAP ===")
print(f"Tamanho do chunk: {MAX_TOKENS}")
print(f"Overlap: {OVERLAP}")
print(f"Passo entre chunks: {passo}")
print(f"Quantidade de chunks: {len(chunks_com_overlap)}")
print(f"Tokens processados: {tokens_processados_com_overlap}")
print(f"Tokens repetidos: {tokens_repetidos}")


print("\n=== TAMANHO DOS CHUNKS COM OVERLAP ===")
for i, chunk in enumerate(chunks_com_overlap, start=1):
    print(f"Chunk {i}: {len(chunk)} tokens")

print("\n=== FINAL DO CHUNK 1 ===")
print(tokenizer.decode(chunks_com_overlap[0][-50:],skip_special_tokens=True))

print("\n=== INÍCIO DO CHUNK 2 ===")
print(tokenizer.decode(chunks_com_overlap[1][:50],skip_special_tokens=True))