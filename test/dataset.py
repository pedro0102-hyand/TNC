import pandas as pd
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
df = pd.read_parquet("data/raw/bbc_news.parquet")
print(f"Quantidade de registros no dataset: {len(df)}")

MAX_TOKENS = 510
OVERLAP = 50
PASSO = MAX_TOKENS - OVERLAP

total_tokens = 0
total_chunks_sem_overlap = 0
total_chunks_com_overlap = 0
total_tokens_processados_sem_overlap = 0
total_tokens_processados_com_overlap = 0

for _, row in df.iterrows():

    titulo = row["titulo"]
    texto = row["texto"]
    texto_completo = titulo + " " + texto

    tokens = tokenizer.encode(texto_completo, add_special_tokens=False)
    total_tokens += len(tokens)

    # Sem overlap
    chunks_sem_overlap = [
        tokens[i:i + MAX_TOKENS]
        for i in range(0, len(tokens), MAX_TOKENS)
    ]
    total_chunks_sem_overlap += len(chunks_sem_overlap)
    total_tokens_processados_sem_overlap += sum(len(chunk) for chunk in chunks_sem_overlap)

    # Com overlap
    chunks_com_overlap = [
        tokens[i:i + MAX_TOKENS]
        for i in range(0, len(tokens), PASSO)
    ]
    total_chunks_com_overlap += len(chunks_com_overlap)
    total_tokens_processados_com_overlap += sum(len(chunk) for chunk in chunks_com_overlap)

print(f"Total de tokens no dataset: {total_tokens}")
print(f"Total de chunks sem overlap: {total_chunks_sem_overlap}")
print(f"Total de tokens processados sem overlap: {total_tokens_processados_sem_overlap}")
print(f"Total de chunks com overlap: {total_chunks_com_overlap}")
print(f"Total de tokens processados com overlap: {total_tokens_processados_com_overlap}")

print(f"Chunks sem overlap: {total_chunks_sem_overlap:,}")
print(f"Chunks com overlap: {total_chunks_com_overlap:,}")
print(f"Chunks adicionais: {total_chunks_com_overlap - total_chunks_sem_overlap:,}")
print(f"Tokens repetidos com overlap: {total_tokens_processados_com_overlap - total_tokens:,}")