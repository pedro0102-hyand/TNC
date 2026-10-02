import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer

DATA_PATH = Path("data/raw/bbc_news_ptbr.parquet")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

# Carregar o dataset
df = pd.read_parquet(DATA_PATH)
print(f"Dataset carregado: {DATA_PATH}")

# concatenando titulo e texto
df["texto_completo"] = df["titulo"] + " " + df["texto"]
print(f"Dataset com texto completo: {df.shape[0]} registros")

# tamanho em caracteres
print("Caracteres por texto:")
print(df["texto_completo"].str.len().describe().round(0))

# tamanho em tokens
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
ids = tokenizer(df["texto_completo"].tolist())["input_ids"]
token_lengths = pd.Series([len(x) for x in ids])
print("Tokens por texto:")
print(token_lengths.describe().round(0))
print("\nTokens por texto (percentis):")
print(token_lengths.quantile([0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).astype(int))

# percentual de textos acima dos limites de tokens
for limit in (128, 256, 512):
    pct = (token_lengths > limit).mean() * 100
    print(f"Textos acima de {limit} tokens: {pct:.1f}%")

# Gráfico: distribuição do tamanho dos textos (tokens)
plt.hist(token_lengths, bins=50)
plt.axvline(256, color="orange", linestyle="--", label="256")
plt.axvline(512, color="red", linestyle="--", label="512 (limite do BERT)")
plt.title("Distribuição do tamanho dos textos (tokens)")
plt.xlabel("Tokens")
plt.ylabel("Quantidade de textos")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "tokens_hist.png", dpi=150)
plt.close()

# duplicados
print("\nDuplicados:")
for col in ("titulo", "texto", "link"):
    print(f"  {col}: {df[col].duplicated().sum()}")

# Período coberto
datas = pd.to_datetime(df["data"])
print(f"\nPeríodo: {datas.min().date()} -> {datas.max().date()}")



