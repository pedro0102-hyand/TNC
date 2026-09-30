import pandas as pd

DATA_PATH = "data/raw/bbc_news.parquet"
data = pd.read_parquet(DATA_PATH)

empty_texts = data["texto"].str.strip() == ""
print("\n=== TEXTOS VAZIOS ===")
print(f"Quantidade: {empty_texts.sum()}")

if empty_texts.sum() > 0:
    print("\nRegistros:")
    print(
        data.loc[
            empty_texts,
            ["categoria", "titulo", "texto", "link"]
        ]
    )

short_texts = data["texto"].str.strip().str.len() <= 50

print("\n=== TEXTOS COM ATÉ 50 CARACTERES ===")
print(f"Quantidade: {short_texts.sum()}")

if short_texts.sum() > 0:
    print("\nRegistros:")
    print(
        data.loc[
            short_texts,
            ["categoria", "titulo", "texto", "link"]
        ]
    )

print("\n=== TEXTOS CURTOS POR CATEGORIA ===")
print(
    data.loc[short_texts, "categoria"]
    .value_counts()
)