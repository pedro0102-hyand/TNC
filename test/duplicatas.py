import pandas as pd

DATA_PATH = "data/raw/bbc_news.parquet"
data = pd.read_parquet(DATA_PATH)

complete_duplicates = data.duplicated().sum()

print("\n=== DUPLICATAS COMPLETAS ===")
print(f"Registros duplicados: {complete_duplicates}")

title_duplicates = data["titulo"].duplicated().sum()
unique_duplicate_titles = data.loc[data["titulo"].duplicated(keep=False), "titulo"].nunique()

print("\n=== DUPLICIDADE DE TÍTULOS ===")
print(f"Registros com título duplicado: {title_duplicates}")
print(f"Títulos duplicados distintos: {unique_duplicate_titles}")

text_duplicates = data["texto"].duplicated().sum()
unique_duplicate_texts = data.loc[data["texto"].duplicated(keep=False), "texto"].nunique()

print("\n=== DUPLICIDADE DE TEXTOS ===")
print(f"Registros com texto duplicado: {text_duplicates}")
print(f"Textos duplicados distintos: {unique_duplicate_texts}")
text_category_counts = data.groupby("texto")["categoria"].nunique()
texts_multiple_categories = (text_category_counts > 1).sum()

print("\n=== TEXTOS EM MÚLTIPLAS CATEGORIAS ===")
print(
    f"Textos associados a mais de uma categoria: "
    f"{texts_multiple_categories}"
)

title_category_counts = data.groupby("titulo")["categoria"].nunique()
titles_multiple_categories = (title_category_counts > 1).sum()

print("\n=== TÍTULOS EM MÚLTIPLAS CATEGORIAS ===")
print(
    f"Títulos associados a mais de uma categoria: "
    f"{titles_multiple_categories}"
)

text_category_counts = (data.groupby("texto")["categoria"].nunique())
print("\n=== DISTRIBUIÇÃO DE CATEGORIAS POR TEXTO ===")
print(text_category_counts.value_counts().sort_index())

ambiguous_texts = text_category_counts[text_category_counts > 1].index
examples = (data[data["texto"].isin(ambiguous_texts)][["categoria", "titulo", "texto"]].drop_duplicates(subset=["texto"]))

print("\n=== QUANTIDADE DE TEXTOS AMBÍGUOS ===")
print(len(ambiguous_texts))

unique_texts = (data.groupby("texto")["categoria"].nunique())
unambiguous_texts = unique_texts[unique_texts == 1]

print("\n=== TEXTOS NÃO AMBÍGUOS ===")
print(f"Textos com apenas uma categoria: {len(unambiguous_texts)}")
print(
    f"Percentual do dataset: "
    f"{len(unambiguous_texts) / len(data) * 100:.2f}%"
)