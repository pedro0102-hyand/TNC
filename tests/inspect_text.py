import pandas as pd
from collections import Counter
from pathlib import Path

# Carregar o dataset de textos únicos
df = pd.read_parquet(Path("data/processed/unique_texts.parquet"))

# Contagem de categorias
df["cat"] = df["categorias"].map(lambda c : c[0])

# analise de textos mais curtos
print("== 10 textos mais curtos ==")
curtos = df.assign(n=df["texto"].str.len()).nsmallest(10, "n")
for _, r in curtos.iterrows():
    print(f"{r['n']:>4} | {r['cat']:<14} | {r['titulo'][:50]} | {r['texto'][:100]!r}")

# analise de textos mais longos
print("\n== 10 textos mais longos ==")
longos = df.assign(n=df["texto"].str.len()).nlargest(10, "n")
for _, r in longos.iterrows():
    print(f"{r['n']:>4} | {r['cat']:<14} | {r['titulo'][:50]} | {r['texto'][:100]!r}")

# prefixos nos títulos (ex.: "Vídeo,")
print("\n== Prefixos de título mais comuns ==")
prefixos = df["titulo"].str.extract(r"^([^,]{1,20}),")[0].dropna()
print(prefixos.value_counts().head(10))
video = df["titulo"].str.startswith("Vídeo,")
print("\n'Vídeo,' por categoria:")
print(video.groupby(df["cat"]).sum())

# Começos e finais repetidos nos textos (indício de rodapé ou boilerplate)
print("\n== Começos mais repetidos (40 caracteres) ==")
print(Counter(df["texto"].str[:40]).most_common(8))
print("\n== Finais mais repetidos (60 caracteres) ==")
print(Counter(df["texto"].str[-60:]).most_common(8))

# Padrões de texto que podem indicar problemas de limpeza
padroes = {
    "url": r"https?://\S+",
    "quebra de linha": r"\n",
    "espaços duplos": r"[ \t]{2,}",
    "crédito/legenda": r"Crédito,|Legenda da foto|Getty Images",
    "entidade html": r"&\w+;",
}
print("\n== % de textos com cada padrão ==")
for nome, padrao in padroes.items():
    pct = df["texto"].str.contains(padrao, regex=True).mean() * 100
    print(f"{nome:<18} {pct:5.1f}%")