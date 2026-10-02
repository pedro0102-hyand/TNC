import re
from pathlib import Path
import pandas as pd

df = pd.read_parquet(Path("data/processed/unique_texts.parquet"))
df["n_chars"] = df["texto"].str.len()

# Quantos textos abaixo de cada limite
print("== Textos curtos ==")
for limite in (1, 50, 100, 200, 300, 500, 1000):
    print(f"< {limite:>4} caracteres: {(df['n_chars'] < limite).sum()}")

# Páginas de erro
erro = df["texto"].str.contains("Lo sentimos", regex=False)
print(f"\nPáginas de erro ('Lo sentimos'): {erro.sum()}")

# Amostra de textos curtos (para escolher o limite)
print("\n== Amostra entre 100 e 400 caracteres ==")
faixa = df[df["n_chars"].between(100, 400)]
amostra = faixa.sample(min(8, len(faixa)), random_state=42)
for _, r in amostra.iterrows():
    print(f"{r['n_chars']:>4} | {r['titulo'][:50]} | {r['texto'][:150]!r}")

# Contexto dos espaços duplos
print("\n== Contexto dos espaços duplos ==")
mostrados = 0
for t in df["texto"].sample(200, random_state=1):
    m = re.search(r".{30}[ \t]{2,}.{30}", t)
    if m:
        print(repr(m.group()))
        mostrados += 1
    if mostrados == 8:
        break

# Palavras coladas (minúscula + Maiúscula + minúscula)
cola = re.compile(r"[a-zà-ú][A-ZÀ-Ú][a-zà-ú]")
pct = df["texto"].map(lambda t: bool(cola.search(t))).mean() * 100
print(f"\nTextos com padrão minúscula+Maiúscula colado: {pct:.1f}%")

print("\n== Exemplos (com contexto) ==")
mostrados = 0
for t in df["texto"].sample(500, random_state=2):
    m = re.search(r".{15}[a-zà-ú][A-ZÀ-Ú][a-zà-ú].{15}", t)
    if m:
        print(repr(m.group()))
        mostrados += 1
    if mostrados == 10:
        break