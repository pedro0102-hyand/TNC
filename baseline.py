import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline  
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time

DATA_PATH = Path("data/processed/clean_texts.parquet")
df = pd.read_parquet(DATA_PATH)
print(f"Carregadas : {len(df)} noticias")
SEED = 42

# removendo noticias que possuem mais de uma categoria da fase de treino do modelo
ambiguas = df[df["n_categorias"]> 1].reset_index(drop=True)

# categorias unicas servem como base de treinamento do modelo, validação e teste
unicas = df[df["n_categorias"]==1].copy()
unicas["categoria"] = unicas["categorias"].map(lambda c: c[0])
unicas = unicas.reset_index(drop=True)

# concatenando titulo e texto para o modelo
unicas["entrada"] = unicas["titulo"] + " " + unicas["texto"]

print(f"Categoria única (treino/val/teste): {len(unicas)}")
print(f"Ambíguas (avaliação à parte): {len(ambiguas)}")
print(f"Categorias: {sorted(unicas['categoria'].unique())}")
print(f"\nExemplo de entrada:\n{unicas['entrada'].iloc[0][:200]}...")
print(f"\nTamanho médio da entrada: {unicas['entrada'].str.len().mean():.0f} caracteres")

# primeiro corte para separar treino e teste
treino_val, teste = train_test_split(unicas, test_size = 0.15, random_state = SEED, stratify = unicas["categoria"])

# segundo corte para separar treino e validação
treino, val = train_test_split(treino_val, test_size = 0.15 / 0.85, random_state = SEED, stratify = treino_val["categoria"])

# reunindo os datasets em um dicionário
datasets = {"treino": treino, "val": val, "teste": teste}
assert len(treino) + len(val) + len(teste) == len(unicas)

# conferindo se há vazamento
nomes = list(datasets)
for i, a in enumerate(nomes):
    for b in nomes[i + 1:]:
        comum = set(datasets[a]["texto"]) & set(datasets[b]["texto"])
        assert not comum, f"Vazamento entre {a} e {b}: {len(comum)} textos"

# conferindo se as categorias estão balanceadas
todas = set(unicas["categoria"])
for nome, parte in datasets.items():
    faltando = todas - set(parte["categoria"])
    assert not faltando, f"Categorias ausentes em {nome}: {faltando}"

print(f"\ntreino: {len(treino)} | val: {len(val)} | teste: {len(teste)}")
dist = pd.DataFrame({nome: parte["categoria"].value_counts(normalize=True) for nome, parte in datasets.items()})
print((dist * 100).round(1))
print(teste["categoria"].value_counts())

# criando o pipeline do modelo
# pipeline para evitar vazamento de dados entre treino e validação
modelo = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase = True, ngram_range = (1, 2), min_df = 2, max_df = 0.9, sublinear_tf = True)),
    ("clf", LogisticRegression(max_iter = 1000, random_state = SEED))
])

inicio = time.time()
modelo.fit(treino["entrada"], treino["categoria"])
print(f"\nTreino do baseline concluído em {time.time() - inicio:.1f} s")
vocab = modelo.named_steps["tfidf"].vocabulary_
print(f"Tamanho do vocabulário: {len(vocab)} termos")
print(f"Ordem das classes: {list(modelo.classes_)}")

# avaliando o modelo
pred_treino = modelo.predict(treino["entrada"])
pred_val = modelo.predict(val["entrada"])

# métricas de avaliação
acc_treino = accuracy_score(treino["categoria"], pred_treino)
acc_val = accuracy_score(val["categoria"], pred_val)
f1_val = f1_score(val["categoria"], pred_val, average="macro")
piso = val["categoria"].value_counts(normalize=True).max()

print("\n===== Baseline: resultados =====")
print(f"Piso (chutar sempre a maior classe): {piso:.3f}")
print(f"Accuracy no treino:     {acc_treino:.3f}")
print(f"Accuracy na validação:  {acc_val:.3f}")
print(f"F1 macro na validação:  {f1_val:.3f}")
print("\nRelatório por classe (validação):")
print(classification_report(val["categoria"], pred_val, digits=3))