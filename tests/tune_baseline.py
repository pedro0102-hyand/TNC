import time
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

DATA_PATH = Path("data/processed/clean_texts.parquet")
REPORTS_DIR = Path("reports")
SEED = 42

df = pd.read_parquet(DATA_PATH)

# Categoria única: base do treino, validação e teste
unicas = df[df["n_categorias"] == 1].copy()
unicas["categoria"] = unicas["categorias"].map(lambda c: c[0])
unicas = unicas.reset_index(drop=True)

# Entrada do modelo: título + texto, a mesma junção do treino
unicas["entrada"] = unicas["titulo"] + " " + unicas["texto"]

treino_val, teste = train_test_split(unicas, test_size=0.15, stratify=unicas["categoria"], random_state=SEED)
treino, val = train_test_split(treino_val,test_size=0.15 / 0.85,stratify=treino_val["categoria"],random_state=SEED)

conjuntos = {"treino": treino, "val": val, "teste": teste}

# Conferência 1: nada se perdeu nem se repetiu
assert len(treino) + len(val) + len(teste) == len(unicas)

# Conferência 2: nenhum texto em dois conjuntos (vazamento)
nomes = list(conjuntos)
for i, a in enumerate(nomes):
    for b in nomes[i + 1:]:
        comum = set(conjuntos[a]["texto"]) & set(conjuntos[b]["texto"])
        assert not comum, f"Vazamento entre {a} e {b}: {len(comum)} textos"

# Conferência 3: as 9 categorias aparecem em cada conjunto
todas = set(unicas["categoria"])
for nome, parte in conjuntos.items():
    faltando = todas - set(parte["categoria"])
    assert not faltando, f"Categorias ausentes em {nome}: {faltando}"

# busca por validacao cruzada nos dados de treino
pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(lowercase=True, min_df=2, max_df=0.9, sublinear_tf=True),
        ),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
    ]
)

grade = {
    "tfidf__ngram_range": [(1, 1), (1, 2)], # sequencia continua de n itens
    "clf__C": [1, 10, 100], # controle na capacidade de computacao
    "clf__class_weight": [None, "balanced"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
busca = GridSearchCV(pipeline,grade,scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},refit="f1_macro",cv=cv,n_jobs=1,verbose=2)

n_config = len(grade["tfidf__ngram_range"]) * len(grade["clf__C"]) * len(grade["clf__class_weight"])
print(f"\nBusca: {n_config} configurações x {cv.get_n_splits()} partes = {n_config * cv.get_n_splits()} treinos")

inicio = time.time()
busca.fit(treino["entrada"], treino["categoria"])
print(f"\nBusca concluída em {(time.time() - inicio) / 60:.1f} min")

res = pd.DataFrame(busca.cv_results_)
tabela = pd.DataFrame(
    {
        "ngram": res["param_tfidf__ngram_range"].astype(str),
        "C": res["param_clf__C"],
        "class_weight": res["param_clf__class_weight"].astype(str),
        "f1_macro_cv": res["mean_test_f1_macro"],
        "desvio": res["std_test_f1_macro"],
        "acc_cv": res["mean_test_accuracy"],
        "seg_por_treino": res["mean_fit_time"],
    }
).round(3)
tabela = tabela.sort_values("f1_macro_cv", ascending=False)

print("\n===== Validação cruzada no treino (ordenado por F1 macro) =====")
print(tabela.to_string(index=False))
REPORTS_DIR.mkdir(exist_ok=True)
tabela.to_csv(REPORTS_DIR / "baseline_tuning_cv.csv", index=False)

v1 = tabela[(tabela["ngram"] == "(1, 2)") & (tabela["C"] == 1) & (tabela["class_weight"] == "None")]
print("\nA configuração do baseline v1, nesta mesma busca:")
print(v1.to_string(index=False))
print(f"\nMelhor configuração: {busca.best_params_}")