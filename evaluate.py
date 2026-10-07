import argparse
import sys
import joblib
import torch
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

DATA_PATH = Path("data/processed/clean_texts.parquet")
BASELINE_PATH = Path("models/baseline.joblib")
BERT_DIR = "models/bert"
REPORTS_DIR = Path("reports")
MAX_LENGTH = 256
SEED = 42
df = pd.read_parquet(DATA_PATH)

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["val", "teste"], default="val")
parser.add_argument("--abrir-teste",action="store_true",help="confirma que a avaliação final no teste deve ser feita agora")
args = parser.parse_args()

if args.split == "teste" and not args.abrir_teste:
    sys.exit(
        "O conjunto de teste só abre com --abrir-teste "
        "(avaliação final, uma única vez). Use --split val para o dia a dia."
    )

# Categoria única: base do treino, validação e teste
unicas = df[df["n_categorias"] == 1].copy()
unicas["categoria"] = unicas["categorias"].map(lambda c: c[0])
unicas = unicas.reset_index(drop=True)

unicas["entrada"] = unicas["titulo"] + " " + unicas["texto"]

treino_val, teste = train_test_split(unicas, test_size=0.15, stratify=unicas["categoria"], random_state=SEED)
treino, val = train_test_split(treino_val,test_size=0.15 / 0.85,stratify=treino_val["categoria"],random_state=SEED)

conjuntos = {"treino": treino, "val": val, "teste": teste}
assert len(treino) + len(val) + len(teste) == len(unicas)

nomes = list(conjuntos)
for i, a in enumerate(nomes):
    for b in nomes[i + 1:]:
        comum = set(conjuntos[a]["texto"]) & set(conjuntos[b]["texto"])
        assert not comum, f"Vazamento entre {a} e {b}: {len(comum)} textos"

todas = set(unicas["categoria"])
for nome, parte in conjuntos.items():
    faltando = todas - set(parte["categoria"])
    assert not faltando, f"Categorias ausentes em {nome}: {faltando}"

print(f"treino: {len(treino)} | val: {len(val)} | teste: {len(teste)}")
avaliacao = conjuntos[args.split]
print(f"\nConjunto avaliado: {args.split} ({len(avaliacao)} notícias)")

entradas = avaliacao["entrada"].tolist()
baseline = joblib.load(BASELINE_PATH)
probs_baseline = baseline.predict_proba(entradas)

# BERT: carregado de models/bert, em lotes e sem gradientes
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(device).eval()

# As duas numerações de classe precisam ser a mesma (a alfabética)
classes = [model.config.id2label[i] for i in range(model.config.num_labels)]
assert list(baseline.classes_) == classes, "Ordem de classes diferente entre os modelos"

def prever_bert(textos, tamanho_lote=32):
    saidas = []
    for i in range(0, len(textos), tamanho_lote):
        lote = tokenizer(textos[i : i + tamanho_lote],truncation=True,max_length=MAX_LENGTH,padding=True,return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**lote).logits
        saidas.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(saidas)

probs_bert = prever_bert(entradas)


# Tabela com uma linha por notícia (sem o texto, para o arquivo ficar pequeno)
resultados = avaliacao[["titulo", "categoria", "formato", "data", "link"]].reset_index(drop=True)
resultados["pred_baseline"] = np.array(classes)[probs_baseline.argmax(axis=1)]
resultados["conf_baseline"] = probs_baseline.max(axis=1)
resultados["pred_bert"] = np.array(classes)[probs_bert.argmax(axis=1)]
resultados["conf_bert"] = probs_bert.max(axis=1)
REPORTS_DIR.mkdir(exist_ok=True)
saida = REPORTS_DIR / f"predicoes_{args.split}.parquet"
resultados.to_parquet(saida, index=False)
print(f"\nPrevisões salvas em {saida}")

# Conferência: quem acertou o quê
acerto_b = resultados["pred_baseline"] == resultados["categoria"]
acerto_n = resultados["pred_bert"] == resultados["categoria"]
print(f"\nAccuracy no conjunto '{args.split}':")
print(f"  Baseline: {acerto_b.mean():.3f}")
print(f"  BERT:     {acerto_n.mean():.3f}")
print(f"\nAmbos acertaram:        {(acerto_b & acerto_n).sum()}")
print(f"Só o BERT acertou:      {(~acerto_b & acerto_n).sum()}")
print(f"Só o baseline acertou:  {(acerto_b & ~acerto_n).sum()}")
print(f"Ambos erraram:          {(~acerto_b & ~acerto_n).sum()}")


y = resultados["categoria"]
print("\n===== Geral =====")
for nome, col in (("Baseline", "pred_baseline"), ("BERT", "pred_bert")):
    acc = accuracy_score(y, resultados[col])
    f1 = f1_score(y, resultados[col], average="macro")
    print(f"{nome:<9} accuracy {acc:.3f} | F1 macro {f1:.3f}")


def por_classe(coluna):
    p, r, f, _ = precision_recall_fscore_support(y, resultados[coluna], labels=classes, zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f}, index=classes)

pb = por_classe("pred_baseline")
pn = por_classe("pred_bert")

tabela = pd.DataFrame(
    {
        "suporte": y.value_counts().reindex(classes),
        "f1_base": pb["f1"],
        "f1_bert": pn["f1"],
        "delta_f1": pn["f1"] - pb["f1"],
        "rec_base": pb["recall"],
        "rec_bert": pn["recall"],
        "prec_base": pb["precision"],
        "prec_bert": pn["precision"],
    }
).round(3)

print("\n===== Por classe (ordenado pelo ganho de F1 do BERT) =====")
print(tabela.sort_values("delta_f1", ascending=False).to_string())
tabela.to_csv(REPORTS_DIR / f"metricas_por_classe_{args.split}.csv")

# Comparação pareada: só contam as notícias em que os modelos discordam
b = int((~acerto_b & acerto_n).sum())  # só o BERT acertou
c = int((acerto_b & ~acerto_n).sum())  # só o baseline acertou
n = len(resultados)
dif = (b - c) / n
erro_padrao = np.sqrt(b + c - (b - c) ** 2 / n) / n
p = binomtest(b, b + c, 0.5).pvalue
print("\n===== Comparação pareada (BERT - baseline) =====")
print(f"Diferença de accuracy: {dif:+.3f}")
print(f"Intervalo de 95%: [{dif - 1.96 * erro_padrao:+.3f}, {dif + 1.96 * erro_padrao:+.3f}]")
print(f"Teste exato de McNemar: p = {p:.4f}")



