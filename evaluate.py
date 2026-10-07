import argparse
import sys
import joblib
import torch
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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




