import argparse
import sys
import json
import joblib
import torch
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, ConfusionMatrixDisplay, confusion_matrix

DATA_PATH = Path("data/processed/clean_texts.parquet")
BASELINE_PATH = Path("models/baseline.joblib")
BERT_DIR = "models/bert"
REPORTS_DIR = Path("reports")
MAX_LENGTH = 256 # padrao, usado se o modelo nao tiver treino.json
SEED = 42
df = pd.read_parquet(DATA_PATH)

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["val", "teste"], default="val")
parser.add_argument("--abrir-teste",action="store_true",help="confirma que a avaliação final no teste deve ser feita agora")
parser.add_argument("--bert-dir", default=BERT_DIR, help="pasta do BERT a avaliar")
parser.add_argument("--baseline", default=str(BASELINE_PATH), help="arquivo do baseline a avaliar")
parser.add_argument("--comparar-com", default=None, help="pasta de outro BERT, já avaliado neste mesmo split, para comparar")
args = parser.parse_args()

if args.split == "teste" and not args.abrir_teste:
    sys.exit(
        "O conjunto de teste só abre com --abrir-teste "
        "(avaliação final, uma única vez). Use --split val para o dia a dia."
    )

BERT_DIR = args.bert_dir # pasta do modelo BERT a avaliar
BASELINE_PATH = Path(args.baseline)
TAG = Path(BERT_DIR).name # nome do modelo, usado nos nomes dos arquivos de saida

# o max_length vem do treino.json do modelo (sem o arquivo, fica o padrao de 256)
config_treino = Path(BERT_DIR) / "treino.json"
if config_treino.exists():
    MAX_LENGTH = json.loads(config_treino.read_text(encoding="utf-8"))["max_length"]
else:
    print(f"Aviso: {config_treino} não existe, assumindo max_length={MAX_LENGTH}")
print(f"Modelo avaliado: {BERT_DIR} (max_length={MAX_LENGTH}) | baseline: {BASELINE_PATH}")

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

# BERT: carregado da pasta escolhida em --bert-dir, em lotes e sem gradientes
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

probs_bert = prever_bert(entradas, 32 if MAX_LENGTH <= 256 else 16) # lote menor com 512 tokens, so para economizar memoria

# Tabela com uma linha por notícia (sem o texto, para o arquivo ficar pequeno)
resultados = avaliacao[["titulo", "categoria", "formato", "data", "link"]].reset_index(drop=True)
resultados["pred_baseline"] = np.array(classes)[probs_baseline.argmax(axis=1)]
resultados["conf_baseline"] = probs_baseline.max(axis=1)
resultados["pred_bert"] = np.array(classes)[probs_bert.argmax(axis=1)]
resultados["conf_bert"] = probs_bert.max(axis=1)
REPORTS_DIR.mkdir(exist_ok=True)
saida = REPORTS_DIR / f"predicoes_{args.split}_{TAG}.parquet"
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
tabela.to_csv(REPORTS_DIR / f"metricas_por_classe_{args.split}_{TAG}.csv")

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

FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
cm_base = confusion_matrix(y, resultados["pred_baseline"], labels=classes)
cm_bert = confusion_matrix(y, resultados["pred_bert"], labels=classes)

def erro_mais_comum(cm):
    """Para cada classe real, a categoria que mais recebe os erros dela."""
    saida = []
    for i in range(len(classes)):
        erros = cm[i].copy()
        erros[i] = 0
        j = erros.argmax()
        saida.append(f"{classes[j]} ({erros[j]})" if erros[j] > 0 else "-")
    return saida

resumo = pd.DataFrame(
    {
        "suporte": cm_bert.sum(axis=1),
        "erros_base": cm_base.sum(axis=1) - np.diag(cm_base),
        "erros_bert": cm_bert.sum(axis=1) - np.diag(cm_bert),
        "mais_comum_base": erro_mais_comum(cm_base),
        "mais_comum_bert": erro_mais_comum(cm_bert),
    },
    index=classes,
)
print("\n===== Para onde vão os erros de cada classe =====")
print(resumo.to_string())

confusoes = [
    (cm_bert[i, j], classes[i], classes[j])
    for i in range(len(classes))
    for j in range(len(classes))
    if i != j
]
print("\n===== Confusões mais frequentes do BERT (real -> prevista) =====")
for qtd, real, prevista in sorted(confusoes, reverse=True)[:8]:
    qtd_base = cm_base[classes.index(real), classes.index(prevista)]
    print(f"  {real} -> {prevista}: {qtd}  (baseline: {qtd_base})")

# Figura: as duas matrizes lado a lado, normalizadas por linha
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for ax, col, titulo in zip(
    axes, ["pred_baseline", "pred_bert"], ["Baseline (TF-IDF + LR)", f"BERT ({TAG})"]
):
    ConfusionMatrixDisplay.from_predictions(y,resultados[col],labels=classes,normalize="true",xticks_rotation=45,values_format=".2f",cmap="Blues",ax=ax,colorbar=False)
    ax.set_title(f"{titulo}: {args.split}")

plt.tight_layout()
plt.savefig(FIGURES_DIR / f"confusao_{args.split}_{TAG}.png", dpi=150)
plt.close()
print(f"\nFigura salva em {FIGURES_DIR / f'confusao_{args.split}_{TAG}.png'}")


resultados["acerto_baseline"] = acerto_b
resultados["acerto_bert"] = acerto_n
resultados["ano"] = pd.to_datetime(resultados["data"]).dt.year
resultados["formato_agrupado"] = resultados["formato"].replace({"video": "video/audio", "audio": "video/audio"})
resultados["n_chars"] = avaliacao["entrada"].str.len().values
resultados["faixa_tamanho"] = pd.qcut(resultados["n_chars"], 4, labels=["Q1 (curtos)", "Q2", "Q3", "Q4 (longos)"])

def corte(coluna):
    t = resultados.groupby(coluna, observed=True).agg(n=("categoria", "size"),acc_baseline=("acerto_baseline", "mean"),acc_bert=("acerto_bert", "mean"))
    t["delta"] = t["acc_bert"] - t["acc_baseline"]
    return t.round(3)

for coluna, titulo in (
    ("formato_agrupado", "formato"),
    ("ano", "ano"),
    ("faixa_tamanho", "tamanho do texto"),
):
    print(f"\n===== Accuracy por {titulo} =====")
    print(corte(coluna).to_string())

print("\nFaixas de tamanho (caracteres da entrada):")
print(resultados.groupby("faixa_tamanho", observed=True)["n_chars"].agg(["min", "max"]))


textos = avaliacao["texto"].reset_index(drop=True)
ambos = resultados[~acerto_b & ~acerto_n].copy()
ambos["texto"] = textos[ambos.index]
print(f"\n===== Erros que ambos cometeram: {len(ambos)} =====")

mesma = ambos["pred_baseline"] == ambos["pred_bert"]
print(f"Os dois erraram para a MESMA categoria: {mesma.sum()} ({mesma.mean():.0%})")

taxa = (ambos["categoria"].value_counts() / y.value_counts()).dropna().round(2)
print("\nFração de cada classe em que ambos erram:")
print(taxa.sort_values(ascending=False).to_string())
print("\nPares mais comuns (real -> previsão do BERT):")
print((ambos["categoria"] + " -> " + ambos["pred_bert"]).value_counts().head(8).to_string())
conf_erros = ambos["conf_bert"].mean()
conf_acertos = resultados.loc[acerto_n, "conf_bert"].mean()
print(f"\nConfiança média do BERT: {conf_erros:.2f} nesses erros, {conf_acertos:.2f} nos acertos")

# Arquivo para leitura, do erro mais confiante ao menos confiante
ambos = ambos.sort_values("conf_bert", ascending=False)

print("\nOs 5 erros mais confiantes do BERT:")
for _, r in ambos.head(5).iterrows():
    print(f"  [{r['categoria']} -> {r['pred_bert']} {r['conf_bert']:.0%}] {r['titulo'][:90]}")

linhas = [
    f"# Erros que baseline e BERT ({TAG}) cometeram ({args.split}): {len(ambos)}",
    "",
    "Ordenados pela confiança do BERT (o mais confiante primeiro).",
]
for k, (_, r) in enumerate(ambos.iterrows(), start=1):
    linhas += [
        "",
        f"### {k}. {r['titulo']}",
        f"- Real: **{r['categoria']}** | Baseline: {r['pred_baseline']} "
        f"({r['conf_baseline']:.0%}) | BERT: {r['pred_bert']} ({r['conf_bert']:.0%})",
        f"- Formato: {r['formato']} | Data: {r['data']} | {r['link']}",
        f"> {r['texto'][:400]}...",
    ]

saida_md = REPORTS_DIR / f"erros_ambos_{args.split}_{TAG}.md"
saida_md.write_text("\n".join(linhas), encoding="utf-8")
print(f"\nArquivo para leitura salvo em {saida_md}")


# comparacao com outro BERT (regra de escolha entre 256 e 512 tokens)
if args.comparar_com:
    tag_outro = Path(args.comparar_com).name
    arquivo_outro = REPORTS_DIR / f"predicoes_{args.split}_{tag_outro}.parquet"
    if not arquivo_outro.exists():
        sys.exit(f"Falta {arquivo_outro}. Rode antes: python evaluate.py --bert-dir {args.comparar_com}")
    outro = pd.read_parquet(arquivo_outro)

    # as duas tabelas precisam ser das mesmas noticias, na mesma ordem
    assert (outro["link"].values == resultados["link"].values).all(), "Notícias diferentes"

    acerto_este = acerto_n.values
    acerto_outro = (outro["pred_bert"] == outro["categoria"]).values
    acc_este = accuracy_score(y, resultados["pred_bert"])
    f1_este = f1_score(y, resultados["pred_bert"], average="macro")
    acc_outro = accuracy_score(y, outro["pred_bert"])
    f1_outro = f1_score(y, outro["pred_bert"], average="macro")

    print(f"\n===== {TAG} contra {tag_outro} ({args.split}) =====")
    print(f"{TAG:<10} accuracy {acc_este:.3f} | F1 macro {f1_este:.3f}")
    print(f"{tag_outro:<10} accuracy {acc_outro:.3f} | F1 macro {f1_outro:.3f}")

    b2 = int((acerto_este & ~acerto_outro).sum())  # só este acertou
    c2 = int((~acerto_este & acerto_outro).sum())  # só o outro acertou
    dif2 = (b2 - c2) / n
    erro_padrao2 = np.sqrt(b2 + c2 - (b2 - c2) ** 2 / n) / n
    p2 = binomtest(b2, b2 + c2, 0.5).pvalue if (b2 + c2) > 0 else 1.0
    print(f"\nSó {TAG} acertou: {b2} | só {tag_outro} acertou: {c2}")
    print(f"Diferença de accuracy: {dif2:+.3f}")
    print(f"Intervalo de 95%: [{dif2 - 1.96 * erro_padrao2:+.3f}, {dif2 + 1.96 * erro_padrao2:+.3f}]")
    print(f"Teste exato de McNemar: p = {p2:.4f}")

    # regra definida antes: F1 macro pelo menos +0,01 e accuracy sem cair
    regra = (f1_este - f1_outro >= 0.01) and (acc_este >= acc_outro)
    print(f"\nRegra (F1 macro pelo menos +0,01 e accuracy sem cair): {'CUMPRIDA' if regra else 'NÃO cumprida'} por {TAG}")

