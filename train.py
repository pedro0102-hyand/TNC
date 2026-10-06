import pandas as pd
import torch
import numpy as np
import time
import argparse
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding, Trainer, TrainingArguments)

SEED = 42
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LENGTH = 256
DATA_PATH = Path("data/processed/clean_texts.parquet")
df = pd.read_parquet(DATA_PATH)

unicas = df[df["n_categorias"] == 1].copy()
unicas["categoria"] = unicas["categorias"].map(lambda c: c[0])
unicas = unicas.reset_index(drop=True)

unicas["entrada"] = unicas["titulo"] + " " + unicas["texto"]
print(f"Notícias de categoria única: {len(unicas)}")

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
print("\nExemplos por categoria no teste:")
print(teste["categoria"].value_counts())

# convertendo categoria em numero e numero em categoria
classes = sorted(unicas["categoria"].unique())
label2id = {c: i for i, c in enumerate(classes)}
id2label = {i: c for c, i in label2id.items()}
print(f"\nlabel2id: {label2id}")
print(f"\nid2label: {id2label}")

# tokenizacao
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenizar(lote):
    return tokenizer(lote["entrada"], truncation = True, max_length = MAX_LENGTH)

def para_dataset(parte):
    ds = Dataset.from_dict(
        {
            "entrada": parte["entrada"].tolist(),
            "labels": parte["categoria"].map(label2id).tolist(),
        }
    )
    return ds.map(tokenizar, batched=True, remove_columns=["entrada"])

ds_treino = para_dataset(treino)
ds_val = para_dataset(val)
print(f"\nTreino tokenizado: {len(ds_treino)} | Validação tokenizada: {len(ds_val)}")
print(f"Colunas: {ds_treino.column_names}")

ids = ds_treino[0]["input_ids"]
print(f"\nTokens do primeiro exemplo: {len(ids)}")
print(f"Primeiros tokens: {tokenizer.convert_ids_to_tokens(ids[:15])}")
print(f"Último token: {tokenizer.convert_ids_to_tokens(ids[-1])}")
print(f"Rótulo: {ds_treino[0]['labels']} -> {id2label[ds_treino[0]['labels']]}")
no_limite = sum(len(x) == MAX_LENGTH for x in ds_treino["input_ids"]) / len(ds_treino)
print(f"Exemplos que chegaram ao limite de {MAX_LENGTH} tokens: {no_limite:.1%}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

# definindo o modelo de classificacao
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = len(classes), id2label = id2label, label2id = label2id)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parâmetros: {n_params / 1e6:.1f}M")

# Monta os lotes, completando cada um só até o seu texto mais longo
data_collator = DataCollatorWithPadding(tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, pred),
        "f1_macro": f1_score(labels, pred, average="macro"),
    }

# Verificação: um lote de 8 notícias passa pelo modelo, sem treinar
lote = data_collator([ds_treino[i] for i in range(8)])
lote = {k: v.to(device) for k, v in lote.items()}

model.to(device)
model.eval()
with torch.no_grad():
    saida = model(**lote)

print(f"\nFormato do lote: {tuple(lote['input_ids'].shape)}")
print(f"Formato dos logits: {tuple(saida.logits.shape)}")
print(f"Loss do lote (sem treino): {saida.loss.item():.3f}  (referência ln 9 = {np.log(9):.3f})")
if device.type == "mps":
    print(f"Memória no mps: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="teste rápido: 200 exemplos, 1 época")
args = parser.parse_args()

EPOCAS = 1 if args.smoke else 3

if args.smoke:
    ds_treino_run = ds_treino.shuffle(seed=SEED).select(range(200))
    ds_val_run = ds_val.shuffle(seed=SEED).select(range(100))
    print("\n*** MODO SMOKE: 200 exemplos de treino, 100 de validação, 1 época ***")
else:
    ds_treino_run, ds_val_run = ds_treino, ds_val

training_args = TrainingArguments(
    output_dir="models/checkpoints",
    num_train_epochs=EPOCAS,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no" if args.smoke else "epoch",
    load_best_model_at_end=not args.smoke,
    metric_for_best_model="f1_macro",
    save_total_limit=2,
    logging_steps=5 if args.smoke else 50,
    seed=SEED,
    report_to="none",
)

trainer = Trainer(model=model,args=training_args,train_dataset=ds_treino_run,eval_dataset=ds_val_run,processing_class=tokenizer,data_collator=data_collator,compute_metrics=compute_metrics)
print(f"Dispositivo usado pelo Trainer: {trainer.args.device}")

inicio = time.time()
resultado = trainer.train()
duracao = time.time() - inicio

m = resultado.metrics
print(f"\nTreino concluído em {duracao / 60:.1f} min")
print(f"Amostras por segundo: {m['train_samples_per_second']:.2f}")
if args.smoke:
    estimativa = len(ds_treino) * 3 / m["train_samples_per_second"] / 60
    print(f"Estimativa do treino completo (3 épocas): ~{estimativa:.0f} min")
    print("(pessimista: os primeiros passos são mais lentos)")
if device.type == "mps":
    print(f"Memória reservada no mps: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")

