import pandas as pd
import torch
import numpy as np
import time
import sys
import argparse
import json
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

SEED = 42
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
DATA_PATH = Path("data/processed/clean_texts.parquet")

parser = argparse.ArgumentParser(description="Fine-tuning do BERT para classificar notícias")
parser.add_argument("--max-length", type=int, default=256, help="tokens por notícia")
parser.add_argument("--saida", default="models/bert", help="pasta do modelo final")
parser.add_argument("--batch-size", type=int, default=8, help="notícias por mini-lote")
parser.add_argument("--grad-acc", type=int, default=1, help="mini-lotes somados por atualização dos pesos")
parser.add_argument("--smoke", action="store_true", help="teste rápido: 200 exemplos, 1 época")
parser.add_argument("--sobrescrever", action="store_true", help="permite refazer uma saída que já existe")
args = parser.parse_args()

MAX_LENGTH = args.max_length # limite de tamanho dos textos em tokens
SAIDA = Path(args.saida) 
CHECKPOINTS_DIR = SAIDA.parent / f"checkpoints_{SAIDA.name}"

# trava para nao sobrescrever um modelo ja treinado sem querer
if SAIDA.exists() and not args.smoke and not args.sobrescrever:
    sys.exit(f"{SAIDA} já existe. Use outra --saida ou --sobrescrever para refazer.")
print(f"Configuração: max_length={MAX_LENGTH} | lote {args.batch_size} x acumulação {args.grad_acc} = {args.batch_size * args.grad_acc} | saída={SAIDA}")

df = pd.read_parquet(DATA_PATH)

# filtrando noticias que pertencem a uma unica categoria
unicas = df[df["n_categorias"] == 1].copy()
unicas["categoria"] = unicas["categorias"].map(lambda c: c[0])
unicas = unicas.reset_index(drop=True)

# concatenando titulo e texto da noticia
unicas["entrada"] = unicas["titulo"] + " " + unicas["texto"]
print(f"Notícias de categoria única: {len(unicas)}")

treino_val, teste = train_test_split(unicas, test_size=0.15, stratify=unicas["categoria"], random_state=SEED)
treino, val = train_test_split(treino_val,test_size=0.15 / 0.85,stratify=treino_val["categoria"],random_state=SEED)

# criando um dicionário com os três subconjuntos
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

# baixando o tokenizer do modelo BERT
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# tokeniza a entrada (texto + titulo)
def tokenizar(lote):
    return tokenizer(lote["entrada"], truncation = True, max_length = MAX_LENGTH)

def para_dataset(parte):
    ds = Dataset.from_dict(
        {
            "entrada": parte["entrada"].tolist(),
            "labels": parte["categoria"].map(label2id).tolist(),
        }
    )

    # aplicando tokenizacao em lote do dataset
    return ds.map(tokenizar, batched=True, remove_columns=["entrada"])

# gera os datasets de treino e validacao já tokenizados
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

# calculando as métricas de avaliacao do modelo
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

EPOCAS = 1 if args.smoke else 3 # definindo a quantidade de vezes que passamos pelos dados de treino

if args.smoke:
    ds_treino_run = ds_treino.shuffle(seed=SEED).select(range(200))
    ds_val_run = ds_val.shuffle(seed=SEED).select(range(100))
    print("\n*** MODO SMOKE: 200 exemplos de treino, 100 de validação, 1 época ***")
else:
    ds_treino_run, ds_val_run = ds_treino, ds_val

# definindo as configuracoes do treinamento do modelo
training_args = TrainingArguments(
    output_dir=str(CHECKPOINTS_DIR),
    num_train_epochs=EPOCAS,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_acc, # soma o gradiente de varios mini-lotes antes de atualizar os pesos
    per_device_eval_batch_size=16 if MAX_LENGTH <= 256 else 8, # na validacao, nao há atualizacao dos pesos do modelo
    learning_rate=2e-5,
    weight_decay=0.01, # regularizacao L2, penaliza valores muito grandes para evitar overfitting
    eval_strategy="epoch",
    save_strategy="no" if args.smoke else "epoch",
    load_best_model_at_end=not args.smoke,
    metric_for_best_model="f1_macro",
    save_total_limit=2,
    logging_steps=5 if args.smoke else 50,
    seed=SEED,
    report_to="none",
)

# treinando o BERT
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

if not args.smoke:
    trainer.save_model(str(SAIDA))
    tokenizer.save_pretrained(str(SAIDA))

    # guardando a configuracao do treino ao lado do modelo
    config_treino = {
        "max_length": MAX_LENGTH,
        "epocas": EPOCAS,
        "batch_size": args.batch_size,
        "grad_acc": args.grad_acc,
        "learning_rate": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
        "seed": SEED,
        "modelo_base": MODEL_NAME,
        "f1_macro_val": trainer.state.best_metric,
        "tempo_min": round(duracao / 60, 1),
    }
    (SAIDA / "treino.json").write_text(json.dumps(config_treino, indent=2), encoding="utf-8")
    print(f"\nModelo salvo em {SAIDA}")

