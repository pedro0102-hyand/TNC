import argparse
import json
import re
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def limpar_texto(t):

    return re.sub(r"\s+", " ", t).strip()

def limpar_titulo(t):

    t = re.sub(r"Duration,\s*[\d,.:]+\s*$", "", t)
    t = re.sub(r"^(?:Vídeo|Áudio),\s*(?:Em áudio\s*\|\s*)?", "", t)
    return limpar_texto(t)

def carregar_modelo(bert_dir):

    bert_dir = Path(bert_dir)
    config = bert_dir / "treino.json"
    max_length = json.loads(config.read_text(encoding="utf-8"))["max_length"] if config.exists() else 256
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(bert_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(bert_dir)).to(device).eval()
    return tokenizer, model, device, max_length

def prever(titulo, texto, tokenizer, model, device, max_length, top=3):

    entrada = f"{limpar_titulo(titulo)} {limpar_texto(texto)}"
    n_tokens = len(tokenizer(entrada, verbose=False)["input_ids"])
    inputs = tokenizer(entrada, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    melhores = torch.topk(probs, top)
    previsoes = [(model.config.id2label[i.item()], p.item()) for p, i in zip(melhores.values, melhores.indices)]
    return previsoes, n_tokens

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--titulo")
    parser.add_argument("--texto")
    parser.add_argument("--arquivo", help="arquivo .txt com o texto da notícia")
    parser.add_argument("--bert-dir", default="models/bert_512", help="pasta do modelo BERT")
    args = parser.parse_args()

    # entrada: argumentos ou, se faltarem, perguntas no terminal
    titulo = args.titulo or input("Título: ")
    if args.arquivo:
        texto = Path(args.arquivo).read_text(encoding="utf-8")
    else:
        texto = args.texto or input("Texto (em uma linha só): ")

    if len(limpar_texto(texto)) < 200:
        print("Aviso: o modelo foi treinado com textos de pelo menos 200 caracteres.")

    tokenizer, model, device, max_length = carregar_modelo(args.bert_dir)
    previsoes, n_tokens = prever(titulo, texto, tokenizer, model, device, max_length)

    print(f"\nTítulo: {limpar_titulo(titulo)}")
    print(f"Tokens da entrada: {n_tokens} (o modelo leu os primeiros {min(n_tokens, max_length)})")
    print("\nPrevisão:")
    for classe, prob in previsoes:
        print(f"  {classe:<15} {prob:6.1%}  {'█' * int(prob * 30)}")

if __name__ == "__main__":
    main()