from transformers import AutoTokenizer, AutoModel

model_name = "neuralmind/bert-base-portuguese-cased"

print("Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Carregando modelo...")
model = AutoModel.from_pretrained(model_name)

print("BERTimbau carregado com sucesso!")
print(f"Parâmetros: {model.num_parameters():,}")