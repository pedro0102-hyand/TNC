import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
NUM_LABELS = 5

# Check if MPS (Metal Performance Shaders) is available and set the device accordingly
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(device)
model.eval()

# Count the number of parameters in the model
n_params = sum(p.numel() for p in model.parameters())
print(f"Modelo carregado: {MODEL_NAME}")
print(f"Número de parâmetros: {n_params/1e6:.2f} milhões")

# Test the tokenizer with a sample text
texto = "O Brasil é um país de dimensões continentais e possui uma diversidade cultural impressionante."
inputs = tokenizer(texto, return_tensors="pt").to(device)
print("Tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

with torch.no_grad():
    logits = model(**inputs).logits

print("Forma dos logits:", tuple(logits.shape))
if device.type == "mps":
    print(f"Memória no mps: {torch.mps.current_allocated_memory() / 1e6:.0f} MB")