import torch

print("PyTorch:", torch.__version__)
print("MPS disponível:", torch.backends.mps.is_available())
print("MPS compilado:", torch.backends.mps.is_built())