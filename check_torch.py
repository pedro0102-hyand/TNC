import torch

print("PyTorch version:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())
print("mps built:", torch.backends.mps.is_built())

if torch.backends.mps.is_available():
    x = torch.ones(3, device="mps")
    print("tensor no mps:", x * 2)
else:
    print("mps indisponível: o treino cairia na CPU")