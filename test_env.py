import torch
import tokenizers
import numpy as np

print("Torch version:", torch.__version__)
print("Tokenizer loaded")
print("Numpy version:", np.__version__)
print("GPU Available:", torch.cuda.is_available())
