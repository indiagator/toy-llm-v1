import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class TextDataset(Dataset):
    def __init__(self, text, tokenizer_path, seq_len=64):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.seq_len = seq_len

        ids = self.tokenizer.encode(text).ids
        self.data = ids

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]

        return torch.tensor(x), torch.tensor(y)
