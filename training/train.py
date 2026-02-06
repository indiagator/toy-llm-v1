import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path

from dataset.dataset import TextDataset
from model.transformer_lm import TransformerLM
from tokenizers import Tokenizer

SEQ_LEN = 64
BATCH = 32
EPOCHS = 5
LR = 3e-4

def load_text():
    return Path("data/corpus.txt").read_text()

def main():
    text = load_text()

    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    dataset = TextDataset(text, "tokenizer/tokenizer.json", SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    model = TransformerLM(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for x, y in loader:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: {loss.item()}")

    torch.save(model.state_dict(), "model.pt")

if __name__ == "__main__":
    main()
