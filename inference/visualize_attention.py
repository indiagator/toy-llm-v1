import torch
import matplotlib.pyplot as plt
import seaborn as sns

from tokenizers import Tokenizer
from model.transformer_lm import TransformerLM

def visualize(prompt):

    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    model = TransformerLM(vocab_size)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    ids = tokenizer.encode(prompt).ids
    x = torch.tensor(ids).unsqueeze(0)

    logits, attentions = model(x, return_attention=True)

    tokens = tokenizer.encode(prompt).tokens

    # Visualize first layer, first head
    attn = attentions[0][0, 0].detach().numpy()

    # Swap axes: what used to be on X is now on Y, and vice versa
    attn_swapped = attn.T

    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_swapped, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title("Attention Map (Axes Swapped)")
    plt.xlabel("Current Token")
    plt.ylabel("Attended Tokens")
    plt.show()

if __name__ == "__main__":
    prompt = input("Enter prompt: ")
    visualize(prompt)