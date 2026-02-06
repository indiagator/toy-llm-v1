import torch
from tokenizers import Tokenizer
from model.transformer_lm import TransformerLM

def generate(model, tokenizer, prompt, max_len=50):
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor(ids).unsqueeze(0)

    for _ in range(max_len):
        logits = model(x)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()

        x = torch.cat([x, torch.tensor([[next_token]])], dim=1)

    return tokenizer.decode(x.squeeze().tolist())

def main():
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    model = TransformerLM(vocab_size)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    while True:
        prompt = input("\nEnter prompt (or type 'exit'): ")

        if prompt.lower() == "exit":
            break

        output = generate(model, tokenizer, prompt)
        print("\nGenerated:\n", output)


if __name__ == "__main__":
    main()
