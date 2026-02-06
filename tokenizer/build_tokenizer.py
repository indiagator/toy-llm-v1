from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "corpus.txt"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"


def build_tokenizer():
    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=1000,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )

    tokenizer.train([str(DATA_PATH)], trainer)
    tokenizer.save(str(TOKENIZER_PATH))

if __name__ == "__main__":
    build_tokenizer()
