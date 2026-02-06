import torch
import torch.nn as nn
import math

# ---- Self Attention ----
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_dim // heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, return_attention=False):
        B, T, C = x.shape

        q = self.q(x).view(B, T, self.heads, self.head_dim)
        k = self.k(x).view(B, T, self.heads, self.head_dim)
        v = self.v(x).view(B, T, self.heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out(out)

        if return_attention:
            return out, attn

        return out


# ---- Feed Forward ----
class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

# ---- Transformer Block ----
class Block(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.attn = SelfAttention(embed_dim, heads)
        self.ff = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.ln1(x), True)
            x = x + attn_out
            x = x + self.ff(self.ln2(x))
            return x, attn_weights

        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ---- Full LM ----
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, heads=4, layers=2, seq_len=64):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(seq_len, embed_dim)

        self.blocks = nn.Sequential(*[
            Block(embed_dim, heads) for _ in range(layers)
        ])

        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, return_attention=False):
        B, T = x.shape

        tok = self.token_embed(x)
        pos = self.pos_embed(torch.arange(T, device=x.device))

        x = tok + pos

        attentions = []

        for block in self.blocks:
            if return_attention:
                x, attn = block(x, True)
                attentions.append(attn)
            else:
                x = block(x)

        x = self.ln(x)
        logits = self.fc(x)

        if return_attention:
            return logits, attentions

        return logits

