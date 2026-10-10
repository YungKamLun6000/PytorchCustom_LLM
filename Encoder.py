import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, max_length, hidden_size, head):
        super(Encoder, self).__init__()
        self.d_k = embed_size//head

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encode = nn.Embedding(max_length, embed_size)
        self.Q = nn.Linear(embed_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)

        self.feed_forward = nn.Linear(hidden_size, embed_size)
        self.LayerNorm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        seq_len = x.size(1)
        encoded = self.embed(x)
        pos = self.pos_encode(torch.arange(seq_len, device=x.device).unsqueeze(0))
        x = encoded + pos
        skip = x
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        attention_weights = F.softmax((Q @ K.transpose(1, 2) / self.d_k), dim=-1)
        z = attention_weights @ V
        z = z + skip
        z = self.LayerNorm(z)

        skip2 = z
        out = self.feed_forward(z)
        out = out + skip2
        out = self.LayerNorm2(out)
        return out


