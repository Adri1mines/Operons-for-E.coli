import torch
import torch.nn as nn
import lightning as pl
from math import sqrt

class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, n_head):
        self.n_head = n_head
        self.d_k = d_model//n_head
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, x, pad_mask = None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q.view(x.size(0), x.size(1), self.n_head, -1).transpose(1, 2)
        k = k.view(x.size(0), x.size(1), self.n_head, -1).transpose(1, 2)
        v = v.view(x.size(0), x.size(1), self.n_head, -1).transpose(1, 2)
        qk = q@k.transpose(2, 3)/sqrt(self.d_k)
        mask = torch.ones((qk.size(2), qk.size(3)))
        qk = qk.masked_fill(torch.triu(mask) == 1, float('-inf'))
        qk = qk.masked_fill(pad_mask == 1, float('-inf'))
        att = torch.softmax(qk)@v


class DNATransformer(pl.LightningModule):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.emb_to_head = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        token_emb = self.embedding(x)
        pos_tokens = torch.arange(x.size(1), device = x.device)
        pos_emb = self.pos_embedding(pos_tokens)
        emb = token_emb + pos_emb
        to_attention = self.emb_to_head(emb)
