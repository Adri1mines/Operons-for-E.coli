import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
from tokenizers import Tokenizer



class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()
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
        mask = torch.ones((qk.size(2), qk.size(3)), device = x.device)
        mask_pad = pad_mask.unsqueeze(1).unsqueeze(2)
        qk = qk.masked_fill(torch.triu(mask, diagonal = 1) == 1, float('-inf'))
        qk = qk.masked_fill(mask_pad == 1, float('-inf'))
        att = torch.softmax(qk, dim = -1)@v
        att = att.transpose(1, 2)
        att = att.reshape(x.size(0), x.size(1), -1)
        return att


class DNATransformer(nn.Module):
    def __init__(self, vocab_size, d_model,n_head, max_len, tokenizer):
        super().__init__()
        tokenizer = Tokenizer.from_file(tokenizer)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_token = tokenizer.token_to_id("[PAD]")
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.emb_to_head = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.attention_head = CausalSelfAttention(d_model, n_head)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.proj = nn.Linear(d_model, vocab_size)


    def forward(self, x):
        token_emb = self.embedding(x)
        pos_tokens = torch.arange(x.size(1), device = x.device)
        pos_emb = self.pos_embedding(pos_tokens)
        emb = token_emb + pos_emb
        norm_emb = self.norm(emb)
        to_attention = self.emb_to_head(norm_emb)
        pad_mask = torch.zeros(x.size(), device = x.device).masked_fill(x == self.pad_token, 1)
        attention = self.attention_head(to_attention, pad_mask)
        emb = emb + attention
        feed_forwarded = self.feed_forward(emb)
        emb = emb + feed_forwarded
        logits = self.proj(emb)
        return logits






