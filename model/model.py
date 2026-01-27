import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
from tokenizers import Tokenizer


class DNATransformer(nn.Module):
    def __init__(self, vocab_size, d_model,n_head, max_len, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, 
                                                    nhead=n_head, 
                                                    dim_feedforward = 4 * d_model,
                                                    batch_first = True,
                                                    norm_first = True,
                                                    activation = "gelu")
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        token_emb = self.embedding(x)
        pos_tokens = torch.arange(x.size(1), device = x.device)
        pos_emb = self.pos_embedding(pos_tokens)
        x = token_emb + pos_emb
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        x = self.transformer_decoder(x, tgt_mask=causal_mask, is_causal=True)
        logits = self.output_head(x)
        return logits






