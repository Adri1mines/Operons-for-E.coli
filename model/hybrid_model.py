import torch
import torch.nn as nn
from model.model_encoder import ProteinEncoder
from model.model_decoder import DNATransformerLlama

class ProteinGuidedGen(nn.Module):

    def __init__(self, vocab_size, d_model, n_head, num_layers, esm_model_name, use_encoder = False):
        super().__init__()
        self.use_encoder = use_encoder
        
        self.decoder = DNATransformerLlama(vocab_size, d_model, n_head, num_layers)

        if self.use_encoder:
            self.encoder = ProteinEncoder(esm_model_name, d_model)
        else:
            self.encoder = None

                
    def forward(self, x, protein_ids = None, prot_mask = None):

        if self.use_encoder:
            context = self.encoder(protein_ids, prot_mask)
        else:
            context = None
        x = self.decoder(x, context)

        return x

