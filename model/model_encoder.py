import torch
import torch.nn as nn
from transformers import EsmModel

class ProteinEncoder(nn.Module):

    def __init__(self, esm_model_name, d_model, freeze_esm=True):
        super().__init__()
        

        print(f"Loading ESM: {esm_model_name}...")
        self.esm = EsmModel.from_pretrained(esm_model_name)       

        self.esm_dim = self.esm.config.hidden_size

        self.projector = nn.Linear(self.esm_dim, d_model)
        
        if freeze_esm:
            for param in self.esm.parameters():
                param.requires_grad = False

                
    def forward(self, protein_input, protein_attention_mask, decoder_input):
        """
        protein_input_ids: Tokens des protéines (Tokenisés par ESM tokenizer)
        dna_input_ids: Tokens de l'opéron (Tokenisés par TON tokenizer DNA)
        """

        with torch.no_grad(): # Pas de gradient pour ESM
            esm_outputs = self.esm(input_ids=protein_input, attention_mask=protein_attention_mask)
        
        # On récupère les embeddings de la dernière couche cachée
        # Shape: (Batch, Prot_Len, ESM_Dim)
        prot_emb = esm_outputs.last_hidden_state

        outputs = self.projector(prot_emb)
        
        return outputs