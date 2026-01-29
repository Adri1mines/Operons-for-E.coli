import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

class RoPEAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()

        self.n_head = n_head
        self.head_dim = d_model//n_head

        self.attn = nn.Linear(d_model, 3*d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.rotary_emb = RotaryEmbedding(dim = self.head_dim)

    def forward(self, x, mask=None):
        B, T, C = x.size() # Batch, Time, Channels
        
        # 1. Calcul Q, K, V
        qkv = self.attn(x)
        q, k, v = qkv.chunk(3, dim=2)
        
        # 2. Reshape pour les têtes [Batch, Heads, Time, Head_Dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # 3. APPLICATION DU ROPE (Rotation des Q et K)
        # C'est ça qui remplace l'embedding de position absolu
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        
        # 4. Attention classique (Scaled Dot Product)
        # On utilise l'optimisation Flash Attention de PyTorch 2.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)
        
        # 5. Réassemblage
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)
    


class TransformerBlock(nn.Module):
    """Un bloc complet : Norm -> Attention -> Add -> Norm -> FFN -> Add"""
    def __init__(self, d_model, n_head):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEAttentionBlock(d_model, n_head)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # Connexion résiduelle 1 (Pre-Norm architecture)
        x = x + self.attn(self.norm1(x))
        # Connexion résiduelle 2
        x = x + self.mlp(self.norm2(x))
        return x

class DNATransformerRoPE(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Création de la liste de couches (Empilement)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x : [Batch, Time]
        x = self.token_embedding(x)
        
        # Passage dans toutes les couches
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits


if __name__ == "__main__":
    # Paramètres de test
    vocab_size = 100 # Petit vocab
    d_model = 64     # Doit être divisible par n_head
    n_head = 4       # 64 / 4 = 16 (head_dim)
    layers = 2
    seq_len = 50     # Longueur factice
    batch_size = 2

    print("--- Test de l'architecture RoPE ---")
    
    # 1. Instanciation
    try:
        model = DNATransformerRoPE(vocab_size, d_model, n_head, layers)
        print("✅ Modèle instancié avec succès.")
    except Exception as e:
        print(f"❌ Erreur instanciation : {e}")
        exit()

    # 2. Forward Pass (Test des dimensions)
    try:
        # Création d'un batch bidon d'entiers (indices tokens)
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        output = model(dummy_input)
        
        print(f"✅ Forward pass réussi !")
        print(f"   Input shape: {dummy_input.shape}  (Batch, Seq)")
        print(f"   Output shape: {output.shape} (Batch, Seq, Vocab)")
        
        assert output.shape == (batch_size, seq_len, vocab_size), "Erreur de dimension en sortie !"
        print("✅ Dimensions de sortie valides.")
        
    except Exception as e:
        print(f"❌ Erreur pendant le forward : {e}")
        import traceback
        traceback.print_exc()

