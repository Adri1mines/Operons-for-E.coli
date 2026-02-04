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

                
    def forward(self, protein_input, protein_attention_mask):


        with torch.no_grad(): # Pas de gradient pour ESM
            esm_outputs = self.esm(input_ids=protein_input, attention_mask=protein_attention_mask)
        
        # On récupère les embeddings de la dernière couche cachée
        # Shape: (Batch, Prot_Len, ESM_Dim)
        prot_emb = esm_outputs.last_hidden_state

        outputs = self.projector(prot_emb)
        
        return outputs


if __name__ == "__main__":
    # --- Paramètres de Test ---
    # Utilise un petit modèle ESM pour que le test soit rapide (même sur CPU)
    ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D" 
    D_MODEL = 256  # Dimension cible de ton décodeur
    BATCH_SIZE = 2
    SEQ_LEN_PROT = 50
    
    print("\n--- 🧪 TEST DE PROTEIN ENCODER ---")

    # 1. Instanciation
    print(f"1. Instanciation du modèle avec {ESM_MODEL_NAME} -> d_model={D_MODEL}...")
    try:
        encoder = ProteinEncoder(
            esm_model_name=ESM_MODEL_NAME, 
            d_model=D_MODEL, 
            freeze_esm=True
        )
        print("✅ Modèle instancié avec succès.")
    except Exception as e:
        print(f"❌ Erreur critique à l'instanciation : {e}")
        exit()

    # 2. Vérification du Gel des Poids (Freezing)
    print("\n2. Vérification des gradients (Freeze)...")
    esm_frozen = all(not p.requires_grad for p in encoder.esm.parameters())
    projector_trainable = all(p.requires_grad for p in encoder.projector.parameters())
    
    if esm_frozen:
        print("✅ Poids ESM bien gelés (requires_grad=False).")
    else:
        print("❌ ATTENTION : Certains poids ESM sont entraînables !")
        
    if projector_trainable:
        print("✅ Projecteur bien entraînable (requires_grad=True).")
    else:
        print("❌ ATTENTION : Le projecteur est gelé !")

    # 3. Test du Forward Pass
    print("\n3. Test du Forward Pass...")
    
    # Création de données factices
    # Le vocabulaire ESM standard a une taille de ~33 tokens
    dummy_prot_ids = torch.randint(0, 33, (BATCH_SIZE, SEQ_LEN_PROT))
    dummy_mask = torch.ones((BATCH_SIZE, SEQ_LEN_PROT)) # Tout est utile (pas de padding ici)
    
    # Note: Ton forward demande 'decoder_input' mais ne l'utilise pas dans le code fourni.
    # On passe None pour tester la robustesse.
    dummy_decoder_input = None 

    try:
        outputs = encoder(dummy_prot_ids, dummy_mask, dummy_decoder_input)
        
        print("✅ Forward pass réussi !")
        print(f"   Input Shape  : {dummy_prot_ids.shape}")
        print(f"   Output Shape : {outputs.shape}")
        
        # Vérification finale des dimensions
        expected_shape = (BATCH_SIZE, SEQ_LEN_PROT, D_MODEL)
        if outputs.shape == expected_shape:
            print("✅ Dimensions de sortie valides.")
        else:
            print(f"❌ Erreur de dimension : Attendu {expected_shape}, Reçu {outputs.shape}")
            
    except Exception as e:
        print(f"❌ Erreur pendant le forward : {e}")
        import traceback
        traceback.print_exc()