import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel
from processor import DNAProcessor

def load_compiled_checkpoint(checkpoint_path, model_class):
    # 1. Charger le fichier brut (sur CPU pour éviter de saturer la VRAM)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 2. Récupérer le dictionnaire de poids
    state_dict = checkpoint["state_dict"]
    
    # 3. Créer un nouveau dictionnaire "propre"
    new_state_dict = {}
    for key, value in state_dict.items():
        # L'astuce est ici : on supprime "_orig_mod." s'il est présent
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value
        
    # 4. Instancier le modèle avec les hyperparamètres sauvegardés
    # (Cela suppose que tu as utilisé self.save_hyperparameters() dans ton __init__)
    hparams = checkpoint.get("hyper_parameters", {})
    model = model_class(**hparams)
    
    # 5. Charger les poids nettoyés
    model.load_state_dict(new_state_dict)
    
    return model

class ProteinGuidedGen(nn.Module):

    def __init__(self, dna_decoder_filepath, esm_model_name="facebook/esm2_t6_8M_UR50D", freeze_esm=True):
        super().__init__()
        

        print(f"Loading ESM: {esm_model_name}...")
        self.esm = EsmModel.from_pretrained(esm_model_name)
        self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)

        pretrained_processor = load_compiled_checkpoint(dna_decoder_filepath, DNAProcessor)
   
        self.decoder = pretrained_processor.model
        

        self.esm_dim = self.esm.config.hidden_size
        self.dna_dim = self.decoder.config.d_model # Assure-toi que ta config a d_model
        
        self.projector = nn.Linear(self.esm_dim, self.dna_dim)
        
        if freeze_esm:
            for param in self.esm.parameters():
                param.requires_grad = False

                
    def forward(self, protein_input, protein_attention_mask, decoder_input):
        """
        protein_input_ids: Tokens des protéines (Tokenisés par ESM tokenizer)
        dna_input_ids: Tokens de l'opéron (Tokenisés par TON tokenizer DNA)
        """
        
        # A. ENCODAGE PROTÉINE
        # On passe les protéines dans ESM
        with torch.no_grad(): # Pas de gradient pour ESM
            esm_outputs = self.esm(input_ids=protein_input, attention_mask=protein_attention_mask)
        
        # On récupère les embeddings de la dernière couche cachée
        # Shape: (Batch, Prot_Len, ESM_Dim)
        prot_embeddings = esm_outputs.last_hidden_state
        
        # B. PROJECTION
        # On transforme les vecteurs ESM en vecteurs "compatibles DNA"
        # Shape: (Batch, Prot_Len, DNA_Dim)
        projected_embeddings = self.projector(prot_embeddings)
        
        # C. PRÉPARATION INPUT DECODEUR
        # Le décodeur a besoin des embeddings de l'ADN
        # Attention: self.decoder.token_embedding est ta couche d'embedding DNA
        dna_embeddings = self.decoder.token_embedding(decoder_input)
        
        # --- LA FUSION (Soft Prompting) ---
        # On colle les vecteurs de protéines AVANT les vecteurs d'ADN
        # Le modèle "voit" la protéine, puis commence à générer l'ADN
        combined_embeddings = torch.cat([projected_embeddings, dna_embeddings], dim=1)
        
        # D. PASSAGE DANS LE TRANSFORMER
        # Note: Il faut adapter ton DNATransformer pour qu'il accepte 'inputs_embeds'
        # au lieu de 'x' (indices). C'est standard dans les LLMs.
        
        # Si ton modèle ne prend que 'x' (indices), tu dois modifier sa méthode forward
        # pour accepter 'inputs_embeds'.
        outputs = self.decoder(inputs_embeds=combined_embeddings)
        
        return outputs