import torch
from torch.utils.data import Dataset
import json
import os
from tokenizers import Tokenizer
from transformers import AutoTokenizer
import pandas as pd
from Bio.Seq import Seq

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, 'r') as f:
    config_param = json.load(f)


class FinetuneDataset(Dataset):

    def __init__(self, finetune_data_path = config_param["data"]["data_clean_filepath"], 
                 tokenizer_decoder = config_param["tokenizer"]["tokenizer_filepath"], 
                 esm_model_name = "facebook/esm2_t6_8M_UR50D",
                 max_len = config_param["model"]["max_len"]):
        esm_model_name="facebook/esm2_t6_8M_UR50D"
        # print(tokenizer_decoder)
        # try:
        #     tok = Tokenizer.from_file(tokenizer_decoder)
        #     print(f"✅ Tokenizer chargé depuis : {tokenizer_decoder}")
        #     print(f"Taille du vocabulaire : {tok.get_vocab_size()}")
            
        #     # Test des tokens
        #     specials = ["[START_CONTEXT]", "[UNK]", "[END_CONTEXT]", "[TERM_INFERRED]", " ", "A", "C", "T", "G"]
        #     for s in specials:
        #         id_ = tok.token_to_id(s)
        #         if id_ is not None:
        #             print(f"  - Token '{s}' : TROUVÉ (ID: {id_})")
        #         else:
        #             print(f"  - Token '{s}' : ❌ NON TROUVÉ (C'est peut-être lui le coupable)")

        # except Exception as e:
        #     print(f"❌ Impossible de charger le fichier : {e}")
        self.tokenizer_decoder = Tokenizer.from_file(tokenizer_decoder)
        self.tokenizer_decoder.unk_token = "[UNK]"
        self.tokenizer_esm = AutoTokenizer.from_pretrained(esm_model_name, padding = 'max_length',
                                                            truncation = 'longest_first',
                                                             max_length = max_len)
        self.tokenizer_decoder.enable_padding(
            direction='right',
            pad_id= self.tokenizer_decoder.token_to_id("[PAD]"), 
            pad_token="[PAD]", 
            length=max_len
        )
        self.tokenizer_decoder.enable_truncation(max_length=max_len)
        self.finetune_data = pd.read_csv(finetune_data_path, sep = ",")


    def __len__(self):
        return len(self.finetune_data)


    def __getitem__(self, idx):
        
        row = self.finetune_data.iloc[idx]
        input_prot = str(row["translated_prot"])
        input_decoder = str(row["Input_context"]) + str(row["Target"])

        print(f"DEBUG:INPUT_DECODER {input_decoder}, et INPUT_PROT {input_prot}")

        encoded_input_decoder = self.tokenizer_decoder.encode(input_decoder)
        input_ids_prot = self.tokenizer_esm.encode(input_prot)

        input_ids_decoder = encoded_input_decoder.ids
        print(input_ids_decoder)
        print(self.tokenizer_decoder.decode(input_ids_decoder))
        input_ids_decoder = torch.tensor(input_ids_decoder, dtype = torch.long)
        input_ids_prot = torch.tensor(input_ids_prot, dtype = torch.long)
        prot_mask = (input_ids_prot != 0).long()
        return({"input_protein": input_ids_prot,
                "protein_attention_mask": prot_mask,
                 "input_decoder": input_decoder})


if __name__ == "__main__":
    import sys
    
    # --- 1. CRÉATION DE DONNÉES FACTICES POUR LE TEST ---
    # On crée un CSV temporaire pour simuler tes données
    dummy_csv_path = "dummy_test_data.csv"
    dummy_data = {
        "translated_prot": ["MKLVINGFGRIGRLVLRACMEKGKVVAVV", "MSEQHVNKVALIAGAG"],
        "Input_context": ["[START_CONTEXT][TERM_INFERRED][END_CONTEXT]", "[START_CONTEXT][TERM_KNOWN][END_CONTEXT]"],
        "Target": ["ATGCGTATCGGCTAG", "ATGGGCCCCTTTTAA"]
    }
    pd.DataFrame(dummy_data).to_csv(dummy_csv_path, index=False)
    print(f"✅ Fichier de test créé : {dummy_csv_path}")

    # --- 2. INITIALISATION DU DATASET ---
    print("\n🚀 Chargement du Dataset...")
    
    # NOTE: Pour le test, on passe les arguments manuellement pour éviter de charger ton 'config.json' réel
    # Assure-toi que 'tokenizer_dna.json' existe ou remplace par un chemin valide
    try:
        dataset = FinetuneDataset(
            finetune_data_path=dummy_csv_path,
            max_len=128 # On met petit pour le test
        )
        
        print(f"Taille du dataset : {len(dataset)}")

        # --- 3. TEST DE __GETITEM__ ---
        print("\n🔍 Inspection du premier élément...")
        sample = dataset[0]

        # Vérification des clés
        print(f"Clés disponibles : {list(sample.keys())}")

        # Vérification des formes (Shapes)
        prot_ids = sample["input_protein"]
        prot_mask = sample["protein_attention_mask"]
        dec_input = sample["input_decoder"] # Attention: vérifie si c'est un tenseur ou int list

        print(f"Shape Protéine IDs : {prot_ids.shape} (Type: {prot_ids.dtype})")
        print(f"Shape Protéine Mask: {prot_mask.shape}")
        print(f"Shape Decoder Input: {dec_input.shape} (Type: {dec_input.dtype})")

        # Vérification visuelle
        print(f"\nExemple de masque (10 premiers) : {prot_mask[:10].tolist()}")
        
        # Test de cohérence
        assert prot_ids.shape == prot_mask.shape, "❌ Erreur : Le masque et l'input protéine n'ont pas la même taille !"
        print("\n✅ Test passé avec succès !")

    except Exception as e:
        print(f"\n❌ ERREUR PENDANT LE TEST : {e}")
        print("Conseil : Vérifie les chemins des tokenizers et les bugs signalés ci-dessous.")
    
    # Nettoyage (optionnel)
    # os.remove(dummy_csv_path)