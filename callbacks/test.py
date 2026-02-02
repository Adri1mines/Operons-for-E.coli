import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

# --- IMPORTS DE TON CODE ---
# Ajuste les chemins selon ta structure de dossiers
from model.processor import DNAProcessor 
from dataset.pretrain.dataset import PretrainDataset
from callbacks.bio_test import BioEvalCallback

# --- MOCK CLASSES (Pour simuler WandB sans le lancer) ---
class MockLogger:
    def log(self, metrics):
        print("\n[MOCK WandB] Metrics reçues :")
        for k, v in metrics.items():
            print(f"  - {k}: {v}")
    
    @property
    def experiment(self):
        return self # Pour gérer logger.experiment.log(...)

    def log(self, *args, **kwargs):
        pass # On ignore les appels complexes pour le test

class MockTrainer:
    """Imite le Trainer Lightning juste pour le callback"""
    def __init__(self):
        self.logger = MockLogger()
        self.global_step = 0
        self.current_epoch = 0

def test_everything():
    print("=== 1. INITIALISATION ===")
    
    # 1. Config bidon pour le test
    config = {
        "tokenizer": {
            "tokenizer_filepath": "tokenizer/tokenizer.json",
             "vocab_size": 1024, # <--- METS TON CHEMIN ICI
        },
        "model": { 
            "d_model": 128,      # Petit modèle pour le test
            "n_head": 4,
            "n_layer": 2,
            "max_len": 128
        },
        "training": {
            "learning_rate": 3e-4
        }
    }

    # 2. Chargement du modèle
    try:
        model = DNAProcessor()
        model.eval() # Important
        print("✅ Modèle chargé avec succès.")
    except Exception as e:
        print(f"❌ Erreur chargement modèle : {e}")
        return

    # 3. Chargement d'un mini dataset (juste 10 items pour aller vite)
    # Remplace par le chemin vers ton vrai fichier .txt ou un petit fichier test
    clean_genome_path = "dataset/processed/clean_genome.txt" # <--- METS TON CHEMIN ICI
    
    try:
        full_dataset = PretrainDataset()
        # On ne garde que 50 exemples pour que les calculs de stats soient instantanés
        mini_dataset = Subset(full_dataset, range(min(50, len(full_dataset))))
        # Hack pour que le callback accède au tokenizer via le subset
        mini_dataset.tokenizer = full_dataset.tokenizer 
        print(f"✅ Dataset chargé (taille test: {len(mini_dataset)})")
    except Exception as e:
        print(f"❌ Erreur chargement dataset : {e}")
        return

    print("\n=== 2. TEST DE GÉNÉRATION (generate_sequences) ===")
    try:
        sequences = model.generate_sequences(n_sequence=3, max_len=50)
        print("Sortie brute du modèle (non entraîné, donc charabia attendu) :")
        for i, seq in enumerate(sequences):
            print(f"  Seq {i+1}: {seq}")
        
        if len(sequences) > 0 and isinstance(sequences[0], str):
            print("✅ La fonction generate_sequences fonctionne !")
        else:
            print("❌ La fonction ne renvoie pas de strings.")
    except Exception as e:
        print(f"❌ Crash pendant la génération : {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n=== 3. TEST DU CALLBACK (BioEvalCallback) ===")
    try:
        # On instancie le callback
        callback = BioEvalCallback(
            val_dataset=mini_dataset,
            num_samples=5,  # Très peu pour que ça aille vite
            max_len=50
        )
        
        # On crée un faux trainer
        mock_trainer = MockTrainer()
        
        print("Lancement de 'on_validation_epoch_end'...")
        # On appelle manuellement la fonction qui est normalement appelée par Lightning
        callback.on_validation_epoch_end(mock_trainer, model)
        
        print("✅ Le Callback a fini sans crasher !")
        
    except Exception as e:
        print(f"❌ Crash dans le Callback : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_everything()