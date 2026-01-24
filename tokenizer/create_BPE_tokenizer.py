import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing



DATA_PARAM_PATH = "dataset/data_parameters.json"
TOKENIZER_PARAM_PATH = "tokenizer/tokenizer_param.json"
with open(DATA_PARAM_PATH, 'r') as f:
    dic_data_param = json.load(f)
with open(TOKENIZER_PARAM_PATH, 'r') as f:
    dic_token_param = json.load(f)

TOKENIZER_PATH = dic_token_param["tokenizer_path"]
CLEAN_GENOME_PATH = dic_data_param["data_clean_filepath"]
VOCAB_SIZE = dic_token_param["vocab_size"]


#On ne peut pas charger en mémoire tout le dataset donc on utilise un itérateur
def training_iterator(train_set_filepath, chunk_size = 10000):
    with open(train_set_filepath, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            else:
                yield chunk


def main():

    print(f"🏗️  Initialisation du Tokenizer BPE...")
    
    # 1. Création du modèle BPE vide
    tokenizer = Tokenizer(BPE())
    
    # 2. Configuration de l'entraîneur
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"] + dic_token_param["special_codons"],
        initial_alphabet=["A", "C", "G", "T"], # On force l'alphabet de base
        show_progress = True)

    # 3. Entraînement
    print(f"Entraînement sur {CLEAN_GENOME_PATH}...")
    tokenizer.train_from_iterator(training_iterator(CLEAN_GENOME_PATH), trainer=trainer)
    
    # 5. SAUVEGARDE !
    tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizer sauvegardé sous : {TOKENIZER_PATH}")

if __name__ == "__main__":
    main()


