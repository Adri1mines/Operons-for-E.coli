import os
import json
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
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
def training_iterator_sampled(train_set_filepath, chunk_size = 10000, max_chunks = dic_token_param["chunks_trained_on"]):
    with open(train_set_filepath, 'r') as f:
        count = 0
        while True and count < max_chunks:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            else:
                count += 1
                yield chunk


def main():

    print(f"🏗️  Initialisation du Tokenizer BPE...")
    
    # 1. Création du modèle BPE vide
    tokenizer = Tokenizer(Unigram())

    bases = ["A", "C", "G", "T"]
    
    # 2. Configuration de l'entraîneur
    trainer = UnigramTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"] + dic_token_param["special_codons"]
        + [dic_data_param["separateur"]],
        initial_alphabet= bases + [ b1 + b2 + b3 for b1 in bases for b2 in bases for b3 in bases], # On force l'alphabet de base et les codons
        show_progress = True)

    # 3. Entraînement
    print(f"Entraînement sur {CLEAN_GENOME_PATH}...")
    tokenizer.train_from_iterator(training_iterator_sampled(CLEAN_GENOME_PATH), trainer=trainer)
    
    # 5. SAUVEGARDE !
    tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizer sauvegardé sous : {TOKENIZER_PATH}")

if __name__ == "__main__":
    main()


