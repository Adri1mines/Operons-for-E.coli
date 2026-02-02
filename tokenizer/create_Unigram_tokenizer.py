import os
import json
from tokenizers import Tokenizer, pre_tokenizers, Regex
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing



CONFIG_PATH = "config.json"
with open(CONFIG_PATH, 'r') as f:
    config_param = json.load(f)

TOKENIZER_PATH = config_param["tokenizer"]["tokenizer_filepath"]
CLEAN_GENOME_PATH = config_param["data"]["data_clean_filepath"]
VOCAB_SIZE = config_param["tokenizer"]["vocab_size"]


#On ne peut pas charger en mémoire tout le dataset donc on utilise un itérateur
def training_iterator_sampled(train_set_filepath, chunk_size = 10000, max_chunks = config_param["tokenizer"]["chunks_trained_on"]):
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
    special_tokens = [
        "[UNK]",       # Inconnu
        "[CLS]",       # Start of Sequence 
        "[SEP]",       # Separator 
        "[PAD]",       # Padding
        "[MASK]",
        "[PROM]",      #tokens pour annotations bio
        "[RBS]",
        "[CDS]",
        "[TERM]"
        "[START_CONTEXT]", 
        "[END_CONTEXT]"
    ]
    # 2. Configuration de l'entraîneur
    trainer = UnigramTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens= special_tokens + config_param["tokenizer"]["special_codons"]
        + [config_param["data"]["separateur"]],
        initial_alphabet= bases + [ b1 + b2 + b3 for b1 in bases for b2 in bases for b3 in bases], # On force l'alphabet de base et les codons
        show_progress = True)

    # 3. Entraînement
    print(f"Entraînement sur {CLEAN_GENOME_PATH}...")
    tokenizer.train_from_iterator(training_iterator_sampled(CLEAN_GENOME_PATH), trainer=trainer)

    #ajout du preprocessing
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),
    pre_tokenizers.Split(
        pattern=Regex(r"\[[^\]]+\]"), 
        behavior="isolated"           
    )
])

    tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizer sauvegardé sous : {TOKENIZER_PATH}")

if __name__ == "__main__":
    main()


