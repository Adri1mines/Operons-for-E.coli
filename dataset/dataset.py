import torch
from torch.utils.data import Dataset
import json
import os
from tokenizers import Tokenizer

DATA_PARAM_PATH = "dataset/data_parameters.json"
TOKENIZER_PARAM_PATH = "tokenizer/tokenizer_param.json"
with open(DATA_PARAM_PATH, 'r') as f:
    dic_data_param = json.load(f)
with open(TOKENIZER_PARAM_PATH, 'r') as f:
    dic_token_param = json.load(f)


class PretrainDataset(Dataset):

    def __init__(self, clean_genome_path = dic_data_param["data_clean_filepath"], 
                 tokenizer = dic_token_param["tokenizer_filepath"], 
                 chunk_size = dic_data_param["chunk_size"], 
                 stride = dic_data_param["stride"],
                 max_len = 1000):
        self.tokenizer = Tokenizer.from_file(tokenizer)
        self.tokenizer.enable_padding(
            direction='right', 
            pad_id=0, 
            pad_token="[PAD]", 
            length=max_len
        )
        self.tokenizer.enable_truncation(max_length=max_len)
        self.chunk_size = chunk_size
        self.clean_genome_path = clean_genome_path
        self.offsets = []
        file_len = os.path.getsize(clean_genome_path)
        #Indexation du fichier
        print(f"Indexation du fichier ({file_len}) en cours...")
        for i in range(0, file_len - chunk_size + 1, stride):
            self.offsets.append(i)
        print("Indexation terminée")

    def __len__(self):
        return len(self.offsets)


    def __getitem__(self, idx):
        
        offset = self.offsets[idx]
        #On read binary parce que l'indice représente l'octet, et on veut éviter un caractère spécial
        #qui viendrait casser l'indexing
        try:
            with open(self.clean_genome_path, 'r', encoding='utf-8') as f:#!!! NE MARCHE PAS EN BINARY POURQUOI????
                f.seek(offset)
                chunk = f.read(self.chunk_size)
        except UnicodeDecodeError as e:
            print("ERREUR: Le fichier contient des caractères spéciaux non tolérés")
        except Exception as e:
            print("Erreur lors du chargement du dataset")
        encoding = self.tokenizer.encode(chunk)
        # On récupère les IDs (les nombres)
        ids = encoding.ids
        #gestion du cas ou on retrouve un séparateur de chunk
        chunk_sep_id = self.tokenizer.token_to_id("###")
        if chunk_sep_id in ids:
            #On tronque au niveau du séparateur de chunk, tout ce qu'il y a après on le remplace
            #par du padding
            pad_id = self.tokenizer.token_to_id("[PAD]")
            ids_to_keep = ids[:ids.index(chunk_sep_id)]
            padding_len = len(ids[ids.index(chunk_sep_id):])
            padding = [pad_id for _ in range(padding_len)]
            ids = ids_to_keep + padding
        #on rajoute les tokens de début et fin de séquence
        cls_id = self.tokenizer.token_to_id("[CLS]")
        sep_id = self.tokenizer.token_to_id("[SEP]")        
        ids = [cls_id] + ids + [sep_id]
        
        return torch.tensor(ids, dtype=torch.long)

