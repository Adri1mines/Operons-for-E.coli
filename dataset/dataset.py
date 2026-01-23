import torch
from torch.utils.data import Dataset
import json
import os

JSON_PATH = "dataset/data_parameters.json"

with open(JSON_PATH, 'r') as f:
    dic_param = json.load(f)


class PretrainDataset(Dataset):

    def __init__(self, clean_genome_path = dic_param["data_clean_filepath"], tokenizer = dic_param["tokenizer_filepath"], chunk_size = dic_param["block_size"], stride = dic_param["stride"]):
        self.tokenizer = tokenizer
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
            with open(self.clean_genome_path, 'rb', encoding='utf-8') as f:
                f.seek(offset)
                chunk = f.read(self.chunk_size)
        except UnicodeDecodeError as e:
            print("ERREUR: Le fichier contient des caractères spéciaux non tolérés")
        except Exception as e:
            print("Erreur lors du chargement du dataset")
        encoding = self.tokenizer.encode(chunk)
    
        # On récupère les IDs (les nombres)
        ids = encoding.ids
        
        return torch.tensor(ids, dtype=torch.long)

