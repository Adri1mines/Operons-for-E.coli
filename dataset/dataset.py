import torch
from torch.utils.data import Dataset, DataLoader # CORRECTION: DataLoader avec majuscule
from torch.nn.utils.rnn import pad_sequence # CORRECTION: rnn au lieu de nn
import json
import os

JSON_PATH = "dataset/data_parameters.json"

with open(JSON_PATH, 'r') as f:
    dic_param = json.load(f)


class PretrainDataset(Dataset):

    def __init__(self, clean_genome_path, tokenizer, chunk_size = dic_param["block_size"], stride = dic_param["stride"]):
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
    
        with open(self.clean_genome_path, 'rb', encoding='utf-8') as f:
            f.seek(offset)
            chunk = f.read(self.chunk_size)
        encoding = self.tokenizer.encode(chunk)

        # On récupère les IDs (les nombres)
        ids = encoding.ids
        
        return torch.tensor(ids, dtype=torch.long)

