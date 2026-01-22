import json
import random
import re
import os
from tqdm import tqdm

JSON_PATH = "dataset/data_parameters.json"

with open(JSON_PATH, 'r') as f:
    dic_param = json.load(f)

FILEPATH_RAW_DATA = dic_param["data_raw_filepath"]
FILEPATH_CLEAN_DATA = dic_param["data_clean_filepath"]
min_len = dic_param["min_len_segment"]

def main():
    print("Nettoyage en cours...")
    if os.path.exists(FILEPATH_CLEAN_DATA):
        os.remove(FILEPATH_CLEAN_DATA)
    with open(FILEPATH_RAW_DATA, 'r') as f:
        data_raw = f.read().split(dic_param["separateur"])

    clean_chunks = []
    total_bp = 0
    for raw_chunk in tqdm(data_raw, desc = "nettoyage", unit = "chunk"):
        #Enlever les retours chariot
        raw_chunk = raw_chunk.replace("\n","")
        #mettre tout en uppercase
        raw_chunk = raw_chunk.upper()
        #changer toutes les lettres autres que ACTG en N
        raw_chunk = re.sub(r'[^ATGC N]', 'N', raw_chunk)
        #séparer la séquence sur les gros blocs de N (plus de 10 N)
        raw_chunk_chunk = re.split(r'N{10,}', raw_chunk)
        
        
        for chunk in raw_chunk_chunk:

            chunk_list = list(chunk)
            for i, char in enumerate(chunk_list):
                if char == 'N':
                    #On remplace les N restants par un acide nucléique random
                    chunk_list[i] = random.choice(['A', 'C', 'G', 'T'])
            
            cleaned_chunk = "".join(chunk_list)
            total_bp += len(cleaned_chunk)
            # 4. Vérification de taille minimale
            if len(cleaned_chunk) >= min_len:
                clean_chunks.append(cleaned_chunk)
    total_chunk = len(clean_chunks)
    avg_len = total_bp/total_chunk
    print(f"Nombre de bp: {total_bp} | Nombre de chunk: {total_chunk} | Longueur moyenne chunk {avg_len}" )
    with open(FILEPATH_CLEAN_DATA, 'a') as f:
        for chunk in clean_chunks:
            f.write(chunk + dic_param["separateur"])

if __name__ == "__main__":
    main()