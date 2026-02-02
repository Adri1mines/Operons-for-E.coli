import requests
import json
import os
from tqdm import tqdm
import json
import gzip
import shutil

CONFIG_PATH = "config.json"
TARGET = 400

with open(CONFIG_PATH, 'r') as f:
    config_param = json.load(f)
# URL de l'index officiel des génomes bactériens RefSeq
SUMMARY_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/assembly_summary.txt"
FILEPATH_DATA = config_param["data"]["data_raw_filepath"]
FILEPATH_META = config_param["data"]["meta_filepath"]


def main():
    print("Téléchargement de l'index NCBI en cours... (ça peut prendre quelques secondes)")

    # On récupère le fichier
    response = requests.get(SUMMARY_URL)
    data = response.text.splitlines()

    urls = []
    count = 0

    if os.path.exists(FILEPATH_META):
        os.remove(FILEPATH_META)

    if os.path.exists(FILEPATH_DATA):
        os.remove(FILEPATH_DATA)

    print("Filtrage et construction des URLs...")
    with tqdm(total = TARGET, desc="Progression", unit="souche") as pbar:
        for line in data:
            if line.startswith("#"):
                continue
            
            parts = line.split("\t")
            
            # Colonnes clés dans assembly_summary.txt :
            # 7: Organism name
            # 11: Assembly level (on veut "Complete Genome" ou "Chromosome")
            # 19: FTP path
            
            organism = parts[7]
            assembly_level = parts[11]
            ftp_path = parts[19]
            
            # On cherche Escherichia coli et on privilégie les génomes complets pour éviter les fragments
            if "Escherichia coli" in organism and assembly_level == "Complete Genome":

                # Le chemin dans le fichier est en ftp://, on le passe en https://
                https_base = ftp_path.replace("ftp://", "https://")
                
                # On récupère le nom du dossier final (ex: GCF_000013425.1_ASM1342v1)
                folder_name = https_base.split("/")[-1]
                
                # On construit le lien complet vers le fichier .fna.gz
                full_url = f"{https_base}/{folder_name}_genomic.fna.gz"
                
                urls.append(full_url)
                with open(FILEPATH_DATA, 'a') as f:
                    with requests.get(full_url, stream = True) as r:
                        r.raise_for_status()
                        with gzip.open(r.raw, mode='rt', encoding='utf-8') as f_in:
        
                            # --- RÉCUPÉRATION DU NOM DE LA SOUCHE ---
                            # On lit juste la première ligne
                            first_line = f_in.readline()
    
                            if first_line.startswith(">"):
                                # On nettoie le chevron '>' et on garde le nom
                                strain_name = first_line.strip().replace(">", "")
                            else:
                                strain_name = "Inconnue"

                            with open(FILEPATH_META, 'a') as f_m:
                                f_m.write(strain_name + "\n")
                                
                            # On lit le reste du flux ligne par ligne ou par blocs
                            # copyfileobj est super optimisé pour ça (il lit/écrit par chunks tout seul)
                            shutil.copyfileobj(f_in, f)
                            f.write(config_param["data"]["separateur"])

                pbar.update(1)
                count += 1
            if count >= TARGET:
                break


if __name__ == "__main__":
    main()
