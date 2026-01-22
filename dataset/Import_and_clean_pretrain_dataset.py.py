import os
import requests
import gzip
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "data/big_genome"
OUTPUT_FILE = "data/processed/big_train_k3.txt"
CHUNK_SIZE = 512   # Longueur des séquences pour le modèle
STRIDE = 256       # On décale de 256 à chaque fois (Overlap de 50%) -> Data Augmentation naturelle

# URLs des génomes complets (RefSeq NCBI)
# E. coli, B. subtilis, P. aeruginosa, S. aureus, M. tuberculosis
GENOME_URLS = {
    "E_coli": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz",
    "B_subtilis": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/009/045/GCF_000009045.1_ASM904v1/GCF_000009045.1_ASM904v1_genomic.fna.gz",
    "P_aeruginosa": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/006/765/GCF_000006765.1_ASM676v1/GCF_000006765.1_ASM676v1_genomic.fna.gz",
    "S_aureus": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/013/425/GCF_000013425.1_ASM1342v1/GCF_000013425.1_ASM1342v1_genomic.fna.gz",
}

def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    filename = url.split("/")[-1]
    filepath = os.path.join(dest_folder, filename)
    
    if os.path.exists(filepath):
        print(f"✅ Déjà présent : {filename}")
        return filepath
        
    print(f"⬇️ Téléchargement de {filename}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return filepath

def read_fasta(filepath):
    """Lit un fichier FASTA compressé ou non et retourne la séquence brute complète."""
    sequences = []
    current_seq = []
    
    # Gestion .gz
    open_func = gzip.open if filepath.endswith(".gz") else open
    
    with open_func(filepath, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line.upper()) # Tout en majuscule
        if current_seq:
            sequences.append("".join(current_seq))
            
    return "".join(sequences) # On colle tous les chromosomes ensemble pour simplifier

def process_genomes():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    all_chunks = []
    
    print("-" * 50)
    for name, url in GENOME_URLS.items():
        # 1. Download
        gz_path = download_file(url, DATA_DIR)
        
        # 2. Read
        print(f"📖 Lecture du génome : {name}...")
        full_dna = read_fasta(gz_path)
        
        # Filtrage basique (On ne garde que ACTG)
        full_dna = "".join([b for b in full_dna if b in "ATCG"])
        
        print(f"   -> Longueur : {len(full_dna)/1e6:.2f} Millions de bases")
        
        # 3. Sliding Window Slicing
        # C'est ici qu'on crée la masse de données
        chunks = [full_dna[i : i + CHUNK_SIZE] for i in range(0, len(full_dna) - CHUNK_SIZE + 1, STRIDE)]
        print(f"   -> Généré {len(chunks)} exemples d'entraînement.")
        all_chunks.extend(chunks)
        
    print("-" * 50)
    print(f"💾 Sauvegarde de {len(all_chunks)} séquences dans {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "w") as f:
        for seq in tqdm(all_chunks):
            f.write(seq + "\n")
            
    print("✅ Dataset Big Data prêt !")

if __name__ == "__main__":
    process_genomes()