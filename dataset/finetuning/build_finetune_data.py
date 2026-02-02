import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np

# CONFIGURATION
GENOME_PATH = "dataset/finetuning/raw/K_12_genome.fasta"
FILE_PATH_TU = "dataset/finetuning/raw/TUSet.tsv"
FILE_PATH_GENE = "dataset/finetuning/raw/Gene_sequence.tsv"
FILE_PATH_PROM = "dataset/finetuning/raw/PromoterSet.tsv"
FILE_PATH_TERM = "dataset/finetuning/raw/TerminatorSet.tsv"
OUTPUT_FILE = "dataset/finetuning/processed/dataset_operons_train.csv"

def translate_dna_to_protein(dna_sequence):
    """
    Traduit une séquence ADN de gène en protéine (Table 11 - Bacterial).
    Gère les Start Codons alternatifs (GTG, TTG deviennent Met en début).
    """
    if pd.isna(dna_sequence) or len(dna_sequence) < 3:
        return ""
    
    # Nettoyage
    seq_obj = Seq(str(dna_sequence).strip().upper())
    
    # Traduction
    # table=11 : Le code génétique des bactéries
    # to_stop=True : S'arrête au premier codon stop (TAG, TAA, TGA)
    # cds=False : On le fait 'manuellement' pour éviter les erreurs si la longueur n'est pas multiple de 3
    protein = seq_obj.translate(table=11, to_stop=True, cds=False)
    
    # CORRECTION BIOLOGIQUE IMPORTANTE :
    # En bactéries, GTG et TTG codent pour la Valine/Leucine, MAIS
    # s'ils sont au DÉBUT du gène, ils codent pour la Méthionine (fMet).
    # Biopython traduit bêtement GTG -> V. On doit forcer M.
    prot_str = str(protein)
    if len(prot_str) > 0:
        # On force le premier AA à être une Méthionine (standard ESM)
        prot_str = "M" + prot_str[1:]
        
    return prot_str

def get_reverse_complement(seq_str):
    return str(Seq(seq_str).reverse_complement())

def main():
    print("📂 Chargement des données...")
    genome_record = SeqIO.read(GENOME_PATH, "fasta")
    genome_seq = genome_record.seq

    # 1. TUSet (Le Hub)
    tu_df = pd.read_csv(FILE_PATH_TU, sep='\t', comment='#', header=None,
        usecols=[0, 1, 4, 5, 6], 
        names=["TUID", "TUName", "Genes", "PromoterName", "TerminatorID"]
    )
    clean_tu = tu_df.dropna(subset=['PromoterName', 'Genes'])

    # 2. GeneProductSet (Contenu)
    # Regex pour nettoyer les noms "(synonym)"
    gene_df = pd.read_csv(FILE_PATH_GENE, sep='\t', comment='#', header=None,
        usecols=[1, 2, 3, 4, 9], 
        names=["GeneName", "Left", "Right", "Strand", "ProteinSequence"]
    )
    gene_df['GeneName'] = gene_df['GeneName'].str.replace(r'\s*\(.*\)', '', regex=True)
    gene_df = gene_df.drop_duplicates(subset = ["GeneName"])
    gene_map = gene_df.set_index("GeneName").to_dict('index')

    # 3. PromoterSet (Position TSS)
    # CRUCIAL : Il faut trouver la colonne "TSS Absolute Position" (Souvent col 4 ou 5)
    # Regarde ton fichier : c'est la coordonnée absolue du +1
    prom_df = pd.read_csv(FILE_PATH_PROM, sep='\t', comment='#', header=None,
        usecols=[1, 3], # <--- ADAPTE L'INDEX 3 SELON TON FICHIER (Position)
        names=["PromoterName", "TSS_Position"]
    )
    # On gère les doublons (garde le premier)
    prom_map = prom_df.drop_duplicates("PromoterName").set_index("PromoterName")['TSS_Position'].to_dict()

    # 4. TerminatorSet (Positions Fin)
    term_df = pd.read_csv(FILE_PATH_TERM, sep='\t', comment='#', header=None,
        usecols=[0, 1, 2], # <--- ADAPTE : ID, Left, Right
        names=["TerminatorID", "Term_Left", "Term_Right"]
    )
    term_map = term_df.set_index("TerminatorID").to_dict('index')

    dataset_entries = []
    print(f"🚀 Reconstruction Hybride sur {len(clean_tu)} TUs...")

    for index, row in clean_tu.iterrows():
        try:
            # --- A. Récupération Infos Basiques ---
            tu_id = row['TUID']
            prom_name = row['PromoterName']
            term_id = row['TerminatorID']
            
            gene_names = [g.strip() for g in str(row['Genes']).split(';') if g.strip() in gene_map]
            if not gene_names: continue

            # --- B. Logique des Coordonnées (Le Cœur du sujet) ---
            
            # 1. On détermine le brin via les gènes (Vote majoritaire)
            coords_genes = [gene_map[g] for g in gene_names]
            strands = [c['Strand'] for c in coords_genes]
            strand = max(set(strands), key=strands.count) # '+' ou '-'

            # 2. Définir le DÉBUT (Promoteur)
            start_coord = None
            
            # Est-ce qu'on a la position exacte du TSS dans PromoterSet ?
            if prom_name in prom_map and not pd.isna(prom_map[prom_name]):
                tss = int(prom_map[prom_name])
                # On recule de 60bp pour inclure les boîtes -10 et -35
                if strand == 'forward':
                    start_coord = tss - 60
                else:
                    start_coord = tss + 60 # Sur reverse, "avant" = plus grand
            else:
                # Fallback : On prend le gène le plus "à gauche" (ou droite) - 100bp
                if strand == 'forward':
                    start_coord = min(int(c['Left']) for c in coords_genes) - 100
                else:
                    start_coord = max(int(c['Right']) for c in coords_genes) + 100

            # 3. Définir la FIN (Terminateur)
            end_coord = None
            
            # Est-ce qu'on a un terminateur valide avec coordonnées ?
            if term_id in term_map:
                t_dat = term_map[term_id]
                if strand == 'forward':
                    end_coord = int(t_dat['Term_Right']) # La fin physique est à droite
                else:
                    end_coord = int(t_dat['Term_Left'])  # La fin physique est à gauche
            else:
                # Fallback : Fin du dernier gène + 60bp
                if strand == 'forward':
                    end_coord = max(int(c['Right']) for c in coords_genes) + 60
                else:
                    end_coord = min(int(c['Left']) for c in coords_genes) - 60

            # --- C. Extraction de l'ADN ---
            
            # Python slicing (min, max)
            # On s'assure que min < max pour le slice
            slice_min = min(start_coord, end_coord)
            slice_max = max(start_coord, end_coord)
            
            # Extraction brute
            # Attention aux bornes < 0 ou > len(genome)
            slice_min = max(0, slice_min - 1) # 0-based index
            seq_raw = genome_seq[slice_min : slice_max]
            
            if strand == 'reverse' or strand == '-':
                target_dna = str(seq_raw.reverse_complement())
            else:
                target_dna = str(seq_raw)

            # --- D. Construction du Prompt Protéine (Idem avant) ---
            # ... (Code de tri des protéines et concaténation identique au précédent) ...
            # Je te remets juste la logique de tri pour rappel :
            gene_objs = [(g, gene_map[g]) for g in gene_names]
            if strand == 'forward':
                gene_objs.sort(key=lambda x: x[1]['Left'])
            else:
                gene_objs.sort(key=lambda x: x[1]['Right'], reverse=True)
            
            context_prompt = "[START_CONTEXT]" 
            context_prompt += "[TERM_KNOWN]" if term_id in term_map else "[TERM_INFERRED]"
            context_prompt += "[END_CONTEXT]"
            translated_prot = ""
            for g_name, g_data in gene_objs:
                p_seq = str(g_data['ProteinSequence'])
                if p_seq != 'nan':
                    translated_prot += f"{translate_dna_to_protein(p_seq)}<eos>"

            # --- E. Ajout au dataset ---
            dataset_entries.append({
                "TUID": tu_id,
                "Input_context": context_prompt,
                "translated_prot": translated_prot,
                "Target": target_dna
            })

        except Exception as e:
            print(f"Error on {row['TUID']}: {e}")
            continue

    final_df = pd.DataFrame(dataset_entries)
    print(f"✅ Terminé : {len(final_df)} opérons.")
    final_df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()