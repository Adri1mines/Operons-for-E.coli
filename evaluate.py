import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner
from torch.utils.data import DataLoader, Dataset, random_split
from dataset.finetuning.finetune_dataset import FinetuneDataset
# Importe ta classe (assure-toi que les chemins sont bons)
from model.processor import DNAProcessor 
# Si DNAProcessor est dans un autre fichier, ajuste l'import ci-dessus

def get_model_fingerprint(model):
    """Récupère un poids au hasard pour servir de signature."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            return param[0,0].item()
    return 0.0

def calculate_gc_content(dna_seq):
    """Calcule le pourcentage de GC d'une séquence string."""
    if len(dna_seq) == 0: return 0.0
    g = dna_seq.count('G')
    c = dna_seq.count('C')
    return (g + c) / len(dna_seq) * 100

def get_amino_acid_identity(seq1, seq2):
    """
    Calcule le % d'identité basé sur le nombre exact de matchs.
    Gère le crash 'OverflowError' quand les séquences sont trop éloignées.
    """
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    
    try:
        # C'est cette ligne qui peut crasher si les séquences sont trop différentes
        alignments = aligner.align(seq1, seq2)
        
        # Vérification de sécurité (liste vide)
        if not alignments:
            return 0.0
            
        # On prend le premier alignement pour compter les identités
        alignment = alignments[0]
        matches = alignment.counts().identities
        
        # Normalisation
        max_len = max(len(seq1), len(seq2))
        
        if max_len == 0: return 0.0
        
        return (matches / max_len) * 100

    except OverflowError:
        # CAS DU CRASH : Trop d'alignements possibles = Séquences sans aucun rapport
        # On considère que l'identité est nulle.
        return 0.0
        
    except Exception as e:
        # Filet de sécurité pour d'autres erreurs bizarres
        print(f"Warning alignment error: {e}")
        return 0.0


def main():
    # --- CONFIGURATION ---
    CHECKPOINT_PATH = "checkpoints/dna_model_llamafied_finetune_causal_mask015.ckpt" # <--- METS TON CHEMIN ICI
    # Si tu n'as pas de GPU dispo pour l'éval, mets "cpu"
    DEVICE = "cuda" 
    BATCH_SIZE = 8
    MAX_BATCHES = 1
    MAX_LEN = 1024
    print(f"🔄 Chargement du modèle depuis {CHECKPOINT_PATH}...")
    
    # Chargement du modèle
    # map_location est important si tu as entraîné sur GPU et évalues sur CPU
    model = DNAProcessor.load_from_checkpoint(CHECKPOINT_PATH, map_location=DEVICE)
    model.to(DEVICE)

    # Affiche l'empreinte unique du modèle chargé
    print(f"🕵️ CHECKPOINT CHARGÉ : {CHECKPOINT_PATH}")
    print(f"🧬 EMPREINTE DU MODÈLE : {get_model_fingerprint(model):.9f}")


    generator = torch.Generator().manual_seed(42)

    full_dataset = FinetuneDataset()

    val_size = int(0.1*len(full_dataset))
    train_size = len(full_dataset)-val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator = generator)
    val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, num_workers = 2)

    print("🚀 Démarrage de l'évaluation...")
    
    metrics = {
        "gc_content_gen": [],
        "gc_content_true": [],
        "orf_validity": [],
        "protein_recovery": []
    }
    
    count = 0
    
    with torch.no_grad():
        sequence_tot = []
        prot_amino_tot = []
        for batch in val_dataloader:
            if count >= MAX_BATCHES: break
            
            # 1. Déballage du Batch
            prompt_decoder = batch['input_decoder'].to(DEVICE)
            prot_ids = batch['input_protein'].to(DEVICE)
            prot_mask = batch['protein_attention_mask'].to(DEVICE)

            prot_ids_list = prot_ids.tolist()
            prot_amino = [model.tokenizer_encoder.decode(prot_ids, skip_special_tokens = True).replace(" ", "") for prot_ids in prot_ids_list]
            
            batch_size_curr = prompt_decoder.size(0)

            prompt_start = torch.tensor([model.tokenizer_decoder.encode("[START_CONTEXT][TERM_KNOWN][END_CONTEXT]").ids for _ in range(BATCH_SIZE)], dtype = torch.long).to(DEVICE)
            
            sequences = model.generate_sequences(n_sequence = BATCH_SIZE,
                                                 max_len = MAX_LEN,
                                                 temp = 0.5,
                                                 prompt_decoder = prompt_start,
                                                 protein_input = prot_ids,
                                                 prot_mask = prot_mask)
            sequences = [x.replace(" ", "") for x in sequences]
            sequence_tot += sequences
            prot_amino_tot += prot_amino
            count +=1
# ... (ton code précédent s'arrête ici) ...
            
            # 3. Boucle d'analyse séquence par séquence
        print(f"   🧬 Analyse du batch {count+1}...")
        
        for gen_dna, true_prot in zip(sequence_tot, prot_amino_tot):
            
            # A. Calcul du GC Content
            metrics["gc_content_gen"].append(calculate_gc_content(gen_dna))
            
            # B. Traduction (ADN -> Protéine)
            # On utilise Biopython. 
            # table=11 est spécifique aux bactéries (E. Coli). 
            # Utilise table=1 pour le code génétique standard si besoin.
            dna_obj = Seq(gen_dna)
            
            translated_prot = ""
            is_valid_orf = False
            
            try:
                # to_stop=True : Arrête la traduction dès qu'un codon STOP est rencontré
                # cds=False : Ne force pas le start codon (ATG), traduit tout tel quel
                translated_prot = str(dna_obj.translate(table=11, to_stop=True))
                
                # Une séquence est considérée "valide" si elle a pu être traduite
                # et qu'elle n'est pas vide
                if len(translated_prot) > 0:
                    is_valid_orf = True
                    
            except Exception as e:
                # Biopython peut râler si la longueur n'est pas un multiple de 3
                # (bien que translate() gère souvent ça en tronquant)
                pass

            metrics["orf_validity"].append(1 if is_valid_orf else 0)

            # C. Protein Recovery Rate (Comparaison Input vs Output Traduit)
            # C'est ici qu'on vérifie si l'ADN généré recrée bien la protéine demandée
            if is_valid_orf:
                recovery_score = get_amino_acid_identity(true_prot, translated_prot)
            else:
                recovery_score = 0.0
            
            metrics["protein_recovery"].append(recovery_score)

    
    # --- 4. RAPPORT FINAL ET SAUVEGARDE ---
    print("\n" + "="*50)
    print("📊 RAPPORT D'ÉVALUATION DU MODÈLE")
    print("="*50)

    # Calcul des moyennes (avec gestion des listes vides au cas où)
    def safe_mean(l): return np.mean(l) if len(l) > 0 else 0.0

    avg_recovery = safe_mean(metrics["protein_recovery"])
    avg_gc = safe_mean(metrics["gc_content_gen"])
    avg_validity = safe_mean(metrics["orf_validity"]) * 100

    print(f"🧬 GC Content Moyen (Généré) : {avg_gc:.2f}%")
    print(f"✅ Validité ORF (Traductible)  : {avg_validity:.2f}%")
    print("-" * 30)
    print(f"🏆 PROTEIN RECOVERY RATE     : {avg_recovery:.2f}% (Identité AA)")
    print("-" * 30)

    if avg_recovery > 90:
        print("🌟 EXCELLENT : Le modèle a parfaitement appris le code génétique.")
    elif avg_recovery > 50:
        print("⚠️ MOYEN : Le modèle respecte partiellement la contrainte protéique.")
    else:
        print("❌ ÉCHEC : Le modèle génère de l'ADN mais ignore la protéine cible.")

    # Sauvegarde des résultats bruts dans un CSV
    try:
        df_res = pd.DataFrame({
            "Generated_GC": metrics["gc_content_gen"],
            "ORF_Valid": metrics["orf_validity"],
            "Recovery_Score": metrics["protein_recovery"]
        })
        output_file = "evaluation_results015.csv"
        df_res.to_csv(output_file, index=False)
        print(f"\n📄 Résultats détaillés sauvegardés dans '{output_file}'")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde CSV: {e}")

if __name__ == "__main__":
    main()