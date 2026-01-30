from model.processor import DNAProcessor
import matplotlib.pyplot as plt
from Bio.Seq import Seq
import numpy as np

MODEL_PATH = ""

def find_first_orf(dna_sequence):
    """
    Cherche le premier ORF valide commençant au début de la séquence.
    Retourne: (est_valide, longueur_bp, sequence_proteine)
    """
    seq = Seq(dna_sequence)
    
    # 1. Vérification basique : est-ce que ça commence par un Start ?
    # (Théoriquement oui puisque c'est ton prompt, mais on vérifie)
    if not str(seq).startswith("ATG"):
        return False, 0, None

    # 2. Traduction jusqu'au premier stop
    # to_stop=True arrête la traduction au premier codon stop rencontré
    try:
        protein = seq.translate(to_stop=True)
    except Exception:
        return False, 0, None # Cas de caractères invalides

    # 3. Vérification de la terminaison
    # La longueur de l'ADN utilisé = len(prot) * 3
    dna_used_len = len(protein) * 3
    
    # Si la longueur utilisée est inférieure à la longueur totale de la seq,
    # c'est qu'on a trouvé un STOP avant la fin (Trunkation du tokenizer).
    # Il faut vérifier s'il reste assez de place pour un stop (3 bases)
    if dna_used_len + 3 <= len(seq):
        # On vérifie si les 3 bases suivantes sont bien un stop valide
        stop_codon = str(seq)[dna_used_len : dna_used_len+3]
        if stop_codon in ["TAA", "TAG", "TGA"]:
            return True, dna_used_len, str(protein)
    
    # Si on arrive ici, c'est que le modèle a généré jusqu'à la fin de sa max_len
    # sans jamais mettre de codon stop. C'est un "Runaway ORF".
    return False, len(seq), None

def analyze_generations(sequences):
    valid_orfs = []
    runaway_orfs = []
    lengths_bp = []
    
    print(f"🔍 Analyse de {len(sequences)} séquences...")
    
    for seq in sequences:
        is_valid, length, protein = find_first_orf(seq)
        
        if is_valid:
            valid_orfs.append(seq)
            lengths_bp.append(length)
        else:
            runaway_orfs.append(seq)

    # --- Statistiques ---
    total = len(sequences)
    valid_pct = (len(valid_orfs) / total) * 100
    avg_len = np.mean(lengths_bp) if lengths_bp else 0
    
    print("\n" + "="*30)
    print("📊 RÉSULTATS DU DIAGNOSTIC")
    print("="*30)
    print(f"Total séquences testées : {total}")
    print(f"✅ ORFs Valides (Stop trouvé) : {len(valid_orfs)} ({valid_pct:.2f}%)")
    print(f"❌ Runaway / Invalides      : {len(runaway_orfs)} ({100-valid_pct:.2f}%)")
    
    if lengths_bp:
        print(f"📏 Longueur Moyenne (bp)    : {avg_len:.1f}")
        print(f"📏 Longueur Médiane (bp)    : {np.median(lengths_bp):.1f}")
        print(f"🧬 Plus petit ORF           : {min(lengths_bp)} bp")
        print(f"🧬 Plus grand ORF           : {max(lengths_bp)} bp")
    
    # --- Visualisation ---
    if lengths_bp:
        plt.figure(figsize=(10, 6))
        plt.hist(lengths_bp, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Distribution des longueurs des ORFs générés')
        plt.xlabel('Longueur (paires de bases)')
        plt.ylabel('Nombre de séquences')
        plt.axvline(avg_len, color='red', linestyle='dashed', linewidth=1, label=f'Moyenne: {avg_len:.0f}bp')
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        plt.show()

# ==========================================
# EXEMPLE D'UTILISATION (Simulation)
# ==========================================
if __name__ == "__main__":
    # Remplace cette liste par les sorties de ton modèle
    # Exemple : generated_seqs = model.generate(prompt="ATG", num_samples=100)
    lightning_module = DNAProcessor.load_from_checkpoint(checkpoint_path=MODEL_PATH)
    orfs = lightning_module.generate_sequences(n_sequence = 100, max_len = 1024, temp = 0.7, prompt = "ATG")

    
    analyze_generations(orfs)