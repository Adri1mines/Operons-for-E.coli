import lightning as pl
from lightning.pytorch.callbacks import Callback
import torch
import numpy as np
import wandb
from collections import Counter
import re
from tokenizers import Tokenizer

class BioEvalCallback(Callback):
    def __init__(self, tokenizer_path, val_dataset, num_samples=50, max_len=1024):
        super().__init__()
        self.val_dataset = val_dataset
        self.num_samples = num_samples
        self.max_len = max_len
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        # On pré-calcule les stats du VRAI dataset pour comparer
        print("Calcul des statistiques de référence (Validation Set)...")
        print("Analyse du dataset de validation (K-mers & ORFs)...")
        real_seqs = self._sample_real_sequences(count=200)
        self.real_gc = self._compute_gc_stats(real_seqs)
        
        self.ref_kmer_dist = self._get_kmer_distribution(real_seqs, k=3)
        self.ref_orf_len = np.mean([self._get_max_orf_length(s) for s in real_seqs])
        print(f"Ref Dataset -> Max ORF Avg: {self.ref_orf_len:.1f}bp")
        
    def _compute_gc_stats(self, seqs):
        """Calcule le contenu GC des séquences"""
        gc_contents = []
        for seq in seqs:
            if len(seq) > 0:
                gc = (seq.count('G') + seq.count('C')) / len(seq)
                gc_contents.append(gc)
        return np.mean(gc_contents)
    
    def _sample_real_sequences(self, count):
        """Récupère des séquences réelles décodées"""
        indices = np.random.choice(len(self.val_dataset), size=min(len(self.val_dataset), count), replace=False)
        seqs = []
        for i in indices:
            ids = self.val_dataset[i].tolist()
            txt = self.tokenizer.decode(ids, skip_special_tokens=True)
            seqs.append(txt.replace(" ", ""))
        return seqs
    
    def _get_kmer_distribution(self, sequences, k=3):
        """Compte la fréquence de chaque k-mer (ex: ATG, CCC...)"""
        counter = Counter()
        total_kmers = 0
        for seq in sequences:
            for i in range(len(seq) - k + 1):
                kmer = seq[i : i + k]
                counter[kmer] += 1
                total_kmers += 1
        
        # Normalisation pour avoir des probabilités
        dist = {k: v / total_kmers for k, v in counter.items()}
        return dist
    
    def _get_max_orf_length(self, sequence):
        """Trouve la longueur de la plus longue protéine potentielle (simplifié)"""
        # Pattern: ATG (Start) ... (3 par 3) ... TAA|TAG|TGA (Stop)
        # C'est une approx rapide sans Biopython
        max_len = 0
        seq_len = len(sequence)
        
        # On cherche les START codons
        starts = [m.start() for m in re.finditer('ATG', sequence)]
        
        for start in starts:
            # Pour chaque start, on cherche le premier STOP en phase (multiple de 3)
            for i in range(start + 3, seq_len - 2, 3):
                codon = sequence[i : i + 3]
                if codon in ['TAA', 'TAG', 'TGA']:
                    length = (i + 3) - start
                    if length > max_len:
                        max_len = length
                    break # On a trouvé le stop de cet ORF, on passe au start suivant
        return max_len

    def on_validation_epoch_end(self, trainer, pl_module):
        # 1. Génération
        print(f"\n[BioEval] Génération de {self.num_samples} séquences...")
        gen_seqs = pl_module.generate_sequences(
            n_sequence=self.num_samples, 
            max_len=self.max_len
        )
        
        # 2. Calcul des métriques biologiques
        gc_values = []
        lengths = []
        
        for seq in gen_seqs:
            # Nettoyage basique (retirer espaces)
            clean_seq = seq.replace(" ", "")
            if len(clean_seq) == 0: continue
            
            # GC Content
            gc = (clean_seq.count('G') + clean_seq.count('C')) / len(clean_seq)
            gc_values.append(gc)
            lengths.append(len(clean_seq))

            #

        avg_gen_gc = np.mean(gc_values) if gc_values else 0
        
        # 3. Logging WandB
        # On logge la différence (Error) et la valeur absolue
        gc_error = abs(avg_gen_gc - self.real_gc)
        
        # 2. Métrique K-Mers (Distance Euclidienne)
        gen_kmer_dist = self._get_kmer_distribution(gen_seqs, k=3)
        
        # On calcule la différence entre la distribution générée et la réelle
        all_kmers = set(self.ref_kmer_dist.keys()) | set(gen_kmer_dist.keys())
        kmer_error = 0.0
        for kmer in all_kmers:
            diff = self.ref_kmer_dist.get(kmer, 0) - gen_kmer_dist.get(kmer, 0)
            kmer_error += diff ** 2
        kmer_score = np.sqrt(kmer_error) # Plus c'est bas, mieux c'est

        # 3. Métrique ORF
        orf_lengths = [self._get_max_orf_length(s) for s in gen_seqs]
        avg_orf_len = np.mean(orf_lengths)
        
        # 4. Métrique Shine-Dalgarno (Bonus E. coli)
        sd_count = sum([s.count("AGGAGG") for s in gen_seqs])
        sd_density = sd_count / len(gen_seqs)

        # 5. Logging
        metrics = {
            "bio/kmer_distance": kmer_score,   # Doit descendre
            "bio/avg_orf_length": avg_orf_len, # Doit monter (se rapprocher de ~300-500)
            "bio/real_orf_ref": self.ref_orf_len,
            "bio/shine_dalgarno_density": sd_density,
            "bio/generated_gc_avg": avg_gen_gc,
            "bio/real_gc_ref": self.real_gc,
            "bio/gc_error": gc_error,
            "bio/avg_length": np.mean(lengths) if lengths else 0
        }
        # Log des scalaires
        pl_module.log_dict(metrics)
            