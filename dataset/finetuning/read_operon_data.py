import pandas as pd

FILE_PATH_TU = "finetuning/TUSet.tsv"
FILE_PATH_GENE = "finetuning/GeneProductSet.tsv"
FILE_PATH_PROM = "finetuning/PromoterSet.tsv"
FILE_PATH_TERM = "finetuning/TerminatorSet.tsv"



def main():
# 1. Chargement du fichier TUSet
    # On saute les lignes de commentaires (qui commencent par #)
    tu_df = pd.read_csv(FILE_PATH_TU, 
        sep='\t', 
        comment='#',
        header = 0,
        usecols=[0, 1, 2, 4, 5, 6, 7],
        names=["TUID", "TUName", "OperonID", "Genes", "PromoterName", "TerminatorID", "confidence"]
        )
    gene_df = pd.read_csv(
        FILE_PATH_GENE, 
        sep='\t', 
        comment='#',
        header = 0,
        usecols=[1, 4, 9],
        names=["Genes", "read", "GeneSequence"]
    )
    prom_df = pd.read_csv(
        FILE_PATH_PROM, 
        sep='\t', 
        comment='#',
        header = 0,
        usecols=[1, 4, 9],
        names=["PromoterName", "read", "PromoterSequence"]
    )
    term_df = pd.read_csv(
        FILE_PATH_TERM, 
        sep='\t', 
        comment='#',
        header = 0,
        usecols=[0, 3, 4],
        names=["TerminatorID", "read", "TerminatorSequence"]
    )


    # 2. Nettoyage
    # On vire les lignes où il n'y a pas de Promoteur (NaN) car inutilisables pour la génération
    clean_tu = tu_df.dropna(subset=['PromoterName', 'Genes'])
    clean_tu = clean_tu[clean_tu["confidence"] != "W"]
