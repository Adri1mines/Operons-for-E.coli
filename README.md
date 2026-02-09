# 🧬 Prot2DNA: Generative Operon Modeling

> **Status:** En développement actif  
> **Objectif :** Génération de séquences d'ADN bactérien type E.Coli (Opérons/Unités de Transcription) à partir de séquences protéiques via une architecture Transformer Hybride.

BONJOUR GEORGES-ANDRE, CE README EST POUR TOI !

Ce projet explore l'utilisation de modèles de langage (LLMs) pour la "traduction inverse" biologique : prédire l'ADN codant à partir d'une ou plusieurs séquences d'acides aminés, en prenant en compte la structure polycistronique (opérons) des bactéries. Chez les bactéries, les protéines sont codées sous formes d'unités de transcription, un long morceau d'ADN qui sera lu d'une traite par un complexe enzymatique pour créer l'ARN messager qui donnera ensuite la protéine. Les unités de transcription prennent la forme d'opérons, formé d'un promoteur, qui sert d'accroche au complexe et qui va gérer à quel point l'unité de transcription s'exprime, la séquence codante, et enfin le terminateur. Pour être biologiquement viable, l'opéron doit respecter beaucoup de règles, dont par exemple la ditribution GC de la bactérie (quantité de guanine et de cytosine dans l'ADN), la distribution de k-mer dans l'ADN, mettre un promoteur qui inhibe un peu l'opéron pour les protéines avec une haute toxicité métabolique etc... cette complexité peut donc donner un intérêt à l'utilisation du ML
---

## Etape 1, se procurer les données et créer une architecture

Partant de zéro pour ce projet il a fallu faire un peu d'exploration pour trouver comment accomplir la tâche voulue. D'abord, trouver le dataset. Mon plan de bataille était le suivant: Pre-train le modèle sur de l'ADN bactérien pour qu'il comprenne la synthaxe globale, et ensuite finetuner sur des données annotées avec en input la protéine et en target l'unité de transcription correspondante. J'ai choisi de me concentrer sur E.coli car c'est la bactérie la mieux documentée, donc celle ou l'on a le plus de donnée publiquement accessible. Le site de la NHI recense le génome complet de BEAUCOUP de souches de E.coli, c'est qui nous servira de base pour le pretrain. Le script import_raw_data permet d'aller chercher le génome des souches sur le web, et clean_raw_data permet de nettoyer les données (enlever les retours chariot, les espaces, les morceaux d'ADN de mauvaise qualité etc...)

Le choix du tokenizer est très important, il va grandement influencer les performances du modèle en terme de taille de contexte, et en terme de granularité de ce contexte. Plus le tokenizer est petit, meilleure est la "vision" du modèle à l'échelle du nucléotide mais sa fenêtre de contexte courte. Le tokenizer naïf composé de [A, T, C, G] pourrait être une bonne solution si l'on possède beaucoup de capacité de calcul (ou qu'on implémente une attention linéaire par exemple ça pourrait être intéressant) et que l'on veut un modèle capable de voir à l'échelle de la mutation d'un nucléotide. On pourrait aussi envisager un tokenizer composé des codons, i.e. des triplets d'ADN ce qui serait un peu le tokenizer le plus naturel pour ce que l'on veut faire, mais les tests que j'ai effectué n'ont pas été très concluants et le taux de compression du tokenizer n'est pas assez gros pour avoir un contexte qui encapsule tout un opéron. J'ai donc opté pour un tokenizer type BPE/Unigram, ou l'on va fixer une taille de vocabulaire et ou on va construire le tokenizer en se basant sur le dataset. J'ai choisi un taille de vocabulaire de 1024 et un tokenizer Unigram (pour pouvoir faire de la data augmentation si besoin).

Pour le modèle j'ai d'abord choisis un nombre assez faible de paramètres (~1M de paramètres) pour pouvoir faire des tests d'architecture à un coût d'entraînement moindre. La première architecture que j'ai testé a été le transformer classique, le bloc décodeur défini dans le papier "Attention is all you need" (à l'exception que j'ai appliqué la normalisation avant les blocs d'attention et de feed-forward, pas en aval), histoire d'avoir une baseline en terme de temps d'entraînement et de loss. La loss de validation tourne autour de 5,1 (ce qui est mauvais pour un tokenizer de taille 1024), et le temps d'entraînement de 2h30. J'ai voulu voir les effets des nouveaux standards sur le modèle de transformer, j'ai donc successivement implémenté le RoPE, la flash attention, la SwiGLU et la RMSNorm. Le RoPE a eu des bons effets sur la loss, avec pour le même nombre de paramètres une loss de validation de 4.9, soit un gain de 5% à peu près. La flash attention a été l'ajout le plus impactant, avec une temps d'entaînement qui est passé de 3h30 à 1h15. Enfin la RMSNorm et la SwiGLU n'ont pas eu d'effets notable, mais je pense que ce sont des ajouts qui ont un intérêt pour stabiliser le modèle et améliorer ses performances quand on augmente sa taille et sa profondeur, donc le gain ne se voit pour un bloc décodeur seul avec 1M de paramètres. Au final je me suis rendu compte que j'avais implémenté l'architecture llama, étant donné que c'est l'état de l'art actuel pour le transformer, j'ai décidé de rester sur cette architecture et de scale-up.

Je suis passé sur un bloc décodeur avec d_model = 512, n_head = 8, hidden_size = 4*d_model, et j'ai stacké 12 bloc de décodeur les uns sur les autres, pour un total de 42M de paramètres, ce qui est gérable pour un L4 (surtout en bp16). Pour le dataset d'entraînement j'ai utilisé 400 souches de E.coli différentes, pour un total de ~700M de tokens (ce qui est normalement optimal si on s'en réfère aux scaling laws Chinchilla). Le temps d'entraînement a été de 12h et la loss de validation était de 1, donc on imagine que le modèle commence à bien comprendre.

Cependant l'une des difficultés de l'ADN est que contrairement au langage humain, on ne peut pas simplement regarder les réponses du modèle et voir si il nous raconte n'importe quoi, j'ai donc implémenté des métriques biologiques avec le fichier bio_test.py dans le fichier callback, pour voir à chaque epoch les performances "réelles" du modèle. J'ai décidé d'utiliser la distribution GC que j'ai évoqué plus haut, qui est ~53% chez E.coli en général et la distribution de 3-mers, i.e. quels sont les codons que le modèle génère, et plus précisément j'ai regardé le delta entre la distribution de 3-mers chez E.coli et celle de mon modèle. Passer d'un transformer basique à une architecture llama-like et augmenter la taille du modèle à amélioré ces métriques, ce qui était un bon indicateur que j'allais dans la bonne direction. 

Une fois que j'avais mon décodeur entraîné, j'ai voulu voir si ces performances sur la génération d'opérons étaient bonnes. Elles ne l'étaient pas du tout, mais ce n'était pas une surprise car les inputs que je lui donnais étaient des morceaux d'ADN de taille fixe coupés à des endroits arbitraires, il ne pouvait donc pas avoir de notion de ce qu'était un opéron entier bien formé, enfin il aurait peut-être pu mais seulement si j'avais donné des chunks assez gros pour encapsuler des opérons entiers, et encore. Il a donc fallu passer à l'étape suivante, le finetuning.

## Etape 2, finetuner le modèle

J'avais maintenant un décodeur qui comprenait plutôt bien la synthaxe de l'ADN, mais l'objectif final pour rappel est de donner en entrée une protéine et d'avoir un opéron en sortie. J'avais pour ça un bloc essentiel manquant: pouvoir donner à manger à mon modèle des protéines, donc une séquence d'acides aminés. Pour cela j'ai décidé d'utiliser un deuxième modèle déjà entraîné, ESM2 de chez Meta. Afin de relier mon modèle et ESM2 j'ai envisagé deux méthodes. La première est le soft prompting, où je viens concaténer l'embedding de mon prompt initial avec l'embedding généré par ESM2 à partir de la protéine. La deuxième est d'utiliser ESM2 comme un encodeur, et d'injecter les embeddings de protéine dans mon décodeur en utilisant une cross-attention. J'ai essayé d'implémenter la première, mais étant donné que je n'avais pas prévu d'utiliser ESM2 dans mon plan inital, j'ai dû refaire au propre mon code pour éviter le bricolage, donc j'ai décidé d'opter pour la cross-attention quitte à tout refaire. J'ai donc recodé mon architecture en incluant la cross-attention et les connexions avec l'encodeur mais en me laissant la capacité de "débrancher" la partie encodeur/cross-attention pendant le pré-entraînement.

L'autre diffculté a été de se procurer le dataset, qui cette fois devait être annoté, donc j'ai trouvé le dataset RegulonDB, qui avec un peu de manipulation permet de créer le dataset voulu (script build_finetune_dataset.py). Après moulte débuggage et déboirs j'ai réussis le finetuning en boostant le learning rate des couches de cross-attention et de projection de ESM2 vers d_model, car celles-ci partaient "à froid", et en faisant d'abord un entraînement causal purement pour la première epoch, puis en appliquant un masquage sur ~30% des batches (avec un taux de masquage de 0.3) pour le forcer à utiliser le contexte de la protéine.

Pour tester les résultats j'ai utilisé comme métrique ce qu'on pourrait appeler le "protein recovery rate", i.e. est-ce qu'on retrouve dans l'opéron la séquence codante de la protéine qu'on lui a donné en entrée. les résultats étaient catastrophiques (~3% de recovery rate), et j'ai compris pourquoi quand j'ai testé le dataset lui-même qui avait en fait un protein recovery rate de 7%. Mon dataset de finetuning était donc de très mauvaisa qualité, en raison d'erreurs dans le dataset genomeDB, qui sont réparables mais que je n'avais pas envisagé

## Etape 3, La suite

Le projet n'est pas encore fini mais je pense que ce qui a été fait est très prometteur, il ne manqua pas grand chose pour arriver à un très bon résultat. Voici le plan d'action que j'espère finir d'ici mars:

1) Corriger le dataset de finetuning et s'assurer de sa qualité, essayer de trouver d'autres métriques que le protein recovery rate pour juger de la qualité du dataset afin de s'assurer d'avoir de très bonnes données.

2) Refaire le finetuning sur ces données et (je pense) avoir un résultat de qualité. Potentiellement essayer d'implémenter une loss "biologique" avec ce protein recovery rate, même si c'est un peu compliqué étant donné le caractère relatif de la lecture d'ADN.

3) Multiplier les inputs, j'aimerais bien pouvoir prompter le modèle pour pouvoir demander un opéron très expressif, ou peu expressif, bref pouvoir augmenter le montant de customisation sur l'opéron généré, sans pour atant compromettre sa viabilité biologique

4) Refaire le modèle en utilisant un vocabulaire de 4 tokens (A, C, G, T) mais en utilisant une attention linéaire pour pouvoir un gros contexte sans faire exploser le temps de calcul et d'inférence, et en même temps une précision à l'échelle du nucléotide

Et si les résultats sont bons et que j'ai le temps dans le futur:

5) Augmenter la taille du modèle de nouveau, car le décodeur avait une bonne loss mais les métriques biologiques pouvaient encore être meilleure,

6) Tester le résultat en vrai en modifiant génétiquement une bactérie et en lui greffant un opéron généré par le modèle, test ULTIME de la viabilité biologique.

---

## 🏗️ Architecture Technique (dernière en date)

Le modèle repose sur une architecture **Encoder-Decoder** asymétrique :

* **Encoder (Protéine) :** Modèle **ESM-2**
    * *État :* Frozen (gelé) pour préserver les features biochimiques apprises.
* **Decoder (ADN) :** Architecture "Llama-style" custom (`DNAProcessor`).
    * *Tokenization :* Unigram 1024 de vocab_size.
    * *Attention :* **RoPE** (Rotary Positional Embeddings) pour une meilleure extrapolation de longueur.
    * *Activation :* **SwiGLU**.
    * *Connexion :* Couches de **Cross-Attention** insérées dans les blocs du décodeur.
* **Objectif d'Entraînement :** * Causal Language Modeling (CLM).
    * **Input Masking (15-30%)** : Technique cruciale pour forcer le décodeur à utiliser l'information de l'encodeur (et éviter le "Posterior Collapse").

---

## ✅ Best Practices Identifiées / leçons retenues

* **Lightning et wandb:** J'ai découvert la libraire lightning pytorch pendant le projet et je la trouve très puissante. J'ai aussi découvert weights and bias pour log les entraînement, c'est très pratique !
* **Pytorch compile:** Quand on commence à empiler beaucoup de blocs dans l'architectures et que le modèle devient gros, la compilation torch permet un gain significatif de temps d'entraînement
* **Overfit on 1 Batch :** Ne jamais lancer un training long sans avoir vérifié que le modèle peut mémoriser 4 exemples par cœur.
* **L'importance de l'informatique:** Pour une même architecture, le temps de calcul peut varier massivement en fonction d'à quel point celle-ci est optimisé niveau du code
* **Garbage in, Garbage out:** Le modèle ne sera jamais meilleur que son dataset, comprendre les données sur lesquelles on travaill et s'assurer de leur qualité est donc l'étape zéro de n'importe quel projet de ML.