#!/bin/bash

echo "Lancement de l'entraînement du DNA Transformer..."


python3 train.py \
    --project_name="Operons for E.coli" \
    --wandb_team_name = "adrien-le_marchand-mines-paris-alumni" \
    --config_path="config.json" \
    --num_epochs=20 \
    --batch_size=64 \
    --model_save_name="dna_model_test" \
    --seed=42 \
    --num_workers=4 
