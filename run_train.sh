#!/bin/bash

echo "Lancement de l'entraînement du DNA Transformer..."


python3 train.py \
    --project_name="Operons" \
    --wandb_team_name= "Adrien_LM" \
    --config_path="config.json" \
    --num_epochs=15 \
    --batch_size=64 \
    --model_save_name="dna_model_llamafied" \
    --seed=42 \
    --num_workers=4 
