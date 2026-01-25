#!/bin/bash

echo "Lancement de l'entraînement du DNA Transformer..."


python3 train.py \
    --project_name="Operons for E.coli" \
    --training_parameters_path="training_params.json" \
    --num_epochs=20 \
    --batch_size=64 \
    --init_lr=0.0005 \
    --model_save_name="dna_model_test" \
    --seed=42 \
    --num_workers=4 
