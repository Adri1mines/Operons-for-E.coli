#!/bin/bash

echo "Lancement du préentraînement du DNA Transformer..."


python3 train.py \
    --project_name="Operons" \
    --wandb_team_name= "Adrien_LM" \
    --config_path="config.json" \
    --num_epochs=6 \
    --batch_size=48 \
    --learning_rate=5e-4 \
    --model_save_name="dna_model_llamafied_pretrain" \
    --seed=42 \
    --num_workers=4 \
    --resume_training=True \
    --model_path="dna_model_llamafied_pretrain" \
    --finetuning=False
