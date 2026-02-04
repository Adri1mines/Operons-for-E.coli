#!/bin/bash

echo "Lancement du finetuning du DNA Transformer..."


python3 train.py \
    --project_name="Operons" \
    --wandb_team_name= "Adrien_LM" \
    --config_path="config.json" \
    --num_epochs=10 \
    --batch_size=16 \
    --learning_rate=1e-4 \
    --model_save_name="dna_model_llamafied_finetunedwmask5" \
    --seed=42 \
    --num_workers=4 \
    --finetuning=True \
    --pretrain_model_path="checkpoints/dna_model_llamafied_pretrain.ckpt" \
