#!/bin/bash

echo "Lancement du test du DNA Transformer..."


python3 tests/test_run.py \
    --project_name="Operons" \
    --wandb_team_name= "Adrien_LM" \
    --config_path="config.json" \
    --num_epochs=2000 \
    --batch_size=64 \
    --model_save_name="dna_model_baseline" \
    --seed=42 \
    --num_workers=4 
