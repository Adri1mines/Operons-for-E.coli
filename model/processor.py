import os
import lightning as pl
import torch
import torch.nn.functional as F
import json
from model import DNATransformer

CONFIG_PATH = "config.json"

with open(CONFIG_PATH, 'r') as fp:
    config_param = json.load(fp)

model_param = config_param["model"]

class DNAProcessor(pl.LightningModule):

    def __init__(self, training_params):
        super().__init__()
        self.save_hyperparameters()
        if os.path.isfile(training_params):
            with open(training_params, 'r') as fp:
                training_params = json.load(fp)
        self.model = DNATransformer(vocab_size = model_param["vocab_size"]
                                    ,d_model = model_param["d_model"]
                                    ,n_head = model_param["n_head"]
                                    ,max_len = model_param["max_len"]
                                    ,tokenizer = model_param["tokenizer"])
        self.params = training_params
        self.loss = F.cross_entropy
        self.wandb_run_id = None

    def forward (self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x = batch
        input = x[:, :-1]
        target = x[:, 1:]
        logits = self.model(input) # On appelle le modèle interne
        loss = self.loss(logits.view(-1, logits.size(-1)), target.view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, _):
        x = batch
        input = x[:, :-1]
        target = x[:, 1:]
        logits = self.model(input)
        loss = self.loss(logits.view(-1, logits.size(-1)), target.view(-1))
=======
        loss = self.loss(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        # Important : le nom ici "val_loss" doit correspondre au monitor du checkpoint
>>>>>>> 5c11b32053781d3bf2cceda28369d13d12c62cee
        self.log("val_loss", loss, prog_bar=True)
        return loss
    def on_save_checkpoint(self, checkpoint):
        checkpoint["wandb_run_id"] = self.wandb_run_id

    def on_load_checkpoint(self, checkpoint):
        self.wandb_run_id = checkpoint.get("wandb_run_id")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-3)
    