import os
import lightning as pl
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from model.hybrud_model import ProteinGuidedGen
from tokenizers import Tokenizer

CONFIG_PATH = "config.json"

with open(CONFIG_PATH, 'r') as fp:
    config_param = json.load(fp)

class DNAProcessor_hybrid(pl.LightningModule):

    def __init__(self, dna_decoder_filepath = config_param["tokenizer"]["tokenizer_filepath"], esm_model_name="facebook/esm2_t6_8M_UR50D", freeze_esm=True):
        super().__init__()
        self.save_hyperparameters()
        self.model = ProteinGuidedGen(
        self.params = training_params
        self.tokenizer = Tokenizer.from_file(training_params["tokenizer"]["tokenizer_filepath"])
        self.wandb_run_id = None
        self.pad_token = self.tokenizer.token_to_id("[PAD]")
        self.loss = nn.CrossEntropyLoss(ignore_index = self.pad_token)

    def forward (self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x = batch
        input = x[:, :-1]
        target = x[:, 1:]
        logits = self.model(input) # On appelle le modèle interne
        loss = self.loss(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
    @torch.no_grad()
    def generate_sequences(self, n_sequence = 1, max_len = 1024, temp = 0.7, prompt = ""):
        self.model.eval()
        device = self.device
        input = self.tokenizer.encode("[CLS]" + prompt).ids
        seqs = torch.tensor([input]*n_sequence, device = device)
        length = seqs.size(1)
        print("Début de la génération de séquences...")
        for i in tqdm(range(max_len-length)):
            logits = self.model(seqs)
            last_logits = logits[:, -1, :]/temp
            probs = torch.softmax(last_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            seqs = torch.cat([seqs, next_tokens], dim=1)
            length += 1

        generated_strings = []
        sep_token_id = self.tokenizer.token_to_id("[SEP]")
        special_tokens = [self.tokenizer.token_to_id(token) for token
                         in ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]]
        for seq in seqs:
            seq_list = seq.tolist()
            for special_token in special_tokens:
                if special_token in seq_list:
                    seq_list.remove(special_token)
            decoded = self.tokenizer.decode(seq_list, skip_special_tokens = False) 
            generated_strings.append(decoded)
            if sep_token_id in seq_list:
                # On coupe tout ce qui dépasse après le premier [SEP]
                end_index = seq_list.index(sep_token_id)
                seq_list = seq_list[:end_index]

        self.model.train() 
        return generated_strings
    
    def validation_step(self, batch, _):
        x = batch
        input = x[:, :-1]
        target = x[:, 1:]
        logits = self.model(input)
        loss = self.loss(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        loss = self.loss(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint["wandb_run_id"] = self.wandb_run_id

    def on_load_checkpoint(self, checkpoint):
        self.wandb_run_id = checkpoint.get("wandb_run_id")

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=config_param["training"]["learning_rate"], 
                                        weight_decay = config_param["training"]["weight_decay"])
        total_steps = self.trainer.estimated_stepping_batches

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.params["training"]["learning_rate"],
            total_steps=total_steps,
            pct_start=0.1,  # 10% du temps en Warmup (montée), 90% en descente
            div_factor=25,  # Le LR de départ sera max_lr / 25
            final_div_factor=1000 # Le LR final sera minime
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # IMPORTANT : On met à jour à chaque BATCH, pas chaque époque
                "frequency": 1
            },
        }
        

    