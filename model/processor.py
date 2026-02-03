import os
import lightning as pl
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from hybrid_model import ProteinGuidedGen
from tokenizers import Tokenizer
from transformers import AutoTokenizer

class DNAProcessor(pl.LightningModule):

    def __init__(self, config_param, lr, weight_decay, use_encoder = False, pretrained_checkpoint_path = None,):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = ProteinGuidedGen(vocab_size = config_param["tokenizer"]["vocab_size"]
                                    ,d_model = config_param["model"]["d_model"]
                                    ,n_head = config_param["model"]["n_head"]
                                    ,num_layers = config_param["model"]["num_layers"]
                                    ,esm_model_name = config_param["finetuning"]["esm_model_name"]
                                    ,use_encoder = use_encoder)
        
        if pretrained_checkpoint_path is not None:
            ckpt = torch.load(pretrained_checkpoint_path, map_location = "cpu")
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("model.", "") # On retire le préfixe du wrapper Lightning
                new_state_dict[new_key] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
        
        self.params = config_param
        self.tokenizer_decoder = Tokenizer.from_file(config_param["tokenizer"]["tokenizer_filepath"])
        self.tokenizer_encoder = AutoTokenizer.from_pretrained(config_param["finetuning"]["esm_model_name"], padding = 'max_length',
                                                            truncation = 'longest_first',
                                                             max_length = config_param["model"]["max_len"])
        self.wandb_run_id = None
        self.loss = nn.CrossEntropyLoss(ignore_index = self.pad_token)
        self.use_encoder = use_encoder

    def forward (self, x, protein_input = None, prot_mask = None):
        return self.model(x, protein_input, prot_mask)

    def training_step(self, batch, _):
        if self.use_encoder:
            # Mode Finetune (Dataset complet)
            x = batch['input_decoder']
            prot_input = batch['input_protein']
            prot_mask = batch['protein_attention_mask']
            input = x[:, :-1]
            # Le modèle gère l'injection de contexte
            logits = self(input, prot_input, prot_mask)
        else:
            x = batch
            input = x[:, :-1]
            logits = self(input)

        target = x[:, 1:]
        loss = self.loss(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
    @torch.no_grad()
    def generate_sequences(self, n_sequence = 1, max_len = 1024, temp = 0.7,
                            prompt_decoder = "", protein_input_not_tokenized = None):
        self.model.eval()
        device = self.device
        input = self.tokenizer_decoder.encode("[CLS]" + prompt_decoder).ids
        seqs = torch.tensor([input]*n_sequence, device = device)
        length = seqs.size(1)
        print("Début de la génération de séquences...")
        if protein_input_not_tokenized is not None:
            protein_input = self.tokenizer_encoder(protein_input_not_tokenized)
        else:
            protein_input = None
        for i in tqdm(range(max_len-length)):
            logits = self.model(seqs, protein_input)
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
        if self.use_encoder:
            # Mode Finetune (Dataset complet)
            x = batch['input_decoder']
            prot_input = batch['input_protein']
            prot_mask = batch['protein_attention_mask']
            input = x[:, :-1]
            # Le modèle gère l'injection de contexte
            logits = self(input, prot_input, prot_mask)
        else:
            x = batch
            input = x[:, :-1]
            logits = self(input)

        target = x[:, 1:]
        loss = self.loss(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint["wandb_run_id"] = self.wandb_run_id

    def on_load_checkpoint(self, checkpoint):
        self.wandb_run_id = checkpoint.get("wandb_run_id")

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr= self.lr, 
                                        weight_decay = self.weight_decay)
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
        

    