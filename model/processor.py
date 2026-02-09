import os
import lightning as pl
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from model.hybrid_model import ProteinGuidedGen
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
        self.loss = nn.CrossEntropyLoss(ignore_index = config_param["tokenizer"]["pad_token_id"])
        self.use_encoder = use_encoder

    def forward (self, x, protein_ids = None, prot_mask = None):
        return self.model(x, protein_ids, prot_mask)

    def _apply_word_dropout(self, input_ids, prob=0.5):
        """
        Applique le masquage aléatoire sur les inputs.
        Args:
            input_ids: Tensor (Batch, Seq_Len)
            prob: Probabilité de masquer un token
        Returns:
            masked_input_ids: Tensor avec des [MASK]
        """
        # 1. On clone pour ne surtout pas toucher à l'original (qui sert de target)
        masked_ids = input_ids.clone()
        
        # 2. Création du masque de probabilité
        # On crée une matrice de probas sur le même device que les données
        probs = torch.full(masked_ids.shape, prob, device=self.device)
        
        # 3. Protection des tokens spéciaux (OPTIONNEL mais recommandé)
        # On ne veut pas masquer [CLS], [SEP], [PAD] car ils structurent la phrase
        special_tokens = [
            self.tokenizer_decoder.token_to_id("[CLS]"),
            self.tokenizer_decoder.token_to_id("[SEP]"),
            self.tokenizer_decoder.token_to_id("[PAD]"),
            self.tokenizer_decoder.token_to_id("[UNK]")
        ]
        
        for special_id in special_tokens:
            probs.masked_fill_(masked_ids == special_id, 0.0)
            
        # 4. Génération du masque booléen (Bernoulli)
        mask_indices = torch.bernoulli(probs).bool()
        
        # 5. Remplacement par [MASK]
        mask_token_id = self.tokenizer_decoder.token_to_id("[MASK]")
        # Si pas de token MASK, on utilise un token rare ou 0
        if mask_token_id is None: 
            mask_token_id = 0 
            
        masked_ids[mask_indices] = mask_token_id
        
        return masked_ids


    def training_step(self, batch, _):
        if self.use_encoder:
            # Mode Finetune (Dataset complet)
            x = batch['input_decoder']
            prot_input = batch['input_protein']
            prot_mask = batch['protein_attention_mask']
            do_masking = (torch.rand(1).item() < 0.30)
            input = x[:, :-1]
            if do_masking:
                processed_input = self._apply_word_dropout(input, prob=0.3)
            # Le modèle gère l'injection de contexte
            else:
                processed_input = input
            logits = self(x = processed_input, protein_ids = prot_input, prot_mask = prot_mask)
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
                            prompt_decoder = None, protein_input = None, prot_mask = None):
        self.model.eval()
        device = self.device
        if prompt_decoder is None:
            input = self.tokenizer_decoder.encode("[CLS]").ids
            seqs = torch.tensor([input]*n_sequence, device = device)
        else:
            seqs = prompt_decoder
        length = seqs.size(1)
        print("Début de la génération de séquences...")
        if protein_input is not None:
            protein_input = protein_input
            prot_mask = prot_mask
        else:
            protein_input = None
            prot_mask = None
        for i in tqdm(range(max_len-length)):
            logits = self(seqs, protein_input, prot_mask)
            last_logits = logits[:, -1, :]/temp
            probs = torch.softmax(last_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            seqs = torch.cat([seqs, next_tokens], dim=1)
            length += 1

        generated_strings = []
        sep_token = "[SEP]"
        special_tokens = [self.tokenizer_decoder.token_to_id(token) for token
                         in ["[UNK]", "[CLS]", "[PAD]", "[MASK]","[START_CONTEXT]", "[TERM_INFERRED]", "[TERM_KNOWN]", "[END_CONTEXT]"]]
        seqs_list = seqs.tolist()
        for seq in seqs_list:
            for special_token in special_tokens:
                if special_token in seq:
                    seq = [x for x in seq if x != special_token]
            decoded = self.tokenizer_decoder.decode(seq, skip_special_tokens = False) 
            if sep_token in decoded:
                # On coupe tout ce qui dépasse après le premier [SEP]
                end_index = decoded.index(sep_token_id)
                seq_list = decoded[:end_index]
            generated_strings.append(decoded)


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
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        
        
        for k, v in state_dict.items():
            # Le préfixe ajouté par torch.compile est généralement "_orig_mod."
            if "_orig_mod." in k:
                clean_key = k.replace("_orig_mod.", "")
                new_state_dict[clean_key] = v
            else:
                new_state_dict[k] = v
        
        # On remplace le dictionnaire par la version propre
        checkpoint["state_dict"] = new_state_dict

    def configure_optimizers(self):

        if self.use_encoder:
            connector_params = [] # Projecteur + Cross Attention
            base_params = []      # Le reste (Décodeur, ESM...)

            for name, param in self.model.named_parameters():
                # Si le paramètre appartient à la "nouvelle" connexion
                if "projector" in name or "cross_attn" in name:
                    connector_params.append(param)
                else:
                    base_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': self.lr},           # Vitesse normale
            {'params': connector_params, 'lr': self.lr * 20}],
                                        weight_decay = self.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=0.1,  # 10% du temps en Warmup (montée), 90% en descente
            div_factor=25,  # Le LR de départ sera max_lr / 25
            final_div_factor=1000 # Le LR final sera minime
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
                "frequency": 1
            },
        }
        

    