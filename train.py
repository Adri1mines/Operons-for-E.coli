import os
from dataset.pretrain.dataset import PretrainDataset
from dataset.finetune.dataset import FinetuneDataset
from model.processor import DNAProcessor
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import wandb
from absl import app, flags
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from absl import app, flags
from callbacks.bio_test import BioEvalCallback
import json

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")

FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", "my_project", "Name of the WandB project")
flags.DEFINE_string("wandb_team_name", "my_team", "Name of the team")
flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("num_workers", 2, "Number of DataLoader workers")
flags.DEFINE_integer("prefetch_factor", 2, "Number of batches to prefetch")
flags.DEFINE_integer("learning_rate", 1e-4, "max learning rate")
flags.DEFINE_integer("weight_decay", 1e-2, "weight decay")
flags.DEFINE_string(
    "model_save_name", None, "Name to save the checkpoint during training"
)
flags.DEFINE_string(
    "model_path", None, "Path to the checkpoint (.ckpt) to resume training from"
)
flags.DEFINE_bool(
    "resume_training", False, "Whether to resume an unfinished training or not"
)
flags.DEFINE_string(
    "config_path", None, "Path to the training parameters JSON file"
)
flags.DEFINE_string(
    "finetuning", False, "if training is in finetuning mode or not"
)

def main(argv):
    del argv

    # Check that the training parameters path is provided
    if not FLAGS.config_path:
        raise ValueError("The 'training_parameters_path' flag must be provided.")

    # Load training parameters from JSON file
    config_path = FLAGS.config_path
    logger.info(f"Opening training parameters from {config_path}")
    try:
        with open(config_path, "r") as fp:
            config_param = json.load(fp)
    except Exception as e:
        logger.error(f"Error reading training parameters: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_project_name = FLAGS.project_name
    num_epochs = FLAGS.num_epochs
    batch_size = FLAGS.batch_size
    num_workers = FLAGS.num_workers
    prefetch_factor = FLAGS.prefetch_factor
    model_save_name = FLAGS.model_save_name
    model_path = FLAGS.model_path
    resume_training = FLAGS.resume_training
    wandb_team_name = FLAGS.wandb_team_name
    seed = FLAGS.seed
    finetuning = FLAGS.finetuning
    lr = FLAGS.learning_rate
    weight_decay = FLAGS.weight_decay

    generator = torch.Generator().manual_seed(seed)

    if not finetuning:
        full_dataset = PretrainDataset()
    else:
        full_dataset = FinetuneDataset()

    val_size = int(config_param["data"]["val_dataset_size"]*len(full_dataset))
    train_size = len(full_dataset)-val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator = generator)

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, generator = generator, num_workers = num_workers, prefetch_factor = prefetch_factor)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers, prefetch_factor = prefetch_factor)

    if model_path and os.path.isfile(model_path):
        logger.info(f"Loading model from checkpoint: {model_path}")
        lightning_module = DNAProcessor.load_from_checkpoint(checkpoint_path=model_path)
        logger.info(f"Resuming WandB run: {lightning_module.wandb_run_id}")
    else:
        logger.info("Initializing new model")
        lightning_module = DNAProcessor(config_param, lr = lr, weight_decay= weight_decay, use_encoder = finetuning)

    # Initialize WandbLogger
    if resume_training:
        wandb_run = wandb.init(
            project=wandb_project_name, entity = wandb_team_name, id=lightning_module.wandb_run_id, resume="allow"
        )
    else:
        wandb_run = wandb.init(project=wandb_project_name, entity = wandb_team_name)

    wandb_logger = WandbLogger(experiment=wandb_run)
    lightning_module.wandb_run_id = wandb_logger.experiment.id
    if model_save_name is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/", filename=model_save_name
        )
    else:
        checkpoint_callback = ModelCheckpoint(dirpath="checkpoints")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    bio_eval_callback = BioEvalCallback(tokenizer_path = config_param["tokenizer"]["tokenizer_filepath"], 
                                        val_dataset=val_dataset)
    early_stop_callback = EarlyStopping(
    monitor="bio/global_score",  # On surveille la loss de validation
    min_delta=0.00,      # Il faut que ça s'améliore un minimum
    patience=2,          # Si ça ne s'améliore pas pendant 2 checks (epochs), on coupe
    verbose=True,
    mode="min")

    wandb_logger.experiment.config.update(
        {
            "architecture": config_param["model"]["type"],
            "d_model": config_param["model"]["d_model"],
            "n_head": config_param["model"]["n_head"],
            "context_size": config_param["model"]["max_len"],
            "max_lr": lr,
            "batch_size": batch_size,
        }
    )
    trainer = Trainer(
        logger=wandb_logger,                 # Connecte WandB
        callbacks=[checkpoint_callback, lr_monitor, bio_eval_callback, early_stop_callback], # Connecte la sauvegarde et le moniteur de LR
        max_epochs=num_epochs,
        accelerator="gpu",                  # Choisit GPU/CPU tout seul
        devices=1,
        precision = "bf16-mixed",
        log_every_n_steps=10,                # Fréquence de log pour WandB
        val_check_interval=1.0,            # Vérifie la validation à chaque fin d'époque
    )
    logger.info("Starting training...")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=model_path if resume_training else None # Gère la reprise automatique
    )

if __name__ == "__main__":
    app.run(main)








