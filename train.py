import os
from model.model import DNATransformer
from dataset.dataset import PretrainDataset
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import wandb
from absl import app, flags
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from absl import app, flags
import json

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")

FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", "my_project", "Name of the WandB project")
flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_float("init_lr", 0.001, "Initial learning rate")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("warmup", 1000, "Learning rate warmup steps")
flags.DEFINE_integer("num_workers", 2, "Number of DataLoader workers")
flags.DEFINE_integer("prefetch_factor", 2, "Number of batches to prefetch")
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
    "training_parameters_path", None, "Path to the training parameters JSON file"
)

def main(argv):
    del argv

    # Check that the training parameters path is provided
    if not FLAGS.training_parameters_path:
        raise ValueError("The 'training_parameters_path' flag must be provided.")

    # Load training parameters from JSON file
    training_parameters_path = FLAGS.training_parameters_path
    logger.info(f"Opening training parameters from {training_parameters_path}")
    try:
        with open(training_parameters_path, "r") as fp:
            parameters = json.load(fp)
    except Exception as e:
        logger.error(f"Error reading training parameters: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_project_name = FLAGS.project_name
    num_epochs = FLAGS.num_epochs
    initial_lr = FLAGS.init_lr
    batch_size = FLAGS.batch_size
    warmup = FLAGS.warmup
    num_workers = FLAGS.num_workers
    prefetch_factor = FLAGS.prefetch_factor
    model_save_name = FLAGS.model_save_name
    model_path = FLAGS.model_path
    resume_training = FLAGS.resume_training

    full_dataset = PretrainDataset()

    val_size = int(parameters["val_dataset_size"]*len(full_dataset))
    train_size = len(full_dataset)-val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size)

    if model_path and os.path.isfile(model_path):
        logger.info(f"Loading model from checkpoint: {model_path}")
        lightning_module = DNATransformer.load_from_checkpoint(
            checkpoint_path=model_path,
            training_params=parameters,
        )
        logger.info(f"Resuming WandB run: {lightning_module.wandb_run_id}")
    else:
        logger.info("Initializing new model")
        lightning_module = DNATransformer(training_params=parameters)

    # Initialize WandbLogger
    if resume_training:
        wandb_run = wandb.init(
            project=wandb_project_name, id=lightning_module.wandb_run_id, resume="allow"
        )
    else:
        wandb_run = wandb.init(project=wandb_project_name)

    wandb_logger = WandbLogger(experiment=wandb_run)
    lightning_module.wandb_run_id = wandb_logger.experiment.id
    if model_save_name is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/", filename=model_save_name
        )
    else:
        checkpoint_callback = ModelCheckpoint(dirpath="checkpoints")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    wandb_logger.experiment.config.update(
        {
            "architecture": parameters["model"]["type"],
            "#_layers": parameters["model"]["message_passing_num"],
            "#_neurons": parameters["model"]["hidden_size"],
            "max_lr": initial_lr,
            "batch_size": batch_size,
        }
    )
    trainer = Trainer(
        logger=wandb_logger,                 # Connecte WandB
        callbacks=[checkpoint_callback, lr_monitor], # Connecte la sauvegarde et le moniteur de LR
        max_epochs=num_epochs,
        accelerator="auto",                  # Choisit GPU/CPU tout seul
        devices="auto",
        log_every_n_steps=10,                # Fréquence de log pour WandB
        val_check_interval=1.0,              # Vérifie la validation à chaque fin d'époque
    )
    logger.info("Starting training...")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=model_path if resume_training else None # Gère la reprise automatique
    )








