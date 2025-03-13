# train.py
import sys
# Preprocess sys.argv to remove extra dashes for Hydra overrides.
new_args = []
for arg in sys.argv[1:]:
    if arg.startswith("--hyperparameters."):
        new_args.append(arg.lstrip("-"))
    else:
        new_args.append(arg)
sys.argv = [sys.argv[0]] + new_args

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf

# Import our custom data module and the HMNNLightning model.
from data import GameDataModule
from model import HMNNLightning  # HMNNLightning should be defined in model.py with the BayesianNN

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Print configuration for verification
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set the random seed for reproducibility.
    torch.manual_seed(cfg.hyperparameters.seed)
    
    # Initialize the GameDataModule with parameters from the configuration.
    data_dir = cfg.hyperparameters.get("data_dir", "data/raw")
    csv_filename = cfg.hyperparameters.get("csv_filename", "game.csv")
    batch_size = cfg.hyperparameters.batch_size
    data_module = GameDataModule(data_dir=data_dir, csv_filename=csv_filename, batch_size=batch_size)
    data_module.setup()
    
    # Dynamically determine input and output dimensions from one batch.
    train_loader = data_module.train_dataloader()
    x, y = next(iter(train_loader))
    input_dim = x.shape[1]
    output_dim = y.shape[1]
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
    
    # Retrieve model hyperparameters from config.
    prior_std = cfg.hyperparameters.get("prior_std", 1.0)
    lr = cfg.hyperparameters.get("lr", 1e-3)
    task = cfg.hyperparameters.get("task", "regression")
    
    # Instantiate the HMNNLightning model.
    model = HMNNLightning(
        input_dim=input_dim,
        output_dim=output_dim,
        prior_std=prior_std,
        lr=lr,
        task=task
    )
    
    # Setup Wandb logger.
    wandb_project = cfg.hyperparameters.get("wandb_project", "hmnn_project")
    logger = pl.loggers.WandbLogger(project=wandb_project)
    
    # Instantiate the Trainer.
    trainer = Trainer(
        default_root_dir="my_logs_dir",
        max_epochs=cfg.hyperparameters.n_epochs,
        logger=logger,
        gradient_clip_val=cfg.hyperparameters.get("gradient_clip_val", 0.5)
    )
    
    # Start training.
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
