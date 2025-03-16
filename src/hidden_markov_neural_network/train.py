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
# from data import GameDataModule
from data2 import GameDataModule
from model import HMNNLightning, HMNNModel  # HMNNLightning should be defined in model.py with the BayesianNN

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
    # data_module = GameDataModule(data_dir=data_dir, csv_filename=csv_filename, batch_size=batch_size)
    data_module = GameDataModule(
        data_dir="data/raw",
        csv_filename="game.csv",
        batch_size=64,
        start_year=1980,
        end_year=1994
    )

    data_module.setup()
    
    # Dynamically determine input and output dimensions from one batch.
    train_loader = data_module.train_dataloader()
    x, y, _ = next(iter(train_loader))
    input_dim = x.shape[1]
    output_dim = y.shape[1]
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
    
    # Retrieve model hyperparameters from config.
    prior_std = cfg.hyperparameters.get("prior_std", 1.0)
    lr = cfg.hyperparameters.get("lr", 1e-3)
    task = cfg.hyperparameters.get("task", "regression")
    hidden_dim = cfg.hyperparameters.get("hidden_dim", 64)
    
    # Instantiate the HMNNLightning model.
    # model = HMNNLightning(
    #    input_dim=input_dim,
    #    output_dim=output_dim,
    #    prior_std=prior_std,
    #    lr=lr,
    #    task=task
    #)

    # Instantiate the HMNNModel (the full HMM version) instead of HMNNLightning.
    model = HMNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        prior_initial_sigma=prior_std,
        prior_large_sigma=cfg.hyperparameters.get("prior_large_sigma", 1.0),
        drop_rate=cfg.hyperparameters.get("drop_rate", 0.1),
        lr=lr,
        n_train=data_module.train_size  # Pass the size of the training set for KL scaling.
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
    # after trainer.fit(...)
    torch.save(model.state_dict(), "initial_hmnn_model.pth")


    # if cfg.hyperparameters.use_sequential_updates:
       # seasons = sorted(data_module.train_dataset.times.unique())
       # for season in seasons:
           # print(f"Training on season: {season}")
            
           # indices = [i for i, t in enumerate(data_module.train_dataset.times) if t == season]
           # season_dataset = torch.utils.data.Subset(data_module.train_dataset, indices)
           # season_loader = torch.utils.data.DataLoader(season_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)

           # trainer.fit(model, train_dataloaders=season_loader, val_dataloaders=data_module.val_dataloader())
            
           # model.hmm_update_model_weights()
           # print(f"Priors updated after season {season}.")
   # else:
       # trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
