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

from data import GameDataModule
from model import NeuralNetwork

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.hyperparameters.seed)

    data_module = GameDataModule(
        data_dir=cfg.hyperparameters.get("data_dir", "data/raw"),
        csv_filename=cfg.hyperparameters.get("csv_filename", "game.csv"),
        batch_size=cfg.hyperparameters.batch_size,
        start_year=1980,
        end_year=1994
    )

    data_module.prepare_data()
    data_module.setup()

    train_loader = data_module.train_dataloader()
    x, y, _ = next(iter(train_loader))
    input_dim = x.shape[1]
    print(f"Input dimension: {input_dim}")

    lr = cfg.hyperparameters.get("lr", 0.001)
    start_year = data_module.start_year

    model = NeuralNetwork(input_size=input_dim, lr=lr, start_year=start_year)

    wandb_project = cfg.hyperparameters.get("wandb_project", "nn_project")
    logger = pl.loggers.WandbLogger(project=wandb_project)

    trainer = Trainer(
        default_root_dir="my_logs_dir",
        max_epochs=cfg.hyperparameters.n_epochs,
        logger=logger,
        gradient_clip_val=cfg.hyperparameters.get("gradient_clip_val", 0.5)
    )

    trainer.fit(model, datamodule=data_module)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/neural_network_model.pth")
    print("Model saved at: models/neural_network_model.pth")

if __name__ == "__main__":
    main()
