import sys
import os
# Handle Hydra overrides from CLI
new_args = [arg.lstrip("-") if arg.startswith("--hyperparameters.") else arg for arg in sys.argv[1:]]
sys.argv = [sys.argv[0]] + new_args

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf
from model import HMNNModel
from data import GameDataModule

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Print configuration for verification
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Load checkpoint path from config
    # checkpoint_path = cfg.hyperparameters.get("checkpoint_path", "path_to_your_checkpoint.ckpt")

    # Load previously trained initial model
    # model = HMNNModel.load_from_checkpoint(checkpoint_path)
    # Setup data module with new incoming data (use hyperparameters from config)
    update2_data_module = GameDataModule(
        data_dir="data/raw",
        csv_filename="game.csv",
        batch_size=64,
        start_year=2013,
        end_year=2020
    )
    update2_data_module.prepare_data()
    update2_data_module.setup()


    # Dynamically determine input and output dimensions from one batch.
    train_loader = update2_data_module.train_dataloader()
    x, y, _ = next(iter(train_loader))
    input_dim = x.shape[1]
    output_dim = y.shape[1]
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")

    # Retrieve model hyperparameters from config.
    prior_std = cfg.hyperparameters.get("prior_std", 1.0)
    lr = cfg.hyperparameters.get("lr", 1e-3)
    task = cfg.hyperparameters.get("task", "regression")
    hidden_dim = cfg.hyperparameters.get("hidden_dim", 64)
    prior_large_sigma = cfg.hyperparameters.get("prior_large_sigma", 1.0)
    drop_rate = cfg.hyperparameters.get("drop_rate", 0.1)
    start_year = update2_data_module.start_year  # Extract from the data module

    # Initialize the model with correct hyperparameters
    model = HMNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        prior_initial_sigma=prior_std,
        prior_large_sigma=prior_large_sigma,
        drop_rate=drop_rate,
        lr=lr,
        start_year=start_year,
        n_train=update2_data_module.train_size  # Needed for KL scaling
    )

    pth_checkpoint_path = "models/updated_hmnn_model_1995.pth"
    model.load_state_dict(torch.load(pth_checkpoint_path))

    # Set the model to train mode to allow updating
    model.train()

    # Trainer for incremental update
    trainer = Trainer(
        max_epochs=cfg.hyperparameters.epochs_update,
        gradient_clip_val=cfg.hyperparameters.get("gradient_clip_val", 0.5)
    )

    # Incrementally train the model on new data
    trainer.fit(model, datamodule=update2_data_module)

    # Update priors after incremental training
    model.hmm_update_model_weights()
    print("Priors updated after incremental training.")

    # Save the updated model (recommended)
    # updated_checkpoint = cfg.hyperparameters.get("updated_checkpoint_path", "updated_hmnn_model.pth")
    # torch.save(model.state_dict(), updated_checkpoint)

    # Generate filename dynamically based on start_year
    filename = f"updated_hmnn_model_{start_year}.pth"
    os.makedirs("models", exist_ok=True)
    filepath = os.path.join("models", filename)

    # Save the updated model
    torch.save(model.state_dict(), filepath)
    print(f"Updated model saved to {filepath}")

if __name__ == "__main__":
    main()
