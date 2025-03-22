import sys
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf
from model import HMNNModel
from data import GameDataModule

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Define the initial training window and sliding window size.
    initial_start_year = 1980
    initial_end_year = 1994
    window_size = initial_end_year - initial_start_year + 1  # e.g., 15 years

    # Load the initial model trained on 1980-1994.
    # Assume the initial model has been saved at 'models/initial_hmnn_model.pth'
    # and that it has the correct architecture.
    # For determining input and output dimensions, we load one batch from the initial window.
    initial_data_module = GameDataModule(
        data_dir="data/raw",
        csv_filename="game.csv",
        batch_size=64,
        start_year=initial_start_year,
        end_year=initial_end_year
    )
    initial_data_module.prepare_data()
    initial_data_module.setup()
    train_loader = initial_data_module.train_dataloader()
    x, y, _ = next(iter(train_loader))
    input_dim = x.shape[1]
    output_dim = y.shape[1]
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")

    # Initialize the model.
    prior_std = cfg.hyperparameters.get("prior_std", 1.0)
    lr = cfg.hyperparameters.get("lr", 1e-3)
    hidden_dim = cfg.hyperparameters.get("hidden_dim", 64)
    prior_large_sigma = cfg.hyperparameters.get("prior_large_sigma", 1.0)
    drop_rate = cfg.hyperparameters.get("drop_rate", 0.1)
    # Use the start year from the initial training window
    model = HMNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        prior_initial_sigma=prior_std,
        prior_large_sigma=prior_large_sigma,
        drop_rate=drop_rate,
        lr=lr,
        start_year=initial_start_year,
        n_train=initial_data_module.train_size  # For KL scaling
    )

    # Load the pretrained checkpoint.
    initial_ckpt = "models/initial_hmnn_model.pth"
    model.load_state_dict(torch.load(initial_ckpt))
    model.train()

    # Define the final year until which you want to update.
    final_year = cfg.hyperparameters.get("final_year", 2020)

    # Loop: update the model one new year at a time (using only the new year's data)
    for current_year in range(initial_end_year + 1, final_year + 1):
        print(f"\nIncremental update on year {current_year}")

        # Create a new data module for the current year.
        update_data_module = GameDataModule(
            data_dir="data/raw",
            csv_filename="game.csv",
            batch_size=64,
            start_year=current_year,
            end_year=current_year  # Only the new year's data
        )
        update_data_module.prepare_data()
        update_data_module.setup()

        # Update the model's training size for proper KL scaling.
        model.n_train = update_data_module.train_size

        # Create a new Trainer instance for this update.
        trainer = Trainer(
            max_epochs=cfg.hyperparameters.epochs_update,
            gradient_clip_val=cfg.hyperparameters.get("gradient_clip_val", 0.5)
        )

        # Set window_info to current year (for CSV logging, etc.)
        model.window_info = f"{current_year}"
        trainer.fit(model, datamodule=update_data_module)

        # Update priors after training on the new year's data.
        model.hmm_update_model_weights()
        print(f"Priors updated after year {current_year}.")

        # Save the updated model.
        os.makedirs("models", exist_ok=True)
        checkpoint_filename = f"updated_hmnn_model_{current_year}.pth"
        checkpoint_filepath = os.path.join("models", checkpoint_filename)
        torch.save(model.state_dict(), checkpoint_filepath)
        print(f"Updated model saved to {checkpoint_filepath}")

if __name__ == "__main__":
    main()
