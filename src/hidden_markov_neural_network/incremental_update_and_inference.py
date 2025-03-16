import sys
# Handle Hydra overrides from CLI
new_args = [arg.lstrip("-") if arg.startswith("--hyperparameters.") else arg for arg in sys.argv[1:]]
sys.argv = [sys.argv[0]] + new_args

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf
from model import HMNNModel
from data2 import GameDataModule

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Print configuration for verification
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Load checkpoint path from config
    checkpoint_path = cfg.hyperparameters.get("checkpoint_path", "path_to_your_checkpoint.ckpt")

    # Load previously trained initial model
    model = HMNNModel.load_from_checkpoint(checkpoint_path)

    # Set the model to train mode to allow updating
    model.train()

    # Setup data module with new incoming data (use hyperparameters from config)
    new_data_module = GameDataModule(
        data_dir=cfg.hyperparameters.new_data_dir, 
        csv_filename=cfg.hyperparameters.new_csv_filename, 
        batch_size=cfg.hyperparameters.batch_size
    )
    new_data_module.setup()

    # Trainer for incremental update
    trainer = Trainer(
        max_epochs=cfg.hyperparameters.epochs_update,
        gradient_clip_val=cfg.hyperparameters.get("gradient_clip_val", 0.5)
    )

    # Incrementally train the model on new data
    trainer.fit(model, datamodule=new_data_module)

    # Update priors after incremental training
    model.hmm_update_model_weights()
    print("Priors updated after incremental training.")

    # Save the updated model (recommended)
    updated_checkpoint = cfg.hyperparameters.get("updated_checkpoint_path", "updated_hmnn_model.pth")
    torch.save(model.state_dict(), updated_checkpoint)

    # Perform inference after incremental update
    model.eval()
    example_input, _, _ = next(iter(new_data_module.test_dataloader()))
    predictions, _, _ = model(example_input)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
