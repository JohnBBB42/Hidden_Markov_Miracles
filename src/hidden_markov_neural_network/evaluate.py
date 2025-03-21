import os
import torch
import pytorch_lightning as pl
from model import HMNNModel
from data import GameDataModule

def evaluate_models():
    """
    Loads and evaluates the HMNN models for:
      - initial_hmnn_model.pth (evaluated on 1980-1994)
      - updated_hmnn_model_1995.pth (evaluated on 1995-1999)
      - updated_hmnn_model_2000.pth (evaluated on 2000-2004)
      - ...
      - updated_hmnn_model_2020.pth (evaluated on 2020-2023)
    """
    # Define which models to evaluate.
    # We'll include 'initial' plus every 5 years from 1995..2020.
    model_years = [None, 1995, 2000, 2005, 2010, 2015, 2020]
    model_paths = []
    for y in model_years:
        if y is None:
            # initial model
            model_paths.append(("initial", "models/initial_hmnn_model.pth"))
        else:
            # updated model from that year
            model_paths.append((str(y), f"models/updated_hmnn_model_{y}.pth"))

    

    # We'll create a single Trainer to run .test() on each model in turn.
    trainer = pl.Trainer()

    # Evaluate each model on its corresponding interval.
    for label, model_path in model_paths:
        if not os.path.isfile(model_path):
            print(f"Skipping {label} (file not found: {model_path})")
            continue

        # Determine the test interval based on the label.
        if label == "initial":
            test_start_year = 1980
            test_end_year = 1994
        else:
            year = int(label)
            test_start_year = year
            test_end_year = min(year + 5, 2023)
            
        print(f"\n=== Evaluating model: {label} on interval {test_start_year}-{test_end_year} ===")

        # Create a DataModule for the corresponding test interval.
        data_module = GameDataModule(
            data_dir="data/raw",
            csv_filename="game.csv",
            batch_size=64,
            start_year=test_start_year,
            end_year=test_end_year,
            # Optionally, pass num_workers if desired:
            # num_workers=7
        )
        data_module.prepare_data()
        data_module.setup()

        # Dynamically determine input and output dimensions from one batch.
        train_loader = data_module.train_dataloader()
        x, y, _ = next(iter(train_loader))
        input_dim = x.shape[1]
        output_dim = y.shape[1]

        # Create a fresh instance of HMNNModel with the same architecture as your saved model.
        model = HMNNModel(
            input_dim=input_dim,
            hidden_dim=256,
            output_dim=output_dim,
            prior_initial_sigma=1.0,
            prior_large_sigma=1.0,
            alpha_k=0.5,
            sigma_k=0.2,   # adjust if needed
            c=54.6,        # adjust if needed
            pi=0.2,        # adjust if needed
            drop_rate=0.1,
            lr=1e-3,
            start_year=test_start_year,  # use the start of the interval
            n_train=data_module.train_size,
            pos_weight=data_module.pos_weight
        )
        # Load the saved weights.
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        # Evaluate on the test set for this interval.
        results = trainer.test(model, datamodule=data_module)
        print(f"Results for model {label} on {test_start_year}-{test_end_year}: {results}")

if __name__ == "__main__":
    evaluate_models()

