import os
import torch
import pytorch_lightning as pl
from model import NeuralNetwork, HMNNModel
from data import GameDataModule

def evaluate_models():
    """
    Loads and evaluates the same model on datasets from different years and intervals.
    """

    # Define which years/intervals to evaluate on
    model_intervals = [
        ("1980-1994", 1980, 1994),
        ("1995-2000", 1995, 2000),
        ("2000-2005", 2000, 2005),
        ("2005-2010", 2005, 2010),
        ("2010-2015", 2010, 2015),
        ("2015-2020", 2015, 2020),
        ("2020-2023", 2020, 2023)
    ]

    # Load the fixed model path (the same model for all evaluations)
    # model_path = "models/neural_network_model.pth"
    model_path = "models/updated_hmnn_model_2020.pth"
    if not os.path.isfile(model_path):
        print(f"Model file not found: {model_path}")
        return

    # Load the fixed model
    print(f"Loading model from: {model_path}")

    # Create a single trainer to evaluate the model on each dataset
    trainer = pl.Trainer()

    # Loop over each interval and evaluate the model on the corresponding data
    for label, start_year, end_year in model_intervals:
        print(f"\n=== Evaluating on data from interval: {label} ===")

        # Create a DataModule for the specific interval
        data_module = GameDataModule(
            data_dir="data/raw",
            csv_filename="game.csv",
            batch_size=64,
            start_year=start_year,
            end_year=end_year
        )
        data_module.prepare_data()
        data_module.setup()
        
        # Dynamically determine input and output dimensions from one batch.
        train_loader = data_module.train_dataloader()
        x, y, _ = next(iter(train_loader))
        input_dim = x.shape[1]
        output_dim = y.shape[1]

        #model = NeuralNetwork(input_size=4, lr=1e-3, start_year=1980)  # Adjust input_size if needed
        # Create a fresh instance of HMNNModel with the same architecture as your saved model.
        model = HMNNModel(
            input_dim=input_dim,
            hidden_dim=256,
            output_dim=output_dim,
            prior_initial_sigma=1.0,
            prior_large_sigma=1.0,
            alpha_k=0.5,
            sigma_k=0.5,   # adjust if needed
            c=148.4132,        # adjust if needed
            pi=0.5,        # adjust if needed
            drop_rate=0.1,
            lr=1e-3,
            start_year=1980,  # use the start of the interval
            n_train=data_module.train_size,
            pos_weight=data_module.pos_weight
        )

        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        # Evaluate the model on the specific interval data
        results = trainer.test(model, datamodule=data_module)
        print(f"Results for interval {label}: {results}")

if __name__ == "__main__":
    evaluate_models()
