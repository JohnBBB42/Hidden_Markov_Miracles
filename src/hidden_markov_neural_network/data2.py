import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

# Custom Dataset for Game Data (Now Sequential)
class GameDataset(Dataset):
    def __init__(self, dataframe, target_columns, time_column="season_id"):
        """
        - dataframe: Data containing input features and targets.
        - target_columns: List of column names to predict.
        - time_column: Column to track time (e.g., 'season_id' for sequential learning).
        """
        self.time_column = time_column
        self.X = dataframe.drop(target_columns, axis=1).values.astype(np.float32)
        self.y = dataframe[target_columns].values.astype(np.float32)
        self.times = dataframe[self.time_column].values.astype(np.int64)  # Time-step index

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.y[idx]),
            torch.tensor(self.times[idx])  # Return time-step index
        )

# PyTorch Lightning DataModule for HMNN
class GameDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='data/raw', csv_filename='game.csv',
                 batch_size=32, test_size=0.3, val_size=0.5, random_state=42):
        super().__init__()
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        # Target variables (3-pointers made, attempted, %)
        self.target_columns = ['fg3m_home', 'fg3a_home', 'fg3_pct_home', 'fg3m_away', 'fg3a_away', 'fg3_pct_away']
        self.time_column = "season_id"  # Season ID is our time index
        # Placeholders for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # No downloading required; data is local.
        pass

    def setup(self, stage=None):
        file_path = os.path.join(self.data_dir, self.csv_filename)
        game_df = pd.read_csv(file_path)

        # Ensure game_date is in datetime format
        game_df["game_date"] = pd.to_datetime(game_df["game_date"], errors='coerce')

        # Filter games from 1980 onwards
        game_df = game_df[game_df["game_date"].dt.year >= 1980]

        # Extract season_id (year) from game_date
        game_df["season_id"] = game_df["game_date"].dt.year

        # Define columns to keep
        columns_to_keep = [
            'season_id', 'team_id_home', 'team_name_home', 'game_id', 'game_date', 'matchup_home',
            'wl_home', 'min', 'fg3m_home', 'fg3a_home', 'fg3_pct_home',
            'team_id_away', 'team_name_away', 'matchup_away', 'wl_away',
            'fg3m_away', 'fg3a_away', 'fg3_pct_away', 'season_type'
        ]
        game_df = game_df[columns_to_keep]

        # Drop rows with missing values
        game_df = game_df.dropna()

        # Convert datetime to Unix timestamp (integer)
        game_df["game_date"] = game_df["game_date"].astype('int64') // 10**9

        # One-hot encode categorical variables
        categorical_columns = ['team_name_home', 'matchup_home', 'wl_home',
                               'team_name_away', 'matchup_away', 'wl_away', 'season_type']
        game_df = pd.get_dummies(game_df, columns=categorical_columns, drop_first=True)

        # Convert team IDs and game IDs to numeric values
        game_df["game_id"] = pd.to_numeric(game_df["game_id"], errors='coerce')
        game_df["team_id_home"] = pd.to_numeric(game_df["team_id_home"], errors='coerce')
        game_df["team_id_away"] = pd.to_numeric(game_df["team_id_away"], errors='coerce')

        # Ensure all data is numeric (float)
        game_df = game_df.astype(float)

        # Sort data by season_id to maintain chronological order
        game_df = game_df.sort_values(by=["season_id", "game_date"])

        # Split into input features and target labels
        X = game_df.drop(self.target_columns, axis=1)
        y = game_df[self.target_columns]
        df_full = pd.concat([X, y], axis=1)

        # Split into training, validation, and test sets
        train_df, temp_df = train_test_split(df_full, test_size=self.test_size, random_state=self.random_state)
        val_df, test_df = train_test_split(temp_df, test_size=self.val_size, random_state=self.random_state)

        # Create datasets (GameDataset now includes time step tracking)
        self.train_dataset = GameDataset(train_df, self.target_columns, self.time_column)
        self.val_dataset = GameDataset(val_df, self.target_columns, self.time_column)
        self.test_dataset = GameDataset(test_df, self.target_columns, self.time_column)
        # Store the training set size for KL scaling in the model
        self.train_size = len(self.train_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
