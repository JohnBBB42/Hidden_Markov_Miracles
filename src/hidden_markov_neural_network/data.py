import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

# Custom Dataset for our game data
class GameDataset(Dataset):
    def __init__(self, dataframe, target_columns):
        # Separate features and targets
        self.X = dataframe.drop(target_columns, axis=1).values.astype(np.float32)
        self.y = dataframe[target_columns].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# PyTorch Lightning DataModule
class GameDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='data/raw', csv_filename='game.csv', batch_size=32, test_size=0.3, val_size=0.5, random_state=42):
        super().__init__()
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.batch_size = batch_size
        self.test_size = test_size      # Proportion of the dataset to include in the temporary test split
        self.val_size = val_size        # Proportion of temporary split to use for validation (the rest is test)
        self.random_state = random_state
        # Define the target columns we want to predict
        self.target_columns = [
            'fg3m_home', 'fg3a_home', 'fg3_pct_home',
            'fg3m_away', 'fg3a_away', 'fg3_pct_away'
        ]
        # Placeholders for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # This is called only from a single GPU in distributed settings
        # Here you might download data if needed
        pass

    def setup(self, stage=None):
        # Load CSV file
        file_path = os.path.join(self.data_dir, self.csv_filename)
        game_df = pd.read_csv(file_path)

        # Convert game_date to datetime if not already
        game_df["game_date"] = pd.to_datetime(game_df["game_date"], errors='coerce')

        # Filter games from 1980 onward
        game_df = game_df[game_df["game_date"].dt.year >= 1980]

        # Define columns to keep
        columns_to_keep = [
            'team_id_home', 'team_name_home', 'game_id', 'game_date', 'matchup_home',
            'wl_home', 'min', 'fg3m_home', 'fg3a_home', 'fg3_pct_home',
            'team_id_away', 'team_name_away', 'matchup_away', 'wl_away',
            'fg3m_away', 'fg3a_away', 'fg3_pct_away', 'season_type'
        ]
        game_df = game_df[columns_to_keep]

        # Remove rows with NA values
        game_df = game_df.dropna()

        # Convert datetime to numeric (Unix timestamp)
        game_df["game_date"] = game_df["game_date"].astype('int64') // 10**9

        # One-hot encode categorical columns
        categorical_columns = ['team_name_home', 'matchup_home', 'wl_home', 
                               'team_name_away', 'matchup_away', 'wl_away', 'season_type']
        game_df = pd.get_dummies(game_df, columns=categorical_columns, drop_first=True)

        # Convert string columns that should be numeric to numeric
        game_df["game_id"] = pd.to_numeric(game_df["game_id"], errors='coerce')
        game_df["team_id_home"] = pd.to_numeric(game_df["team_id_home"], errors='coerce')
        game_df["team_id_away"] = pd.to_numeric(game_df["team_id_away"], errors='coerce')

        # Force all data to be numeric (float)
        game_df = game_df.astype(float)

        # Split the data into training, validation, and test sets.
        X = game_df.drop(self.target_columns, axis=1)
        y = game_df[self.target_columns]
        df_full = pd.concat([X, y], axis=1)

        # First split into train and temporary test set
        train_df, temp_df = train_test_split(df_full, test_size=self.test_size, random_state=self.random_state)
        # Then split the temporary set into validation and test sets equally
        val_df, test_df = train_test_split(temp_df, test_size=self.val_size, random_state=self.random_state)

        # Create datasets
        self.train_dataset = GameDataset(train_df, self.target_columns)
        self.val_dataset = GameDataset(val_df, self.target_columns)
        self.test_dataset = GameDataset(test_df, self.target_columns)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

