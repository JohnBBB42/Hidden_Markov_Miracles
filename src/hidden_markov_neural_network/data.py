import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Custom Dataset for Game Data (Now Sequential)
class GameDataset(Dataset):
    def __init__(self, dataframe, target_columns, time_column="season_id"):
        """
        - dataframe: Data containing input features and targets.
        - target_columns: List of column names to predict.
        - time_column: Column to track time (e.g., 'season_id' for sequential learning).
        """
        self.time_column = time_column
        # X includes all features except the target and the time column.
        self.X = dataframe.drop(target_columns + [time_column], axis=1).values.astype(np.float32)
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
                 batch_size=32, test_size=0.3, val_size=0.5, random_state=42,
                 start_year=1980, end_year=None):
        super().__init__()
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.start_year = start_year
        self.end_year = end_year
        # We want to predict home wins (binary classification)
        self.target_columns = ['home_win']
        # Season ID is used as a time index.
        self.time_column = "season_id"
        # Placeholders
        self.full_df = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_size = None

    def prepare_data(self):
        file_path = os.path.join(self.data_dir, self.csv_filename)
        game_df = pd.read_csv(file_path)

        # Ensure game_date is in datetime format
        game_df["game_date"] = pd.to_datetime(game_df["game_date"], errors='coerce')

        # Filter games from 1980 onwards
        game_df = game_df[game_df["game_date"].dt.year >= 1980]

        # Remove duplicate rows
        game_df = game_df.drop_duplicates()

        # Remove All-Star games.
        # (This example assumes All-Star games are played on specific days in February.
        # Adjust the condition as needed if you have a 'game_type' column or a known list of dates.)
        # For example, if All-Star games are on the 12th and 13th of February:
        game_df = game_df[~((game_df["game_date"].dt.month == 2) & (game_df["game_date"].dt.day.isin([12, 13])))]

        # Extract season_id (year) from game_date
        game_df["season_id"] = game_df["game_date"].dt.year

        # Keep only the columns needed for transformation.
        # If there are extra columns (e.g. a game type flag) that you want to remove, add them here.
        columns_to_keep = [
            'game_date', 'season_id', 'fg3a_home', 'fga_home', 'fg3m_home', 'wl_home',
            'fg3a_away', 'fga_away', 'fg3m_away', 'wl_away'
        ]
        #columns_to_keep = [
        #    'game_date', 'season_id', 'fg3a_home', 'fga_home', 'wl_home',
        #    'fg3a_away', 'fga_away', 'wl_away'
        #]

        game_df = game_df[columns_to_keep]

        # Drop rows with missing values
        game_df = game_df.dropna()

        # Create target: home_win = 1 if home team wins, 0 otherwise.
        game_df['home_win'] = game_df['wl_home'].map({'W': 1, 'L': 0})
        
        # Compute each season's home win rate
        # season_win_rate = game_df.groupby("season_id")['home_win'].transform('mean')
        # Include the season's win rate as an extra feature. This way the model "knows" the baseline home-court advantage.
        # game_df["season_home_win_rate"] = season_win_rate
        
        # (Optional) If you prefer to use a residual target, you could create one:
        # game_df["win_resid"] = game_df["home_win"] - season_win_rate

        # Keep only the features we want.
        # Here we use only the field goal attempts and three-point attempts from both home and away,
        # along with the season's baseline win rate, as inputs.
        final_df = game_df[['season_id', 'fg3a_home', 'fga_home', 'fg3m_home',
           'fg3a_away', 'fga_away', 'fg3m_away', 'home_win']]
        #final_df = game_df[['season_id', 'fg3a_home', 'fga_home',
        #    'fg3a_away', 'fga_away', 'home_win']]
        final_df = final_df.sort_values(by="season_id").reset_index(drop=True)

        # Scale numeric features so that differences in scale do not affect training.
        scaler = StandardScaler()
        feature_cols = ['fg3a_home', 'fga_home', 'fg3m_home',
            'fg3a_away', 'fga_away', 'fg3m_away']
        #feature_cols = ['fg3a_home', 'fga_home',
        #    'fg3a_away', 'fga_away']
        final_df[feature_cols] = scaler.fit_transform(final_df[feature_cols])

        self.full_df = final_df

    def setup(self, stage=None):
        """
        Called each time the dataloaders are created.
        Here we slice self.full_df by [start_year, end_year] and then do train/val/test splits.
        """
        if self.full_df is None:
            raise RuntimeError("You must call prepare_data() before setup().")

        # Slice data for the given season range.
        sliced_df = self.full_df.copy()
        if self.start_year:
            sliced_df = sliced_df[sliced_df["season_id"] >= self.start_year]
        if self.end_year:
            sliced_df = sliced_df[sliced_df["season_id"] <= self.end_year]

        # For modeling, we are using only the feature columns (all columns except the target and season_id).
        # In this example, the inputs will be: fg3a_home, fga_home, fg3a_away, fga_away, season_home_win_rate.
        df_full = sliced_df.copy()  # already contains the proper columns

        # Split into training, validation, and test sets.
        train_df, temp_df = train_test_split(df_full, test_size=self.test_size, random_state=self.random_state)
        val_df, test_df = train_test_split(temp_df, test_size=self.val_size, random_state=self.random_state)

        # -------------------------------
        # 1) Compute pos_weight from train_df
        #    'home_win' is the binary target (1=home, 0=away)
        pos_count = train_df['home_win'].sum()  # number of positive samples
        neg_count = len(train_df) - pos_count   # number of negative samples

        # Avoid division by zero
        if pos_count == 0 or neg_count == 0:
            self.pos_weight = 1.0  # fallback if data is extremely imbalanced
        else:
            self.pos_weight = neg_count / pos_count  # ratio of negatives to positives


        # Create datasets. Note that the target remains 'home_win' for binary classification.
        self.train_dataset = GameDataset(train_df, self.target_columns, self.time_column)
        self.val_dataset = GameDataset(val_df, self.target_columns, self.time_column)
        self.test_dataset = GameDataset(test_df, self.target_columns, self.time_column)
        # Save the training set size for use in your model (e.g., KL scaling)
        self.train_size = len(self.train_dataset)

    def train_dataloader(self):
        # You might want to shuffle here, but if sequential order matters, set shuffle=False.
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=7)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=7)