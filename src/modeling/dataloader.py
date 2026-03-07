import torch
from torch.utils.data import Dataset
import numpy as np

class ProductionStockDataset(Dataset):
    """
    Standard window-based dataset for stock forecasting.
    Takes a dataframe and returns (X_seq, y_label).
    """
    def __init__(self, df, feature_cols, target_col='target_binary', window_size=10):
        self.window_size = window_size
        self.feature_cols = feature_cols
        
        # Data and Labels
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.int64)
        
    def __len__(self):
        # Sequences are shifted by window_size
        return len(self.X) - self.window_size
        
    def __getitem__(self, idx):
        # Slice the features sequence
        x_seq = self.X[idx : idx + self.window_size]
        # Label is the target for the day immediately AFTER the window
        label = self.y[idx + self.window_size]
        return torch.tensor(x_seq), torch.tensor(label)
