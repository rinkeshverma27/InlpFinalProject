import torch
import torch.nn as nn

class BinaryLSTM(nn.Module):
    """
    3-layer LSTM architecture for binary stock direction prediction (UP/DOWN).
    Includes a deep classification head with ReLUs and Dropout.
    """
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        """
        Input: (B, seq_len, input_size)
        Output: (B, 2) logits
        """
        _, (h_n, _) = self.lstm(x)
        # Use last hidden state of the top LSTM layer
        return self.head(h_n[-1])
