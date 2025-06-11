import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=52, hidden_dim=64, output_dim=2):
        super(LSTMClassifier, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]
        x, _ = self.lstm1(x)
        # After lstm1: [batch_size, sequence_length, hidden_dim]
        # For the second LSTM, we want all outputs
        x, _ = self.lstm2(x)
        # After lstm2: [batch_size, sequence_length, hidden_dim], but we only need the last output
        x = x[:, -1, :]  # Take the output from the last time step
        # Shape: [batch_size, hidden_dim]
        x = self.fc(x)
        # Shape: [batch_size, output_dim]
        return x







