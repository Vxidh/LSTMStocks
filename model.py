import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=100, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_layer_size * 2, hidden_layer_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_layer_size * 2, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_size, hidden_layer_size // 2)
        self.fc3 = nn.Linear(hidden_layer_size // 2, 1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(hidden_layer_size * 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_layer_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_layer_size // 2)
    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout(lstm_out1)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout(lstm_out2)
        out = lstm_out2[:, -1, :]
        out = self.batch_norm1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.batch_norm2(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.batch_norm3(out)
        out = self.fc3(out)
        return out
