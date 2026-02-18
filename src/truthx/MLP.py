import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        if x.size() == residual.size():
            x = x + residual
        
        residual = x
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        if x.size() == residual.size():
            x = x + residual
        
        x = self.output_layer(x)
        return x
