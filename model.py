import torch
import torch.nn as nn


class SimpleNN(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, hidden_dim2: int, num_classes: int=2):      
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim2, bias=True),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim2, num_classes, bias=True)
        )

    def forward(self, X):
        return self.output(X)
