from itertools import combinations
from distutils.util import strtobool

import torch
import torch.nn as nn


class SimpleNN(nn.Module):

    def __init__(self, input_dim: int=64, hidden_dim: int=16, num_classes: int=2):
        
        super().__init__()

        self.output = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes, bias=True)
        )

    def forward(self, X):
        return self.output(X)
