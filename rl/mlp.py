"""Shared MLP module used by both SAC and Dreamer."""

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """3-layer MLP with ReLU activations and small output init."""

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.out.weight.data.uniform_(-3e-3, 3e-3)
        self.out.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)
