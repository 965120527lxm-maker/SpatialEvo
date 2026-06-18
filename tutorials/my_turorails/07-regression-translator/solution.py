"""Solution for Lesson 07: Regression Translator."""
import torch
import torch.nn as nn


class RegressionTranslator(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int):
        super(RegressionTranslator, self).__init__()
        self.fc = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)
