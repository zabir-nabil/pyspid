import torch.nn.functional as F
import torch

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        pass

    def forward(self, input: torch.tensor) -> torch.tensor:
        pass