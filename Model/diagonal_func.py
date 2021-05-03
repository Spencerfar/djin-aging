import torch
import torch.nn as nn
from torch.nn import functional as F


# N separate independent neural networks 
class DiagonalFunc(nn.Module):

    def __init__(self, N, input_size, hidden_size1):
        super(DiagonalFunc, self).__init__()

        self.N = N
        
        f = []
        for i in range(self.N):
            f.append(nn.Sequential(
                nn.Linear(input_size, hidden_size1),
                nn.ELU(),
                nn.Linear(hidden_size1, 1)
            ))
        self.f = nn.ModuleList(f)
        
    def forward(self, x, z):
        return torch.cat([self.f[i](torch.cat((x[..., i].unsqueeze(-1), z), dim=-1)) for i in range(self.N)], dim = -1)
