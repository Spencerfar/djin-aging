import torch
import torch.nn as nn

class Memory(nn.Module):
    def __init__(self, N, context_size, hidden_size):
        super(Memory, self).__init__()
        
        self.memory0 = nn.Sequential(
            nn.Linear(context_size + N + 1, 3*hidden_size),
            nn.ELU(),
            nn.Linear(3*hidden_size, hidden_size + hidden_size - 15)
        )
    
    def forward(self, t, x0, v):
        return self.memory0(torch.cat((x0, v, t), dim=-1))
