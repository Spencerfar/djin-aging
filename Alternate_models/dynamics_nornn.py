import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory)) 

from Model.diagonal_func import DiagonalFunc

class SDEModel(nn.Module):

    def __init__(self, N, device, context_size, gamma_size, f_nn_size, mean_T, std_T):
        super(SDEModel, self).__init__()

        self.N = N
        self.mean_T = mean_T
        self.std_T = std_T
        self.device = device
        
        med_size = 10
        
        self.f = DiagonalFunc(N, 2 + context_size, f_nn_size)
        
        self.sigma_nn = nn.Sequential(
            nn.Linear(N, N),
            nn.ELU(),
            nn.Linear(N, N),
            nn.Sigmoid()
        )
        
        self.w_mask = (torch.ones(N,N) - torch.eye(N)).to(device)
        
        self.hazard = nn.Sequential(
            nn.Linear(N+1+context_size, gamma_size),
            nn.ELU(),
            nn.Linear(gamma_size, gamma_size-15),
            nn.ELU(),
            nn.Linear(gamma_size-15, 1)
        )
        
        self.g = nn.Sequential(
            nn.Linear(N + 1 + context_size, 8),
            nn.ELU(),
            nn.Linear(8, N, bias=False),
        )

    def sigma_x(self, x): # lower bound of 1e-5
        return self.sigma_nn(x) + 1e-5
    
    def prior_drift(self, x, z, W):
        return torch.matmul(x, self.w_mask*W) + self.f(x,z)

    def posterior_drift(self, x, z):
        return torch.matmul(x, self.w_mask*W) + self.g(torch.cat((x,z),dim=-1)) + self.f(x,z)
    
    def log_Gamma(self, x):
        return self.hazard(x)
    
    def forward(self, x, h, t, context, W):
        
        z_RNN = torch.cat(((t.unsqueeze(-1) - self.mean_T)/self.std_T, context), dim=-1)
        x_ = x.clone()
        
        log_Gamma = self.log_Gamma(torch.cat((x, context, (t.unsqueeze(-1) - self.mean_T)/self.std_T),dim=-1))
        
        h = None
        dx = dx = torch.matmul(x.unsqueeze(1), self.w_mask*W).squeeze(1) + self.f(x, z_RNN) + self.g(torch.cat((x,z_RNN),dim=-1))
        
        log_dS = -torch.exp(log_Gamma).reshape(x.shape[0])
        
        return dx, log_dS, log_Gamma, h, self.sigma_x(x_)
