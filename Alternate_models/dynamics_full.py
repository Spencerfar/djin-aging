import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class SDEModel(nn.Module):

    def __init__(self, N, device, context_size, gamma_size, mean_T, std_T):
        super(SDEModel, self).__init__()

        self.N = N
        self.mean_T = mean_T
        self.std_T = std_T
        self.device = device

        med_size = 10
        
        self.hazard1 = nn.GRU(N + 1, gamma_size, batch_first=True)
        self.hazard2 = nn.GRU(gamma_size, gamma_size - 15, batch_first=True) 

        self.hazard_out = nn.Sequential(
            nn.ELU(),
            nn.Linear(gamma_size-15, 1)
        )

        self.f = nn.Sequential(
            nn.Linear(N + 1 + context_size, 50),
            nn.ELU(),
            nn.Linear(50, N),
        )
        
        self.g = nn.Sequential(
            nn.Linear(N + 1 + context_size, 50),
            nn.ELU(),
            nn.Linear(50, N),
        )
        
        self.sigma_nn = nn.Sequential(
            nn.Linear(N, N),
            nn.ELU(),
            nn.Linear(N, N),
            nn.Sigmoid()
        )
        

    def sigma_x(self, x): # lower bound of 1e-5
        return self.sigma_nn(x) + 1e-5
    
    def prior_drift(self, x, z):
        return self.f(torch.cat((x,z),dim=-1))

    def posterior_drift(self, x, z):
        return self.g(torch.cat((x,z),dim=-1))
    
    def log_Gamma(self, x, h):
        g, h1 = self.hazard1(x.unsqueeze(1), h[0])
        g, h2 = self.hazard2(g, h[1])
        h = (h1, h2)
        return self.hazard_out(g).squeeze(1), h
    
    def forward(self, x, h, t, context):
        
        z_RNN = torch.cat(((t.unsqueeze(-1) - self.mean_T)/self.std_T, context), dim=-1)
        x_ = x.clone()
        
        log_Gamma, h = self.log_Gamma(torch.cat((x, (t.unsqueeze(-1) - self.mean_T)/self.std_T),dim=-1), h)
        
        dx = self.posterior_drift(x_, z_RNN)

        log_dS = -torch.exp(log_Gamma).reshape(x.shape[0])
        
        return dx, log_dS, log_Gamma, h, self.sigma_x(x_)
