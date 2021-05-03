import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# function for f
from .diagonal_func import DiagonalFunc

class SDEModel(nn.Module):

    def __init__(self, N, device, context_size, gamma_size, f_nn_size, mean_T, std_T):
        super(SDEModel, self).__init__()

        self.N = N
        self.mean_T = mean_T
        self.std_T = std_T
        self.device = device
        self.w_mask = (torch.ones(N,N) - torch.eye(N)).to(device) # mask for off-diagonal weights in network
        med_size = 10

        # diagonal neural net in dynamics
        self.f = DiagonalFunc(N, 2 + context_size, f_nn_size)

        # neural net for sigma_x
        self.sigma_nn = nn.Sequential(
            nn.Linear(N, N),
            nn.ELU(),
            nn.Linear(N, N),
            nn.Sigmoid()
        )

        # rnn and nn for mortality
        self.hazard1 = nn.GRU(N + 1, gamma_size, batch_first=True)
        self.hazard2 = nn.GRU(gamma_size, gamma_size - 15, batch_first=True)
        self.hazard_out = nn.Sequential(
            nn.ELU(),
            nn.Linear(gamma_size-15, 1)
        )

        # posterior drift nn
        self.g = nn.Sequential(
            nn.Linear(N + 1 + context_size, 8),
            nn.ELU(),
            nn.Linear(8, N, bias=False)
        )

    def sigma_x(self, x): # lower bound of 1e-5
        return self.sigma_nn(x) + 1e-5
    
    def prior_drift(self, x, z, W):
        return torch.matmul(x, self.w_mask*W) + self.f(x,z)

    def posterior_drift(self, x, z, W):
        return torch.matmul(x, self.w_mask*W) + self.g(torch.cat((x,z),dim=-1)) + self.f(x,z)
    
    def log_Gamma(self, x, h):
        g, h1 = self.hazard1(x.unsqueeze(1), h[0])
        g, h2 = self.hazard2(g, h[1])
        h = (h1, h2)
        return self.hazard_out(g).squeeze(1), h

    # output one step of posterior SDE and survival model
    def forward(self, x, h, t, context, W):
        
        z_RNN = torch.cat(((t.unsqueeze(-1) - self.mean_T)/self.std_T, context), dim=-1)
        x_ = x.clone()
        
        log_Gamma, h = self.log_Gamma(torch.cat((x, (t.unsqueeze(-1) - self.mean_T)/self.std_T),dim=-1), h)
        dx = torch.matmul(x.unsqueeze(1), self.w_mask*W).squeeze(1) + self.f(x, z_RNN) + self.g(torch.cat((x,z_RNN),dim=-1))
        log_dS = -torch.exp(log_Gamma).reshape(x.shape[0])
        
        return dx, log_dS, log_Gamma, h, self.sigma_x(x_)

    # output one step of prior SDE and survival model
    def prior_sim(self, x, h, t, context, W):
        
        z_RNN = torch.cat(((t.unsqueeze(-1) - self.mean_T)/self.std_T, context), dim=-1)
        x_ = x.clone()

        log_Gamma, h = self.log_Gamma(torch.cat((x, (t.unsqueeze(-1) - self.mean_T)/self.std_T),dim=-1), h)
        dx = torch.matmul(x.unsqueeze(1), self.w_mask*W).squeeze(1) + self.f(x_, z_RNN)
        log_dS = -torch.exp(log_Gamma).reshape(x.shape[0])
        
        return dx, log_dS, log_Gamma, h, self.sigma_x(x_)
