import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class SolveSDE(nn.Module):

    def __init__(self, N, device, dt = 0.5, length = 25):
        super(SolveSDE, self).__init__()

        self.device = device
        self.N = N
        
        self.dt = dt
        self.num_t = int(length/dt)
    
    def _solve(self, model, x0, t0, M, context, h):

        X = torch.zeros((M, self.num_t, self.N)).to(self.device)
        log_S = torch.zeros((M, self.num_t)).to(self.device)
        log_Gammas = torch.zeros((M, self.num_t)).to(self.device)
        sigma_xs = torch.zeros((M, self.num_t, self.N)).to(self.device)

        times = torch.zeros((M, self.num_t)).to(self.device)
        
        X[:,0,:] = x0
        times[:,0] = t0
        log_Gammas[:,0] = -1e5
        
        for i in range(1, self.num_t):
            
            dx,  log_dS, log_Gamma, h, sigma_x = model(X[:, i-1, :], h, times[:,i-1], context)

            x_tilde = X[:, i-1, :] + self.dt*dx + sigma_x*np.sqrt(self.dt)
            X[:, i, :] = X[:, i-1, :] + self.dt*dx + torch.randn_like(X[:,i-1,:])*sigma_x*np.sqrt(self.dt) + 0.5*(model.sigma_x(x_tilde) - sigma_x)*(self.dt*torch.randn_like(X[:,i-1,:]).pow(2) - self.dt)/np.sqrt(self.dt)
            
            log_S[:, i] = log_S[:, i-1] + self.dt*log_dS
            log_Gammas[:,i] = log_Gamma.reshape(M)
            
            times[:,i] = times[:,i-1] + self.dt
            sigma_xs[:, i] = sigma_x
            
        sigma_xs[:,0] = sigma_xs[:,1]
        return times, X, log_S, log_Gammas, sigma_xs
