import torch
from torch.nn import functional as F
import numpy as np

def log_gaussian(pred, data, sigma_y):
    return -0.5*torch.log(2*np.pi*sigma_y.pow(2)) - 0.5*(pred - data).pow(2)/sigma_y.pow(2)

def loss(X, recon_x0, log_Gamma, log_S, survival_mask, dead_mask, after_dead_mask, times, data, censored, mask, sigma_y, sigma_y0, batch_weights):

    # longitudinal
    log_longitudinal1 = torch.sum(log_gaussian(X[:,1:], data[:,1:], sigma_y)*mask[:,1:], dim = -1)
    log_longitudinal0 = torch.sum(log_gaussian(recon_x0, data[:,0,:], sigma_y0)*mask[:,0,:], dim = -1).unsqueeze(-1)
    log_longitudinal = torch.cat((log_longitudinal0, log_longitudinal1), dim=-1)
    batch_log_likelihood = torch.sum((batch_weights*log_longitudinal.permute(1,0)).permute(1,0), dim=0)

    # survival
    batch_log_S_likelihood = torch.sum(survival_mask*log_S +
                                        ((1 - censored)*dead_mask.permute(1,0)).permute(1,0)*log_Gamma +
          ((1 - censored)*after_dead_mask.permute(1,0)).permute(1,0)*torch.log(1-log_S.exp()+1e-8), dim = -1)
    
    return -1*( torch.sum( batch_log_likelihood,dim=0) + torch.sum(batch_weights*batch_log_S_likelihood,dim=0))

def sde_KL_loss(X, times, context, survival_mask, posterior, prior, sigma_x, dt, mean_T, std_T, batch_weights, med):
    
    T = X.shape[1]
    c =  torch.cat([context[:,None,:]]*(T),dim=1)   

    f_result = prior(X, torch.cat(((times.unsqueeze(-1) - mean_T)/std_T, c),dim=-1))
    
    g_result = posterior(X, torch.cat(((times.unsqueeze(-1) - mean_T)/std_T, c),dim=-1))

    full_integral = dt*torch.sum( torch.norm((g_result[:,1:-1] - f_result[:,1:-1])/sigma_x[:,1:-1], dim = -1).pow(2), dim = 1)
    first = dt/2 * (((g_result[:,0] - f_result[:,0])/sigma_x[:,0]).pow(2)).sum(dim = -1)
    last = dt/2 * (((g_result[:,-1] - f_result[:,-1])/sigma_x[:,-1]).pow(2)).sum(dim = -1)
    
    return 0.5*torch.sum(batch_weights*(full_integral + first + last))

def vae_loss(mu, logvar, batch_weights):#, mu_b, logvar_b, sigma_y):
    KLD = 0.5 * torch.sum(batch_weights*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    return -1*KLD
