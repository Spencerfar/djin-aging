import torch
import torch.nn as nn

from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory)) 

from .dynamics_full import SDEModel
from Model.memory_model import Memory
from .solver import SolveSDE

class Model(nn.Module):
    def __init__(self, device, N, gamma_size, z_size, decoder_size, mean_T, std_T, dt = 0.5, length = 25):
        super(Model, self).__init__()

        self.N = N
        self.gamma_size = gamma_size
        self.mean_T = mean_T
        self.std_T = std_T
        self.z_size = z_size
        self.dt = dt
        self.length = length
        
        self.device = device

        # sigma_y params
        self.register_parameter(name='logalpha', param = nn.Parameter(torch.log(10.0*torch.ones(N))))
        self.register_parameter(name='logbeta', param = nn.Parameter(torch.log(1000.0*torch.ones(N))))
        
        self.register_parameter(name='logalphax', param = nn.Parameter(torch.log(10.0*torch.ones(z_size))))
        self.register_parameter(name='logbetax', param = nn.Parameter(torch.log(100.0*torch.ones(z_size))))
        
        self.encoder = nn.Sequential(
            nn.Linear(2*N + 26 + 10 + 1, 95),
            nn.BatchNorm1d(95),
            nn.ELU(),
            nn.Linear(95, 70),
            nn.BatchNorm1d(70),
            nn.ELU(),
            nn.Linear(70, 2*self.z_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_size + 3 + 1, decoder_size),
            nn.LayerNorm(decoder_size),
            nn.ELU(),
            nn.Linear(decoder_size, N)
        )
        
        self.memory0 = Memory(z_size, 3, self.gamma_size).to(device)
        self.dynamics = SDEModel(z_size, device, 3, self.gamma_size, mean_T, std_T).to(device)
        self.solver = SolveSDE(z_size, device, dt=dt, length=length).to(device)
    
    def forward(self, data, sigma_y, test = False):
        
        # get batch
        y = data['Y'].to(self.device)
        times = data['times'].to(self.device)
        mask = data['mask'].to(self.device)
        mask0 = data['mask0'].to(self.device)
        survival_mask = data['survival_mask'].to(self.device)
        dead_mask = data['dead_mask'].to(self.device)
        after_dead_mask = data['after_dead_mask'].to(self.device)
        censored = data['censored'].to(self.device)
        env = data['env'].to(self.device)
        med = data['med'].to(self.device)
        sample_weights = data['weights'].to(self.device)
        predict_missing = data['missing'].to(self.device)
        pop_std = data['pop std'].to(self.device)
        
        batch_size = y.shape[0]
        
        # create initial timepoints
        y0_ = y[:, 0, :]
        t0 = times[:, 0]
        med0 = med[:,0,:]
        trans_t0 = (t0.unsqueeze(-1) - self.mean_T)/self.std_T
        
        # create input state
        if test:
            y0 = mask[:,0,:]*(y0_) \
            + (1 - mask[:,0,:])*(predict_missing + pop_std*torch.randn_like(y0_))
        else:
            y0 = mask0*(y0_ ) \
            + (1 - mask0)*(predict_missing + pop_std*torch.randn_like(mask0))

        #sample VAE
        if test:
            encoder_input = torch.cat((y0, mask[:,0,:], trans_t0, env, med0),dim=-1)
            params = self.encoder(encoder_input)
            mu0 = params[...,:self.z_size]
            logvar0 = params[...,self.z_size:]
        else:
            encoder_input = torch.cat((y0, mask0, trans_t0, env, med0),dim=-1)
            params = self.encoder(encoder_input)
            mu0 = params[...,:self.z_size]
            logvar0 = params[...,self.z_size:]
        

        z_sample = mu0 + torch.exp(0.5*logvar0)*torch.randn_like(mu0)
            
        # compute context
        context = env[:,[8,12,13]]
        context_full = torch.cat(int(self.length/self.dt)*[context.unsqueeze(1)], dim = 1)
        
        #compute memory0
        h = self.memory0(trans_t0, z_sample, env[:,[8,12,13]]) 
        h1 = h[:,:self.gamma_size]
        h2 = h[:,self.gamma_size:]
        h = (h1.unsqueeze(0).contiguous(), h2.unsqueeze(0).contiguous())

        t, pred_Z, pred_S, pred_logGamma, pred_sigma_X = self.solver._solve(self.dynamics, z_sample, t0, batch_size, context, h)
        
        trans_t = (t - self.mean_T)/self.std_T
        
        pred_X  = self.decoder(torch.cat((pred_Z, trans_t.unsqueeze(-1), context_full), dim=-1))
        
        recon_mean_x0 = pred_X[:,0]
        
        return pred_X, pred_Z, t, pred_S, pred_logGamma, pred_sigma_X, context, y, times, mask, survival_mask, dead_mask, after_dead_mask, censored, sample_weights, med, env, mu0, logvar0, recon_mean_x0, mask0
