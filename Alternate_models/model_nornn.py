import torch
import torch.nn as nn

from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))

from .dynamics_nornn import SDEModel
from Model.vae_flow import VAEImputeFlows
from Model.solver import SolveSDE

#@torch.jit.script
class Model(nn.Module):
    def __init__(self, device, N, gamma_size, z_size, decoder_size, Nflows, flow_hidden, f_nn_size, mean_T, std_T, dt = 0.5, length = 25):
        super(Model, self).__init__()

        self.N = N
        self.gamma_size = gamma_size
        self.mean_T = mean_T
        self.std_T = std_T
        self.z_size = z_size
        
        self.device = device

        self.register_parameter(name='mean', param = nn.Parameter(0.03*torch.randn(N,N)))
        self.register_parameter(name='logscale', param = nn.Parameter(torch.log(0.03*torch.ones(N,N))))
        self.register_parameter(name='logalpha', param = nn.Parameter(torch.log(10.0*torch.ones(N))))
        self.register_parameter(name='logbeta', param = nn.Parameter(torch.log(100.0*torch.ones(N))))
        
        # initialize vae, model, solver
        self.impute = VAEImputeFlows(N, z_size, decoder_size, Nflows, flow_hidden, device).to(device)
        self.dynamics = SDEModel(N, device, 10 + 26, self.gamma_size, f_nn_size, mean_T, std_T).to(device)
        self.solver = SolveSDE(N, device, dt=dt, length=length).to(device)
        
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
        
        if test:
            y0 = mask[:,0,:]*(y0_) \
            + (1 - mask[:,0,:])*(predict_missing + pop_std*torch.randn_like(y0_))
        else:
            y0 = mask0*(y0_ ) \
            + (1 - mask0)*(predict_missing + pop_std*torch.randn_like(mask0))

        #sample VAE
        if test:
            sample0, z_sample, mu0, logvar0, prior_entropy, log_det = self.impute(trans_t0, y0, mask[:,0,:], env, med0)
        else:
            sample0, z_sample, mu0, logvar0, prior_entropy, log_det = self.impute(trans_t0, y0, mask0, env, med0)
        
        
        recon_mean_x0 = self.impute.decoder(torch.cat((z_sample, trans_t0, env, med0), dim=-1))
        
        x0 = mask[:,0,:] * (y0_ ) + (1 - mask[:,0,:]) * recon_mean_x0
        
        # compute context
        context = torch.cat((env, med0), dim = -1)
        
        h = None
        t, pred_X, pred_S, pred_logGamma, pred_sigma_X, drifts = self.solver._solve(self.dynamics, x0, t0, batch_size, context, h, self.mean)
            
        return pred_X, t, pred_S, pred_logGamma, pred_sigma_X, context, y, times, mask, survival_mask, dead_mask, after_dead_mask, censored, sample_weights, med, env, z_sample, prior_entropy, log_det, recon_mean_x0, drifts, mask0, self.mean
