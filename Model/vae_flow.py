import torch
import torch.nn as nn
from torch.nn import functional as F
from .realnvp_flow import AffineHalfFlow, NormalizingFlowModel

class VAEImputeFlows(nn.Module):
    def __init__(self, N, z_size, decoder_size, Nflows, flow_hidden, device):
        super(VAEImputeFlows, self).__init__()
        
        med_size = 10
        self.N = N
        self.z_size = z_size
        self.device = device
        self.Nflows = Nflows
        
        self.encoder = nn.Sequential(
            nn.Linear(2*N + 26 + med_size + 1, 95),
            nn.BatchNorm1d(95),
            nn.ELU(),
            nn.Linear(95, 70),
            nn.BatchNorm1d(70),
            nn.ELU(),
            nn.Linear(70, 2*z_size + z_size//2)
        )
        
        flows = [AffineHalfFlow(dim=z_size, h_size = z_size//2, parity=i%2, device=self.device, nh = flow_hidden) for i in range(Nflows)]
        self.flow_model = NormalizingFlowModel(flows, self.device)
        
        self.decoder_net = nn.Sequential(
            nn.Linear(z_size + 26 + med_size + 1, decoder_size),
            nn.BatchNorm1d(decoder_size),
            nn.ELU(),
            nn.Linear(decoder_size, N)
        )
    
    def forward(self, t, x, m, env, med):
        
        y = torch.cat((x, m, t, env, med),dim=-1)
        
        params = self.encoder(y)
        sum_mask = torch.clamp(torch.sum(m, dim=-1), 0., 1.)
        mu0, logvar0 = (sum_mask*params[..., :self.z_size].permute(1,0)).permute(1,0), (sum_mask*params[..., self.z_size:2*self.z_size].permute(1,0)).permute(1,0)
        h0 = (sum_mask*params[..., 2*self.z_size:].permute(1,0)).permute(1,0)
        
        prior = torch.distributions.normal.Normal(mu0, torch.exp(0.5*logvar0))
        sample0 = prior.rsample()
        
        zs, prior_logprob, log_det = self.flow_model(sample0, h0, prior)
        
        return sample0, zs[-1], mu0, logvar0, prior.entropy(), log_det
    
    def decoder(self, z):
        return self.decoder_net(z)
