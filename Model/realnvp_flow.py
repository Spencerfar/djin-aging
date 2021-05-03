import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, nin, nout, nh):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.BatchNorm1d(nh),
            nn.Tanh(),
            nn.Linear(nh, nout)
        )
    def forward(self, x, h0):
        return self.net(torch.cat((x, h0), dim=-1))

class AffineHalfFlow(nn.Module):
    
    def __init__(self, dim, h_size, parity, device, net_class=MLP, nh=24):
        super().__init__()
        self.dim = dim
        self.h_size = h_size
        self.parity = parity
        self.device = device
        
        #scale neural net
        self.s_cond = net_class(self.dim // 2 + h_size, self.dim // 2, nh).to(device)

        #shift neural net
        self.t_cond = net_class(self.dim // 2 + h_size, self.dim // 2, nh).to(device)
        
    def forward(self, x, h0):
        x0, x1 = x[:,::2], x[:,1::2]
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(x0, h0)
        t = self.t_cond(x0, h0)
        z0 = x0 # untouched half
        z1 = torch.exp(s) * x1 + t # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det, h0
    
    def backward(self, z, h0):
        z0, z1 = z[:,::2], z[:,1::2]
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(z0, h0)
        t = self.t_cond(z0, h0)
        x0 = z0 # this was the same
        x1 = (z1 - t) * torch.exp(-s) # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det, h0


class NormalizingFlow(nn.Module):
    def __init__(self, flows, device):
        super().__init__()
        self.device = device
        self.flows = nn.ModuleList(flows)

    def forward(self, x, h0):
        m, _ = x.shape
        log_det = torch.zeros(m).to(self.device)
        zs = [x]
        for flow in self.flows:
            x, ld, h0 = flow.forward(x, h0)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z, h0):
        m, _ = z.shape
        log_det = torch.zeros(m).to(self.device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z, h0)
            log_det += ld
            xs.append(z)
        return xs, log_det, h0

    
class NormalizingFlowModel(nn.Module):
    def __init__(self, flows, device):
        super().__init__()
        self.device = device
        self.flow = NormalizingFlow(flows, device).to(device)
    
    def forward(self, x, h0, prior):
        zs, log_det = self.flow.forward(x, h0)
        prior_logprob = prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z, h0):
        xs, log_det = self.flow.backward(z)
        return xs, log_det
    
    def sample(self, num_samples, h0):
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z, h0)
        return xs
