import argparse
import torch
import numpy as np
from scipy.stats import sem
from pandas import read_csv

from torch.utils import data

from Model.model import Model

from DataLoader.dataset import Dataset
from DataLoader.collate import custom_collate


parser = argparse.ArgumentParser('Generate baseline')
parser.add_argument('--job_id', type=int)
parser.add_argument('--epoch', type=int)
parser.add_argument('--gamma_size', type=int, default = 25)
parser.add_argument('--z_size', type=int, default = 20)
parser.add_argument('--decoder_size', type=int, default = 65)
parser.add_argument('--Nflows', type=int, default = 3)
parser.add_argument('--flow_hidden', type=int, default = 24)
parser.add_argument('--f_nn_size', type=int, default = 12)
parser.add_argument('--W_prior_scale', type=float, default = 0.1)
args = parser.parse_args()

torch.set_num_threads(4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 29
sims = 100
dt = 0.5
length = 1

pop_avg = np.load('Data/Population_averages.npy')
pop_avg_env = np.load('Data/Population_averages_env.npy')
pop_std = np.load('Data/Population_std.npy')
pop_avg_ = torch.from_numpy(pop_avg[...,1:]).float()
pop_avg_env = torch.from_numpy(pop_avg_env).float()
pop_std = torch.from_numpy(pop_std[...,1:]).float()

test_name = 'Data/test.csv'
test_set = Dataset(test_name, N, pop=True, min_count=1)
num_test = 400
test_generator = data.DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))

mean_T = test_set.mean_T
std_T = test_set.std_T

model = Model(device, N, args.gamma_size, args.z_size, args.decoder_size, args.Nflows, args.flow_hidden, args.f_nn_size, mean_T, std_T, dt, length).to(device)
model.load_state_dict(torch.load('Parameters/train%d_Model_DJIN_epoch%d.params'%(args.job_id, args.epoch),map_location=device))

model = model.eval()

with torch.no_grad():

    sigma_posterior = torch.distributions.gamma.Gamma(model.logalpha.exp(), model.logbeta.exp())
    W_posterior = torch.distributions.laplace.Laplace(model.mean, model.logscale.exp())

    X = torch.zeros(sims, test_set.__len__(), N)
    Y = torch.zeros(test_set.__len__(), N)
    mask = torch.zeros(test_set.__len__(), N)
    start = 0
    for data in test_generator:
        
        size = data['Y'].shape[0]
        env = data['env'].to(device)
        med = data['med'][:,0,:].to(device)
        t0 = data['times'][:,0].to(device)
    
        for s in range(sims):
        
            sigma_y = sigma_posterior.sample((size, int(length/dt))) 
            W = W_posterior.sample((data['Y'].shape[0],))
        
            x0, t, pred_X, pred_S,_ = model.generate(t0, env, med, sigma_y, W)
            X[s,start:start+size] = x0
        Y[start:start+size] = data['Y'][:,0]
        mask[start:start+size] = data['mask'][:,0]
        start += size

np.save('Analysis_Data/generated_baseline_pop_job_id%d_epoch%d.npy'%(args.job_id, args.epoch), X.numpy())
