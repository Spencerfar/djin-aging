import argparse
import torch
import numpy as np
from scipy.stats import sem
from pandas import read_csv

from torch.utils import data
from torch.nn import functional as F

from Model.model import Model
from Utils.record import record

from DataLoader.dataset import Dataset
from DataLoader.collate import custom_collate

parser = argparse.ArgumentParser('Predict classifier data')
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

torch.set_num_threads(6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 29
sims = 10
dt = 0.5
length = 20

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
min_values = test_set.min_values
max_values = test_set.max_values


ids = test_set.id_names
np.save('Analysis_Data/Synthetic_classifier_ids%d_epoch%d.npy'%(args.job_id, args.epoch), ids)

indiv_weights = test_set.weights
np.save('Analysis_Data/Synthetic_classifier_weights%d_epoch%d.npy'%(args.job_id, args.epoch), indiv_weights)

model = Model(device, N, args.gamma_size, args.z_size, args.decoder_size, args.Nflows, args.flow_hidden, args.f_nn_size, mean_T, std_T, dt, length).to(device)
model.load_state_dict(torch.load('Parameters/train%d_Model_DJIN_epoch%d.params'%(args.job_id, args.epoch),map_location=device))

with torch.no_grad():

    sigma_posterior = torch.distributions.gamma.Gamma(model.logalpha.exp(), model.logbeta.exp())
    W_posterior = torch.distributions.laplace.Laplace(model.mean, model.logscale.exp())
    
    fake = np.zeros((sims, test_set.__len__(), length, N + 1 + 26)) * np.nan
    real = np.zeros((test_set.__len__(), length, N + 1 + 26)) * np.nan
    mask = np.zeros((test_set.__len__(), length, N)) * np.nan
    start = 0
    for data in test_generator:

        size = data['Y'].shape[0]
        env = data['env'].to(device)
        med = data['med'][:,0,:].to(device)
        times = data['times'][:,:length]
        t0 = data['times'][:,0].to(device)
        env_long = torch.cat(length*[data['env'].unsqueeze(1)], dim = 1)
        
        pop_avg_bins = np.arange(40, 105, 3)[:-2]
        
        sex_index = env[:,12].long().cpu().numpy()
        
        avg_missing = []
        for t in range(length):
            t_index = np.digitize(times[:,t], pop_avg_bins, right=True) - 1
            t_index[t_index < 0] = 0
            predict_missing = pop_avg[sex_index, t_index][:,1:]
            avg_missing.append(predict_missing[:,np.newaxis])
        avg_missing = np.concatenate(avg_missing, axis = 1)
        
        mask[start:start+size] = data['mask'][:,:length].cpu().numpy()
        real[start:start+size,:,1:N+1] = (data['Y'][:,:length].cpu().numpy() + 0.1*np.random.randn(*(data['Y'][:,:length].cpu().numpy()).shape)) * mask[start:start+size] + (1-mask[start:start+size])*avg_missing
        
        alive = torch.ones(sims, size, int(length/dt)).to(device)
        
        for s in range(sims):
            
            sigma_y = sigma_posterior.sample((data['Y'].shape[0], length*2))
            W = W_posterior.sample((data['Y'].shape[0],))
            
            x0, t, pred_X, pred_S, pred_logGamma = model.generate(t0, env, med, sigma_y, W)
            
            alive[s,:,1:] = torch.cumprod(torch.bernoulli(torch.exp(-1*pred_logGamma.exp()[:,:-1]*dt)), dim=1)
            
            fake[s, start:start+size,:,1:N+1] = (pred_X[:,::2]).cpu().numpy() * mask[start:start+size] + (1-mask[start:start+size]) * avg_missing
            
            fake[s, start:start+size,:,N+1:] = env_long.cpu().numpy()
            fake[s, start:start+size,:,0] = t[:,::2].cpu().numpy()
        real[start:start+size,:,0] = t[:,::2].cpu().numpy()
        real[start:start+size,:,N+1:] = env_long.cpu().numpy()
        start += size

real = np.concatenate(sims*[real[np.newaxis]], axis=0)
labels = np.concatenate((np.ones((real.shape[0], real.shape[1])), np.zeros((fake.shape[0], fake.shape[1]))), axis=0)

realfake = np.concatenate((real, fake), axis=0) 
np.save('Analysis_Data/Synthetic_classifier_data%d_epoch%d.npy'%(args.job_id, args.epoch), realfake)
np.save('Analysis_Data/Synthetic_classifier_labels%d_epoch%d.npy'%(args.job_id, args.epoch), labels)
np.save('Analysis_Data/Synthetic_classifier_mask%d_epoch%d.npy'%(args.job_id, args.epoch), mask)
