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

parser = argparse.ArgumentParser('Predict population stats')
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
sims = 250
dt = 0.5
length = 60

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

pop_S_sum = np.zeros(120)
pop_S_sum2 = np.zeros(120)
pop_S_count = np.zeros(120)
pop_sum = np.zeros((120, N+1))
pop_sum2 = np.zeros((120, N+1))
pop_count = np.zeros(120)

mean_results = np.zeros((test_set.__len__(), 20, N+1)) * np.nan
corr_results = np.zeros((test_set.__len__(), 20, N*(N-1)//2)) * np.nan
std_results = np.zeros((test_set.__len__(), 20, N+1)) * np.nan

with torch.no_grad():

    sigma_posterior = torch.distributions.gamma.Gamma(model.logalpha.exp(), model.logbeta.exp())

    death_ages = torch.zeros(sims, test_set.__len__())
    
    start = 0
    for data in test_generator:

        size = data['Y'].shape[0]
        env = data['env'].to(device)
        med = data['med'][:,0,:].to(device)
        t0 = data['times'][:,0].to(device)
        
        X = torch.zeros(sims, size, int(length/dt), N)
        X_std = torch.zeros(sims, size, int(length/dt), N)
        S = torch.zeros(sims, size, int(length/dt))
        alive = torch.ones(sims, size, int(length/dt))
        
        # sample network
        W_posterior = torch.distributions.laplace.Laplace(model.mean, model.logscale.exp())
        
        for s in range(sims):
            
            sigma_y = sigma_posterior.sample((data['Y'].shape[0], length*2))
            W = W_posterior.sample((data['Y'].shape[0],))
            
            x0, t, pred_X, pred_S, pred_logGamma = model.generate(t0, env, med, sigma_y, W)
            X[s] = pred_X.cpu()
            X_std[s] = pred_X.cpu()
            S[s] = pred_S.exp().cpu()
            
            alive[s,:,1:] = torch.cumprod(torch.bernoulli(torch.exp(-1*pred_logGamma.exp()[:,:-1]*dt)), dim=1).cpu()
            death_ages[s,start:start+size] = torch.max(t.cpu()*alive[s], dim = -1)[0].cpu()
            
        
        t0 = t[:,0]
        t=t.cpu()
        record_times = [torch.from_numpy(np.arange(t0[b].cpu(), t0[b].cpu()+20, 1)) for b in range(size)]
        X_record, S_record = record(t, X, S, record_times, dt)
        X_std_record, alive_record = record(t, X_std, alive, record_times, dt)
        t0 = t0.cpu()
        
        
        X_sum = []
        X_sum_std = []
        XX_sum_std = [[] for i in range(N*(N-1)//2)]
        X_sum2 = []
        X_count = []
        for b in range(size):
            
            if t0[b] >= 65 and t0[b] < 100:
                
                pop_sum[int(t0[b]):int(t0[b])+20,1:] += torch.sum( X_record[b].permute(2,0,1)*alive_record[b], dim = 1).permute(1,0).cpu().numpy()
                pop_sum2[int(t0[b]):int(t0[b])+20,1:] += torch.sum( X_std_record[b].pow(2).permute(2,0,1)*alive_record[b], dim = 1).permute(1,0).cpu().numpy()
                pop_count[int(t0[b]):int(t0[b])+20] += torch.sum(alive_record[b], dim = 0).cpu().numpy()

                pop_S_sum[int(t0[b]):int(t0[b])+20] += S_record[b].sum(dim = 0).cpu().numpy()
                pop_S_sum2[int(t0[b]):int(t0[b])+20] += (S_record[b].pow(2)).sum(dim = 0).cpu().numpy()
                pop_S_count[int(t0[b]):int(t0[b])+20] += sims
                
            
                X_sum.append(torch.sum(X_record[b].permute(2,0,1)*alive_record[b], dim = 1).cpu())
                X_sum_std.append(torch.sum(X_std_record[b].permute(2,0,1)*alive_record[b], dim = 1).cpu())
                X_sum2.append(torch.sum(X_std_record[b].pow(2).permute(2,0,1)*alive_record[b], dim = 1).cpu())
                X_count.append(torch.sum(alive_record[b], dim = 0).cpu())
            
        start += size
        
pop_mean_results = np.zeros((120, N+1)) * np.nan
pop_std_results = np.zeros((120, N+1)) * np.nan

pop_S_results = pop_S_sum/pop_S_count
pop_S_std_results = np.sqrt(pop_S_sum2/pop_S_count - pop_S_results**2)

for i in range(N):
    pop_mean_results[:,i+1] = pop_sum[:,i+1]/pop_count
    pop_std_results[:,i+1] = np.sqrt(pop_sum2[:,i+1]/pop_count - (pop_mean_results[:,i+1])**2)

pop_mean_results[:,0] = np.arange(0, 120, 1)
pop_std_results[:,0] = np.arange(0, 120, 1)


np.save('Analysis_Data/Mean_pop_job_id%d_epoch%d_DJIN.npy'%(args.job_id, args.epoch), pop_mean_results)
np.save('Analysis_Data/Std_pop_job_id%d_epoch%d_DJIN.npy'%(args.job_id, args.epoch), pop_std_results)
np.save('Analysis_Data/S_pop_job_id%d_epoch%d_DJIN.npy'%(args.job_id, args.epoch), pop_S_results)
np.save('Analysis_Data/S_pop_std_job_id%d_epoch%d_DJIN.npy'%(args.job_id, args.epoch), pop_S_std_results)

np.save('Analysis_Data/Death_ages_pop_job_id%d_epoch%d_DJIN.npy'%(args.job_id, args.epoch), death_ages.numpy())
np.save('Analysis_Data/Mean_pop_trajectories_job_id%d_epoch%d_DJIN.npy'%(args.job_id, args.epoch), mean_results)
np.save('Analysis_Data/Std_pop_trajectories_job_id%d_epoch%d_DJIN.npy'%(args.job_id, args.epoch), std_results)
