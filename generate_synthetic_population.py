import argparse
import torch
import numpy as np
from scipy.stats import sem
from pandas import read_csv

from torch.utils import data
from torch.nn import functional as F

from Model.model import Model
from Utils.record import record

from Utils.transformation import Transformation

from DataLoader.dataset import Dataset
from DataLoader.collate import custom_collate

parser = argparse.ArgumentParser('Predict change')
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

if torch.cuda.is_available():
    num_workers = 8
    torch.set_num_threads(12)
    test_after = 10
    test_average = 5
else:
    num_workers = 0
    torch.set_num_threads(4)
    test_after = 10
    test_average = 3

N = 29
sims = 1
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
num_test = test_set.__len__()
test_generator = data.DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))

mean_T = test_set.mean_T
std_T = test_set.std_T

mean_deficits = torch.Tensor(read_csv('Data/mean_deficits.txt', index_col=0,sep=',',header=None, names = ['variable']).values.flatten())
std_deficits = torch.Tensor(read_csv('Data/std_deficits.txt', index_col=0,sep=',',header=None, names = ['variable']).values.flatten())

psi = Transformation(mean_deficits[1:-3], std_deficits[1:-3], [15, 16, 23, 25, 26, 28])


model = Model(device, N, args.gamma_size, args.z_size, args.decoder_size, args.Nflows, args.flow_hidden, args.f_nn_size, mean_T, std_T, dt, length).to(device)
model.load_state_dict(torch.load('Parameters/train%d_Model_DJIN_epoch%d.params'%(args.job_id, args.epoch),map_location=device))

pop_avg_bins = np.arange(40, 105, 5)
env_mean = torch.Tensor(np.load('Data/Population_averages_test_env_all.npy'))
env_std = torch.Tensor(np.load('Data/Population_std_test_env_all.npy'))

for data in test_generator:
    break

size = 10000
for baseline_age in [65., 75., 85.]:
    
    with torch.no_grad():
        
        sigma_posterior = torch.distributions.gamma.Gamma(model.logalpha.exp(), model.logbeta.exp())
        W_dist = torch.distributions.laplace.Laplace(model.mean, model.logscale.exp())
        
        start = 0
        for group in range(300):
            
            X_output = np.zeros((size, 20, N))*np.nan
            alive_output = np.zeros((size, 20))*np.nan
            times_output = np.zeros((size, 20))*np.nan
            u_output = np.zeros((size, 19))*np.nan
            
            X = torch.zeros(sims, size, int(length/dt), N)
            S = torch.zeros(sims, size, int(length/dt))
            alive = torch.ones(sims, size, int(length/dt))
            death_ages = torch.zeros(sims, size)
            
            t0 = torch.ones(size) * baseline_age
            env = torch.zeros(size, 19)
            env_mask = torch.ones(size, 7)
            env = torch.cat((env, env_mask), dim=-1)

            med = torch.cat((torch.zeros(size,5), torch.ones(size,5)), dim=-1)

            t0_index = np.digitize(baseline_age, pop_avg_bins, right=True)

            # sex
            env[:,12] = 1*(torch.rand(size) < data['env'][:,12].mean())

            # ethnicity
            env[:,13] = 1*(torch.rand(size) < data['env'][:,13].mean())
            
            # long standing illness
            env[:,0] = 1*(torch.rand(size) < env_mean[env[:,12].long(),t0_index,0])

            # long-standing illness limits activities
            env[env[:,0]>0,1] = 1.*(torch.rand(env[env[:,0]>0,1].shape[0]) < env_mean[env[env[:,0]>0,12].long(),t0_index,1])

            # everything is an effort, smoking ever, smoking now, mobility, country of birth, joint replacement, and fractures
            for i in [2,3,4, 7,8, 10, 11]:
                env[:,i] = 1.*(torch.rand(size) < env_mean[env[:,12].long(),t0_index,i])
                
            # height, bmi, and alcohol
            minimums = [0, 0, 1]
            maximums = [np.inf, np.inf, 6]
            means = [mean_deficits[-3], mean_deficits[-2], mean_deficits[-1]]
            stds = [std_deficits[-3], std_deficits[-2], std_deficits[-1]]
            for j, i in enumerate([5,6,9]):
                env[:,i] = env_std[env[:,12].long(),t0_index,i]*torch.randn(size) + env_mean[env[:,12].long(),t0_index,i]
                env[:,i] = torch.clamp(env[:,i], (minimums[j]-means[j])/stds[j], (maximums[j]-means[j])/stds[j])
        
            env = env.to(device)
            med = med.to(device)
            t0 = t0.to(device)
            for s in range(sims):
                
                sigma_y = sigma_posterior.sample((size, length*2))
                W = W_dist.sample((size,))
                
                x0, t, pred_X, pred_S, pred_logGamma = model.generate(t0, env, med, sigma_y, W)
                X[s] = pred_X.cpu()
                alive[s,:,1:] = torch.cumprod(torch.bernoulli(torch.exp(-1*pred_logGamma.exp()[:,:-1]*dt)), dim=1).cpu()
                death_ages[s] = torch.max(t.cpu()*alive[s], dim = -1)[0].cpu()
                
            t0 = t[:,0]
            t=t.cpu()
            record_times = [torch.from_numpy(np.arange(t0[b].cpu(), t0[b].cpu()+20, 1)) for b in range(size)]
            X_record, alive_record = record(t, X, alive, record_times, dt)
            t0 = t0.cpu()
            
            for b in range(size):
                
                X_output[b] = ((X_record[b].permute(2,0,1)*alive_record[b]).permute(1,2,0) + (1-alive_record[b]).unsqueeze(-1)*(-10000)).cpu().numpy()
                X_output[b][X_output[b]<-5000] = np.nan
                alive_output[b] = alive_record[b].cpu().numpy()
                times_output[b] = record_times[b]
            
            u_output[:,:14] = env[:,:14].cpu().numpy()
            u_output[:,14:] = med[:,:5].cpu().numpy()      
            start += size

            # transform
            X_output = psi.untransform(X_output).numpy()
            
            np.save('Analysis_Data/Generated_population/Population_health_baseline%d_group%d.npy'%(baseline_age, group), X_output)
            np.save('Analysis_Data/Generated_population/Population_alive_baseline%d_group%d.npy'%(baseline_age, group), alive_output)
            np.save('Analysis_Data/Generated_population/Population_times_baseline%d_group%d.npy'%(baseline_age, group), times_output)
            np.save('Analysis_Data/Generated_population/Population_background_baseline%d_group%d.npy'%(baseline_age, group), u_output)
