import argparse
import torch
import numpy as np
from scipy.stats import sem
from pandas import read_csv
from torch.utils import data

from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))  

from DataLoader.dataset import Dataset
from DataLoader.collate import custom_collate
from Utils.cindex import cindex_td

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cm = plt.get_cmap('Set1')


parser = argparse.ArgumentParser('Cindex')
parser.add_argument('--job_id', type=int)
parser.add_argument('--epoch', type=int)
args = parser.parse_args()

device = 'cpu'

N = 29
dt = 0.5
length = 50

pop_avg = np.load('../Data/Population_averages.npy')
pop_avg_env = np.load('../Data/Population_averages_env.npy')
pop_std = np.load('../Data/Population_std.npy')
pop_avg_ = torch.from_numpy(pop_avg[...,1:]).float()
pop_avg_env = torch.from_numpy(pop_avg_env).float()
pop_std = torch.from_numpy(pop_std[...,1:]).float()

test_name = '../Data/test.csv'
test_set = Dataset(test_name, N,  pop=False, min_count=10)
num_test = test_set.__len__()
test_generator = data.DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))


with torch.no_grad():

    survival = np.load('../Analysis_Data/Survival_trajectories_job_id%d_epoch%d_DJIN.npy'%(args.job_id,args.epoch))
    linear = np.load('../Comparison_models/Results/Survival_trajectories_baseline_id1_rfmice_test.npy')
    
    start = 0
    for data in test_generator:
        break

    death_ages = data['death age'].numpy()
    censored = data['censored'].numpy()
    times = data['times'].numpy()
    ages = times[:,0]
    death_ages = np.array([death_ages[m] if death_ages[m] > 0 else times[m].max() for m in range(death_ages.size)])
    
    sample_weight = data['weights'].numpy()
    sex_index = data['env'][:,12].long().numpy()
    

age_bins = np.arange(40, 105, 3)
bin_centers = age_bins[1:] - np.diff(age_bins)
c_index_list = np.ones(bin_centers.shape)*np.nan

for j in range(len(age_bins)-1):

    selected = []
    for i in range(death_ages.shape[0]):
        if age_bins[j] <= ages[i] and ages[i] < age_bins[j+1]:
            selected.append(i)
    
    c_index = cindex_td(death_ages[selected], survival[selected,:,1], survival[selected,:,0], 1 - censored[selected], weights = sample_weight)
    c_index_list[j] = c_index


c_index_linear = np.ones(bin_centers.shape)*np.nan

for j in range(len(age_bins)-1):

    selected = []
    for i in range(death_ages.shape[0]):
        if age_bins[j] <= ages[i] and ages[i] < age_bins[j+1]:
            selected.append(i)
    
    c_index = cindex_td(death_ages[selected], linear[selected,:,1], linear[selected,:,0], 1 - censored[selected], weights = sample_weight)
    c_index_linear[j] = c_index


#### Plot C index
fig,ax = plt.subplots(figsize=(4.5,4.5))
    
plt.plot(bin_centers, c_index_list, marker = 'o',color=cm(0), markersize=8, linestyle = '', label = 'DJIN model')
plt.plot(bin_centers, c_index_linear, marker = 's',color=cm(2), markersize=7, linestyle = '', label = 'Elastic-net Cox model')
    
plt.plot(bin_centers, cindex_td(death_ages, survival[:,:,1], survival[:,:,0], 1 - censored)*np.ones(bin_centers.shape), color = cm(0), linewidth = 2.5, label = '')

plt.plot(bin_centers, cindex_td(death_ages, linear[:,:,1], survival[:,:,0], 1 - censored)*np.ones(bin_centers.shape), color = cm(2), linewidth = 2.5, label = '', linestyle = '--')

print(cindex_td(death_ages, survival[:,:,1], survival[:,:,0], 1 - censored))

plt.ylabel('Survival C-index', fontsize = 14)
plt.xlabel('Baseline age (years)', fontsize = 14)
plt.legend(loc = 'lower right')

ax.text(-0.05, 1.05, 'a', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 16, zorder=1000000,
        fontweight='bold')

plt.ylim(0.5, 1.02)
plt.xlim(60,100)
ax.tick_params(labelsize=12)

ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))

plt.tight_layout()
plt.savefig('../Plots/Survival_Cindex_job_id%d_epoch%d.pdf'%(args.job_id, args.epoch))
