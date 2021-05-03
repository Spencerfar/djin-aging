import argparse
import torch
import numpy as np
from scipy.stats import sem, spearmanr
from pandas import read_csv

from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))  

from DataLoader.dataset import Dataset
from DataLoader.collate import custom_collate
from Utils.record import record

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cm = plt.get_cmap('Set1')
cm2 = plt.get_cmap('Set2')

parser = argparse.ArgumentParser('Predict longitudinal all times')
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
pop_avg_bins = np.arange(40, 105, 3)[:-2]

missing = [[[] for y in range(10)] for i in range(N)]
notmissing = [[[] for y in range(10)] for i in range(N)]
linear_notmissing = [[[] for y in range(10)] for i in range(N)]
exact_missing = [[[] for y in range(10)] for i in range(N)]
exact_notmissing = [[[] for y in range(10)] for i in range(N)]
weights_notmissing = [[[] for y in range(10)] for i in range(N)]
weights_missing = [[[] for y in range(10)] for i in range(N)]

first_notmissing = [[[] for y in range(10)] for i in range(N)]
pop_missing = [[[] for y in range(10)] for i in range(N)]
pop_notmissing = [[[] for y in range(10)] for i in range(N)]

test_name = '../Data/test.csv'
test_set = Dataset(test_name, N, pop=False, min_count = 10)
num_test = test_set.__len__()
test_generator = torch.utils.data.DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))

with torch.no_grad():

    mean = np.load('../Analysis_Data/Mean_trajectories_job_id%d_epoch%d_DJIN.npy'%(args.job_id,args.epoch))
    linear = np.load('../Comparison_models/Results/Longitudinal_predictions_baseline_id21_rfmice_test.npy')
    
    mean_impute = np.zeros(mean.shape)
    
    for yi, years in enumerate([0, 2, 4, 6, 8, 10, 12, 14, 16, 18]):
        
        
        start = 0
        for data in test_generator:
            break
        
        y = data['Y']
        times = data['times']
        mask = data['mask']
        sample_weight = data['weights'].numpy()
        sex_index = data['env'][:,12].long().numpy()
        
        
        record_times = []
        record_y = []
        record_mask = []
        for b in range(num_test):
            observed = torch.sum(mask[b,:, :], dim = -1) > 0
            record_times.append(times[b, observed].numpy().astype(int))
            record_y.append(y[b, observed, :].numpy())
            record_mask.append(mask[b, observed, :].numpy().astype(int))

        if yi == 0:
            continue
        else:
            
            for b in range(num_test):
                t = 0
                for t_rec in range(len(record_times[b])):
                    
                    t_index = np.digitize(record_times[b][t_rec], pop_avg_bins, right=True)
                    pop_data_t = pop_avg_[sex_index[b], t_index].numpy()
                    
                    while t < min(40, int(np.sum(~np.isnan(mean[b,:,1])))):
                        
                        if record_times[b][t_rec] == mean[b, t, 0].astype(int):
                            
                            for n in range(N):
                                
                                if record_mask[b][t_rec, n] > 0 and int(record_times[b][t_rec] - record_times[b][0]) < years + 1 and int(record_times[b][t_rec] - record_times[b][0]) >= years -1:
                                    
                                    # missing
                                    if record_mask[b][0, n] < 1:
                                        missing[n][yi].append(mean[b, t, n+1])
                                        exact_missing[n][yi].append(record_y[b][t_rec, n])
                                        weights_missing[n][yi].append(sample_weight[b])
                                        pop_missing[n][yi].append(pop_data_t[n])
                                    else:
                                        notmissing[n][yi].append(mean[b, t, n+1])
                                        exact_notmissing[n][yi].append(record_y[b][t_rec, n])
                                        weights_notmissing[n][yi].append(sample_weight[b])
                                        first_notmissing[n][yi].append(record_y[b][0, n])
                                        pop_notmissing[n][yi].append(pop_data_t[n])
                                        linear_notmissing[n][yi].append(linear[b, t, n+1])
                            break
                        t += 1

R2_missing = np.zeros((10, N))
R2_notmissing = np.zeros((10, N))
R2_first_notmissing = np.zeros((10, N))
R2_pop_missing = np.zeros((10, N))
R2_pop_notmissing = np.zeros((10, N))

RMSE_notmissing = np.zeros((10, N))
RMSE_linear_notmissing = np.zeros((10, N))
RMSE_pop_notmissing = np.zeros((10, N))
RMSE_first = np.zeros((10, N))
for yi in range(1,10):
    for n in range(N):
        
        # missing
        weights_missing[n][yi] = np.array(weights_missing[n][yi])
        exact_missing[n][yi] = np.array(exact_missing[n][yi])
        missing[n][yi] = np.array(missing[n][yi])
        
        # not missing
        weights_notmissing[n][yi] = np.array(weights_notmissing[n][yi])
        exact_notmissing[n][yi] = np.array(exact_notmissing[n][yi])
        notmissing[n][yi] = np.array(notmissing[n][yi])
        linear_notmissing[n][yi] = np.array(linear_notmissing[n][yi])
        
        # population and first 
        pop_notmissing[n][yi] = np.array(pop_notmissing[n][yi])
        pop_missing[n][yi] = np.array(pop_missing[n][yi])
        
        if yi > 0:
            first_notmissing[n][yi] = np.array(first_notmissing[n][yi])
        
        if len(weights_notmissing[n][yi]) > 25:
            RMSE_notmissing[yi][n] = np.sqrt((weights_notmissing[n][yi] * ((exact_notmissing[n][yi] - notmissing[n][yi]))**2).mean())
            RMSE_linear_notmissing[yi][n] = np.sqrt((weights_notmissing[n][yi] * ((exact_notmissing[n][yi] - linear_notmissing[n][yi]))**2).mean())
            RMSE_pop_notmissing[yi][n] = np.sqrt((weights_notmissing[n][yi] * ((exact_notmissing[n][yi] - pop_notmissing[n][yi]))**2).mean())
            RMSE_first[yi][n] = np.sqrt((weights_notmissing[n][yi] * ((exact_notmissing[n][yi] - first_notmissing[n][yi]))**2).mean())
        else:
            RMSE_notmissing[yi][n] = np.nan
            RMSE_linear_notmissing[yi][n] = np.nan
            RMSE_pop_notmissing[yi][n] = np.nan
            RMSE_first[yi][n] = np.nan


##### Predict longitudinal change average NOT MISSING
fig,ax = plt.subplots(figsize=(6.2,5))

RMSE = RMSE_notmissing/RMSE_pop_notmissing
RMSE_linear = RMSE_linear_notmissing/RMSE_pop_notmissing
RMSE_first = RMSE_first/RMSE_pop_notmissing


sort_RMSE = RMSE * np.ones((10,N))
sort_RMSE = np.nan_to_num(sort_RMSE, nan = 1000)


sort_RMSE_linear = RMSE_linear * np.ones((10,N))
sort_RMSE_linear = np.nan_to_num(sort_RMSE_linear, nan = 1000)


deficits_small = ['Gait', 'Grip str dom', 'Grip str ndom','ADL score', 'IADL score', 'Chair rise', 'Leg raise','Full tandem',
                      'SRH', 'Eyesight','Hearing', 'Function', 'Dias BP', 'Sys BP', 'Pulse', 'Trig', 'CRP','HDL','LDL',
                      'Gluc','IGF-1','HGB','Fib','Fer', 'Chol', 'WBC', 'MCV', 'hba1c', 'VIT-D']

non_nurse = [0, 3, 4, 8, 9, 10, 11]
nurse = [1, 2, 5, 6, 7, 12, 13, 14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]


posy = [np.nan, 1.05, 1.05, 1.45, 1.55, 1.05, 1.1]
for yi, years in enumerate([0, 2, 4, 6, 8, 10, 12, 14]): 

    
    if years > 0:
        
        parts_nn = ax.boxplot(x=[RMSE[yi][non_nurse][~np.isnan(RMSE[yi][non_nurse])]],positions=[years-0.2],showmeans=False,widths=0.27,showfliers=False,patch_artist=True,boxprops=dict(facecolor=cm2(2), alpha=0.6,color='k'), medianprops=dict(color=cm(0),zorder=100000,linewidth=2))
        parts_n = ax.boxplot(x=[RMSE[yi][nurse][~np.isnan(RMSE[yi][nurse])]],positions=[years+0.2],showmeans=False,widths=0.27,showfliers=False,patch_artist=True,boxprops=dict(facecolor=cm2(3), alpha=0.6,color='k'), medianprops=dict(color=cm(0),zorder=100000,linewidth=2))
        

        
        if yi > 0 and yi <= 7:
            mean_nn = np.median(RMSE[yi][non_nurse][~np.isnan(RMSE[yi][non_nurse])])
            mean_n = np.median(RMSE[yi][nurse][~np.isnan(RMSE[yi][nurse])])
            mean_next_nn = np.median(RMSE[yi+1][non_nurse][~np.isnan(RMSE[yi+1][non_nurse])])
            mean_next_n = np.median(RMSE[yi+1][nurse][~np.isnan(RMSE[yi+1][nurse])])
            mean_linear_nn = np.median(RMSE_linear[yi][non_nurse][~np.isnan(RMSE_linear[yi][non_nurse])])
            mean_linear_n = np.median(RMSE_linear[yi][nurse][~np.isnan(RMSE_linear[yi][nurse])])
            
            if yi == 1:
                

                ax.plot([years-0.2]*2, [mean_nn]*2, color = cm(0), marker='o', linestyle = '', label = '', zorder=100000)
                cnm = ax.plot([years+0.2]*2, [mean_n]*2, color = cm(0), marker='o', linestyle = '', label = 'Network model median', zorder=100000)
                
                ax.plot([years-0.2]*2, [mean_linear_nn]*2, color = cm(2), marker='s', linestyle = '', label = '', zorder=10)
                enm = ax.plot([years+0.2]*2, [mean_linear_n]*2, color = cm(2), marker='s', linestyle = '', label = 'Elastic net linear median', zorder=10)
            else:
                
                ax.plot([years-0.2]*2, [mean_linear_nn]*2, color = cm(2), marker='s', linestyle = '', zorder=10)
                ax.plot([years+0.2]*2, [mean_linear_n]*2, color = cm(2), marker='s', linestyle = '', zorder=10)
                
                ax.plot([years-0.2]*2, [mean_nn]*2, color = cm(0), marker='o', linestyle = '', label = '', zorder=100000)
                ax.plot([years+0.2]*2, [mean_n]*2, color = cm(0), marker='o', linestyle = '', label = '', zorder=100000)


ax.set_ylabel(r'Relative RMSE',fontsize = 14)
ax.set_xlabel(r'Years from baseline',fontsize = 14)
ax.set_ylim(0.6, 1.2)

ax.set_xticks([2,4,6,8,10,12, 14])
ax.set_xticklabels(['[1,2]','[3,4]','[5,6]','[7,8]','[9,10]','[11,12]', '[13,14]'])

pop_m = ax.plot([0,16], [1,1], linestyle = '--', color = 'k', label = 'Population')


ax.legend([parts_nn["boxes"][0], parts_n["boxes"][0], cnm[0], enm[0], pop_m[0]],
                         ['2 year Self report waves', '4 year Nurse evaluation waves', 'DJIN model median', 'Elastic-net linear median', 'Population'], loc='upper left')
ax.set_xlim(1, 15)

ax.yaxis.set_minor_locator(MultipleLocator(0.025))


ax.text(-0.05, 1.05, 'f', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 16, zorder=1000000,
        fontweight='bold')


plt.tight_layout()
plt.savefig('../Plots/Longitudinal_times_notmissing_RMSE_job_id%d_epoch%d.pdf'%(args.job_id,args.epoch))
