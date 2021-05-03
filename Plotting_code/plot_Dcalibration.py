import argparse
import torch
import numpy as np
from scipy.stats import sem, binned_statistic, chi2
from pandas import read_csv
from torch.utils import data

from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))  

from DataLoader.dataset import Dataset
from DataLoader.collate import custom_collate

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cm = plt.get_cmap('Set1')


parser = argparse.ArgumentParser('Dcalibration')
parser.add_argument('--job_id', type=int)
parser.add_argument('--epoch', type=int)
args = parser.parse_args()

torch.set_num_threads(4)

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

test_name = '../Data/test.csv'
test_set = Dataset(test_name, N, pop=False, min_count = 10)
num_test = test_set.__len__()
test_generator = data.DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))

with torch.no_grad():

    survival = np.load('../Analysis_Data/Survival_trajectories_job_id%d_epoch%d_DJIN.npy'%(args.job_id,args.epoch))
    linear=np.load('../Comparison_models/Results/Survival_trajectories_baseline_id1_rfmice_test.npy')

    #print(linear[0,:,0], survival[0,:,0])
    #print(linear.shape, survival.shape)
    #asdasdasd
    start = 0
    for data in test_generator:
        break

    death_ages = data['death age'].numpy()
    censored = data['censored'].numpy()
    times = data['times'].numpy()
    dead_mask = data['dead_mask'].numpy()
    ages = times[:,0]
    death_ages = np.array([death_ages[m] if death_ages[m] > 0 else times[m].max() for m in range(death_ages.size)])
    
    sample_weight = data['weights'].numpy()
    sex_index = data['env'][:,12].long().numpy()
    

dead_mask = np.concatenate((dead_mask, np.zeros((dead_mask.shape[0], dead_mask.shape[1]*3))), axis = 1)

uncensored_list = survival[censored<1,:,1][(dead_mask[censored<1]*survival[censored<1,:,1])>0].flatten()

bin_edges = np.linspace(0,1,11)
uncen_buckets = np.zeros(10)
cen_buckets = np.zeros(10)

# uncensored
uncen_buckets += np.histogram(uncensored_list, bins=bin_edges)[0]

for i in range(len(ages)):
    
    if censored[i] == 1:

        survival[i,:,1][survival[i,:,1] < 1e-10] = 1e-10
        survival[i,:,1][survival[i,:,1] > 1 ] = 1
        
        Sc = survival[i,:,1][(dead_mask[i]*survival[i,:,1]) > 0][0]
        
        bin = (np.digitize(Sc, bin_edges, right=True) - 1)
        
        cen_buckets[bin] += (1 - bin_edges[bin]/Sc)
        
        total = 0
        for j in range(bin-1, -1, -1):
            cen_buckets[j] += 0.1/Sc
            total += 0.1/Sc
        
        
            

buckets = cen_buckets + uncen_buckets

uncen_buckets /= buckets.sum()
cen_buckets /= buckets.sum()

error_buckets = np.sqrt(buckets)/buckets.sum()

statistic = 10./len(ages) * np.sum( (buckets - len(ages)/10.)**2 )

pval = 1 - chi2.cdf(statistic, 9)


####linear baseline
linear_uncensored_list = linear[censored<1,:,1][(dead_mask[censored<1]*linear[censored<1,:,1])>0].flatten()

bin_edges = np.linspace(0,1,11)
linear_uncen_buckets = np.zeros(10)
linear_cen_buckets = np.zeros(10)

# uncensored
linear_uncen_buckets += np.histogram(linear_uncensored_list, bins=bin_edges)[0]


for i in range(len(ages)):
    
    if censored[i] == 1:

        linear[i,:,1][linear[i,:,1] < 1e-10] = 1e-10
        linear[i,:,1][linear[i,:,1] > 1 ] = 1
        
        Sc = linear[i,:,1][(dead_mask[i]*linear[i,:,1]) > 0][0]
        
        bin = (np.digitize(Sc, bin_edges, right=True) - 1)
        
        linear_cen_buckets[bin] += 1 - bin_edges[bin]/Sc
        
        total = 0
        for j in range(bin-1, -1, -1):
            linear_cen_buckets[j] += 0.1/Sc
            total += 0.1/Sc

linear_buckets = linear_cen_buckets + linear_uncen_buckets

linear_uncen_buckets /= linear_buckets.sum()
linear_cen_buckets /= linear_buckets.sum()

linear_statistic = 10./len(ages) * np.sum( (linear_buckets - len(ages)/10.)**2 )

linear_pval = 1 - chi2.cdf(linear_statistic, 9)


print('Elastic net chi-sq %.2f, p-val %.5f'%(linear_statistic, linear_pval))
print('DJIN chi-sq %.2f, p-val %.5f'%(statistic, pval))





# DJIN model
fig, ax = plt.subplots(figsize=(4.5,4.5))


bin_labels = ['[0.,.1)', '[.1,.2)', '[.2,.3)', '[.3,.4)',
                  '[.4,.5)', '[.5,.6)', '[.6,.7)', '[.7,.8)', '[.8,.9)',
                  '[.9,1.]']
bin_index = np.arange(0,len(bin_labels))

plt.barh(bin_labels,  uncen_buckets + cen_buckets, height=0.98, color = cm(0), label = 'DJIN')

plt.errorbar(uncen_buckets + cen_buckets, bin_index, xerr=error_buckets, color = 'k', zorder=10000000, linestyle = '')

ax.text(.7, 0.77, r'DJIN model', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 12, zorder=1000000)
ax.text(.7, 0.71, r'$\chi^2 = {{%.1f}}$'%(statistic), horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 14, zorder=1000000)
ax.text(.7, 0.65, r'$p={{%.1f}}$'%(pval), horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 14, zorder=1000000)

ax.text(.71, 0.55, r'E-net Cox', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 12, zorder=1000000)
ax.text(.71, 0.49, r'$\chi^2 = {{%.1f}}$'%(linear_statistic), horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 14, zorder=1000000)
ax.text(.71, 0.43, r'$p={{%.1f}}$'%(linear_pval), horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 14, zorder=1000000)


ax.text(-0.05, 1.05, 'c', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 16, zorder=1000000,
        fontweight='bold')


plt.plot(2*[0.1], [-0.5,9.5], linestyle = '--', color = 'k', zorder=100, linewidth = 2, label='Uniform')

plt.legend(handlelength=0.75, handletextpad=0.6)


plt.xlim(0, 0.15)
plt.ylim(-0.5,9.5)
plt.ylabel('Survival probability', fontsize=14)
plt.xlabel('Fraction in bin', fontsize=14)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('../Plots/D-Calibration_job_id%d_epoch%d.pdf'%(args.job_id, args.epoch))




# Elastic net Cox
fig, ax = plt.subplots(figsize=(4.5,4.5))


bin_labels = ['[0.,.1)', '[.1,.2)', '[.2,.3)', '[.3,.4)',
                  '[.4,.5)', '[.5,.6)', '[.6,.7)', '[.7,.8)', '[.8,.9)',
                  '[.9,1.]']
bin_index = np.arange(0,len(bin_labels))

plt.barh(bin_labels,  linear_uncen_buckets + linear_cen_buckets, height=0.98, color = cm(2), label = 'Enet Cox')
plt.errorbar(linear_uncen_buckets + linear_cen_buckets, bin_index, xerr=error_buckets, color = 'k', zorder=10000000, linestyle = '')


ax.text(.7, 0.77, r'DJIN model', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 12, zorder=1000000)
ax.text(.7, 0.71, r'$\chi^2 = {{%.1f}}$'%(statistic), horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 14, zorder=1000000)
ax.text(.7, 0.65, r'$p={{%.1f}}$'%(pval), horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 14, zorder=1000000)

ax.text(.71, 0.55, r'E-net Cox', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 12, zorder=1000000)
ax.text(.71, 0.49, r'$\chi^2 = {{%.1f}}$'%(linear_statistic), horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 14, zorder=1000000)
ax.text(.71, 0.43, r'$p={{%.1f}}$'%(linear_pval), horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 14, zorder=1000000)


ax.text(-0.05, 1.05, 'b', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 16, zorder=1000000,
        fontweight='bold')

plt.plot(2*[0.1], [-0.5,9.5], linestyle = '--', color = 'k', zorder=100, linewidth = 2, label='Uniform')


plt.legend(handlelength=0.75, handletextpad=0.6)


plt.xlim(0, 0.175)
plt.ylim(-0.5,9.5)
plt.ylabel('Survival probability', fontsize=14)
plt.xlabel('Fraction in bin', fontsize=14)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('../Plots/D-Calibration_cox.pdf')
