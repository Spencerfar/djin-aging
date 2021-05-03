import argparse
import torch
import numpy as np
from scipy.stats import sem, binned_statistic
from pandas import read_csv

from torch.utils import data

from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))  

from DataLoader.dataset import Dataset
from DataLoader.collate import custom_collate

from Utils.transformation import Transformation

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cm = plt.get_cmap('Set1')

parser = argparse.ArgumentParser('Plot population')
parser.add_argument('--job_id', type=int)
parser.add_argument('--epoch', type=int)
args = parser.parse_args()

device = 'cpu'

N = 29
sims = 250
dt = 0.5
length = 50


pop_avg = np.load('../Data/Population_averages.npy')
pop_avg_env = np.load('../Data/Population_averages_env.npy')
pop_std = np.load('../Data/Population_std.npy')
pop_avg_ = torch.from_numpy(pop_avg[...,1:]).float()
pop_avg_env = torch.from_numpy(pop_avg_env).float()


test_name = '../Data/test.csv'
test_set = Dataset(test_name, N, pop=True, min_count=1)
num_test = test_set.__len__()
test_generator = data.DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))

model_bins = np.arange(0, 120, 1)
bin_centers = model_bins[1:] - np.diff(model_bins)/2

pop_avg = np.load('../Data/Population_averages_test.npy')
pop_std = np.load('../Data/Population_std_test.npy')
pop_avg_bins = np.arange(40, 105, 5)

with torch.no_grad():

    mean = np.load('../Analysis_Data/Mean_pop_job_id%d_epoch%d_DJIN.npy'%(args.job_id,args.epoch))
    std = np.load('../Analysis_Data/Std_pop_job_id%d_epoch%d_DJIN.npy'%(args.job_id,args.epoch))
    
    start = 0
    for data in test_generator:
        break

    y = data['Y'].numpy()
    times = data['times'].numpy()
    mask = data['mask'].numpy()
    sample_weight = data['weights'].numpy()
    sex_index = data['env'][:,12].long().numpy()
    death_ages = data['death age'].numpy()
    censored = data['censored'].numpy()
    ages = times[:,0]
    death_ages = np.array([death_ages[m] if death_ages[m] > 0 else times[m].max() for m in range(death_ages.size)])


#####
fig,ax = plt.subplots(8, 4, figsize = (9, 12))
ax = ax.flatten()


deficits_small = ['Gait', 'Grip str dom', 'Grip str ndom','ADL score', 'IADL score', 'Chair rise', 'Leg raise','Full tandem',
                      'SRH', 'Eyesight','Hearing', 'Walking ability', 'Dias BP', 'Sys BP', 'Pulse', 'Trig', 'CRP','HDL','LDL',
                      'Gluc','IGF-1','HGB','Fib','Fer', 'Chol', 'WBC', 'MCH', 'hba1c', 'VIT-D']

# transform
mean_deficits = read_csv('../Data/mean_deficits.txt', index_col=0,sep=',',header=None, names = ['variable']).values[1:].flatten()
std_deficits = read_csv('../Data/std_deficits.txt', index_col=0,sep=',',header=None, names = ['variable']).values[1:].flatten()


deficits_units = np.array(['Gait speed (m/s)', 'Grip strength (kg)', 'Ndom grip str (kg)', 'ADL score','IADL score', '5 Chair rises (s)','Leg raise (s)','Full tandem (s)', 'SRH', 'Eyesight','Hearing', 'Walking ability score', 'Diastolic BP (mmHg)', 'Systolic BP (mmHg)', 'Pulse (bpm)', 'Trig (mmol/L)','C-RP (mg/L)','HDL (mmol/L)','LDL cholesterol (mmol/L)','Gluc (mmol/L)','IGF-1 (nmol/L)','Hgb (g/dL)','Fibrinogen (g/L)','Ferr (ng/mL)', 'Total cholesterol (mmol/L)', r'WBC ($10^{9}$ cells/L)', 'MCH (pg)', 'HgbA1c (%)', 'Vit-D (ng/mL)'])

for n in range(N):

    pop_age = pop_avg[:,0]
    age = mean[:,0]
    
    pop_trans = pop_avg[:,n+1]*std_deficits[n] + mean_deficits[n]
    pop_std_trans = pop_std[:,n+1]*std_deficits[n]
    
    mean_trans = mean[:,n+1]*std_deficits[n] + mean_deficits[n]
    std_trans = std[:,n+1]*std_deficits[n]
    
    if n in [15, 16, 23, 25, 26, 28, 29]:
        
        mean_trans = np.exp(mean_trans)
        std_trans = mean_trans*std_trans
        
        pop_trans = np.exp(pop_trans)
        pop_std_trans = pop_trans*pop_std_trans

    mean_trans = mean_trans[(age>= 65) & (age<=90)]
    std_trans = std_trans[(age>= 65) & (age<=90)]
    age = age[(age>= 65) & (age<=90)]
        
    ax[n].plot(age, mean_trans, color = cm(0), label = 'Synthetic population', linewidth=3, zorder=10000)
    ax[n].fill_between(age, mean_trans-std_trans, mean_trans+std_trans, color = cm(0), alpha = 0.5, zorder=1000)
    ax[n].plot(pop_age, pop_trans, color = cm(1), label = 'Observed population', linewidth=3)
    ax[n].fill_between(pop_age, pop_trans-pop_std_trans, pop_trans+pop_std_trans, color = cm(1), alpha = 0.5)
    ax[n].set_xlim(65, 90)
    ax[n].set_ylabel(deficits_units[n])
    ax[n].set_xlabel('Age (years)')

ax[-3].set_xlim(65, 90)
ax[-3].set_ylim(65, 90)
ax[-3].plot([0,0],[0,0], color = cm(0), label = 'Synthetic population', linewidth = 3)
ax[-3].plot([0,0],[0,0], color = cm(1), label = 'Observed population', linewidth = 3)
    
ax[-3].legend(loc='center', handlelength=0.5, fontsize=12)

for i in [-1,-2,-3]:
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['left'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['bottom'].set_visible(False)
    ax[i].tick_params(left = False, top=False, right=False, bottom=False)
    ax[i].set_xticklabels([])
    ax[i].set_yticklabels([])

plt.legend()

plt.tight_layout()
plt.subplots_adjust(hspace=0.48)
plt.savefig('../Plots/Population_trajectories_mean_job_id%d_epoch_%d.pdf'%(args.job_id, args.epoch))



fig,ax = plt.subplots(figsize=(6.2,5))

for t in range(0,20,2):
    
    accuracy = np.load('../Analysis_Data/Classifier_accuracy_time%d_job_id%d_epoch%d_acc.npy'%(t,args.job_id,args.epoch))
    
    parts = ax.boxplot(x=[accuracy], positions=[t],showmeans=False,widths=1,showfliers=False,patch_artist=True,boxprops=dict(facecolor=cm(0), alpha=0.6,color='k'), medianprops=dict(color=cm(0),zorder=100000,linewidth=2), capprops={'linewidth' : 2})

ax.set_ylabel(r'Observed/synthetic classifier accuracy',fontsize = 14)
ax.set_xlabel(r'Years from baseline',fontsize = 14)

ax.plot([0, 20], [0.5,0.5], color = 'k', linestyle = '--')

ax.set_xlim(-0.2, 19.2)


plt.tight_layout()
plt.savefig('../Plots/Population_classification_times_job_id%d_epoch%d.pdf'%(args.job_id, args.epoch))



from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()

#load population survival
survival = np.load('../Analysis_Data/S_pop_job_id%d_epoch%d_DJIN.npy'%(args.job_id, args.epoch))
survival_std = np.load('../Analysis_Data/S_pop_std_job_id%d_epoch%d_DJIN.npy'%(args.job_id, args.epoch))
death_ages_pop = np.load('../Analysis_Data/Death_ages_pop_job_id%d_epoch%d_DJIN.npy'%(args.job_id, args.epoch))

fig,ax = plt.subplots(figsize=(3.4, 2.35))

kmf.fit(death_ages[(ages >= 65) & (ages <=100)], event_observed = 1 - censored[(ages >= 65) & (ages <=100)])
plt.plot(kmf.survival_function_.index.values,kmf.survival_function_.values.flatten(),label='Observed population',color=cm(1), linestyle = '-',linewidth=2)

ax.fill_between(kmf.confidence_interval_.index.values, kmf.confidence_interval_.values[:,0],kmf.confidence_interval_.values[:,1],alpha=0.5,color=cm(1))

# censoring distribution
kmf_G = KaplanMeierFitter()
kmf_G.fit(death_ages[(ages >= 65) & (ages <=90)], event_observed = censored[(ages >= 65) & (ages <=90)], timeline = np.arange(0, 200, 1))
G = kmf_G.survival_function_.values.flatten()

death_ages = death_ages[(ages >= 65) & (ages <= 100)]
death_ages_pop = death_ages_pop[:,(ages >= 65) & (ages <= 100)]
ages = ages[(ages >= 65) & (ages <= 100)]


survival = np.zeros((death_ages_pop.shape[0], 120))
for s in range(death_ages_pop.shape[0]):
    #apply G by sampling censoring binary value at given age (with age index)
    
    censoring_ages = []
    for i in range(death_ages.shape[0]):
        if len(death_ages[death_ages >ages[i]]) > 0:
            censoring_ages.append(np.random.choice(death_ages[death_ages > ages[i]], size=1)[0])
        else:
            print(death_ages[i], ages[i])
    censoring_ages = np.array(censoring_ages)
    
    generated_censoring = (death_ages_pop[s] < censoring_ages).astype(int)#np.random.choice(censoring_ages, replace=True, size = death_ages_pop.shape[1])
    death_ages_pop_censored = np.minimum(death_ages_pop[s], censoring_ages)
    
    kmf = KaplanMeierFitter()
    kmf.fit(death_ages_pop_censored, event_observed = generated_censoring, timeline=np.arange(0,120,1))#, entry = ages
    
    survival[s] = kmf.survival_function_.values.flatten()

avg_survival = survival.mean(0)
lower_survival = np.percentile(survival, q=2.5, axis=0)
upper_survival = np.percentile(survival, q=97.5, axis=0)
plt.plot(np.arange(0,120,1), avg_survival, linewidth=2, label = 'Synthetic population', color=cm(0), zorder=1000000)
plt.fill_between(np.arange(0,120,1), lower_survival, upper_survival, linewidth=2, color=cm(0), alpha = 0.3)

plt.xlim(65,100)

plt.legend(handlelength=0.5)
plt.xlabel('Age (years)')
plt.ylabel('Survival probability')
plt.tight_layout()
plt.savefig('../Plots/Population_survival_job_id%d_epoch%d.pdf'%(args.job_id, args.epoch))





deficits_units = np.array(['Gait speed (m/s)', 'Grip strength (kg)', 'Ndom grip str (kg)', 'ADL score','IADL score', '5 Chair rises (s)','Leg raise (s)','Full tandem (s)', 'SRH', 'Eyesight','Hearing', 'Walking ability score', 'Diastolic BP (mmHg)', 'Systolic BP (mmHg)', 'Pulse (bpm)', 'Trig (mmol/L)','C-RP (mg/L)','HDL (mmol/L)','LDL cholesterol (mmol/L)','Gluc (mmol/L)','IGF-1 (nmol/L)','Hgb (g/dL)','Fibrinogen (g/L)','Ferr (ng/mL)', 'Total cholesterol (mmol/L)', r'WBC ($10^{9}$ cells/L)', 'MCH (pg)', 'HgbA1c (%)', 'Vit-D (ng/mL)'])

# plot baseline 
fig,ax = plt.subplots(8, 4, figsize = (9, 12))
ax = ax.flatten()


mean_deficits = torch.Tensor(read_csv('../Data/mean_deficits.txt', index_col=0,sep=',',header=None).values[1:-3].flatten())
std_deficits = torch.Tensor(read_csv('../Data/std_deficits.txt', index_col=0,sep=',',header=None, names = ['variable']).values[1:-3].flatten())
psi = Transformation(mean_deficits, std_deficits, [15, 16, 23, 25, 26, 28])


X = psi.untransform(np.load('../Analysis_Data/generated_baseline_pop_job_id%d_epoch%d.npy'%(args.job_id, args.epoch))).numpy()
Y = psi.untransform(y).numpy()[:,0]
mask = mask[:,0]
print(X.shape, Y.shape, mask.shape)
for i in range(N):
    ax[i].set_xlabel(deficits_units[i], fontsize=12)
    print(deficits_units[i], np.mean(Y[:,i][mask[:,i] > 0]), sem(Y[:,i][mask[:,i] > 0]),  np.mean(X[...,i].flatten()), sem(X[...,i].flatten()))
    
    if i in [3,4,8,9,10,11]:
        hist, bin_edges = np.histogram(Y[:,i][mask[:,i] > 0], density = True, bins = len(np.unique(Y[:,i][mask[:,i] > 0])))
        ax[i].bar(bin_edges[:-1], hist, alpha = 0.5, label = 'Observed population', width = bin_edges[1] - bin_edges[0], color = cm(1))
        hist, bin_edges = np.histogram(X[:,mask[:,i] > 0,i].flatten(), density = True, bins = bin_edges)
        ax[i].bar(bin_edges[:-1], hist, alpha = 0.5, label = 'Synthetic population', width = bin_edges[1] - bin_edges[0], color = cm(0))
            
    elif i in [15, 16, 23, 25, 26, 28]:
            
        hist_obs, bin_edges = np.histogram(np.log(Y[:,i][mask[:,i] > 0]), density = True, bins = 30)
        bin_edges_pred = bin_edges * np.ones(bin_edges.shape)
        bin_edges_pred[0] = -np.inf
        bin_edges_pred[-1] = np.inf
        hist, _ = np.histogram(np.log(X[:,mask[:,i] > 0,i].flatten()), density = True, bins = bin_edges_pred)
        ax[i].bar(np.exp(bin_edges[:-1]), hist_obs, alpha = 0.5, label = 'Observed population', width = np.exp(bin_edges[1:]) - np.exp(bin_edges[:-1]), color = cm(1))
        ax[i].bar(np.exp(bin_edges[:-1]), hist, alpha = 0.5, label = 'Synthetic population', width = np.exp(bin_edges[1:]) - np.exp(bin_edges[:-1]), color = cm(0))
        ax[i].set_xlabel(deficits_units[i] + '*', fontsize=12)
        ax[i].set_xscale('log')

        if i == 19 + 4:
            ax[i].set_xticks([10, 100, 1000])
        
    else:
        hist_obs, bin_edges = np.histogram(Y[:,i][mask[:,i] > 0], density = True, bins = 30)
        hist, bin_edges = np.histogram(X[:,mask[:,i] > 0,i].flatten(), density = True, bins = bin_edges)
        bin_edges_pred = bin_edges * np.ones(bin_edges.shape)
        bin_edges_pred[0] = -np.inf
        bin_edges_pred[-1] = np.inf
        
        ax[i].bar(bin_edges[:-1], hist, alpha = 0.5, label = 'Synthetic population', width = bin_edges[1] - bin_edges[0], color = cm(0),zorder=10000)
        ax[i].bar(bin_edges[:-1], hist_obs, alpha = 0.5, label = 'Observed population', width = bin_edges[1] - bin_edges[0], color = cm(1), zorder=1)
    
ax[-3].set_xlim(65, 90)
ax[-3].set_ylim(65, 90)
ax[-3].plot([0,0],[0,0], color = cm(0), label = 'Synthetic population', linewidth = 3)
ax[-3].plot([0,0],[0,0], color = cm(1), label = 'Observed population', linewidth = 3)
ax[-3].legend(loc='center', handlelength=0.5, fontsize=12)
    
for i in [-1,-2,-3]:
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['left'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['bottom'].set_visible(False)
    ax[i].tick_params(left = False, top=False, right=False, bottom=False)
    ax[i].set_xticklabels([])
    ax[i].set_yticklabels([])

plt.tight_layout()
plt.subplots_adjust(hspace=0.48)
plt.savefig('../Plots/generated_baseline_pop_job_id%d_epoch%d.pdf'%(args.job_id, args.epoch))
