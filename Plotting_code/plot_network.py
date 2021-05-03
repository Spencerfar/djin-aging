import torch
import numpy as np
import argparse
from scipy.stats import laplace

from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))

from Model.model import Model

from scipy.cluster.hierarchy import dendrogram as plot_dendrogam

import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cm = plt.get_cmap('Set1')
cm2 = plt.get_cmap('Set2')

parser = argparse.ArgumentParser('Plot network')
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

deficits = np.array(['Gait speed', 'Dom Grip strength', 'Non-dom grip str', 'ADL score','IADL score', 'Chair rises','Leg raise','Full tandem stance', 'Self-rated health', 'Eyesight','Hearing', 'Walking ability', 'Diastolic blood pressure', 'Systolic blood pressure', 'Pulse', 'Triglycerides','C-reactive protein','HDL cholesterol','LDL cholesterol','Glucose','IGF-1','Hemoglobin','Fibrinogen','Ferritin', 'Total cholesterol', r'White blood cell count', 'MCH', 'Glycated hemoglobin', 'Vitamin-D'])

N = 29

model = Model('cpu', N, args.gamma_size, args.z_size, args.decoder_size, args.Nflows, args.flow_hidden, args.f_nn_size, 0, 0, 0.5)
model.load_state_dict(torch.load('../Parameters/train%d_Model_DJIN_epoch%d.params'%(args.job_id, args.epoch),map_location='cpu'))

mean = model.mean.detach().numpy()*(np.ones((N,N)) - np.eye(N))
scale = model.logscale.exp().detach().numpy()*(np.ones((N,N)) - np.eye(N))


mean_list = mean[~np.eye(mean.shape[0],dtype=bool)]
scale_list = scale[~np.eye(scale.shape[0],dtype=bool)]
robust_network = np.ones(mean.shape)
network = np.ones(mean.shape)*mean
for i in range(N):
    for j in range(N):

        if i!=j:
            posterior = laplace(mean[i,j], scale[i,j])
            interval = posterior.interval(0.99)
            if (interval[0] < 0 and interval[1] > 0):
                robust_network[i,j] = 0

robust_list = robust_network[~np.eye(robust_network.shape[0],dtype=bool)]

fig,ax = plt.subplots(figsize=(6,4))


pd = np.zeros(mean_list.shape)
for i, (m, s) in enumerate(zip(mean_list, scale_list)):
    if m > 0:
        dist = posterior = laplace(m, s)
        pd[i] = posterior.sf(0)
    else:
        dist = posterior = laplace(m, s)
        pd[i] = posterior.cdf(0)

size = 10 + 10*robust_list
color = ['grey' if r==0 else 'black' for r in robust_list]

cax = ax.scatter(pd, mean_list,
                     c = color, s=size, edgecolors='white', linewidths=0.05)
plt.plot([0.995,0.995], [np.min(mean_list), np.max(mean_list)],color='r', linestyle = '--')

ax.set_ylabel(r'Posterior mean weight', fontsize = 14)
ax.set_xlabel(r'Proportion of posterior in direction of mean', fontsize = 14)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('../Plots/Posterior_network_uncertainty_job_id%d_epoch%d.pdf'%(args.job_id, args.epoch))






fig, ax = plt.subplots(figsize=(7,7))

import seaborn as sns
sns.set(style="white")

cmap = sns.color_palette("RdBu_r", 100)

network = np.ones(mean.shape)*mean
network_scale = np.ones(mean.shape)*scale

order = np.array([28, 8, 9, 10, 11, 3, 4, 7, 6, 5, 0, 2, 1, 16, 27, 19, 21, 26, 23, 24, 18, 17, 15, 13, 12, 14, 22, 20, 25])

for i in range(N):
    for j in range(N):

        if i!=j:
            posterior = laplace(mean[i,j], scale[i,j])
            interval = posterior.interval(0.99)
            if (interval[0] < 0 and interval[1] > 0) or np.abs(network[i,j]) < 0.0001:
                network[i,j] = np.nan
                network_scale[i,j] = np.nan

np.save('../Analysis_Data/network_weights_job_id%d_epoch%d.npy'%(args.job_id, args.epoch), network)
                
network = network[order][:,order]        

cbar_ax = fig.add_axes([0.31, 0.09, 0.59, 0.02])

sns.heatmap(network, ax=ax, xticklabels=deficits[order], yticklabels=deficits[order],
                      square=True, mask = np.eye(N) > 0, cmap=cmap, vmax=np.nanmax(np.abs(network)),vmin=-1*np.nanmax(np.abs(network)), cbar_kws={'label': r'Mean connection weight', 'orientation':'horizontal'}, cbar_ax = cbar_ax)

cbar_ax.yaxis.label.set_size(9)
ax.tick_params(labelsize=9)

#ax.text(-0.05, 1.05, 'b', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 16, zorder=1000000,fontweight='bold')

fig.tight_layout()
plt.subplots_adjust(bottom=0.35)
fig.savefig('../Plots/Posterior_network_job_id%d_epoch%d.pdf'%(args.job_id, args.epoch))
