import argparse
import os
import numpy as np
import itertools
import torch

from torch.utils import data
from scipy.stats import pearsonr

from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))  

from DataLoader.dataset import Dataset

device = 'CPU'

N = 29

train_name = '../Data/train.csv'
training_set = Dataset(train_name, N, pop=False, min_count = 6)

data = training_set.longitudinal_data
data = data.reshape(data.shape[0]*data.shape[1], N).numpy()
mask = training_set.mask
mask = mask.reshape(mask.shape[0]*mask.shape[1], N).numpy()

corr = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        if i !=j:
            selected = (mask[:,i] > 0) * (mask[:,j] > 0)
            if np.sum(selected) > 10:
                
                x = data[selected, i]
                y = data[selected, j]
                r, p = pearsonr(x, y)
               
                if p <= 0.01:
                    corr[i,j] = r
                else:
                    corr[i,j] = np.nan
            else:
                corr[i,j] = np.nan 
        else:
            corr[i,j] = np.nan    

deficits = np.array(['Gait speed', 'Dom Grip strength', 'Non-dom grip str', 'ADL score','IADL score', 'Chair rises','Leg raise','Full tandem stance', 'Self-rated health', 'Eyesight','Hearing', 'Walking ability', 'Diastolic blood pressure', 'Systolic blood pressure', 'Pulse', 'Triglycerides','C-reactive protein','HDL cholesterol','LDL cholesterol','Glucose','IGF-1','Hemoglobin','Fibrinogen','Ferritin', 'Total cholesterol', r'White blood cell count', 'MCH', 'Glycated hemoglobin', 'Vitamin-D'])
                
import matplotlib.pyplot as plt

order = np.array([28, 8, 9, 10, 11, 3, 4, 7, 6, 5, 0, 2, 1, 16, 27, 19, 21, 26, 23, 24, 18, 17, 15, 13, 12, 14, 22, 20, 25])

corr = corr[order][:,order]  

fig, ax = plt.subplots(figsize=(7,7))

import seaborn as sns
sns.set(style="white")
cmap = sns.color_palette("RdBu_r", 100)

cbar_ax = sns.heatmap(corr,ax=ax, xticklabels=deficits[order],yticklabels=deficits[order],
                square=True, mask = np.eye(N) > 0, cmap=cmap, vmax=1, vmin=-1, cbar_kws={'label': r'Pearson correlation',"shrink": 0.75})

cbar_ax.figure.axes[-1].yaxis.label.set_size(10)
cbar_ax.figure.axes[-1].yaxis.label.set_rotation(90)

ax.tick_params(labelsize=8)
ax.text(-0.05, 1.05, 'a', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 16, zorder=1000000,
        fontweight='bold')

plt.tight_layout()

plt.savefig('../Plots/pearson_correlations.pdf')
