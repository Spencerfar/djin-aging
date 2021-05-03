import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cm = plt.get_cmap('Set1')

deficits_nice = ['Gait speed', 'Dom grip str', 'Ndom grip str', 'ADL score','IADL score', '5 Chair rises','Leg raise','Full tandem', 'Self-rated health', 'Eyesight', 'Hearing', 'Walking ability', 'Dias BP', 'Sys BP', 'Pulse', 'Trig','C-RP','HDL','LDL','Gluc','IGF-1','Hgb','Fibr','Ferr', 'Total chol', r'WBC', 'MCH', 'HgbA1c', 'Vit-D']
deficits = ['gait speed', 'grip dom', 'grip ndom', 'FI ADL', 'FI IADL', 'chair','leg raise', 'full tandem', 'srh', 'eye',
                    'hear', 'func',
                    'dias', 'sys', 'pulse', 'trig','crp','hdl','ldl','glucose','igf1','hgb','fib','fer', 'chol', 'wbc', 'mch', 'hba1c', 'vitd']


medications = ['BP med', 'anticoagulent med', 'chol med', 'hip/knee treat', 'lung/asthma med']

background = ['longill', 'limitact', 'effort', 'smkevr', 'smknow','height', 'bmi', 'mobility', 'country',
                  'alcohol', 'jointrep', 'fractures', 'sex', 'ethnicity'] + medications

medications = []


background_nice = ['Longterm ill', 'Illness limits', 'Everythings effort', 'Ever smoke', 'Smoke now', 'Height', 'BMI','Mobility', 'Country', 'Alcohol', 'Joint replace', 'Fractures', 'Sex', 'Ethnicity', 'Blood pres med', 'Blood thin med', 'Chol med', 'Hipknee med', 'Lung med']


data = pd.read_csv('../Data/ELSA_cleaned.csv')


for n in range(len(deficits)):
    if n in [15, 16, 23, 25, 26, 28]:
        data = data.rename(columns={deficits[n]:deficits_nice[n]})
        deficits[n] = deficits_nice[n]
    else:
        data = data.rename(columns={deficits[n]:deficits_nice[n]})
        deficits[n] = deficits_nice[n]

deficits_nice = deficits

for n in range(len(background)):
    data = data.rename(columns = {background[n] : background_nice[n]})
background = background_nice

count = 0
total = 0
for label, group in data.groupby('id'):
    total += 1
    if group['death age'].unique()[0] > 0:
        updated = group['death age'].values
        updated[:-1] = -1
        count += 1
        data.loc[data['id']==label, 'death age'] = updated

times = np.arange(-0.5, 20.5, 1)

data['delta_t'] = (data['age'] - data.groupby('id')['age'].transform('first')).astype(int)
data['delta_death_age'] = (data['death age'] - data.groupby('id')['age'].transform('first')).astype(int)


data[data < -100] = np.nan
data.loc[data['delta_death_age'] < 0, 'delta_death_age'] = np.nan

missing = data.drop('delta_t', 1)[deficits].notna().groupby(data.delta_t, sort=False).sum().reset_index()

missing_background = data.drop('delta_t', 1)[background + medications].notna().groupby(data.delta_t, sort=False).sum().reset_index()

deaths = data.drop('delta_t', 1)['delta_death_age'].notna().groupby(data.delta_t, sort=False).sum().reset_index()

non_nurse = [0, 3, 4, 8, 9, 10, 11]
nurse = [1, 2, 5, 6, 7, 12, 13, 14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
(1 - data[np.array(deficits)].notna().mean()).to_csv('../Analysis_Data/ELSA_missing_percent.csv')

missing = missing.sort_values('delta_t')
missing[deficits] = (missing[deficits]).astype(int)
missing.set_index('delta_t', inplace=True)

total_count = pd.DataFrame()
total_count['delta_t'] = missing_background['delta_t']
total_count['Total individuals'] = missing_background['Sex']
total_count.sort_values('delta_t', inplace=True)
total_count = total_count.astype(int)
total_count.set_index('delta_t', inplace=True)

missing_background = missing_background.sort_values('delta_t')
missing_background[background + medications] = (missing_background[background + medications]).astype(int)
missing_background.set_index('delta_t', inplace=True)


death_count = pd.DataFrame()
death_count['delta_t'] = deaths['delta_t']
death_count['Deaths'] = deaths['delta_death_age']
death_count.sort_values('delta_t', inplace=True)
death_count = death_count.astype(int)
death_count.set_index('delta_t', inplace=True)


fig,ax = plt.subplots(4,1, figsize=(15,10),gridspec_kw={'height_ratios': [1, 0.73076923076, 1.0/26, 1.0/26]})

norm = mpl.colors.LogNorm(1, 25290)

import seaborn as sns
sns.set_style("white")

sns.heatmap(missing.transpose(), annot=True, fmt="d",cbar=False,ax=ax[0],cmap="Purples", yticklabels=deficits_nice,xticklabels=23*[''],vmin=1)

sns.heatmap(missing_background.transpose(), annot=True, fmt="d",cbar=False,ax=ax[1],cmap="Greens",yticklabels=background_nice,vmin=1)

g = sns.heatmap(death_count.transpose(), annot=True, fmt="d",cbar=False,ax=ax[2],cmap="Oranges",vmin=1)
g.set_yticklabels(g.get_yticklabels(), rotation=0)

g = sns.heatmap(total_count.transpose(), annot=True, fmt="d",cbar=False,ax=ax[3],cmap="Oranges",vmin=1)
g.set_yticklabels(g.get_yticklabels(), rotation=0)

ax[0].set_xlabel('')
ax[1].set_xlabel('')
ax[2].set_xlabel('')
ax[3].set_xlabel('Followup from baseline (years)', fontsize = 15)
ax[0].set_title('Number of individuals with measurements', fontsize = 15)

ax[0].set_xticks([],[])
ax[1].set_xticks([],[])
ax[2].set_xticks([],[])


for _, spine in ax[0].spines.items():
    if _ != 'bottom':
        spine.set_visible(True)

for _, spine in ax[1].spines.items():
    if _ != 'top' and _ != 'bottom':
        spine.set_visible(True)

for _, spine in ax[2].spines.items():
    if _ != 'top' and _ != 'bottom':
        spine.set_visible(True)

for _, spine in ax[3].spines.items():
    if _ != 'top':
        spine.set_visible(True)


ax[0].text(-0.13, 0.5, 'Health variables', horizontalalignment='center', verticalalignment='center',transform=ax[0].transAxes,fontsize = 20, zorder=1000000, rotation = 90, color = "#88419d", weight='bold')

ax[1].text(-0.13, 0.5, 'Background', horizontalalignment='center', verticalalignment='center',transform=ax[1].transAxes,fontsize = 20, zorder=1000000, rotation = 90, color="#238b45", weight='bold')


plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.subplots_adjust(left=0.15)
plt.savefig('../Plots/missing_values_ELSA.pdf')
