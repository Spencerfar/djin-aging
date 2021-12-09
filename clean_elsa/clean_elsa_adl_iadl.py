import pyreadstat
import numpy as np
import pandas as pd

folder = "~/Downloads/UKDA-5050-spss/spss/spss24/"

adl = []
iadl = []
cond1 = []
cond2 = []
cond3 = []

#wave 1
adl.append(['heada0'+str(i) for i in range(1,10)] + ['heada10'] + ['heada11'])
iadl.append(['headb0'+str(i) for i in range(1,10)] + ['headb1'+str(i) for i in range(0,5)])
cond1.append(['hedib0%s'%i for i in range(1, 10)] + ['hedib10'])
cond2.append(['hedia0%s'%i for i in range(1, 10)] + ['hedia10'])
cond3.append(['heopt%s'%i for i in range(1,6)])

#wave 2
adl.append(['heada0'+str(i) for i in range(1,10)] + ['heada10'])
iadl.append(['headb0'+str(i) for i in range(1,10)] + ['headb1'+str(i) for i in range(0,4)])
cond1.append(['hedib0%s'%i for i in range(1, 10)] + ['hedib10'])
cond2.append(['hedia0%s'%i for i in range(1, 10)]) 
cond3.append(['heopt%s'%i for i in range(1,6)])

#wave3
adl.append(['hemob96','hemobwa','hemobsi','hemobch', 'hemobcs','hemobcl','hemobst','hemobre',
         'hemobpu','hemobli','hemobpi'])
iadl.append(['headl96','headldr','headlwa','headlba',
         'headlea','headlbe','headlwc','headlma','headlpr',
        'headlsh','headlph','headlme','headlho','headlmo'])

#wave 4
adl.append(['hemob96','hemobwa','hemobsi','hemobch', 'hemobcs','hemobcl','hemobst','hemobre',
         'hemobpu','hemobli','hemobpi'])
iadl.append(['headl96','headldr','headlwa','headlba',
         'headlea','headlbe','headlwc','headlma','headlpr',
        'headlsh','headlte','headlme','headlho','headlmo']) 

#wave 5
adl.append(['headl96','headldr','headlwa','headlba','headlea','headlbe',
            'headlwc','headlma','headlda','headlpr','headlsh','headlte',
           'headlco','headlme','headlho','headlmo'])
iadl.append([''])

#wave 6
adl.append(['hemob96','hemobwa','hemobsi','hemobch','hemobcs','hemobcl','hemobst',
           'hemobre','hemobpu','hemobli','hemobpi'])
iadl.append(['headl96','headldr','headlwa','headlba','headlea','headlbe',
            'headlwc','headlma','headlpr','headlsh',
            'headlph','headlme','headlho','headlmo'])

#wave 7
adl.append(['hemob96','hemobwa','hemobsi','hemobch','hemobcs','hemobcl','hemobst',
           'hemobre','hemobpu','hemobli','hemobpi'])
iadl.append(['headl96','headldr','headlwa','headlba','headlea','headlbe',
            'headlwc','headlma','headlpr','headlsh',
            'headlph','headlme','headlho','headlmo'])

#wave 8
adl.append(['hemob96','hemobwa','hemobsi','hemobch','hemobcs','hemobcl','hemobst',
           'hemobre','hemobpu','hemobli','hemobpi'])
iadl.append(['headl96','headldr','headlwa','headlba','headlea','headlbe',
            'headlwc','headlma','headlpr','headlsh',
            'headlph','headlme','headlho','headlmo']) 


waves = []

wave1, _ = pyreadstat.read_sav(folder+"wave_1_core_data_v3.sav",
                                        usecols = ['idauniq','dhager','mmwlka','mmwlkb',
                                                  'dhdobyr'] + adl[0] + iadl[0])
wave1['dhdobyr'].replace(-7.0,1912.0,inplace=True)
waves.append(wave1)


wave2, _ = pyreadstat.read_sav(folder+"wave_2_core_data_v4.sav",
               usecols = ['idauniq','MMWlkA','MMWlkB','dhdobyr',
                         ] + adl[1] + iadl[1])
wave2['dhdobyr'].replace(-7.0,1914.0,inplace=True)
waves.append(wave2)



wave3, _ = pyreadstat.read_sav(folder+"wave_3_elsa_data_v4.sav",
                                        usecols = ['idauniq','dhager','mmwlka','mmwlkb',
                                                  'dhdobyr'] + adl[2] + iadl[2])
waves.append(wave3)


wave4, _ = pyreadstat.read_sav(folder+"wave_4_elsa_data_v3.sav",
                                        usecols = ['idauniq','dhager','mmwlka','mmwlkb'] + adl[3] + iadl[3])
waves.append(wave4)

wave5, _ = pyreadstat.read_sav(folder+"wave_5_elsa_data_v4.sav",
                                        usecols = ['idauniq','indager','indobyr','mmwlka','mmwlkb'] + adl[4])
waves.append(wave5)

wave6, _ = pyreadstat.read_sav(folder+"wave_6_elsa_data_v2.sav",
                                        usecols = ['idauniq','indager','mmwlka','mmwlkb','Indobyr'] + adl[5] + iadl[5])
waves.append(wave6)

wave7, _ = pyreadstat.read_sav(folder+"wave_7_elsa_data.sav",
                                        usecols = ['idauniq','indager','MMWlkA','MMWlkB','Indobyr'] + adl[6] + iadl[6])
waves.append(wave7)

wave8, _ = pyreadstat.read_sav(folder+"wave_8_elsa_data_eul_v2.sav",
                                        usecols = ['idauniq','indager','mmwlka','mmwlkb','indobyr'] + adl[7] + iadl[7])
waves.append(wave8)


indexes = [[] for w in range(len(waves))]

#waves 1 and 2
for w in [0,1]:
    waves[w][adl[w]+iadl[w]] = waves[w][adl[w]+iadl[w]].replace([-8.0,-9.0],[np.nan]*2)
    indexes[w] = [x for x in waves[w].index.values]
    
#waves 3,4 and 6,7   
for w in [2,3,5,6]:
    waves[w][adl[w]+iadl[w]] = waves[w][adl[w]+iadl[w]].replace([-1.0,-2.0,-8.0,-9.0],[np.nan]*4)
    indexes[w] = [x for x in waves[w].index.values]

#wave 5
waves[4][adl[4]] = waves[4][adl[4]].replace([-1.0,-2.0,-8.0,-9.0],[np.nan]*4)
indexes[4] = [x for x in waves[4].index.values]

#waves 8   
waves[7][adl[7]+iadl[7]] = waves[7][adl[7]+iadl[7]].replace([-1.0,-2.0,-3.0,-4.0,-8.0,-9.0],[np.nan]*6)
indexes[7] = [x for x in waves[7].index.values]

N_adl = 10
N_iadl = 13

adl_deficits = {0:{}, 1:{}, 2: {}, 3: {}, 5: {}, 6: {}, 7: {}}
iadl_deficits = {0:{}, 1:{}, 2: {}, 3: {}, 5: {}, 6: {}, 7: {}}

#waves 1 and 2
for w in range(0,2):
    print(w)
    waves[w]['ADL'] = int(-1)
    waves[w]['IADL'] = int(-1)
    waves[w]['ADL count'] = 0
    waves[w]['IADL count'] = 0

    for index in np.unique(indexes[w]):
        
        if waves[w].loc[index,adl[w][0]] == 96.0:
            waves[w].loc[index,'ADL'] = 0
            waves[w].loc[index,'ADL count'] = N_adl
            adl_deficits[w][waves[w].loc[index,'idauniq']] = np.zeros(N_adl)
            
        elif np.all(waves[w].loc[index,adl[w][0]] == -1.0) or np.isnan(waves[w].loc[index,adl[w][0]]):
            waves[w].loc[index,'ADL'] = -1.0
            
        else:
            if w == 0:
                adl_deficits[w][waves[w].loc[index,'idauniq']] = \
                    waves[w].loc[index,adl[w]].ge(0).values.astype(int)[:-1]
            else:
                adl_deficits[w][waves[w].loc[index,'idauniq']] = \
                    waves[w].loc[index,adl[w]].ge(0).values.astype(int)
            
            waves[w].loc[index,'ADL'] = waves[w].loc[index,adl[w]].dropna().ge(0).sum()
            waves[w].loc[index,'ADL count'] = N_adl
            
            if np.any(np.isnan(waves[w].loc[index,adl[w]])):
                print(w, waves[w].loc[index,adl[w]])
        
        if waves[w].loc[index,iadl[w][0]] == 96.0:
            waves[w].loc[index,'IADL'] = 0
            waves[w].loc[index,'IADL count'] = N_iadl
            iadl_deficits[w][waves[w].loc[index,'idauniq']] = np.zeros(N_iadl)
            
        elif np.all(waves[w].loc[index,iadl[w][0]] == -1.0) or np.isnan(waves[w].loc[index,iadl[w][0]]):
            waves[w].loc[index,'IADL'] = -1.0
            
        else:
            if w == 0:
                iadl_deficits[w][waves[w].loc[index,'idauniq']] = \
                    waves[w].loc[index,iadl[w]].ge(0).values.astype(int)[:-1]
            else:
                iadl_deficits[w][waves[w].loc[index,'idauniq']] = \
                    waves[w].loc[index,iadl[w]].ge(0).values.astype(int)
            
            waves[w].loc[index,'IADL'] = waves[w].loc[index,iadl[w]].dropna().ge(0).sum()
            waves[w].loc[index,'IADL count'] = N_iadl

#waves 3,4,6,7,8
for w in [2,3,5,6,7]:
    print(w)
    waves[w]['ADL'] = int(-1)
    waves[w]['IADL'] = int(-1)
    waves[w]['ADL count'] = 0
    waves[w]['IADL count'] = 0

    for index in np.unique(indexes[w]):
        
        if waves[w].loc[index,adl[w][0]] == 1.0:
            waves[w].loc[index,'ADL'] = 0
            waves[w].loc[index,'ADL count'] = N_adl
            adl_deficits[w][waves[w].loc[index,'idauniq']] = np.zeros(N_adl)
            
        elif np.isnan(waves[w].loc[index,adl[w][0]]) or waves[w].loc[index,adl[w][0]] == -1.0:
            waves[w].loc[index,'ADL'] = -1
            
        else:
            if w == 0:
                adl_deficits[w][waves[w].loc[index,'idauniq']] = \
                    waves[w].loc[index,adl[w]].values.astype(int)[:-1]
            elif w == 1:
                adl_deficits[w][waves[w].loc[index,'idauniq']] = \
                    waves[w].loc[index,adl[w]].values.astype(int)
            else:
                adl_deficits[w][waves[w].loc[index,'idauniq']] = \
                    waves[w].loc[index,adl[w]].values.astype(int)[1:]
            
            waves[w].loc[index,'ADL'] = waves[w].loc[index,adl[w]].dropna().sum()
            waves[w].loc[index,'ADL count'] = N_adl
        
        
        if waves[w].loc[index,iadl[w][0]] == 1.0:
            waves[w].loc[index,'IADL'] = 0
            waves[w].loc[index,'IADL count'] = N_iadl
            iadl_deficits[w][waves[w].loc[index,'idauniq']] = np.zeros(N_iadl)
            
        elif np.isnan(waves[w].loc[index,iadl[w][0]]) or waves[w].loc[index,iadl[w][0]] == -1.0:
            waves[w].loc[index,'IADL'] = -1
        else:
            
            if w == 0:
                iadl_deficits[w][waves[w].loc[index,'idauniq']] = \
                    waves[w].loc[index,iadl[w]].values.astype(int)[:-1]
            elif w == 1:
                iadl_deficits[w][waves[w].loc[index,'idauniq']] = \
                    waves[w].loc[index,iadl[w]].values.astype(int)
            else:
                iadl_deficits[w][waves[w].loc[index,'idauniq']] = \
                    waves[w].loc[index,iadl[w]].values.astype(int)[1:]
            
            waves[w].loc[index,'IADL'] = waves[w].loc[index,iadl[w]].dropna().sum()
            waves[w].loc[index,'IADL count'] = N_iadl

def calc_full(row,w):
    if row['ADL'] >= 0 and row['IADL'] >= 0:
        return (row['ADL'] + row['IADL'])/(row['ADL count'] + row['IADL count'])
        
    elif row['ADL'] >= 0 and row['IADL'] < 0:
        return row['ADL']/row['ADL count']            
    elif row['IADL'] >= 0 and row['ADL'] < 0:
        return row['IADL']/row['IADL count']
    else:
        return -1.0
    
def calc_ADL(row,w):
    if row['ADL'] >= 0:
        return row['ADL']/row['ADL count']
    else:
        return -1.0
    
def calc_IADL(row,w):
    if row['IADL'] >= 0:
        return row['IADL']/row['IADL count']
    else:
        return -1.0

#waves 1 to 8, skipping 5
for w in [0,1,2,3,5,6,7]:
    print(w)
    waves[w]['FI ADL'] = waves[w].apply(lambda row: calc_ADL(row,w) ,axis=1)
    waves[w]['FI IADL'] = waves[w].apply(lambda row: calc_IADL(row,w) ,axis=1)

data = pd.concat(waves, keys=[1, 2, 3, 5, 6, 7, 8], sort=False)

for w in [1, 2, 3, 4, 6, 7, 8]:
    waves[w-1][['idauniq', 'ADL', 'IADL',  'FI ADL','FI IADL','ADL count', 'IADL count']].to_csv('../Data/ELSA_Frailty_cleaned_wave'+str(w)+'.csv',index=False)
