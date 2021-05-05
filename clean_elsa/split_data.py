import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import resample

"""
Creates train, validation, and test sets
"""

def create_cv():
    
    def create_data(index, full_data, deficits_val, medications_val, background_val, train = False, mean_deficits = 0, std_deficits = 0):
    
        chosen = X[index, 0]
        random_ids = dict(zip(chosen, np.random.rand(len(chosen))))

        data = full_data.loc[full_data['id'].isin(chosen),['id', 'wave', 'age'] + deficits_val +
                                 medications_val + background_val + ['death age']]
        
        data['new ID'] = data.apply(lambda row: random_ids[int(row['id'])], axis = 1)
        data = data.sort_values(by = ['new ID','wave'])

        # scaling
        if train:
            mean_deficits = pd.Series(index = deficits_val+['height', 'bmi', 'alcohol'])
            std_deficits = pd.Series(index = deficits_val+['height', 'bmi', 'alcohol'])
            for d in deficits_val+['height', 'bmi', 'alcohol']:
                mean_deficits[d] = data.loc[data[d]>-100,d].mean()
                std_deficits[d] = data.loc[data[d]>-100,d].std()
        else:
            for d in deficits_val+['height', 'bmi', 'alcohol']:
                data[d] = data[d].apply(lambda x: (x - mean_deficits[d])/std_deficits[d] if x > -100 else x)
    
        data = data[['id', 'wave', 'age'] + deficits_val + medications_val + background_val + ['death age'] ]#.values
        
        indexes = []
        index_count = -1
        previous_index = -100000
        
        for i in range(len(data)):
            index = data.iloc[i,0]
            if(index != previous_index):
                index_count += 1
            data.iloc[i,0] = index_count
            previous_index = index

        if train:
            return data, mean_deficits, std_deficits
        else:
            return data
    
    ran_state = RandomState(2)
    
    deficits = ['gait speed', 'grip dom', 'grip ndom', 'FI ADL', 'FI IADL', 'chair','leg raise', 'full tandem', 'srh', 'eye',
                    'hear', 'func',
                    'dias', 'sys', 'pulse', 'trig','crp','hdl','ldl','glucose','igf1','hgb','fib','fer', 'chol', 'wbc', 'mch', 'hba1c', 'vitd']

    medications = ['BP med', 'anticoagulent med', 'chol med', 'hip/knee treat', 'lung/asthma med']

    background = ['longill', 'limitact', 'effort', 'smkevr', 'smknow','height', 'bmi', 'mobility', 'country',
                  'alcohol', 'jointrep', 'fractures', 'sex', 'ethnicity']
    
    data = pd.read_csv('../Data/ELSA_cleaned.csv')
    
    unique_indexes = data['id'].unique()
    data['censored'] = data['death age'].apply(lambda x: 0 if x > 0 else 1)
    
    censored = []
    for id in unique_indexes:
        censored.append(data.loc[data['id'] == id, 'censored'].unique()[0])
    
    X = np.array([unique_indexes, censored], int).T
    from sklearn.model_selection import StratifiedKFold
    

    skf_outer = StratifiedKFold(n_splits=5, shuffle = True, random_state = 2)
    skf_inner = StratifiedKFold(n_splits=5, shuffle = True, random_state = 3)
    
    for i, (full_train_index, test_index) in enumerate(skf_outer.split(X[:,0], X[:,1])):
        
        _, mean_deficits, std_deficits = create_data(np.arange(0,len(X[full_train_index,0]),dtype=int), data, deficits, medications, background, train=True)
        
        mean_deficits.to_csv('../Data/mean_deficits.txt')
        std_deficits.to_csv('../Data/std_deficits.txt')
        
        for j, (train_index, valid_index) in enumerate(skf_inner.split(X[full_train_index,0], X[full_train_index,1])):
            
            data_train = create_data(np.random.permutation(train_index), data, deficits, medications, background, mean_deficits = mean_deficits, std_deficits = std_deficits)
            
            #data_train.to_csv('Data/train_outer%d_inner%d.csv'%(i,j), index=False)
            data_train.to_csv('../Data/train.csv', index=False)
            
            data_valid = create_data(np.random.permutation(valid_index), data, deficits, medications, background, mean_deficits = mean_deficits, std_deficits = std_deficits)
            
            #data_valid.to_csv('Data/valid_outer%d_inner%d.csv'%(i,j), index=False)
            data_valid.to_csv('../Data/valid.csv', index=False) 
            break # dont do full cv
        
        data_test = create_data(np.random.permutation(test_index), data, deficits, medications, background, mean_deficits = mean_deficits, std_deficits = std_deficits)
    
        #np.savetxt('Data/test_outer%d.txt'%i,data_test,fmt=s)
        data_test.to_csv('../Data/test_outer%d.csv',index=False)
        break # dont do full cv

if __name__ =="__main__":
    create_cv()
