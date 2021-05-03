import argparse
import numpy as np
import pandas as pd
import pickle

from clean_data import clean_test_data

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from lifelines import CoxPHFitter


parser = argparse.ArgumentParser('Predict change')
parser.add_argument('--param_id', type=int)
parser.add_argument('--alpha', type=float, default = 0.0001)
parser.add_argument('--l1_ratio', type=float, default = 0.0)
parser.add_argument('--max_depth', type=int, default = 10)
parser.add_argument('--dataset', type=str, default = 'test')
args = parser.parse_args()

deficits = ['gait speed', 'grip dom', 'grip ndom', 'FI ADL', 'FI IADL', 'chair','leg raise', 'full tandem', 'srh', 'eye',
          'hear', 'func', 'dias', 'sys', 'pulse', 'trig',
         'crp','hdl','ldl','glucose','igf1','hgb','fib','fer', 'chol', 'wbc', 'mch', 'hba1c', 'vitd']

        
medications = ['BP med', 'anticoagulent med', 'chol med', 'hip/knee treat', 'lung/asthma med']

background = ['longill', 'limitact', 'effort', 'smkevr', 'smknow', 'height', 'bmi', 'mobility', 'country',
              'alcohol', 'jointrep', 'fractures', 'sex', 'ethnicity']
    

def clean_data(data):

    data['status'] = 0
    data['weight'] = 1
    
    
    initial_index = []
    status = []
    for label, group in data.groupby('id'):
        
        # get baseline index
        initial_index.append(group.index.values[0])
        
        # fix death ages of censored
        if group['death age'].values[0] < 0:
            data.loc[data['id'] == label, 'death age'] = group['age'].max()
            data.loc[data['id'] == label, 'status'] = 0
        else:
            data.loc[data['id'] == label, 'status'] = 1

        data.loc[data['id'] == label, 'weight'] = 1./len(group)
    
    X = data[['age'] + deficits + medications + background]
    y = data[['weight','status', 'death age']].values

    return X, y, initial_index
    
train_data = pd.read_csv('../Data/train.csv')
X_train, y_train, initial_index = clean_data(train_data)

min_values = X_train[X_train > -100].min().values
max_values = X_train.max().values
mask_train = (X_train.values > -100).astype(int)
mask_sum = mask_train[:,1:30].sum(-1)
X_train = X_train[mask_sum > 5]
y_train = y_train[mask_sum > 5]


# MICE imputation
imp = IterativeImputer(estimator = RandomForestRegressor(n_estimators=40, max_depth = args.max_depth, n_jobs=40), random_state=0, missing_values = -1000, max_iter=100, verbose=2)

print('starting imputation')

imp.fit(X_train)

print('imputation done')

X_train_imputed = imp.transform(X_train)
X_train_imputed = np.concatenate((X_train_imputed, y_train), axis=1)
df_train = pd.DataFrame(X_train_imputed, columns = ['age'] + deficits + medications + background + ['weight','status', 'death age'])
df_train['age'] = df_train['age']/100.0

print('imputation transformation done')

cph = CoxPHFitter(penalizer = args.alpha, l1_ratio = args.l1_ratio, baseline_estimation_method='breslow')
cph.fit(df_train, duration_col = 'death age', event_col = 'status', weights_col = 'weight')

print('cph trained')

#####test
X_test = clean_test_data(data=args.dataset)
X_test_imputed = imp.transform(X_test)

print('test data ready')

df_test = pd.DataFrame(X_test_imputed, columns = ['age'] + deficits + medications + background)
ages = df_test['age'].values*1.0
df_test['age'] = df_test['age']/100.0

results = np.zeros((df_test.shape[0], 100, 2)) * np.nan
for i in range(df_test.shape[0]):
    
    unconditioned_sf = cph.predict_survival_function(df_test.iloc[i][['age'] + deficits + medications + background], np.arange(ages[i],121,1)).values[:,0]
    predicted = unconditioned_sf/unconditioned_sf[0]
    
    results[i,:len(np.arange(ages[i],121,1)),0] = np.arange(ages[i],121,1)
    results[i,:len(np.arange(ages[i],121,1)),1] = predicted

    
np.save('Predictions/Survival_trajectories_baseline_id%d_rfmice.npy'%(args.param_id), results)
