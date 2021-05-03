import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from clean_data import clean_test_data

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from sklearn.linear_model import ElasticNet


parser = argparse.ArgumentParser('Predict change')
parser.add_argument('--param_id', type=int)
parser.add_argument('--alpha', type=float, default = 1e-4)
parser.add_argument('--l1_ratio', type=float, default = 0.5)
parser.add_argument('--max_depth', type=int, default = 10)
parser.add_argument('--dataset', type=str, default = 'test')
args = parser.parse_args()

deficits = ['gait speed', 'grip dom', 'grip ndom', 'FI ADL', 'FI IADL', 'chair','leg raise', 'full tandem', 'srh', 'eye',
          'hear', 'func', 'dias', 'sys', 'pulse', 'trig',
         'crp','hdl','ldl','glucose','igf1','hgb','fib','fer', 'chol', 'wbc', 'mch', 'hba1c', 'vitd']

        
medications = ['BP med', 'anticoagulent med', 'chol med', 'hip/knee treat', 'lung/asthma med']
        
background = ['longill', 'limitact', 'effort', 'smkevr', 'smknow', 'height', 'bmi', 'mobility', 'country',
              'alcohol', 'jointrep', 'fractures', 'sex', 'ethnicity']
    
N = 29
    
train_data = pd.read_csv('../Data/train.csv')
train_data['weight'] = 1

# train imputation
total_X_train = train_data[['age'] + deficits + background + medications] #

mask_train = (total_X_train.values > -100).astype(int)
mask_sum = mask_train[:,1:30].sum(-1)
total_X_train = total_X_train[mask_sum > 5]


min_values = total_X_train[total_X_train > -100].min().values
max_values = total_X_train.max().values

imp = IterativeImputer(estimator = RandomForestRegressor(n_estimators=40, max_depth = args.max_depth, n_jobs=40), random_state=0, missing_values = -1000, max_iter=100, verbose=2)
imp.fit(total_X_train)

print('imputation done')

X_test, y = clean_test_data(longitudinal=True, data=args.dataset)
predictions = np.zeros((X_test.shape[0], 29, N + 1))

print('test data ready')

for i in range(N-1):
    
    initial_index = []

    X = []
    U = []
    dY = []
    T0 = []
    dt = []

    count = 0
    for label, group in train_data.groupby('id'):
    
        # get baseline index
        initial_index.append(group.index.values[0])
    
        if len(group) >= 2:
            
            selected = group.loc[group[deficits[i]] >-100, deficits[i]]
            
            for pair in itertools.combinations(np.arange(len(selected), dtype=int), 2):
                
                delta = group.loc[group[deficits[i]] >-100, 'age'].values[pair[1]] - \
                  group.loc[group[deficits[i]] >-100, 'age'].values[pair[0]]
                
                dy = selected.values[pair[1]] - selected.values[pair[0]]
                
                dt.append( delta )
                T0.append( group.loc[group[deficits[i]] >-100, 'age'].values[pair[0]] )
                X.append( group.loc[group[deficits[i]] >-100, deficits].values[pair[0]] )
                U.append( group.loc[group[deficits[i]] >-100, background + medications].values[pair[0]] )
                dY.append(dy)
            count += 1
            

    dt = np.array(dt)
    T0 = np.array(T0)
    X = np.array(X)
    U = np.array(U)
    dY = np.array(dY)
    
    X_ = np.concatenate((T0[:,np.newaxis], X), axis=1)
    
    
    X_train = np.concatenate( (X_, U), axis=1)
    y_train = dY
    
    X_train_imputed = imp.transform(X_train)
    X_train_imputed = (X_train_imputed.T*dt).T
    
    model = ElasticNet(fit_intercept = False, alpha=args.alpha, l1_ratio=args.l1_ratio)
    model.fit(X_train_imputed, y_train)
    
    print(i, model.score(X_train_imputed, y_train))
    
    X_test_i = imp.transform(X_test)
    
    for t, delta_t in enumerate(np.arange(0, 29, dtype=int)):
        X_test_i_t = X_test_i * delta_t
        
        predictions[:,t, i + 1] = model.predict(X_test_i_t) + X_test_i[:, i + 1]
        predictions[:,t, 0] = X_test_i[:, 0] + delta_t

np.save('Predictions/Longitudinal_predictions_baseline_id%d_rfmice.npy'%(args.param_id), predictions)
