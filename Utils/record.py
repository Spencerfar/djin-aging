import numpy as np
import torch

def record(t, X, S, record_times, dt):

    sims = X.shape[0]
    M = X.shape[1]
    num_t = X.shape[2]
    N = X.shape[3]
    
    # recorded data containers
    X_record = []
    S_record = []
        
    # find recorded values
    for m in range(M):
        
        X_temp = torch.zeros(sims, len(record_times[m]), N)
        S_temp = torch.zeros(sims, len(record_times[m]))
        
        X_temp[:,0,:] = X[:,m, 0]
        S_temp[:,0] = S[:,m, 0]
        
        i_record = 1
        for i in range(num_t):
            
            if i_record < len(record_times[m]) and \
                (t[m, i] >= record_times[m][i_record]) and \
                (record_times[m][i_record] > t[m, i] - dt):
                    
                X_temp[:, i_record, :] = X[:, m, i, :]
                S_temp[:, i_record] = S[:, m, i]
                    
                i_record = i_record + 1
        X_record.append(X_temp)
        S_record.append(S_temp)

    # these are lists of recorded individuals.   
    return X_record, S_record
