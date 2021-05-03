import numpy as np
import itertools

def cindex_td(death_ages, survival_funcs, survival_ages, observed, weights = []):
    
    num = len(death_ages)
    pairs = itertools.permutations(range(0,num),2)
    if len(weights) == 0:
        weights = np.ones(num)

    
    N = 0.0
    C = 0.0
    for (i,j) in pairs:

        if death_ages[i] < death_ages[j] and observed[i] == 1:

            N += 1.0*weights[i]*weights[j]
            
            index_i = np.searchsorted(survival_ages[i], death_ages[i])
            index_j = np.searchsorted(survival_ages[j], death_ages[i])
            if index_i == len(survival_ages[i]):
                index_i -= 1
            if index_j == len(survival_ages[j]):
                index_j -= 1
            
            S_i = survival_funcs[i].flatten()[index_i]
            S_j = survival_funcs[j].flatten()[index_j]
            
            if S_i < S_j and death_ages[i] < death_ages[j]:
                C += 1.0*weights[i]*weights[j]
                
            elif S_i == S_j and death_ages[i] < death_ages[j]:
                C += 0.5*weights[i]*weights[j]
            
    if N > 0:
        return C/N
    else:
        return np.nan
