import torch
import torch.nn as nn
import numpy as np

class CustomBatch:
    """
    Custom collate function for returning formatted batches
    """
    def __init__(self, data, pop_avg, pop_avg_env, pop_std, p_corruption):
        transposed_data = list(zip(*data))
        
        self.X = torch.stack(transposed_data[0], 0)
        self.times = torch.stack(transposed_data[1], 0)
        self.mask = torch.stack(transposed_data[2], 0)
        self.mask_S = torch.stack(transposed_data[3], 0)
        self.mask_D = torch.stack(transposed_data[4], 0)
        self.mask_AD = torch.stack(transposed_data[5], 0)
        self.censored = torch.stack(transposed_data[6], 0)
        self.death_age = torch.stack(transposed_data[7], 0)
        self.env = torch.stack(transposed_data[8], 0)
        self.med_time = torch.stack(transposed_data[9], 0)
        self.weights = torch.stack(transposed_data[10], 0)

        batch_size = self.X.shape[0]
        N = self.X.shape[-1]
        
        corrupt = torch.ones(batch_size, N)*(torch.rand(batch_size, N) > (1-p_corruption))
        self.mask0 = corrupt * self.mask[:,0,:]
        
        t0 = self.times[:, 0]
        
        pop_avg_bins = np.arange(40, 105, 3)[:-2]

        t0_index = np.digitize(t0.numpy(), pop_avg_bins, right=True) - 1
        t0_index[t0_index < 0] = 0
        sex_index = self.env[:,12].long() 
        
        self.predict_missing = pop_avg[sex_index, t0_index]
        self.pop_std = pop_std[sex_index, t0_index]
        
        
        predict_missing_env = pop_avg_env[sex_index, t0_index]
        
        self.env[:,[5,6]] = self.env[:,[5,6]]*self.env[:,[5+14,6+14]] + (1 - self.env[:,[5+14,6+14]])*predict_missing_env
        
    def __getbatch__(self):
        return {'Y': self.X, 'times': self.times,
                'mask': self.mask, 'mask0': self.mask0, 'survival_mask': self.mask_S, 'dead_mask': self.mask_D,
                'after_dead_mask': self.mask_AD, 'censored': self.censored, 'env': self.env,
                'med': self.med_time, 'weights':self.weights, 'missing':self.predict_missing,
                'pop std': self.pop_std, 'death age': self.death_age}
    
    def pin_memory(self):
        self.X = self.X.pin_memory()
        self.times = self.times.pin_memory()
        self.mask = self.mask.pin_memory()
        self.mask0 = self.mask0.pin_memory()
        self.mask_S = self.mask_S.pin_memory()
        self.mask_D = self.mask_D.pin_memory()
        self.mask_AD = self.mask_AD.pin_memory()
        self.censored = self.censored.pin_memory()
        self.death_age = self.death_age.pin_memory()
        self.env = self.env.pin_memory()
        self.med_time = self.med_time.pin_memory()
        self.weights = self.weights.pin_memory()
        self.predict_missing = self.predict_missing.pin_memory()
        self.pop_std = self.pop_std.pin_memory()
    
def custom_collate(batch, pop_avg, pop_avg_env, pop_std, corruption):
    return CustomBatch(batch, pop_avg, pop_avg_env, pop_std, corruption).__getbatch__()
