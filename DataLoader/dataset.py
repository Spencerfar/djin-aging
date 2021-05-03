import torch
from torch.utils import data
import numpy as np
import pandas as pd
import os


deficits = ['gait speed', 'grip dom', 'grip ndom', 'FI ADL', 'FI IADL', 'chair','leg raise', 'full tandem', 'srh', 'eye',
          'hear', 'func', 'dias', 'sys', 'pulse', 'trig',
         'crp','hdl','ldl','glucose','igf1','hgb','fib','fer', 'chol', 'wbc', 'mch', 'hba1c', 'vitd'] #, 'dheas'
        
medications = ['BP med', 'anticoagulent med', 'chol med', 'hip/knee treat', 'lung/asthma med']
        
background = ['longill', 'limitact', 'effort', 'smkevr', 'smknow', 'height', 'bmi', 'mobility', 'country',
              'alcohol', 'jointrep', 'fractures', 'sex', 'ethnicity']


def build_data_start(full_data, full_times, full_survival, full_env, full_med, M, N, t_length, start, min_count, prune = True):

    # health variables
    data = np.zeros((M, t_length, N))
    mask = np.zeros((M, t_length, N), int)

    # times
    times = np.zeros((M, t_length))

    # background
    env = np.zeros((M, full_env.shape[1] - 1)) #-1 for id
    env_mask = np.zeros((M, full_env.shape[1] - 3), int) #-1 for id, -2 for sex, ethnicity

    # med_time
    med = np.zeros((M, t_length, full_med.shape[1]-1))
    med_mask = np.zeros((M, t_length, full_med.shape[1]-1), int)
    
    # death age
    death_age = np.zeros(M)

    # times before last time known to be alive
    survival_mask = np.zeros((M, t_length), int)
    
    # last time known alive/dead
    dead_mask = np.zeros((M, t_length), int)
    after_dead_mask = np.zeros((M, t_length), int)
    
    # censored 1 if censored
    censored = np.zeros(M, int)

    # ids used
    ids = np.arange(0, M, 1, dtype=int)
    
    zero_mask = np.ones(M, int)
    
    for i,id in enumerate(range(0,M)):
        
        indiv = full_data.loc[full_data['id'] == id, deficits]
        indiv_times = full_times.loc[full_times['id'] == id, 'age']
        indiv_S = full_survival.loc[full_survival['id'] == id, 'death age']
        indiv_env = full_env.loc[full_env['id'] == id, background]
        indiv_med = full_med.loc[full_med['id'] == id, medications]
        
        if start+1 >= len(indiv_env):
            zero_mask[i] = 0
        else:

            env[i] = indiv_env.values[start+1]
            env_mask[i] = (indiv_env.values[start+1][:-2] > -100).astype(int)
            env[i, :-2] *= env_mask[i]
            
            time = indiv_times.values[0]
            i_record = 0
            skiped = 0
            new_start = None
            
            for t in range(t_length):

                
                
                if i_record < len(indiv_times) and np.abs(time - indiv_times.values[i_record]) < 0.1:

                    current_med = indiv_med.values[i_record]
                    current_med_mask = (indiv_med.values[i_record] > -100).astype(int)
                    
                    if i_record > start:
                        if skiped == 0:
                            new_start = t
                        data[i,t - new_start,:] = indiv.values[i_record,:]
                        times[i,t - new_start] = indiv_times.values[i_record]
                        mask[i, t - new_start, :] = (indiv.values[i_record, :] > -100).astype(int)
                        
                        
                        med[i, (t - new_start):,:] = current_med*current_med_mask + 0
                        med_mask[i, (t - new_start):,:] = current_med_mask
                        skiped = 1
                    i_record += 1
                
                if indiv_S.max() > 0:
                    
                    if skiped == 1 and np.abs(time - indiv_S.max()) < 0.1:
                        death_age[i] = indiv_S.max()
                        survival_mask[i, :t - new_start+1] = 1
                        dead_mask[i, t - new_start] = 1
                        after_dead_mask[i, t - new_start:t - new_start+5] = 1
                        censored[i] = 0

                else:
                    if skiped == 1 and np.abs(time -  indiv_times.values[-1]) < 0.1:
                        death_age[i] = -1
                        survival_mask[i, :t - new_start+1] = 1
                        dead_mask[i, t - new_start] = 1
                        censored[i] = 1
                
                if time >= max(indiv_times.values[-1], indiv_S.max()):

                    if prune:
                        if np.sum(dead_mask[i,:]) < 1 or np.sum(mask[i, 0, :]) < min_count:
                            zero_mask[i] = 0
                    else:
                        if np.sum(dead_mask[i,:]) < 1 or np.sum(mask[i, 0, :]) < 1:
                            zero_mask[i] = 0
                    break
                
                time += 1
    
    
    zero_mask = zero_mask.astype(bool)
    
    return data[zero_mask], times[zero_mask], death_age[zero_mask], mask[zero_mask], survival_mask[zero_mask], censored[zero_mask], np.concatenate((env[zero_mask], env_mask[zero_mask]),axis=1), np.concatenate((med[zero_mask], med_mask[zero_mask]),axis=2), dead_mask[zero_mask], after_dead_mask[zero_mask], ids[zero_mask]


class Dataset(data.Dataset):

    def __init__(self, name, N, pop = False, prune = True, min_count = 0):
        #print(name)
        #print(name.rstrip('.csv'))
        folder = name.rstrip('.csv') + '_files/'
        print(folder)
        
        if os.path.isdir(folder) and not pop:
            print('Files exist from ' + folder)
            
            self.longitudinal_data = torch.load(folder + 'longitudinal.pt')
            self.times_data = torch.load(folder + 'times.pt')
            self.death_age = torch.load(folder + 'death_age.pt')
            self.mask = torch.load(folder + 'mask.pt')
            self.survival_mask = torch.load(folder + 'survival_mask.pt')
            self.censored = torch.load(folder + 'censored.pt')
            self.env = torch.load(folder + 'env.pt')
            self.med_time = torch.load(folder + 'med_time.pt')
            self.dead_mask = torch.load(folder + 'dead_mask.pt')
            self.after_dead_mask = torch.load(folder + 'after_dead_mask.pt')
            self.weights = torch.load(folder + 'weights.pt')
            self.list_IDs = np.load(folder + 'IDs.npy')

            if name[:3] != '../':
                self.mean_T, self.std_T = np.load('Data/train_files/' + 'T_stats.npy')
                orig_data = pd.read_csv('Data/train.csv')[deficits]
            else:
                self.mean_T, self.std_T = np.load('../Data/train_files/' + 'T_stats.npy')
                orig_data = pd.read_csv('../Data/train.csv')[deficits]
            
            self.max_values = torch.Tensor(orig_data[orig_data > -1000].quantile(1.0).values).float()
            self.min_values = torch.Tensor(orig_data[orig_data > -1000].quantile(0.0).values).float()
            
        elif os.path.isdir(folder[:-1] + '_pop/') and pop:
            print('Pop files exist')
            
            folder = folder[:-1] + '_pop/'
            
            self.longitudinal_data = torch.load(folder + 'longitudinal.pt')
            self.times_data = torch.load(folder + 'times.pt')
            self.death_age = torch.load(folder + 'death_age.pt')
            self.mask = torch.load(folder + 'mask.pt')
            self.survival_mask = torch.load(folder + 'survival_mask.pt')
            self.censored = torch.load(folder + 'censored.pt')
            self.env = torch.load(folder + 'env.pt')
            self.med_time = torch.load(folder + 'med_time.pt')
            self.dead_mask = torch.load(folder + 'dead_mask.pt')
            self.after_dead_mask = torch.load(folder + 'after_dead_mask.pt')
            self.weights = torch.load(folder + 'weights.pt')
            self.list_IDs = np.load(folder + 'IDs.npy')

            self.id_names = np.load(folder + 'id_names.npy')

            if name[:3] != '../':
                self.mean_T, self.std_T = np.load('Data/train_files/' + 'T_stats.npy')
                orig_data = pd.read_csv('Data/train.csv')[deficits]
            else:
                self.mean_T, self.std_T = np.load('../Data/train_files/' + 'T_stats.npy')
                orig_data = pd.read_csv('../Data/train.csv')[deficits]
            
            self.max_values = torch.Tensor(orig_data[orig_data > -1000].quantile(1.0).values).float()
            self.min_values = torch.Tensor(orig_data[orig_data > -1000].quantile(0.0).values).float()
            
        else:
            print('Files dont exist')
            if not pop:
                os.mkdir(folder)

            if pop:
                folder = folder[:-1] + '_pop/'
                #folder += 'pop'
                os.mkdir(folder)
            
            orig_data = pd.read_csv(name)
            
            num_indiv = len(orig_data['id'].unique())
            
            t_length = 25
            
            # select correct columns for health variables
            index = [0] + list(np.arange(3, 3+N,dtype=int))
            
            X = orig_data[['id'] + deficits] # health variables
            T = orig_data[['id', 'age']] # times 
            A = orig_data[['id', 'death age']] # death ages
            E = orig_data[['id'] + background] # background variables
            Med = orig_data[['id'] + medications] # medications
            
            self.mean_T = np.mean(T.values[:,1])
            self.std_T = np.std(T.values[:,1])
            
            self.max_values = torch.Tensor(orig_data[deficits][orig_data[deficits] > -1000].quantile(1.0).values).float()
            self.min_values = torch.Tensor(orig_data[deficits][orig_data[deficits] > -1000].quantile(0.0).values).float()

            np.save(folder + 'T_stats.npy', [self.mean_T, self.std_T])
            
            data = []
            times = []
            death_age = []
            mask = []
            survival_mask = []
            censored = []
            env = []
            med_time = []
            dead_mask = []
            after_dead_mask = []
            ids = []
            # create different starting points for training
            for i in range(-1, 10):
                data_, times_, death_age_, mask_, survival_mask_, censored_, env_, med_time_, dead_mask_, after_dead_mask_, ids_ = build_data_start(X, T, A, E, Med, num_indiv, N, t_length, i, min_count, prune)
                
                data.append(data_)
                times.append(times_)
                death_age.append(death_age_)
                mask.append(mask_)
                survival_mask.append(survival_mask_)
                censored.append(censored_)
                env.append(env_)
                med_time.append(med_time_)
                dead_mask.append(dead_mask_)
                after_dead_mask.append(after_dead_mask_)
                ids.append(ids_)
            
            
            # concatenate starting points
            num_indiv = sum([data[i].shape[0] for i in range(len(data))])
            data = np.concatenate(data, axis = 0)
            times = np.concatenate(times, axis = 0)
            death_age = np.concatenate(death_age, axis = 0)
            mask = np.concatenate(mask, axis = 0)
            survival_mask = np.concatenate(survival_mask, axis = 0)
            censored = np.concatenate(censored, axis = 0)
            env = np.concatenate(env, axis = 0)
            #med = np.concatenate(med, axis = 0)
            med_time = np.concatenate(med_time, axis = 0)
            dead_mask = np.concatenate(dead_mask, axis = 0)
            after_dead_mask = np.concatenate(after_dead_mask, axis = 0)
            ids = np.concatenate(ids, axis = 0)
        
            weights = np.ones(ids.shape)
            for i, id in enumerate(ids):
                count = (ids == id).sum()
                weights[i] = 1/count
        
            # turn into tensor
            self.longitudinal_data = torch.from_numpy(data).float()
            self.times_data = torch.from_numpy(times).float()
            self.death_age = torch.from_numpy(death_age).float()
            self.mask = torch.from_numpy(mask).float()
            self.survival_mask = torch.from_numpy(survival_mask).float()
            self.censored = torch.from_numpy(censored).float()
            self.env = torch.from_numpy(env).float()
            self.med_time = torch.from_numpy(med_time).float()
            self.dead_mask = torch.from_numpy(dead_mask).float()
            self.after_dead_mask = torch.from_numpy(after_dead_mask).float()
            self.weights = torch.from_numpy(weights).float()
        
            self.list_IDs = np.arange(num_indiv, dtype = int)
                
            torch.save(self.longitudinal_data, folder + 'longitudinal.pt')
            torch.save(self.times_data, folder + 'times.pt')
            torch.save(self.death_age, folder + 'death_age.pt')
            torch.save(self.mask, folder + 'mask.pt')
            torch.save(self.survival_mask, folder + 'survival_mask.pt')
            torch.save(self.censored, folder + 'censored.pt')
            torch.save(self.env, folder + 'env.pt')
            torch.save(self.med_time, folder + 'med_time.pt')
            torch.save(self.dead_mask, folder + 'dead_mask.pt')
            torch.save(self.after_dead_mask, folder + 'after_dead_mask.pt')
            torch.save(self.weights, folder + 'weights.pt')
            np.save(folder + 'IDs.npy', self.list_IDs)
            np.save(folder + 'id_names.npy', ids)
        
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        selected = self.list_IDs[index]
        
        return self.longitudinal_data[selected], self.times_data[selected], self.mask[selected], self.survival_mask[selected], self.dead_mask[selected], self.after_dead_mask[selected], self.censored[selected], self.death_age[selected], self.env[selected], self.med_time[selected], self.weights[selected]
