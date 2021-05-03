import argparse
import os
import numpy as np
from pandas import read_csv
import itertools
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import data

from Alternate_models.model_nornn import Model
from Model.loss import loss, sde_KL_loss

from DataLoader.dataset import Dataset
from DataLoader.collate import custom_collate
from Utils.schedules import LinearScheduler, ZeroLinearScheduler

parser = argparse.ArgumentParser('Train')
parser.add_argument('--job_id', type=int)
parser.add_argument('--batch_size', type=int, default = 1000)
parser.add_argument('--niters', type=int, default = 2000)
parser.add_argument('--learning_rate', type=float, default = 1e-2)
parser.add_argument('--corruption', type=float, default = 0.9)
parser.add_argument('--gamma_size', type=int, default = 25)
parser.add_argument('--z_size', type=int, default = 20)
parser.add_argument('--decoder_size', type=int, default = 65)
parser.add_argument('--Nflows', type=int, default = 3)
parser.add_argument('--flow_hidden', type=int, default = 24)
parser.add_argument('--f_nn_size', type=int, default = 12)
parser.add_argument('--W_prior_scale', type=float, default = 0.05)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    num_workers = 8
    torch.set_num_threads(12)
    test_after = 10
    test_average = 5
else:
    num_workers = 0
    torch.set_num_threads(4)
    test_after = 10
    test_average = 3

# folders for output
params_folder = 'Parameters/'
output_folder = 'Output/'

# setting up file for loss outputs
loss_file = '%svalidation%d_nornn.loss'%(output_folder, args.job_id)
open(loss_file, 'w')

# output hyperparameters
hyperparameters_file = '%strain%d_nornn.hyperparams'%(output_folder, args.job_id)
with open(hyperparameters_file, 'w') as hf:
    hf.writelines('batch_size, %d\n'%args.batch_size)
    hf.writelines('niters, %d\n'%args.niters)
    hf.writelines('learning_rate, %.3e\n'%args.learning_rate)
    hf.writelines('corruption, %.3f\n'%args.corruption)
    hf.writelines('gamma_size, %d\n'%args.gamma_size)
    hf.writelines('z_size, %d\n'%args.z_size)
    hf.writelines('decoder_size, %d\n'%args.decoder_size)
    hf.writelines('Nflows, %d\n'%args.Nflows)
    hf.writelines('flow_hidden, %d\n'%args.flow_hidden)
    hf.writelines('f_nn_size, %d\n'%args.f_nn_size)
    hf.writelines('W_prior_scale, %.4f\n'%args.W_prior_scale)
    

N = 29
batch_size = args.batch_size
dt = 0.5

pop_avg = np.load('Data/Population_averages.npy')
pop_avg_env = np.load('Data/Population_averages_env.npy')
pop_std = np.load('Data/Population_std.npy')
pop_avg = torch.from_numpy(pop_avg[...,1:]).float()
pop_avg_env = torch.from_numpy(pop_avg_env).float()
pop_std = torch.from_numpy(pop_std[...,1:]).float()


train_name = 'Data/train.csv'
training_set = Dataset(train_name, N, pop=False, min_count = 6)
training_generator = data.DataLoader(training_set,
                                     batch_size = batch_size,
                                     shuffle = True, drop_last = True, num_workers = num_workers, pin_memory=True,
                                     collate_fn = lambda x: custom_collate(x, pop_avg, pop_avg_env, pop_std, args.corruption))

valid_name = 'Data/valid.csv'
validation_set = Dataset(valid_name, N, pop=False, min_count = 6)
validation_generator = data.DataLoader(validation_set,
                                       batch_size = 1000,
                                       shuffle = False, drop_last = False,pin_memory=True,
                                       collate_fn = lambda x: custom_collate(x, pop_avg, pop_avg_env, pop_std, 1.0))

print('Data loaded: %d training examples and %d validation examples'%(training_set.__len__(), validation_set.__len__()))


mean_T = training_set.mean_T
std_T = training_set.std_T

# initialize model
model = Model(device, N, args.gamma_size, args.z_size, args.decoder_size, args.Nflows, args.flow_hidden, args.f_nn_size, mean_T, std_T, dt).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.5, threshold = 1000, threshold_mode ='abs', patience = 4, min_lr = 1e-5, verbose=True)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Model has %d parameters'%params)

matrix_mask = (1 - torch.eye(N).to(device))


kl_scheduler_dynamics = LinearScheduler(300)
kl_scheduler_vae = LinearScheduler(500)
kl_scheduler_network = ZeroLinearScheduler(300, 500)

sigma_prior = torch.distributions.gamma.Gamma(torch.tensor(1.0).to(device), torch.tensor(25000.0).to(device))
W_prior = torch.distributions.laplace.Laplace(torch.tensor(0.0).to(device), torch.tensor(args.W_prior_scale).to(device))
vae_prior = torch.distributions.normal.Normal(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))


niters = args.niters
for epoch in range(niters):

    beta_dynamics = kl_scheduler_dynamics()
    beta_network = kl_scheduler_network()
    beta_vae = kl_scheduler_vae()
    
    for data in training_generator:
        
        optimizer.zero_grad()
        
        W_posterior = torch.distributions.laplace.Laplace(model.mean, model.logscale.exp())
        sigma_posterior = torch.distributions.gamma.Gamma(model.logalpha.exp(), model.logbeta.exp())
        
        
        W = W_posterior.rsample((data['Y'].shape[0],))
        
        sigma_y = sigma_posterior.rsample((data['Y'].shape[0],data['Y'].shape[1])) + 1e-6
        
        pred_X, t, pred_S, pred_logGamma, pred_sigma_X, context, y, times, mask, survival_mask, dead_mask, after_dead_mask, censored, sample_weights, med, env, z_sample, prior_entropy, log_det, recon_mean_x0, drifts, mask0, W_mean = model(data, sigma_y)
        summed_weights = torch.sum(sample_weights)
        
        kl_term = beta_network*torch.sum(matrix_mask*(torch.sum(sample_weights*(W_posterior.log_prob(W).permute(1,2,0)),dim=-1) - torch.sum(sample_weights*(W_prior.log_prob(W).permute(1,2,0)),dim=-1))) + torch.sum(torch.sum(sample_weights*((mask*sigma_posterior.log_prob(sigma_y)).permute(1,2,0)),dim=(1,2)) - torch.sum(sample_weights*((mask*sigma_prior.log_prob(sigma_y)).permute(1,2,0)),dim=(1,2))) - beta_vae*torch.sum(sample_weights*vae_prior.log_prob(z_sample).permute(1,0)) - torch.sum(sample_weights*(prior_entropy.permute(1,0))) - torch.sum(sample_weights*log_det)
        
        # calculate loss
        l = loss(pred_X[:,::2], recon_mean_x0, pred_logGamma[:,::2], pred_S[:,::2], survival_mask, dead_mask, after_dead_mask, t, y, censored, mask, sigma_y[:,1:], sigma_y[:,0], sample_weights) + beta_dynamics*sde_KL_loss(pred_X, t, context, dead_mask, drifts, model.dynamics.prior_drift, pred_sigma_X, dt, mean_T, std_T, sample_weights, med, W*matrix_mask, W_mean*matrix_mask) + kl_term
        
        # calculate gradients and update params
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1E4)
        optimizer.step()
    
    # check loss for whole training set
    if epoch % test_after == 0:

        model = model.eval()
        
        with torch.no_grad():

            total_loss = 0.
            recon_loss = 0.
            kl_loss = 0.
            VAE_loss = 0.
            sde_loss = 0.
            for i in range(test_average):
                
                for data in validation_generator:
                    
                    W_posterior = torch.distributions.laplace.Laplace(model.mean, model.logscale.exp())
                    sigma_posterior = torch.distributions.gamma.Gamma(model.logalpha.exp(), model.logbeta.exp())
                    
                    W = W_posterior.rsample((data['Y'].shape[0],))
                    sigma_y = sigma_posterior.rsample((data['Y'].shape[0],data['Y'].shape[1])) + 1e-6
                    
                    pred_X, t, pred_S, pred_logGamma, pred_sigma_X, context, y, times, mask, survival_mask, dead_mask, after_dead_mask, censored, sample_weights, med, env, z_sample, prior_entropy, log_det, recon_mean_x0, drifts, mask0, W_mean = model(data, sigma_y, test=True)
                    summed_weights = torch.sum(sample_weights)
                    
                    
                    kl_term = torch.sum(matrix_mask*(torch.sum(sample_weights*(W_posterior.log_prob(W).permute(1,2,0)),dim=-1) - torch.sum(sample_weights*(W_prior.log_prob(W).permute(1,2,0)),dim=-1))) + torch.sum(torch.sum(sample_weights*((mask*sigma_posterior.log_prob(sigma_y)).permute(1,2,0)),dim=(1,2)) - torch.sum(sample_weights*((mask*sigma_prior.log_prob(sigma_y)).permute(1,2,0)),dim=(1,2))) - torch.sum(sample_weights*vae_prior.log_prob(z_sample).permute(1,0)) - torch.sum(sample_weights*(prior_entropy.permute(1,0))) - torch.sum(sample_weights*log_det)
                    
                    # calculate loss
                    recon_l = loss(pred_X[:,::2], recon_mean_x0, pred_logGamma[:,::2], pred_S[:,::2], survival_mask, dead_mask, after_dead_mask, t, y, censored, mask, sigma_y[:,1:], sigma_y[:,0], sample_weights)
                    full_l = sde_KL_loss(pred_X, t, context, dead_mask, drifts, model.dynamics.prior_drift, pred_sigma_X, dt, mean_T, std_T, sample_weights, med, W*matrix_mask, W_mean*matrix_mask)
                    
                    kl_loss += kl_term
                    total_loss += full_l + recon_l + kl_term
                    recon_loss += recon_l
                    sde_loss += full_l
                    
            # output loss
            with open(loss_file, 'a') as lf:
                lf.writelines('%d, %.3f, %.3f\n'%(epoch, recon_loss.cpu().numpy()/test_average, total_loss.cpu().numpy()/test_average))
            print('Epoch %d, recon loss %.3f, total loss %.3f, kl loss %.3f, sde loss %.3f, beta dynamics %.3f, network %.3f, vae %.3f) '%(epoch, recon_loss.cpu().numpy()/test_average, total_loss.cpu().numpy()/test_average, kl_loss.cpu().numpy()/test_average, sde_loss.cpu().numpy()/test_average, beta_dynamics, beta_network, beta_vae), pred_sigma_X.cpu().mean(), sigma_y.cpu().mean())
            
        model = model.train()
        
        # step learning rate
        scheduler.step(total_loss/test_average)
    
    # output params
    if epoch % 20 ==0:
        torch.save(model.state_dict(), '%strain%d_Model_DJIN_nornn_epoch%d.params'%(params_folder, args.job_id, epoch))
    
    kl_scheduler_dynamics.step()
    kl_scheduler_network.step()
    kl_scheduler_vae.step()

torch.save(model.state_dict(), '%strain%d_Model_DJIN_nornn_epoch%d.params'%(params_folder, args.job_id, epoch))
