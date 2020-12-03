import torch.nn as nn
import torch
import numpy as np
import copy
"""
this defines the MPNet to be used, which will utilize MLP and AE.
"""
class TSMPPathWithPriorNet(nn.Module):
    def __init__(self, cvae):
        super(TSMPPathWithPriorNet, self).__init__()
        self.cvae = cvae
        self.opt = torch.optim.Adagrad(list(self.cvae.parameters()))

    def set_opt(self, opt, lr=1e-2, momentum=None):
        # edit: can change optimizer type when setting
        if momentum is None:
            self.opt = opt(list(self.cvae.parameters()), lr=lr)
        else:
            self.opt = opt(list(self.cvae.parameters()), lr=lr, momentum=momentum)

    def train_forward(self, x, x_cond, obs=None, L=10):
        # x_step_dis: ratio of (distance from start) / (distance of trajectory)
        return self.cvae.train_forward(x, x_cond, L=L)

    def gen_forward(self, x_cond, obs=None, obs_z=None):
        return self.cvae.gen_forward(x_cond)

    def loss(self, x, forward_output, beta=1.0):
        prior_zx_mu, prior_zs_mu, prior_zc_mu, \
            prior_zx_log_sigma_pow2, prior_zs_log_sigma_pow2, prior_zc_log_sigma_pow2, \
            recog_zx_mu, recog_zs_mu, recog_zc_mu, \
            recog_zx_log_sigma_pow2, recog_zs_log_sigma_pow2, recog_zc_log_sigma_pow2,\
            zx, zs, zc, x_mu, s_p = forward_output
        kl_divergence = self.cvae.prior.kl_divergence(recog_zx_mu, recog_zx_log_sigma_pow2, \
                                                        prior_zx_mu, prior_zx_log_sigma_pow2).unsqueeze(1)
        kl_divergence += self.cvae.prior.kl_divergence(recog_zs_mu, recog_zs_log_sigma_pow2, \
                                                        prior_zs_mu, prior_zs_log_sigma_pow2).unsqueeze(1)
        kl_divergence += self.cvae.prior.kl_divergence(recog_zc_mu, recog_zc_log_sigma_pow2, \
                                                        prior_zc_mu, prior_zc_log_sigma_pow2).unsqueeze(1)
        # here we assume the three latent variables have the same length

        generation_loss = self.cvae.decoder.generation_loss(x, x_mu, s_p)
        loss_i = -generation_loss + beta * kl_divergence#.expand(-1,self.cvae.x_latent_size)
        #loss_i = torch.mean(loss_i)
        return loss_i
    
    def generation_loss(self, x, forward_output):
        prior_zx_mu, prior_zs_mu, prior_zc_mu, \
            prior_zx_log_sigma_pow2, prior_zs_log_sigma_pow2, prior_zc_log_sigma_pow2, \
            recog_zx_mu, recog_zs_mu, recog_zc_mu, \
            recog_zx_log_sigma_pow2, recog_zs_log_sigma_pow2, recog_zc_log_sigma_pow2,\
            zx, zs, zc, x_mu, s_p = forward_output
        generation_loss = self.cvae.decoder.generation_loss(x, x_mu, s_p)
        return -generation_loss

    def kl_divergence(self, x, forward_output):
        prior_zx_mu, prior_zs_mu, prior_zc_mu, \
            prior_zx_log_sigma_pow2, prior_zs_log_sigma_pow2, prior_zc_log_sigma_pow2, \
            recog_zx_mu, recog_zs_mu, recog_zc_mu, \
            recog_zx_log_sigma_pow2, recog_zs_log_sigma_pow2, recog_zc_log_sigma_pow2,\
            zx, zs, zc, x_mu, s_p = forward_output
        kl_divergence = self.cvae.prior.kl_divergence(recog_zx_mu, recog_zx_log_sigma_pow2, \
                                                        prior_zx_mu, prior_zx_log_sigma_pow2).unsqueeze(1)
        kl_divergence += self.cvae.prior.kl_divergence(recog_zs_mu, recog_zs_log_sigma_pow2, \
                                                        prior_zs_mu, prior_zs_log_sigma_pow2).unsqueeze(1)
        kl_divergence += self.cvae.prior.kl_divergence(recog_zc_mu, recog_zc_log_sigma_pow2, \
                                                        prior_zc_mu, prior_zc_log_sigma_pow2).unsqueeze(1)
        return kl_divergence

    def step(self, x, x_cond, obs=None, beta=1.0, L=10):
        # given a batch of data, optimize the parameters by one gradient descent step
        # assume here x and y are torch tensors, and have been
        self.zero_grad()
        # edited: loss now returns a D dimension vector recording loss on each input dimension
        loss = torch.mean(self.loss(x, self.train_forward(x, x_cond, obs, L=L), beta=beta))
        loss.backward()
        self.opt.step()