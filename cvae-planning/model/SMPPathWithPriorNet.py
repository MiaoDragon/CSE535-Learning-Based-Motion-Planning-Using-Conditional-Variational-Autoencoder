import torch.nn as nn
import torch
import numpy as np
import copy
"""
this defines the MPNet to be used, which will utilize MLP and AE.
"""
class SMPPathWithPriorNet(nn.Module):
    def __init__(self, e_net, cvae):
        super(SMPPathWithPriorNet, self).__init__()
        self.e_net = e_net
        self.cvae = cvae
        self.opt = torch.optim.Adagrad(list(self.e_net.parameters())+list(self.cvae.parameters()))

    def set_opt(self, opt, lr=1e-2, momentum=None):
        # edit: can change optimizer type when setting
        if momentum is None:
            self.opt = opt(list(self.e_net.parameters())+list(self.cvae.parameters()), lr=lr)
        else:
            self.opt = opt(list(self.e_net.parameters())+list(self.cvae.parameters()), lr=lr, momentum=momentum)

    def train_forward(self, x, x_cond, obs, L=10):
        # x_step_dis: ratio of (distance from start) / (distance of trajectory)
        obs_z = self.e_net(obs)
        cond = torch.cat([x_cond, obs_z], 1)

        return self.cvae.train_forward(x, cond, L=L)

    def gen_forward(self, x_cond, obs=None, obs_z=None):
        assert(obs is not None or obs_z is not None)
        if obs_z is None:
            obs_z = self.e_net(obs)
        cond = torch.cat([x_cond, obs_z], 1)
        return self.cvae.gen_forward(cond)

    def loss(self, x, forward_output, beta=1.0):
        prior_z_mu, prior_z_log_sigma_pow2, recog_z_mu, recog_z_log_sigma_pow2, z, x_mu = forward_output
        kl_divergence = self.cvae.prior.kl_divergence(recog_z_mu, recog_z_log_sigma_pow2, \
                                                        prior_z_mu, prior_z_log_sigma_pow2).unsqueeze(1)
        generation_loss = self.cvae.decoder.generation_loss(x, x_mu)
        loss_i = -generation_loss + beta * kl_divergence.expand(-1,self.cvae.input_size)
        #loss_i = torch.mean(loss_i)
        return loss_i
    
    def generation_loss(self, x, forward_output):
        prior_z_mu, prior_z_log_sigma_pow2, recog_z_mu, recog_z_log_sigma_pow2, z, x_mu = forward_output
        generation_loss = self.cvae.decoder.generation_loss(x, x_mu)
        return -generation_loss

    def kl_divergence(self, x, forward_output):
        prior_z_mu, prior_z_log_sigma_pow2, recog_z_mu, recog_z_log_sigma_pow2, z, x_mu = forward_output
        kl_divergence = self.cvae.prior.kl_divergence(recog_z_mu, recog_z_log_sigma_pow2, \
                                                        prior_z_mu, prior_z_log_sigma_pow2).unsqueeze(1)
        return kl_divergence

    def step(self, x, x_cond, obs, beta=1.0, L=10):
        # given a batch of data, optimize the parameters by one gradient descent step
        # assume here x and y are torch tensors, and have been
        self.zero_grad()
        # edited: loss now returns a D dimension vector recording loss on each input dimension
        loss = torch.mean(self.loss(x, self.train_forward(x, x_cond, obs, L=L), beta=beta))
        loss.backward()
        self.opt.step()