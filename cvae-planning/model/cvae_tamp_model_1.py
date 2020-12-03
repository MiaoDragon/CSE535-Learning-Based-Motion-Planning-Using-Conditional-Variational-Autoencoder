"""
CVAE model for s2d environment
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
# define CVAE model

# recognition network: z <- x, y
# here we implement encoder by the reverse connection of decoder
class Encoder(nn.Module):
    def __init__(self, x_input_size, s_input_size, cond_size, x_latent_size, s_latent_size, c_latent_size):
        super(Encoder, self).__init__()
        # x -> z_c
        self.x_zc_mu_network = \
            nn.Sequential(
                nn.Linear(x_input_size, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, c_latent_size)
            )
        self.x_zc_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(x_input_size, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, c_latent_size)
            )
        # x -> z_s
        self.x_zs_mu_network = \
            nn.Sequential(
                nn.Linear(x_input_size, 256), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(256, 128), nn.PReLU(),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, s_latent_size)
            )
        self.x_zs_log_sigma_pow2_network  = \
            nn.Sequential(
                nn.Linear(x_input_size, 256), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(256, 128), nn.PReLU(),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, s_latent_size)
            )
        # x -> z_x
        self.x_zx_mu_network = \
            nn.Sequential(
                nn.Linear(x_input_size, 256), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(256, 128), nn.PReLU(),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, 32), nn.PReLU(),
                nn.Linear(32, x_latent_size)
            )
        self.x_zx_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(x_input_size, 256), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(256, 128), nn.PReLU(),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, 32), nn.PReLU(),
                nn.Linear(32, x_latent_size)
            )
        # s -> z_c
        self.s_zc_mu_network = \
            nn.Sequential(
                nn.Linear(s_input_size, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, c_latent_size)
            )
        self.s_zc_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(s_input_size, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, c_latent_size)
            )
        # s -> z_s
        self.s_zs_mu_network = \
            nn.Sequential(
                nn.Linear(s_input_size, 256), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(256, 128), nn.PReLU(),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, 32), nn.PReLU(),
                nn.Linear(32, s_latent_size)
            )
        self.s_zs_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(s_input_size, 256), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(256, 128), nn.PReLU(),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, 32), nn.PReLU(),
                nn.Linear(32, s_latent_size)
            )

        # c -> zx
        self.c_zx_mu_network = \
            nn.Sequential(
                nn.Linear(cond_size, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, x_latent_size)
            )
        self.c_zx_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(cond_size, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, x_latent_size)
            )
        # c -> zs
        self.c_zs_mu_network = \
            nn.Sequential(
                nn.Linear(cond_size, 256), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(256, 128), nn.PReLU(),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, s_latent_size)
            )
        self.c_zs_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(cond_size, 256), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(256, 128), nn.PReLU(),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, s_latent_size)
            )

        # c -> zc
        self.c_zc_mu_network = \
            nn.Sequential(
                nn.Linear(cond_size, 256), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(256, 128), nn.PReLU(),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, 32), nn.PReLU(),
                nn.Linear(32, c_latent_size)
            )
        self.c_zc_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(cond_size, 256), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(256, 128), nn.PReLU(),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, 32), nn.PReLU(),
                nn.Linear(32, c_latent_size)
            )

        # zx -> zs
        self.zx_zs_mu_network = \
            nn.Sequential(
                nn.Linear(x_latent_size, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, s_latent_size)
            )
        self.zx_zs_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(x_latent_size, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, s_latent_size)
            )

        # zx -> zc
        self.zx_zc_mu_network = \
            self.mu_network = nn.Sequential(
                nn.Linear(x_latent_size, 64), nn.PReLU(),
                nn.Linear(64, c_latent_size)
            )
        self.zx_zc_log_sigma_pow2_network = \
            self.mu_network = nn.Sequential(
                nn.Linear(x_latent_size, 64), nn.PReLU(),
                nn.Linear(64, c_latent_size)
            )


        # zs -> zc
        self.zs_zc_mu_network = \
            nn.Sequential(
                nn.Linear(s_latent_size, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, c_latent_size)
            )
        self.zs_zc_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(s_latent_size, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, c_latent_size)
            )

        # zx
        # combine all inputs
        self.zx_mu_network = \
            nn.Sequential(
                nn.Linear(x_latent_size*2, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, x_latent_size)
            )
        self.zx_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(x_latent_size*2, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, x_latent_size)
            )
        # zs
        self.zs_mu_network = \
            nn.Sequential(
                nn.Linear(s_latent_size*4, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, s_latent_size)
            )
        self.zs_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(s_latent_size*4, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, s_latent_size)
            )
        # zc
        self.zc_mu_network = \
            nn.Sequential(
                nn.Linear(c_latent_size*5, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, c_latent_size)
            )
        self.zc_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(c_latent_size*5, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, c_latent_size)
            )
        self.output_size = x_latent_size
        self.x_latent_size = x_latent_size
        self.s_latent_size = s_latent_size
        self.c_latent_size = c_latent_size

        self.x_input_size = x_input_size
        self.s_input_size = s_input_size

        self.cond_size = cond_size
    def forward(self, x, y):
        # input tensor shape: BxK1, BxK2
        # x: continuous | discrete
        # extract the continuous state and discrete state
        cont_x = x[:,:self.x_input_size]
        disc_s = x[:,self.x_input_size:]
        x_zx_mu = self.x_zx_mu_network(cont_x)
        x_zs_mu = self.x_zs_mu_network(cont_x)
        x_zc_mu = self.x_zc_mu_network(cont_x)

        x_zx_log_sigma_pow2 = self.x_zx_log_sigma_pow2_network(cont_x)
        x_zs_log_sigma_pow2 = self.x_zs_log_sigma_pow2_network(cont_x)
        x_zc_log_sigma_pow2 = self.x_zc_log_sigma_pow2_network(cont_x)


        s_zs_mu = self.s_zs_mu_network(disc_s)
        s_zc_mu = self.s_zc_mu_network(disc_s)

        s_zs_log_sigma_pow2 = self.s_zs_log_sigma_pow2_network(disc_s)
        s_zc_log_sigma_pow2 = self.s_zc_log_sigma_pow2_network(disc_s)

        c_zx_mu = self.c_zx_mu_network(y)
        c_zs_mu = self.c_zs_mu_network(y)
        c_zc_mu = self.c_zc_mu_network(y)

        c_zx_log_sigma_pow2 = self.c_zx_log_sigma_pow2_network(y)
        c_zs_log_sigma_pow2 = self.c_zs_log_sigma_pow2_network(y)
        c_zc_log_sigma_pow2 = self.c_zc_log_sigma_pow2_network(y)

        # combine z
        zx_mu = self.zx_mu_network(torch.cat([x_zx_mu, c_zx_mu], dim=1))
        zx_log_sigma_pow2 = self.zx_log_sigma_pow2_network(torch.cat([x_zx_log_sigma_pow2, c_zx_log_sigma_pow2], dim=1))

        zx_zs_mu = self.zx_zs_mu_network(zx_mu)
        zx_zc_mu = self.zx_zc_mu_network(zx_mu)

        zx_zs_log_sigma_pow2 = self.zx_zs_log_sigma_pow2_network(zx_log_sigma_pow2)
        zx_zc_log_sigma_pow2 = self.zx_zc_log_sigma_pow2_network(zx_log_sigma_pow2)

        zs_mu = self.zs_mu_network(torch.cat([x_zs_mu, s_zs_mu, c_zs_mu, zx_zs_mu], dim=1))
        zs_log_sigma_pow2 = self.zs_log_sigma_pow2_network(torch.cat([x_zx_log_sigma_pow2, s_zs_log_sigma_pow2, \
                                                                      c_zx_log_sigma_pow2, zx_zs_log_sigma_pow2], dim=1))
        zs_zc_mu = self.zs_zc_mu_network(zs_mu)
        zs_zc_log_sigma_pow2 = self.zs_zc_log_sigma_pow2_network(zs_log_sigma_pow2)

        zc_mu = self.zc_mu_network(torch.cat([x_zc_mu, s_zc_mu, c_zc_mu, zx_zc_mu, zs_zc_mu], dim=1))
        zc_log_sigma_pow2 = self.zc_log_sigma_pow2_network(torch.cat([x_zc_log_sigma_pow2, s_zc_log_sigma_pow2, \
                                                                      c_zc_log_sigma_pow2, zx_zc_log_sigma_pow2, \
                                                                      zs_zc_log_sigma_pow2], dim=1))
        return zx_mu, zs_mu, zc_mu, zx_log_sigma_pow2, zs_log_sigma_pow2, zc_log_sigma_pow2

    def sample(self, mu, log_sigma_pow2, L):
        # given the computed mu, and sigma, obtain L samples by reparameterization
        # draw standard normal distribution
        # input: Bxk
        # return: LxBxk
        eps = torch.randn((L,len(mu),len(mu[0])))
        if log_sigma_pow2.is_cuda:
            eps = eps.cuda(log_sigma_pow2.device)
        eps = eps * torch.exp(log_sigma_pow2/2)
        eps = eps + mu
        return eps

    def kl_divergence(self, mu, log_sigma_pow2):
        # given mu and log(sigma^2), obtain the KL divergence relative to N(0,I)
        # using formula from https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes
        # input: BxK
        # output: B
        res = 1.0 / 2 * (-torch.sum(log_sigma_pow2, dim=1)-len(mu[0])+\
                         torch.sum(torch.exp(log_sigma_pow2), dim=1)+torch.sum(mu*mu, dim=1))
        return res

# prior network: c -> z
class PriorNet(nn.Module):
    def __init__(self, cond_size, x_latent_size, s_latent_size, c_latent_size):
        super(PriorNet, self).__init__()

        # c -> zc
        self.c_zc_mu_network = \
            nn.Sequential(
                nn.Linear(cond_size, 256), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(256, 128), nn.PReLU(),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, 32), nn.PReLU(),
                nn.Linear(32, c_latent_size)
            )
        self.c_zc_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(cond_size, 256), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(256, 128), nn.PReLU(),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, 32), nn.PReLU(),
                nn.Linear(32, c_latent_size)
            )
        # c -> zs
        self.c_zs_mu_network = \
            nn.Sequential(
                nn.Linear(cond_size, 256), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(256, 128), nn.PReLU(),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, s_latent_size)
            )
        self.c_zs_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(cond_size, 256), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(256, 128), nn.PReLU(),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, s_latent_size)
            )
        # c -> zx
        self.c_zx_mu_network = \
            nn.Sequential(
                nn.Linear(cond_size, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, x_latent_size)
            )
        self.c_zx_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(cond_size, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, x_latent_size)
            )

        # zc -> zs
        self.zc_zs_mu_network = \
            nn.Sequential(
                nn.Linear(c_latent_size, 64), nn.PReLU(),
                nn.Linear(64, 128), nn.Dropout(0.9), nn.PReLU(),
                nn.Linear(128, s_latent_size)
            )
        self.zc_zs_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(c_latent_size, 64), nn.PReLU(),
                nn.Linear(64, 128), nn.Dropout(0.9), nn.PReLU(),
                nn.Linear(128, s_latent_size)
            )
        # zc -> zx
        self.zc_zx_mu_network = \
            nn.Sequential(
                nn.Linear(c_latent_size, 64), nn.PReLU(),
                nn.Linear(64, x_latent_size)
            )
        self.zc_zx_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(c_latent_size, 64), nn.PReLU(),
                nn.Linear(64, x_latent_size)
            )

        # zs -> zx
        self.zs_zx_mu_network = \
            nn.Sequential(
                nn.Linear(s_latent_size, 64), nn.PReLU(),
                nn.Linear(64, 128), nn.Dropout(0.9), nn.PReLU(),
                nn.Linear(128, x_latent_size)
            )
        self.zs_zx_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(s_latent_size, 64), nn.PReLU(),
                nn.Linear(64, 128), nn.Dropout(0.9), nn.PReLU(),
                nn.Linear(128, x_latent_size)
            )

        # zc
        # self.zc_mu_network = \
        #     nn.Sequential(
        #         nn.Linear(c_latent_size, 128), nn.PReLU(), nn.Dropout(0.9),
        #         nn.Linear(128, 64), nn.PReLU(),
        #         nn.Linear(64, c_latent_size)
        #     )
        # self.zc_log_sigma_pow2_network = \
        #     nn.Sequential(
        #         nn.Linear(c_latent_size*5, 128), nn.PReLU(), nn.Dropout(0.9),
        #         nn.Linear(128, 64), nn.PReLU(),
        #         nn.Linear(64, c_latent_size)
        #     )

        # zs
        self.zs_mu_network = \
            nn.Sequential(
                nn.Linear(s_latent_size*2, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, s_latent_size)
            )
        self.zs_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(s_latent_size*2, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, s_latent_size)
            )
        # zx
        # combine all inputs
        self.zx_mu_network = \
            nn.Sequential(
                nn.Linear(x_latent_size*3, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, x_latent_size)
            )
        self.zx_log_sigma_pow2_network = \
            nn.Sequential(
                nn.Linear(x_latent_size*3, 128), nn.PReLU(), nn.Dropout(0.9),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, x_latent_size)
            )
        self.output_size = x_latent_size
        self.x_latent_size = x_latent_size
        self.s_latent_size = s_latent_size
        self.c_latent_size = c_latent_size

        self.cond_size = cond_size
    def forward(self, y):
        # input tensor shape: BxK1, BxK2
        zc_mu = self.c_zc_mu_network(y)
        zc_log_sigma_pow2 = self.c_zc_log_sigma_pow2_network(y)

        c_zs_mu = self.c_zs_mu_network(y)
        c_zs_log_sigma_pow2 = self.c_zs_log_sigma_pow2_network(y)

        c_zx_mu = self.c_zx_mu_network(y)
        c_zx_log_sigma_pow2 = self.c_zx_log_sigma_pow2_network(y)

        # combine z
        zc_zs_mu = self.zc_zs_mu_network(zc_mu)
        zc_zs_log_sigma_pow2 = self.zc_zs_log_sigma_pow2_network(zc_log_sigma_pow2)
        zc_zx_mu = self.zc_zx_mu_network(zc_mu)
        zc_zx_log_sigma_pow2 = self.zc_zx_log_sigma_pow2_network(zc_log_sigma_pow2)

        zs_mu = self.zs_mu_network(torch.cat([c_zs_mu, zc_zs_mu], dim=1))
        zs_log_sigma_pow2 = self.zs_log_sigma_pow2_network(torch.cat([c_zs_log_sigma_pow2, zc_zs_log_sigma_pow2], dim=1))

        zs_zx_mu = self.zs_zx_mu_network(zs_mu)
        zs_zx_log_sigma_pow2 = self.zs_zx_log_sigma_pow2_network(zs_log_sigma_pow2)

        zx_mu = self.zx_mu_network(torch.cat([c_zx_mu, zc_zx_mu, zs_zx_mu], dim=1))
        zx_log_sigma_pow2 = self.zx_log_sigma_pow2_network(torch.cat([c_zx_log_sigma_pow2, zc_zx_log_sigma_pow2, \
                                                                      zs_zx_log_sigma_pow2], dim=1))
        return zx_mu, zs_mu, zc_mu, zx_log_sigma_pow2, zs_log_sigma_pow2, zc_log_sigma_pow2

    def sample(self, mu, log_sigma_pow2, L):
        # given the computed mu, and sigma, obtain L samples by reparameterization
        # draw standard normal distribution
        # input: Bxk
        # return: LxBxk
        eps = torch.randn((L,len(mu),self.len(mu[0])))
        if log_sigma_pow2.is_cuda:
            eps = eps.cuda(log_sigma_pow2.device)
        eps = eps * torch.exp(log_sigma_pow2/2)
        eps = eps + mu
        return eps

    def kl_divergence(self, recog_mu, recog_log_sigma_pow2, prior_mu, prior_log_sigma_pow2):
        # given recognition mu and Sigma, and prior mu and Sigma, return the KL divergence
        # using formula from https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes
        # input: BxK
        # output: B
        term1 = torch.sum(prior_log_sigma_pow2-recog_log_sigma_pow2, dim=1)
        term2 = -len(recog_mu[0])
        term3 = torch.sum(torch.exp(recog_log_sigma_pow2-prior_log_sigma_pow2), dim=1)
        
        # term4: (u2-u1)T diag(1/sigma_2_i^2,...) (u2-u1)
        term4 = (prior_mu - recog_mu) * (prior_mu - recog_mu) * torch.exp(-prior_log_sigma_pow2)
        term4 = torch.sum(term4, dim=1)
        res = (term1 + term2+ term3 + term4) / 2
        return res




# generation network: x <- z, y
class Decoder(nn.Module):
    def __init__(self, x_latent_size, s_latent_size, c_latent_size, cond_size, x_size, s_size, sigma=0.1):
        super(Decoder, self).__init__()

        # we use an inverse model here

        # c -> x, s
        self.c_x_mu_network = \
            nn.Sequential(
                nn.Linear(cond_size, 64), nn.PReLU(),
                nn.Linear(64, 128), nn.PReLU(),
                nn.Linear(128, x_size)
            )

        # self.c_x_log_sigma_pow2_network = \
        #     nn.Sequential(
        #         nn.Linear(cond_size, 64), nn.PReLU(),
        #         nn.Linear(64, 128), nn.PReLU(),
        #         nn.Linear(128, x_size)
        #     )
        self.c_s_p_network = \
            nn.Sequential(
                nn.Linear(cond_size, 64), nn.PReLU(),
                nn.Linear(64, 128), nn.PReLU(),
                nn.Linear(128, s_size)
            )

        # zc -> x, s
        self.zc_x_mu_network = \
            nn.Sequential(
                nn.Linear(c_latent_size, 64), nn.PReLU(),
                nn.Linear(64, 128), nn.Dropout(0.9), nn.PReLU(),
                nn.Linear(128, x_size)
            )

        # self.zc_x_log_sigma_pow2_network = \
        #     nn.Sequential(
        #         nn.Linear(c_latent_size, 64), nn.PReLU(),
        #         nn.Linear(64, 128), nn.Dropout(0.9), nn.PReLU(),
        #         nn.Linear(128, x_size)
        #     )
        self.zc_s_p_network = \
            nn.Sequential(
                nn.Linear(c_latent_size, 64), nn.PReLU(),
                nn.Linear(64, 128), nn.Dropout(0.9), nn.PReLU(),
                nn.Linear(128, s_size)
            )

        # zs -> x, s
        self.zs_x_mu_network = \
            nn.Sequential(
                nn.Linear(c_latent_size, 64), nn.PReLU(),
                nn.Linear(64, 128), nn.PReLU(),
                nn.Linear(128, 256), nn.Dropout(0.9), nn.PReLU(),
                nn.Linear(256, x_size)
            )

        # self.zs_x_log_sigma_pow2_network = \
        #     nn.Sequential(
        #         nn.Linear(c_latent_size, 64), nn.PReLU(),
        #         nn.Linear(64, 128), nn.PReLU(),
        #         nn.Linear(128, 256), nn.Dropout(0.9), nn.PReLU(),
        #         nn.Linear(256, x_size)
        #     )
        
        self.zs_s_p_network = \
            nn.Sequential(
                nn.Linear(c_latent_size, 32), nn.PReLU(),
                nn.Linear(32, 64), nn.PReLU(),
                nn.Linear(64, 128), nn.PReLU(),
                nn.Linear(128, 256), nn.Dropout(0.9), nn.PReLU(),
                nn.Linear(256, s_size)
            )

        
        # zx -> x
        self.zx_x_mu_network = \
            nn.Sequential(
                nn.Linear(c_latent_size, 32), nn.PReLU(),
                nn.Linear(32, 64), nn.PReLU(),
                nn.Linear(64, 128), nn.PReLU(),
                nn.Linear(128, 256), nn.Dropout(0.9), nn.PReLU(),
                nn.Linear(256, x_size)
            )
        # self.zx_x_log_sigma_pow2_network = \
        #     nn.Sequential(
        #         nn.Linear(c_latent_size, 32), nn.PReLU(),
        #         nn.Linear(32, 64), nn.PReLU(),
        #         nn.Linear(64, 128), nn.PReLU(),
        #         nn.Linear(128, 256), nn.Dropout(0.9), nn.PReLU(),
        #         nn.Linear(256, x_size)
        #     )        

        self.x_mu_network = nn.Sequential(
            nn.Linear(x_size*4, 128), nn.PReLU(),
            nn.Linear(128, x_size)
        )
        # self.x_log_sigma_pow2_network = nn.Sequential(
        #     nn.Linear(x_size*4, 128), nn.PReLU(),
        #     nn.Linear(128, x_size)
        # )

        self.s_p_network = nn.Sequential(
            nn.Linear(s_size*3, 128), nn.PReLU(),
            nn.Linear(128, s_size), nn.Sigmoid()  # convert to probability
        )

        self.x_latent_size = x_latent_size
        self.s_latent_size = s_latent_size
        self.c_latent_size = c_latent_size

        self.x_size = x_size
        self.s_size = s_size

        self.cond_size = cond_size
        self.sigma = sigma
        self.sigma_2 = sigma*sigma

    def forward(self, zx, zs, zc, y):
        c_x_mu = self.c_x_mu_network(y)
        #c_x_log_sigma_pow2 = self.c_x_log_sigma_pow2_network(y)
        c_s_p = self.c_s_p_network(y)

        zc_x_mu = self.zc_x_mu_network(zc)
        #zc_x_log_sigma_pow2 = self.zc_x_log_sigma_pow2_network(zc)
        zc_s_p = self.zc_s_p_network(zc)

        zs_x_mu = self.zs_x_mu_network(zs)
        #zs_x_log_sigma_pow2 = self.zs_x_log_sigma_pow2_network(zs)
        zs_s_p = self.zs_s_p_network(zs)

        zx_x_mu = self.zx_x_mu_network(zx)
        #zx_x_log_sigma_pow2 = self.zx_x_log_sigma_pow2_network(zx)

        x_mu = self.x_mu_network(torch.cat([c_x_mu, zc_x_mu, zs_x_mu, zx_x_mu], dim=1))
        s_p = self.s_p_network(torch.cat([c_s_p, zc_s_p, zs_s_p], dim=1))
        return x_mu, s_p

    def sample(self, mu):
        # use the computed mu to generate sample
        # input: BxN
        eps = torch.randn(mu.size())
        if mu.is_cuda:
            eps = eps.cuda(mu.device)
        eps = eps * self.sigma + mu
        return eps
    def discrete_sample(self, p):
        # input: BxC
        return torch.bernoulli(p)

    def generation_loss(self, x, x_mu, s_p):
        # input:
        # - x_mu: LxBxN1
        # - s_p: LxBxN2
        # - x: BxN1
        # - s: BxN2
        # output: BxN
        # formula: 1/L * sum_z log(N(x; mu, sigma^2I))
        #       => -1/L \sum_z 1/2*(x-mu)^T(x-mu)/(sigma^2)
        cont_x = x[:,:self.x_size]
        disc_s = x[:,self.x_size:]
        x = cont_x
        s = disc_s

        res = - (x-x_mu)*(x-x_mu) #/self.sigma_2  # using normal distribution for p(x|z,y)
        #print(x.repeat(len(mu),1,1).size())
        #print(mu.size())
        #res = -torch.nn.functional.binary_cross_entropy(mu, x.repeat(len(mu),1,1), reduction='none')
        #print(res.cpu())
        #res = torch.sum(res, dim=2) # sum up (x-mu)^2
        # calculate the mean w.r.t. first dimension (L)
        res = torch.mean(res, dim=0)
        print(res.size())
        # symbolic term
        disc_res = -F.binary_cross_entropy(s_p, s.repeat(len(s_p),1,1), reduce=False).mean(dim=0)
        return torch.cat([res, disc_res], dim=1)

class CVAE(nn.Module):
    def __init__(self, x_size, s_size, x_latent_size, s_latent_size, c_latent_size, cond_size):
        super(CVAE, self).__init__()
        self.encoder = Encoder(x_size, s_size, cond_size, x_latent_size, s_latent_size, c_latent_size)
        self.decoder = Decoder(x_latent_size, s_latent_size, c_latent_size, cond_size, x_size, s_size, sigma=0.1)
        self.prior = PriorNet(cond_size, x_latent_size, s_latent_size, c_latent_size)
        self.x_size = x_size
        self.s_size = s_size
        self.x_latent_size = x_latent_size
        self.s_latent_size = s_latent_size
        self.c_latent_size = c_latent_size
        self.cond_size = cond_size

    def train_forward(self, x, y, L=10):
        # get prior distribution
        prior_zx_mu, prior_zs_mu, prior_zc_mu, \
            prior_zx_log_sigma_pow2, prior_zs_log_sigma_pow2, prior_zc_log_sigma_pow2 = self.prior(y)
        # get necessary signals from the input
        recog_zx_mu, recog_zs_mu, recog_zc_mu, \
            recog_zx_log_sigma_pow2, recog_zs_log_sigma_pow2, recog_zc_log_sigma_pow2 = self.encoder(x, y)
        # generate samples of z using the mean and variance
        zx = self.encoder.sample(recog_zx_mu, recog_zx_log_sigma_pow2, L)
        zs = self.encoder.sample(recog_zs_mu, recog_zs_log_sigma_pow2, L)
        zc = self.encoder.sample(recog_zc_mu, recog_zc_log_sigma_pow2, L)

        y_extended = y.repeat(len(zx),1).view(-1,self.cond_size)
        zx = zx.view(-1,self.x_latent_size)  # B and L together first
        zs = zx.view(-1,self.s_latent_size)  # B and L together first
        zc = zx.view(-1,self.c_latent_size)  # B and L together first

        if x.is_cuda:
            zx.cuda(x.device)
            zs.cuda(x.device)
            zc.cuda(x.device)

        # copy y so we have shape: LxBxc
        x_mu, s_p = self.decoder(zx, zs, zc, y_extended)
        x_mu = x_mu.view(L,-1,self.x_size)
        s_p = s_p.view(L,-1,self.s_size)
        return prior_zx_mu, prior_zs_mu, prior_zc_mu, \
            prior_zx_log_sigma_pow2, prior_zs_log_sigma_pow2, prior_zc_log_sigma_pow2, \
            recog_zx_mu, recog_zs_mu, recog_zc_mu, \
            recog_zx_log_sigma_pow2, recog_zs_log_sigma_pow2, recog_zc_log_sigma_pow2, \
            zx, zs, zc, x_mu, s_p

    def gen_forward(self, y):
        # randomly sample a latent z according to prior
        prior_zx_mu, prior_zs_mu, prior_zc_mu, \
            prior_zx_log_sigma_pow2, prior_zs_log_sigma_pow2, prior_zc_log_sigma_pow2 = self.prior(y)
        zx = self.prior.sample(prior_zx_mu, prior_zx_log_sigma_pow2, L=1)[0]
        zs = self.prior.sample(prior_zs_mu, prior_zs_log_sigma_pow2, L=1)[0]
        zc = self.prior.sample(prior_zc_mu, prior_zc_log_sigma_pow2, L=1)[0]

        x_mu, s_p = self.decoder(zx, zs, zc, y)
        x = self.decoder.sample(x_mu)
        s = self.decoder.discrete_sample(s_p)
        # combine input
        return torch.cat([x,s], dim=1)
