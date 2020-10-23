"""
CVAE model for s2d environment
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

# define CVAE model
class Encoder(nn.Module):
    def __init__(self, input_size, cond_size, latent_size):
        super(Encoder, self).__init__()
        self.mu_network = nn.Sequential(
            nn.Linear(input_size+cond_size, 512), nn.PReLU(), nn.Dropout(),
            nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 32), nn.PReLU(),
            nn.Linear(32, latent_size)
        )
        # we set the covariance matrix to be diag([sigma_1,...,sigma_k])
        self.log_sigma_pow2_network = nn.Sequential(
            nn.Linear(input_size+cond_size, 512), nn.PReLU(), nn.Dropout(),
            nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 32), nn.PReLU(),
            nn.Linear(32, latent_size)
        )
        self.output_size = latent_size
        self.latent_size = latent_size
        self.input_size = input_size
        self.cond_size = cond_size
    def forward(self, x, y):
        # input tensor shape: BxK1, BxK2
        input = torch.cat([x,y], dim=1)
        mu = self.mu_network(input)
        log_sigma_pow2 = self.log_sigma_pow2_network(input)
        return mu, log_sigma_pow2

    def sample(self, mu, log_sigma_pow2, L):
        # given the computed mu, and sigma, obtain L samples by reparameterization
        # draw standard normal distribution
        # input: Bxk
        # return: LxBxk
        eps = torch.randn((L,len(mu),self.latent_size))
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
        res = 1.0 / 2 * (-torch.sum(log_sigma_pow2, dim=1)-self.output_size+\
                         torch.sum(torch.exp(log_sigma_pow2), dim=1)+torch.sum(mu*mu, dim=1))
        return res


class Decoder(nn.Module):
    def __init__(self, latent_size, cond_size, output_size, sigma=0.1):
        super(Decoder, self).__init__()
        self.mu_network = nn.Sequential(
            nn.Linear(latent_size+cond_size, 512), nn.PReLU(), nn.Dropout(),
            nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 32), nn.PReLU(),
            nn.Linear(32, output_size)
        )
        self.latent_size = latent_size
        self.cond_size = cond_size
        self.sigma = sigma
        self.sigma_2 = sigma*sigma

    def forward(self, z, y):
        input = torch.cat([z,y],dim=1)
        mu = self.mu_network(input)
        return mu

    def sample(self, mu):
        # use the computed mu to generate sample
        # input: BxN
        eps = torch.randn(mu.size())
        if mu.is_cuda:
            eps = eps.cuda(mu.device)
        eps = eps * self.sigma + mu
        return eps

    def generation_loss(self, x, mu):
        # input:
        # - mu: LxBxN
        # - x: BxN
        # output: BxN
        # formula: 1/L * sum_z log(N(x; mu, sigma^2I))
        #       => -1/L \sum_z 1/2*(x-mu)^T(x-mu)/(sigma^2)

        res = - 1.0/2*(x-mu)*(x-mu)/self.sigma_2  # using normal distribution for p(x|z,y)
        #print(x.repeat(len(mu),1,1).size())
        #print(mu.size())
        #res = -torch.nn.functional.binary_cross_entropy(mu, x.repeat(len(mu),1,1), reduction='none')
        #print(res.cpu())
        #print(res.size())
        res = torch.sum(res, dim=2) # sum up (x-mu)^2
        # calculate the mean w.r.t. first dimension (L)
        res = torch.mean(res, dim=0)
        return res

class CVAE(nn.Module):
    def __init__(self, input_size, latent_size, cond_size):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_size, cond_size, latent_size)
        self.decoder = Decoder(latent_size, cond_size, input_size, sigma=0.1)
        self.input_size = input_size
        self.latent_size = latent_size
        self.cond_size = cond_size

    def train_forward(self, x, y, L=10):
        # get necessary signals from the input
        z_mu, z_log_sigma_pow2 = self.encoder(x, y)
        # generate samples of z using the mean and variance
        z = self.encoder.sample(z_mu, z_log_sigma_pow2, L)
        y_extended = y.repeat(len(z),1).view(-1,self.cond_size)
        z = z.view(-1,self.latent_size)  # B and L together first
        if x.is_cuda:
            z.cuda(x.device)
        # copy y so we have shape: LxBxc
        x_mu = self.decoder(z, y_extended).view(L,-1,self.input_size)
        return z_mu,z_log_sigma_pow2, z, x_mu

    def gen_forward(self, y):
        # randomly sample a latent z
        z = torch.randn(len(y),self.latent_size)
        if y.is_cuda:
            z = z.cuda(y.device)
        x_mean = self.decoder(z, y)
        x = self.decoder.sample(x_mean)
        return x
