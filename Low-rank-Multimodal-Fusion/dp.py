import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
from torch.nn.functional import kl_div, softmax, log_softmax
from torch.distributions import MultivariateNormal

from scipy.stats import beta


class DirichletProcess3DLoss(nn.Module):

    def __init__(self, dim=256, K=3, M=3, rho_scale=-4, eta=1):
        super(DirichletProcess3DLoss, self).__init__()
        """
        !!! One of the variants !!!
        """
        self.theta = nn.Parameter(torch.ones(1) * 1)

        self.pi = nn.Parameter(torch.ones([K * M]) / (K * M))
        self.phi = torch.ones(K * M) / (K * M)

        self.eta = eta
        self.gamma_1 = torch.ones(K * M)
        self.gamma_2 = torch.ones(K * M) * eta

        self.eta = eta

        self.mu_x = nn.Parameter(torch.zeros([K, dim]))
        self.mu_y = nn.Parameter(torch.zeros([K, dim]))
        self.mu_z = nn.Parameter(torch.zeros([K, dim]))
        self.log_cov_x = nn.Parameter(torch.ones([K, dim]) * rho_scale)
        self.log_cov_y = nn.Parameter(torch.ones([K, dim]) * rho_scale)
        self.log_cov_z = nn.Parameter(torch.ones([K, dim]) * rho_scale)

        self.K = K
        self.M = M
        self.n_mixture = K * M

    def _update_gamma(self):
        phi = self.phi

        phi_flipped = torch.flip(phi, dims=[1])
        cum_sum = torch.cumsum(phi_flipped, dim=1) - phi_flipped
        cum_sum = torch.flip(cum_sum, dims=[1])

        self.gamma_1 = 1 + phi.mean(0)
        self.gamma_2 = self.eta + cum_sum.mean(0)

    def forward(self, x, y, z):

        batch_size = x.shape[0]
        beta = self.sample_beta(batch_size)
        pi = self.mix_weights(beta)[:, :-1]

        cov_x = torch.log1p(torch.exp(self.log_cov_x)).clamp(min=1e-15)
        cov_y = torch.log1p(torch.exp(self.log_cov_y)).clamp(min=1e-15)
        cov_z = torch.log1p(torch.exp(self.log_cov_z)).clamp(min=1e-15)

        # loss = torch.cat(c_list)
        u_log_pdf = [self.mvn_pdf(x, self.mu_x[k], cov_x[k]) for k in range(self.K)]
        v_log_pdf = [self.mvn_pdf(y, self.mu_y[k], cov_y[k]) for k in range(self.K)]
        w_log_pdf = [self.mvn_pdf(z, self.mu_z[k], cov_z[k]) for k in range(self.K)]

        u_entropy = [self.mvn_entropy(self.mu_x[k], cov_x[k]) for k in range(self.K)]
        v_entropy = [self.mvn_entropy(self.mu_y[k], cov_y[k]) for k in range(self.K)]
        w_entropy = [self.mvn_entropy(self.mu_z[k], cov_z[k]) for k in range(self.K)]

        assert not torch.isinf(torch.stack(u_log_pdf, dim=1)).any()
        assert not torch.isinf(torch.stack(v_log_pdf, dim=1)).any()
        assert not torch.isinf(torch.stack(w_log_pdf, dim=1)).any()

        loss = torch.stack(u_log_pdf + v_log_pdf + w_log_pdf, dim=1) + torch.stack(u_entropy + v_entropy + w_entropy, dim=0) + torch.log(pi.clamp(min=1e-15))

        self.phi = torch.softmax(loss, dim=-1).clamp(min=1e-15).detach()
        self._update_gamma()

        loss = torch.logsumexp(loss, -1).mean(0)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

        return - loss  #  ELBO = negative of likelihood

    def sample_beta(self, size):
        a = self.gamma_1.detach().cpu().numpy()
        b = self.gamma_2.detach().cpu().numpy()

        samples = beta.rvs(a, b, size=(size, self.n_mixture))
        samples = torch.from_numpy(samples).cuda()

        return samples

    def mvn_pdf(self, x, mu, cov):
        """
        PDF of multivariate normal distribution
        """
        log_pdf = MultivariateNormal(mu, scale_tril=torch.diag(cov)).log_prob(x)

        return log_pdf

    def mvn_entropy(self, mu, cov):
        """
        PDF of multivariate normal distribution
        """
        log_pdf = MultivariateNormal(mu, scale_tril=torch.diag(cov)).entropy()

        return log_pdf

    def mix_weights(self, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        pi = F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)
        return pi