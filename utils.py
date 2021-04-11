import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GRUCell(nn.Module):
    def __init__(self, map1, map2, forget_bias=0, clip_value=np.inf):
        super(GRUCell, self).__init__()
        self.forget_bias = forget_bias  # forget bias for GRU
        self.clip_value = clip_value  # prevent grad explosion

        self.mat_a = nn.Linear(map1[0], map1[1])
        self.mat_b = nn.Linear(map2[0], map2[1])

    def forward(self, inputs, state, graph=None):
        x = inputs
        h = state.to(device)
        xh = torch.cat([x, h], dim=2)

        # xh = batch x neurons x map1[0] and [r,u] = batch x neurons x map1[1]
        ru = self.mat_a(xh) if graph is None else self.mat_a(torch.matmul(graph, xh))  # normal RNN or graph RNN
        r, u = ru.chunk(chunks=2, dim=2)
        r, u = torch.sigmoid(r), torch.sigmoid(u + self.forget_bias)

        xrh = torch.cat([x, r * h], dim=2)
        # c = batch x neurons x map2[1], xrh = batch x neurons x map2[0]
        # if graph is not None:
        #     print(graph)
        c = self.mat_b(xrh) if graph is None else self.mat_b(torch.matmul(graph, xrh))  # normal RNN or graph RNN

        c = torch.tanh(c)
        # c = torch.sigmoid(c)
        new_h = u * h + (1.0 - u) * c
        
        new_h = torch.clamp(new_h, -self.clip_value, self.clip_value)
        return new_h, new_h


class EnCoder(nn.Module):
    def __init__(self, cell, time, neurons, dim):
        super(EnCoder, self).__init__()
        self.forward_cell = cell((dim + 1, 2 * dim), (dim + 1, dim), clip_value=10000.0)
        self.backward_cell = cell((dim + 1, 2 * dim), (dim + 1, dim), clip_value=10000.0)
        self.time = time
        self.dim = dim

        self.w_in = nn.Linear(neurons, neurons)

    def forward(self, inputs):  # inputs with dimension of: batch x time x neurons
        inputs = self.w_in(inputs)  # batch x time x neurons

        enc_state = torch.zeros([inputs.shape[0], inputs.shape[2], self.dim])  # enc_hidden: batch x neurons x dim_e
        enc_outs_forward = [None] * self.time
        inputs = inputs[:, :, :, None]
        for t in range(self.time):
            input_t = inputs[:, t, :]
            enc_out, enc_state = self.forward_cell(input_t, enc_state)  # enc_out==enc_hidden: batch x neurons x dim_e
            enc_outs_forward[t] = enc_out

        enc_state = torch.zeros([inputs.shape[0], inputs.shape[2], self.dim])  # enc_hidden: batch x neurons x dim_e
        enc_outs_backward = [None] * self.time
        for t in range(self.time - 1, -1, -1):
            input_t = inputs[:, t, :]
            enc_out, enc_state = self.backward_cell(input_t, enc_state)
            enc_outs_backward[t] = enc_out

        return enc_outs_forward, enc_outs_backward


class LearnableGraph(nn.Module):
    def __init__(self, node_num):
        super(LearnableGraph, self).__init__()
        random_matrix = torch.randn((1, node_num, node_num), device=device)
        self.random_matrix = torch.nn.Parameter(random_matrix, requires_grad=True)
        self.register_parameter("adjacency", self.random_matrix)
        # self.A_matrix = torch.tril(self.random_matrix) + torch.transpose(torch.tril(self.random_matrix, -1),1,2)
        # self.A_matrix = self.random_matrix

    def forward(self):
        self.A_matrix = self.random_matrix + torch.transpose(self.random_matrix,1,2)
        prob_A = torch.sigmoid(self.A_matrix)
        return prob_A


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.w = nn.Linear(in_features, out_features)

    def forward(self, inputs, adj):
        support = self.w(inputs)
        output = torch.matmul(adj, support)
        return output


class BinaryKLLoss(nn.Module):
    def __init__(self):
        super(BinaryKLLoss, self).__init__()
    
    def forward(self, p, q):
        loss_tensor =  p*torch.log(p/q+1e-45)+(1-p)*torch.log((1-p)/(1-q)+1e-45)
        return torch.mean(loss_tensor)  


def gaussian_pos_log_likelihood(unused_mean, logvar, noise):
    # ln N(z; mean, sigma) = - ln(sigma) - 0.5 ln 2pi - noise^2 / 2
    return - 0.5 * (logvar + np.log(2 * np.pi) + noise**2)


def diag_gaussian_log_likelihood(z, mu=0.0, logvar=0.0):
    return -0.5 * (logvar + np.log(2 * np.pi) + ((z - mu) / torch.exp(0.5 * logvar))**2)


class DiagonalGaussian(nn.Module):
    def __init__(self, mean, logvar):
        super(DiagonalGaussian, self).__init__()
        self.mean = mean  # bxn already
        self.logvar = logvar  # bxn already
        self.noise = noise = torch.randn(logvar.shape, device=device)
        self.sample_ = mean + torch.exp(0.5 * logvar) * noise
        self.logp_ = None

    # def forward(self):
    #     return self.sample_

    def logp(self, z):
        if z is self.sample_:
            self.logp_ = gaussian_pos_log_likelihood(self.mean, self.logvar, self.noise)
            return self.logp_

        self.logp_ = diag_gaussian_log_likelihood(z, self.mean, self.logvar)
        return self.logp_
 
    @property
    def sample(self):
        return self.sample_



class LearnableAutoRegressive1Prior(nn.Module):
    def __init__(self, neurons, dim_d, autocorrelation_taus, noise_variances, time):
        super(LearnableAutoRegressive1Prior, self).__init__()

        log_evar_inits_1 = math.log(noise_variances)
        logevars_nxd = torch.full((neurons, dim_d), log_evar_inits_1, device=device)
        self.logevars_nxd = torch.nn.Parameter(logevars_nxd, requires_grad=True)
        self.register_parameter("logevars_nxd", self.logevars_nxd)

        log_atau_inits_1 = math.log(autocorrelation_taus)
        logataus_nxd = torch.full((neurons, dim_d), log_atau_inits_1, device=device)
        self.logataus_nxd = torch.nn.Parameter(logataus_nxd, requires_grad=True)
        self.register_parameter("logataus_nxd", self.logataus_nxd)

        # phi in x_t = \mu + phi x_tm1 + \eps
        # phi = exp(-1/tau)
        # phi = exp(-1/exp(logtau))
        # phi = exp(-exp(-logtau))
        ## self.phis_nxd = torch.exp(-torch.exp(-self.logataus_nxd))

        # process noise
        # pvar = evar / (1- phi^2)
        # logpvar = log ( exp(logevar) / (1 - phi^2) )
        # logpvar = logevar - log(1-phi^2)
        # logpvar = logevar - (log(1-phi) + log(1+phi))
        ## self.logpvars_nxd = self.logevars_nxd - torch.log(1.0 - self.phis_nxd) - torch.log(1.0 + self.phis_nxd)

        # logevars_1xu, logataus_1xu and logpvars_1xu are key values

        self.pmeans_nxd = torch.zeros((neurons, dim_d),device=device)

        # For sampling from the prior during de-novo generation.
        # self.means_t = [None] * time
        # self.logvars_t = [None] * time
        # self.samples_t = [None] * time
        # self.gaussians_t = [None] * time
        # self.z_mean_pt_nxd_t = [None] * time

        # for t in range(time):
        #     # process variance used here to make process completely stationary
        #     if t == 0:
        #         logvar_pt_nxd = self.logpvars_nxd
        #         self.z_mean_pt_nxd_t[t] = self.pmeans_nxd + self.phis_nxd * torch.zeros((neurons, dim_d),device=device)
        #     else:
        #         logvar_pt_nxd = self.logevars_nxd
        #         self.z_mean_pt_nxd_t[t] = self.pmeans_nxd + self.phis_nxd * self.samples_t[t-1]
            
        #     self.gaussians_t[t] = DiagonalGaussian(mean=self.z_mean_pt_nxd_t[t], logvar=logvar_pt_nxd)
        #     self.samples_t[t] = self.gaussians_t[t].sample
        #     self.logvars_t[t] = logvar_pt_nxd
        #     self.means_t[t] = self.z_mean_pt_nxd_t[t]


    def logp_t(self, z_t_nxd, z_tm1_nxd=None):

        phis_nxd = torch.exp(-torch.exp(-self.logataus_nxd))
        logpvars_nxd = self.logevars_nxd - torch.log(1.0 - phis_nxd) - torch.log(1.0 + phis_nxd)

        if z_tm1_nxd is None:
            return diag_gaussian_log_likelihood(z_t_nxd, self.pmeans_nxd, logpvars_nxd)
        else:
            means_t_nxd = self.pmeans_nxd + phis_nxd * z_tm1_nxd
            return diag_gaussian_log_likelihood(z_t_nxd, means_t_nxd, self.logevars_nxd)


class KLCost_GaussianGaussianProcessSampled(nn.Module):
    def __init__(self):
        super(KLCost_GaussianGaussianProcessSampled, self).__init__()

    def forward(self, post_zs, prior_z_process):
        # L = -KL + log p(x|z), to maximize bound on likelihood
        # -L = KL - log p(x|z), to minimize bound on NLL
        # so 'KL cost' is postive KL divergence
        z0_bxu = post_zs[0].sample
        logq_bxu = post_zs[0].logp(z0_bxu)
        logp_bxu = prior_z_process.logp_t(z0_bxu)
        z_tm1_bxu = z0_bxu
        for z_t in post_zs[1:]:
            # posterior is independent in time, prior is not
            z_t_bxu = z_t.sample
            logq_bxu += z_t.logp(z_t_bxu)
            logp_bxu += prior_z_process.logp_t(z_t_bxu, z_tm1_bxu)
            z_tm1_bxu = z_t_bxu

        kl_bxu = logq_bxu - logp_bxu
        kl_b = torch.mean(kl_bxu, [1, 2])
        kl_cost = torch.mean(kl_b)/len(post_zs)
        return kl_cost

class Poisson(nn.Module):
    def __init__(self, log_rates):
        self.logr = log_rates

    def logp(self, k):
        """Compute the log probability for the counts in the bin, under the model.

        Args:
        bin_counts: array-like integer counts

        Returns:
        The log-probability under the Poisson models for each element of
        bin_counts.
        """
        # log poisson(k, r) = log(r^k * e^(-r) / k!) = k log(r) - r - log k!
        # log poisson(k, r=exp(x)) = k * x - exp(x) - lgamma(k + 1)
        return torch.unsqueeze(k,-1) * self.logr - torch.exp(self.logr) - torch.lgamma(torch.unsqueeze(k,-1) + 1)