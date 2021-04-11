# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import GRUCell,EnCoder,LearnableGraph,GraphConvolution,BinaryKLLoss,DiagonalGaussian, \
LearnableAutoRegressive1Prior,KLCost_GaussianGaussianProcessSampled,Poisson

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LFADSG(nn.Module):
    def __init__(self, arg):
        super(LFADSG, self).__init__()
        # must fit the data
        self.time = arg.time
        self.neurons = arg.neurons
        # hyper parameters
        self.dim_e = arg.dim_e
        self.dim_c = arg.dim_c
        self.dim_d = arg.dim_d
        self.gcn_hidden = arg.gcn_hidden
        self.tau = arg.tau
        self.time_lag = arg.time_lag
        self.dropout_prob = arg.dropout_prob
        self.kl_weight_1 = arg.kl_weight_1
        self.kl_weight_2 = arg.kl_weight_2
        self.autocorrelation_taus = arg.autocorrelation_taus
        self.noise_variances = arg.noise_variances
        self.dynamic_graph = arg.dynamic_graph
        self.discrete_graph = arg.discrete_graph

        # parameters
        self.encoderGRU_1 = EnCoder(GRUCell, self.time, self.neurons, self.dim_e)
        self.encoderGRU_2 = EnCoder(GRUCell, self.time, self.neurons, self.dim_e)
        if self.dynamic_graph:
            self.graph_gru_cell = GRUCell((2 * self.neurons, 2 * self.neurons), (2 * self.neurons, self.neurons))  # generate graph
        self.graph_rnn_gru_cell = GRUCell((2 * self.dim_e + self.dim_c, 2 * self.dim_c), (2 * self.dim_e + self.dim_c, self.dim_c))  # graph rnn
        self.w_mean = nn.Linear(self.dim_c, self.dim_d)
        self.w_var = nn.Linear(self.dim_c, self.dim_d)

        self.gcn_layers = list()
        previous_hidden = self.dim_d
        for n_hidden in self.gcn_hidden:
            self.gcn_layers.append(GraphConvolution(previous_hidden, n_hidden).to(device))
            previous_hidden = n_hidden
        self.gcn_layers = nn.ModuleList(self.gcn_layers)
        self.w_out = nn.Linear(self.gcn_hidden[-1], 1)

        # prior distribution
        # graph_0
        self.prior_graph_0 = LearnableGraph(self.neurons)
        # d
        self.prior_d = LearnableAutoRegressive1Prior(self.neurons, self.dim_d, self.autocorrelation_taus, self.noise_variances, self.time)

        # loss
        # self.poisson_loss = nn.PoissonNLLLoss(log_input=True)
        self.poisson_loss = 0
        self.kl_loss_1 = BinaryKLLoss()
        # self.kl_loss_graph = [None]*self.time
        # for t in range(self.time):
        #     self.kl_loss_graph[t] = BinaryKLLoss()
        self.kl_loss_2 = KLCost_GaussianGaussianProcessSampled()


    def prob_2_graph_sample(self, prob_a_matrix):
        def adjacency_generate(a, tau, xi1, xi2):
            tmp = (torch.log(a+1e-40) + xi1) - (torch.log(1.0 - a + 1e-40) + xi2) #为什么要加到1e-5
            tmp = tmp / tau
            a_ret = torch.sigmoid(tmp)
            return a_ret

        assert torch.max(torch.abs(prob_a_matrix - prob_a_matrix.transpose(1,2))) < 0.001
        noise1 = -torch.log(-torch.log(torch.rand([self.neurons, self.neurons], device=device)+1e-40)+1e-40)
        noise1 = torch.triu(noise1)+torch.triu(noise1,diagonal=1).t()
        noise2 = -torch.log(-torch.log(torch.rand([self.neurons, self.neurons], device=device)+1e-40)+1e-40)
        noise2 = torch.triu(noise2)+torch.triu(noise2,diagonal=1).t()
        sample_graph = adjacency_generate(prob_a_matrix, self.tau, noise1, noise2)
        assert torch.max(torch.abs(sample_graph - sample_graph.transpose(1,2))) < 0.001
        return sample_graph

    @staticmethod
    def vector_2_graph_prob(inputs):
        # inputs = batch x neuron x 2*dim_e
        a_matrix = torch.matmul(inputs, torch.transpose(inputs, 1, 2))  # a_matrix = batch x neuron x neuron
        a_matrix = torch.sigmoid(a_matrix)
        assert torch.abs(torch.max(a_matrix-a_matrix.transpose(1,2)))<0.0001
        return a_matrix

    @staticmethod
    def matrix_2_graph_prob(inputs):
        inputs = (inputs+inputs.transpose(1,2))/2
        assert torch.equal(inputs, inputs.transpose(1,2))
        return torch.sigmoid(inputs)

    @staticmethod   
    def normalize(mx):
        """Row-normalize sparse matrix"""
        row_sum = torch.sum(mx, 2)
        row_sum_inv = torch.pow(row_sum,-0.5)
        diag_m = torch.diag_embed(row_sum_inv)
        mx = torch.matmul(torch.matmul(diag_m, mx),diag_m)
        return mx

    @staticmethod
    def get_well_loss(graph):
        return torch.mean(graph**2*(1-graph)**2)

    def forward(self, inputs, return_rates=False, return_graph=False):
        assert inputs.shape[1]==self.time, "data time span does not match the model"

        # left part
        # inputs = F.dropout(inputs.float(), p=self.dropout_prob, training=self.training)

        '''
        In encoderGRU_1 and encoderGRU_2 we utilized GRU.
        input x:     batch x neurons x dim_e
        hidden h:    batch x neurons x dim_e
        r and u:     batch x neurons x dim_e
        c and new h: batch x neurons x dim_e
        
        Two linear map is cat(x, h) -> cat(r, u) and cat(x, r*h) -> c
        which are: (dim_e+1) -> 2*dim_e and (dim_e+1) -> dim_e
        '''
        enc_outs_forward_1, enc_outs_backward_1 = self.encoderGRU_1(inputs)  # both lists of: batch x neurons x dim_e
        in_graph_0 = torch.cat((enc_outs_forward_1[-1], enc_outs_backward_1[0]), dim=2)
        in_graph_0 = F.dropout(in_graph_0, p=self.dropout_prob, training=self.training)
        graph_0_prob = self.vector_2_graph_prob(in_graph_0)  # batch x neurons x neurons
        graph_0 = self.prob_2_graph_sample(graph_0_prob)
        graph_0 = self.normalize(graph_0)

        # right part
        enc_outs_forward_2, enc_outs_backward_2 = self.encoderGRU_2(inputs)
        graph_state = graph_0_prob # 2021.4.6
        # graph_state = graph_0
        # graph_state = F.dropout(graph_state, p=self.dropout_prob, training=self.training)
        graph_rnn_state = torch.zeros(inputs.shape[0], self.neurons, self.dim_c)
        log_rates_list = [None] * self.time
        graph_prob_list = [None] * self.time
        d_post_list = [None] * self.time
        loglikelihood_list = [None] * self.time
        for t in range(self.time):
            # step1: generate e_t
            forward_t = t - self.time_lag
            backward_t = t + self.time_lag
            sigma_in_t_forward = enc_outs_forward_2[forward_t] if forward_t >= 0 \
                else torch.zeros_like(enc_outs_forward_2[0])
            sigma_in_t_backward = enc_outs_backward_2[backward_t] if backward_t < self.time \
                else torch.zeros_like(enc_outs_backward_2[0])
            e_t = torch.cat((sigma_in_t_forward, sigma_in_t_backward), dim=2)  # batch x neurons x 2*dim_e
            # e_t = F.dropout(e_t, p=self.dropout_prob, training=self.training)

            # step2: generate graph
            if self.dynamic_graph:
                sigma_t = self.vector_2_graph_prob(e_t)  # batch x neurons x neurons
                '''
                In graph_gru_cell we utilized GRU.
                input x:     batch x neurons x neurons
                hidden h:    batch x neurons x neurons
                r and u:     batch x neurons x neurons
                c and new h: batch x neurons x neurons

                Two linear map is cat(x, h) -> cat(r, u) and cat(x, r*h) -> c
                which are: 2*neurons -> 2*neurons and 2*neurons -> neurons
                '''
                graph_out, graph_state = self.graph_gru_cell(sigma_t, graph_state)  # RNN unit
                # graph_state = F.dropout(graph_state, p=self.dropout_prob, training=self.training)
                graph_prob_list[t] = self.matrix_2_graph_prob(graph_out)
                graph = self.prob_2_graph_sample(graph_prob_list[t])  # batch x neurons x neurons
                # graph = self.normalize(graph)  # normalization
            else:
                graph_prob = graph_0_prob

            # step3: graph rnn
            '''
            In graph_rnn_gru_cell we utilized GRU.
            input x:     batch x neurons x 2*dim_e
            hidden h:    batch x neurons x dim_c
            r and u:     batch x neurons x dim_c
            c, new h and output: batch x neurons x dim_c

            Two linear map is cat(x, h) -> cat(r, u) and cat(x, r*h) -> c
            which are: 2*dim_e+dim_c -> 2*dim_c and 2*dim_e+dim_c -> dim_c
            '''
            graph_prob = self.normalize(graph_prob) # added on 2021.4.21
            u_prob, graph_rnn_state = self.graph_rnn_gru_cell(e_t, graph_rnn_state, graph_prob)  # Graph RNN
            # graph_rnn_state = F.dropout(graph_rnn_state, p=self.dropout_prob, training=self.training)

            # step4: sample
            graph = self.prob_2_graph_sample(graph_0_prob)
            graph = self.normalize(graph)
            d_mean = self.w_mean(u_prob)
            d_log_var = self.w_var(u_prob)
            d_post_list[t] = DiagonalGaussian(d_mean, d_log_var)
            d_sample = d_post_list[t].sample

            # step5: GCN
            x = d_sample
            for layer in self.gcn_layers:
                # x = F.sigmoid(F.dropout(layer(x, graph), p=self.dropout_prob, training=self.training))
                x = torch.tanh(layer(x, graph))
                # x = torch.sigmoid(layer(x, graph))
                # x = F.relu(layer(x, graph))

            # step6: collect log rates
            x = self.w_out(x)
            x = (torch.tanh(x)+1)/2
            log_rates_list[t] = torch.log(x+1e-40)
            loglikelihood_list[t] = Poisson(log_rates_list[t]).logp(inputs[:,t,:])

        log_rates = torch.cat(log_rates_list, dim=2)
        log_rates = torch.transpose(log_rates, 1, 2)

        # return rates to valid
        if return_rates:
            return torch.exp(log_rates)

        # loss 
        # PoissonNLLLoss
        # poisson_nll_loss = self.poisson_loss(log_rates, inputs)
        poisson_nll_loss = torch.mean(torch.cat(loglikelihood_list,dim=2))
        # KL loss between the prior distribution and posterior distribution
        # print('graph_0_prob', graph_0_prob[0][0])
        # print('self.prior_graph_0()', self.prior_graph_0().shape, self.prior_graph_0()[0][0])
        kl_loss_graph_0 = self.kl_loss_1(graph_0_prob,self.prior_graph_0())
        # kl_loss_graph_0 = nn.MSELoss()(graph_0_prob,self.prior_graph_0())
        # kl_loss_graph = 0
        # for t in range(self.time):
        #     if t==0:
        #         kl_loss_graph += self.kl_loss_graph[t](graph_prob_list[t], graph_0_prob)
        #         print(self.kl_loss_graph[t](graph_prob_list[t], graph_0_prob).item())
        #     else:
        #         kl_loss_graph += self.kl_loss_graph[t](graph_prob_list[t], graph_prob_list[t-1])
        #         print(self.kl_loss_graph[t](graph_prob_list[t], graph_prob_list[t-1]).item())
        # kl_loss_graph = kl_loss_graph/self.time
        kl_cost_d = self.kl_loss_2(d_post_list, self.prior_d)

        # well loss if discrete graph
        well_loss_graph_0 = torch.zeros(1).to(device)
        if self.discrete_graph:
            well_loss_graph_0 = self.get_well_loss(self.prior_graph_0())

        loss = \
        - poisson_nll_loss \
        + kl_loss_graph_0 * self.kl_weight_1  \
        + kl_cost_d * self.kl_weight_2 \
        + well_loss_graph_0 * 5 
        
        # wait for modify # - kl_loss_graph * self.kl_weight_1

        # print('poisson loss: ',poisson_nll_loss.item(), \
        #     'kl loss graph 0:', kl_loss_graph_0.item(), \
        #         'kl loss d:', kl_cost_d.item(), \
        #             'well loss graph 0:', well_loss_graph_0.item())

        return loss

    def get_prior_graph(self):
        if self.dynamic_graph:
            return self.prior_graph_0() # todo
        else:
            return self.prior_graph_0()
        
    
