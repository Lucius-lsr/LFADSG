# -*- coding: utf-8 -*-
"""
@Time    : 9/7/20 2:35 PM
@Author  : Lucius
@FileName: model.py
@Software: PyCharm
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

        # x = batch x neurons x 1, r = batch x neurons x de, h = batch x neurons x de
        xrh = torch.cat([x, r * h], dim=2)
        # c = batch x neurons x map2[1], xrh = batch x neurons x map2[0]
        c = self.mat_b(xrh) if graph is None else self.mat_b(torch.matmul(graph, xrh))  # normal RNN or graph RNN
        c = torch.tanh(c)

        new_h = u * h + (1.0 - u) * c
        new_h = torch.clamp(new_h, -self.clip_value, self.clip_value)
        return new_h, new_h


class EnCoder(nn.Module):
    def __init__(self, cell, time, neurons, dim):
        super(EnCoder, self).__init__()
        self.cell = cell((dim + 1, 2 * dim), (dim + 1, dim), clip_value=10000.0)
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
            enc_out, enc_state = self.cell(input_t, enc_state)  # enc_out==enc_hidden: batch x neurons x dim_e
            enc_outs_forward[t] = enc_out

        enc_state = torch.zeros([inputs.shape[0], inputs.shape[2], self.dim])  # enc_hidden: batch x neurons x dim_e
        enc_outs_backward = [None] * self.time
        for t in range(self.time - 1, -1, -1):
            input_t = inputs[:, t, :]
            enc_out, enc_state = self.cell(input_t, enc_state)
            enc_outs_backward[t] = enc_out

        return enc_outs_forward, enc_outs_backward


class LearnableGraph(nn.Module):
    def __init__(self, node_num):
        super(LearnableGraph, self).__init__()
        A = torch.randn((1, node_num, node_num))
        self.A = torch.nn.Parameter(A, requires_grad=True)
        self.register_parameter("adjacency", self.A)

    def forward(self):
        a = torch.sigmoid(self.A)
        return a


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.w = nn.Linear(in_features, out_features)

    def forward(self, inputs, adj):
        support = self.w(inputs)
        output = torch.matmul(adj, support)
        return output


class LFADSG(nn.Module):
    def __init__(self, time, neurons, dim_e, dim_c, dim_d, gcn_hidden):
        super(LFADSG, self).__init__()
        # hyper parameters
        self.dim_e = dim_e
        self.dim_c = dim_c
        self.dim_d = dim_d
        self.time = time
        self.neurons = neurons
        self.time_lag = 1
        self.dropout_prob = 0.95

        # parameters
        self.encoderGRU_1 = EnCoder(GRUCell, time, neurons, dim_e)
        self.encoderGRU_2 = EnCoder(GRUCell, time, neurons, dim_e)
        self.graph_gru_cell = GRUCell((2 * neurons, 2 * neurons), (2 * neurons, neurons))  # generate graph
        self.graph_rnn_gru_cell = GRUCell((2 * dim_e + dim_c, 2 * dim_c), (2 * dim_e + dim_c, dim_c))  # graph rnn
        self.w_mean = nn.Linear(dim_c, dim_d)
        self.w_var = nn.Linear(dim_c, dim_d)

        self.gcn_layers = list()
        previous_hidden = dim_d
        for n_hidden in gcn_hidden:
            self.gcn_layers.append(GraphConvolution(previous_hidden, n_hidden).to(device))
            previous_hidden = n_hidden

        self.w_out = nn.Linear(gcn_hidden[-1], 1)

        # define noise
        self.noise1 = None
        self.noise2 = None
        self.noise3 = None

    def update_noise(self):  # will be called before every forward
        # generate noise
        self.noise1 = -torch.log(-torch.log(torch.rand([self.neurons, self.neurons], device=device)))
        self.noise2 = -torch.log(-torch.log(torch.rand([self.neurons, self.neurons], device=device)))
        self.noise3 = -torch.log(-torch.log(torch.rand([self.neurons, self.dim_d], device=device)))

    def prob_2_graph_sample(self, prob_a_matrix):
        def adjacency_generate(a, xi1, xi2):
            tau = 0.001
            tmp = (torch.log(a) + xi1) - (torch.log(1.0 - a) + xi2)
            tmp = tmp / tau
            a_ret = torch.sigmoid(tmp)
            return a_ret

        sample_graph = adjacency_generate(prob_a_matrix, self.noise1, self.noise2)
        return sample_graph

    @staticmethod
    def vector_2_graph_prob(inputs):
        # inputs = batch x neuron x 2*dim_e
        a_matrix = torch.matmul(inputs, torch.transpose(inputs, 1, 2))  # a_matrix = batch x neuron x neuron
        a_matrix = torch.sigmoid(a_matrix)
        return a_matrix

    @staticmethod
    def matrix_2_graph_prob(inputs):
        return torch.sigmoid(inputs)

    def forward(self, inputs):

        self.update_noise()

        # left part
        inputs = F.dropout(inputs.float(), p=self.dropout_prob, training=self.training)

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

        # right part
        enc_outs_forward_2, enc_outs_backward_2 = self.encoderGRU_2(inputs)
        # print(inputs, enc_outs_forward_2[0])
        graph_state = graph_0_prob
        graph_rnn_state = torch.zeros(inputs.shape[0], self.neurons, self.dim_c)
        log_rates_list = [None] * self.time
        for t in range(self.time):
            # step1: generate e_t
            forward_t = t - self.time_lag
            backward_t = t + self.time_lag
            sigma_in_t_forward = enc_outs_forward_2[forward_t] if forward_t >= 0 \
                else torch.zeros_like(enc_outs_forward_2[0])
            sigma_in_t_backward = enc_outs_backward_2[backward_t] if backward_t < self.time \
                else torch.zeros_like(enc_outs_forward_2[0])
            e_t = torch.cat((sigma_in_t_forward, sigma_in_t_backward), dim=2)  # batch x neurons x 2*dim_e

            # step2: generate graph
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
            graph_out = self.matrix_2_graph_prob(graph_out)
            graph = self.prob_2_graph_sample(graph_out)  # batch x neurons x neurons

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
            graph_rnn_out, graph_rnn_state = self.graph_rnn_gru_cell(e_t, graph_rnn_state, graph)  # Graph RNN

            # step4: sample
            d_mean = self.w_mean(graph_rnn_out)
            d_log_var = self.w_var(graph_rnn_out)
            d_sample = d_mean + torch.exp(0.5 * d_log_var) * self.noise3

            # step5: GCN
            x = d_sample
            for layer in self.gcn_layers:
                x = F.sigmoid(layer(x, graph))

            x = self.w_out(x)
            x = F.sigmoid(x)
            log_rates_list[t] = torch.log(x)

        rates = torch.cat(log_rates_list, dim=2)
        rates = torch.transpose(rates, 1, 2)

        return rates
