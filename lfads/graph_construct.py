# -*- coding: utf-8 -*-
"""
@Time    : 7/31/20 2:25 PM
@Author  : Lucius
@FileName: graph_construct.py
@Software: PyCharm
"""

import numpy as np
import os
import tensorflow as tf
from gru import linear_map


def adjacency_generate(a, xi1, xi2):
    tau = 1e-8
    a_ret = 1 / (1 + tf.exp((tf.log(1 - a) + xi2) / tau - (tf.log(a) + xi1) / tau))
    return a_ret


class Graph(object):
    """Base class for Graph classes."""
    pass


class GraphFromInput(Graph):
    def __init__(self, data_in):
        """
        The data_in is the input with dim of
        batch x node_num x 2de
        node_num is equal to feature dim for each batch
        """

        shape = data_in.get_shape().as_list()  # [batch, node_num, 2*de]
        batch_size = shape[0]
        node_num = shape[1]

        # get a_ij as an matrix
        # data_in = batch x node_num x 2de x 1
        a_matrix = tf.matmul(data_in, tf.transpose(data_in, [0, 2, 1]))
        a_matrix = tf.sigmoid(a_matrix)

        self.prob_matrix = a_matrix

        # generate noise following gumbel distribution
        noise1 = -tf.log(-tf.log(tf.random_uniform([batch_size, node_num, node_num], 0, 1)))
        noise2 = -tf.log(-tf.log(tf.random_uniform([batch_size, node_num, node_num], 0, 1)))
        self.sample_matrix = adjacency_generate(a_matrix, noise1, noise2)

    @property
    def prob(self):
        return self.prob_matrix

    @property
    def sample(self):
        return self.sample_matrix


class GraphFromProb(Graph):
    def __init__(self, data_m):
        """
        The data_in is the input with dim of
        batch x node_num x 2de
        node_num is equal to feature dim for each batch
        """

        self.prob_matrix = tf.sigmoid(data_m)

        # generate noise following gumbel distribution
        noise1 = -tf.log(-tf.log(tf.random_uniform(data_m.get_shape().as_list(), 0, 1)))
        noise2 = -tf.log(-tf.log(tf.random_uniform(data_m.get_shape().as_list(), 0, 1)))

        self.sample_matrix = adjacency_generate(self.prob_matrix, noise1, noise2)

    @property
    def prob(self):
        return self.prob_matrix

    @property
    def sample(self):
        return self.sample_matrix


class LearnableGraph(Graph):
    def __init__(self, batch_size, node_num, name):
        """
        batch_size: batch size of data
        node_num: number of nodes in the graph
        """
        a_matrix = tf.get_variable(name=(name + "/prior_graph0"), shape=[1, node_num, node_num],
                                   initializer=tf.random_normal_initializer)
        a_matrix = tf.sigmoid(a_matrix)

        batch_matrix = tf.tile(a_matrix, [batch_size, 1, 1])
        self.prob_matrix = batch_matrix

        # generate noise following gumbel distribution
        noise1 = -tf.log(-tf.log(tf.random_uniform([batch_size, node_num, node_num], 0, 1)))
        noise2 = -tf.log(-tf.log(tf.random_uniform([batch_size, node_num, node_num], 0, 1)))

        self.batch_matrix = adjacency_generate(batch_matrix, noise1, noise2)

    @property
    def prob(self):
        return self.prob_matrix

    @property
    def sample(self):
        return self.batch_matrix


class KLCost_BinaryBinary(object):
    def __init__(self, posterior, prior):
        kl_b = 0.0
        for post_z, prior_z in zip(posterior, prior):
            assert isinstance(post_z, Graph)
            assert isinstance(prior_z, Graph)
            p = post_z.prob_matrix
            q = prior_z.prob_matrix
            kl_b += tf.reduce_sum(p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q)), [1, 2])

            self.kl_cost_b = kl_b
            self.kl_cost = tf.reduce_mean(kl_b)


class GraphGaussianFromInput(object):
    def __init__(self, c, name):
        """

        Args:
            c: batch x num_node x dim
        """

        shape = c.get_shape()
        dim = shape.as_list()[2]
        mean = linear_map(c, dim, name=name + '_2_mean')
        log_var = linear_map(c, dim, name=name + '_2_var')

        self.noise = noise = tf.random_normal(shape)
        self.sample_bxn = mean + tf.exp(0.5 * log_var) * noise

    @property
    def sample(self):
        return self.sample_bxn


def graph_convolution(adj, inputs, hidden_dims, output_dim, bias=True):
    """

    Args:
        adj: adjacent matrix, batch x num_node x num_node
        inputs: batch x num_node x feature_dim
        hidden_dim: a list of hidden dim
        output_dim: feature dim of output
        bias: if consider bias

    Returns:

    """

    num_node = int(inputs.get_shape()[1])
    mat_init = tf.random_normal_initializer(0.0, 1.)

    normalized_adj = tf.matmul((tf.matrix_diag(1/(tf.reduce_sum(adj, 2)+1e-8))), adj)
    # normalized_adj = adj

    previous_dim = int(inputs.get_shape()[2])
    features = inputs

    all_dims = hidden_dims[:]
    all_dims.append(output_dim)
    for index, hd in enumerate(all_dims):
        w = tf.get_variable('gcn_W_' + str(index), [previous_dim, hd], initializer=mat_init)
        af = tf.matmul(normalized_adj, features)
        af = tf.reshape(af, [-1, previous_dim])  # reshape
        if bias:
            b = tf.get_variable('gcn_b_' + str(index), [1, hd], initializer=mat_init)
            new_features = tf.matmul(af, w) + b
        else:
            new_features = tf.matmul(af, w)
        new_features = tf.reshape(new_features, [-1, num_node, hd])  # reshape

        new_features = tf.nn.tanh(new_features)

        # update
        previous_dim = hd
        features = tf.tan(new_features)

    return features
