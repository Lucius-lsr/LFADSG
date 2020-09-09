# -*- coding: utf-8 -*-
"""
@Time    : 8/6/20 1:17 PM
@Author  : Lucius
@FileName: gru.py
@Software: PyCharm
"""

import tensorflow as tf
import numpy as np


def linear_map(inputs, out_dim, do_bias=True, alpha=1.0, name=None, collections=None, mat_init_value=None,
               bias_init_value=None):
    """

    Args:
        inputs: the input 3D tensor with batch x num_unit x in_dim.
        out_dim: multiply inputs (ignore batch) with a 2D tensor in_dim x out_dim.
        alpha: weights are scaled by alpha/sqrt(#inputs), with alpha being the weight scale.
        name: name for the variables.
        collections: List of additional collections variables should belong to.

    Returns:
        In the equation, y = x W + b, returns the tensorflow op that yields y.
    """

    num_unit = int(inputs.get_shape()[1])
    in_dim = int(inputs.get_shape()[2])

    if mat_init_value is not None and mat_init_value.shape != (in_dim, out_dim):
        raise ValueError(
            'Provided mat_init_value must have shape [%d, %d].' % (in_dim, out_dim))
    if bias_init_value is not None and bias_init_value.shape != (1, out_dim):
        raise ValueError(
            'Provided bias_init_value must have shape [1,%d].' % (out_dim,))

    mat_init = None
    if mat_init_value is None:
        stddev = alpha / np.sqrt(float(num_unit * in_dim))  # is the equation right?
        mat_init = tf.random_normal_initializer(0.0, stddev)

    w_name = (name + "/W") if name else "/W"

    w_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    if collections:
        w_collections += collections
    if mat_init_value is not None:
        w = tf.Variable(mat_init_value, name=w_name, collections=w_collections)
    else:
        w = tf.get_variable(w_name, [in_dim, out_dim], initializer=mat_init, collections=w_collections)

    b = None
    if do_bias:
        b_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        if collections:
            b_collections += collections
        b_name = (name + "/b") if name else "/b"
        if bias_init_value is None:
            b = tf.get_variable(b_name, [1, out_dim], initializer=tf.zeros_initializer(), collections=b_collections)
        else:
            b = tf.Variable(bias_init_value, name=b_name, collections=b_collections)

    inputs = tf.reshape(inputs, [-1, in_dim])
    if do_bias:
        ret = tf.matmul(inputs, w) + b
    else:
        ret = tf.matmul(inputs, w)
    ret = tf.reshape(ret, [-1, num_unit, out_dim])

    return ret


class EncoderGRU(object):
    """
    Gated Recurrent Unit cell to update E_t).

    """

    def __init__(self, de, forget_bias=1.0, weight_scale=1.0,
                 clip_value=np.inf, collections=None):
        """Create a GRU object.

        Args:
          num_units: Number of units in the GRU
          forget_bias (optional): Hack to help learning.
          weight_scale (optional): weights are scaled by ws/sqrt(#inputs), with
           ws being the weight scale.
          clip_value (optional): if the recurrent values grow above this value,
            clip them.
          collections (optional): List of additonal collections variables should
            belong to.
        """
        self._de = de
        self._forget_bias = forget_bias
        self._weight_scale = weight_scale
        self._clip_value = clip_value
        self._collections = collections

    @property
    def state_size(self):
        return self._de

    @property
    def output_size(self):
        return self._de

    @property
    def state_multiplier(self):
        return 1

    def output_from_state(self, state):
        """Return the output portion of the state."""
        return state

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) function.

        Args:
          inputs: A 3D batch x num_unit x 1 tensor of inputs.
          state: The previous state from the last time step. It is a 3D tensor batch x num_unit x de
          scope (optional): TF variable scope for defined GRU variables.

        Returns:
          A tuple (state, state), where state is the newly computed state at time t.
          It is returned twice to respect an interface that works for LSTMs.
        """

        x = inputs
        h = state
        if inputs is not None:
            xh = tf.concat(axis=2, values=[x, h])
        else:
            xh = h

        with tf.variable_scope(scope or type(self).__name__):  # "GRU"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                # xh = batch x num_unit x (1+de) and [r,u] = batch x num_unit x 2*de
                r, u = tf.split(axis=2, num_or_size_splits=2, value=linear_map(xh, 2 * self._de,
                                                                               alpha=self._weight_scale,
                                                                               name="x_2_ru",
                                                                               collections=self._collections))
                r, u = tf.sigmoid(r), tf.sigmoid(u + self._forget_bias)
            with tf.variable_scope("Candidate"):
                # x = batch x num_unit x 1, r = batch x num_unit x de, h = batch x num_unit x de
                xrh = tf.concat(axis=2, values=[x, r * h])
                # c = batch x num_unit x de, xrh = batch x num_unit x (1+de)
                c = tf.tanh(linear_map(xrh, self._de, name="xrh_2_c", collections=self._collections))
            new_h = u * h + (1 - u) * c
            new_h = tf.clip_by_value(new_h, -self._clip_value, self._clip_value)

        return new_h, new_h


class ControllerGRU(object):
    """
    Gated Recurrent Unit cell to update Z_t).

    """

    def __init__(self, forget_bias=1.0, weight_scale=1.0,
                 clip_value=np.inf, collections=None):
        """
        Create a GRU object.

        Args:
          forget_bias (optional): Hack to help learning.
          weight_scale (optional): weights are scaled by ws/sqrt(#inputs), with
           ws being the weight scale.
          clip_value (optional): if the recurrent values grow above this value,
            clip them.
          collections (optional): List of additonal collections variables should
            belong to.
        """
        self._forget_bias = forget_bias
        self._weight_scale = weight_scale
        self._clip_value = clip_value
        self._collections = collections

    @property
    def state_multiplier(self):
        return 1

    def output_from_state(self, state):
        """Return the output portion of the state."""
        return state

    def __call__(self, inputs, state, scope=None):
        """
        Gated recurrent unit (GRU) function.

        Args:
          inputs: A 3D batch x num_unit x num_unit tensor of inputs.
          state: The previous state from the last time step. It is a 3D tensor batch x num_unit x num_unit
          scope (optional): TF variable scope for defined GRU variables.

        Returns:
          A tuple (state, state), where state is the newly computed state at time t.
          It is returned twice to respect an interface that works for LSTMs.
        """

        num_unit = int(inputs.get_shape()[1])

        x = inputs
        h = state
        if inputs is not None:
            xh = tf.concat(axis=2, values=[x, h])
        else:
            xh = h

        with tf.variable_scope(scope or type(self).__name__):  # "GRU"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                # xh = batch x num_unit x 2*num_unit and [r,u] = batch x num_unit x 2*num_unit
                r, u = tf.split(axis=2, num_or_size_splits=2, value=linear_map(xh, 2 * num_unit,
                                                                               alpha=self._weight_scale,
                                                                               name="x_2_ru",
                                                                               collections=self._collections))
                r, u = tf.sigmoid(r), tf.sigmoid(u + self._forget_bias)
            with tf.variable_scope("Candidate"):
                # x = batch x num_unit x num_unit, r = batch x num_unit x num_unit, h = batch x num_unit x num_unit
                xrh = tf.concat(axis=2, values=[x, r * h])
                # c = batch x num_unit x num_unit, xrh = batch x num_unit x 2*num_unit
                c = tf.tanh(linear_map(xrh, num_unit, name="xrh_2_c", collections=self._collections))
            new_h = u * h + (1 - u) * c
            new_h = tf.clip_by_value(new_h, -self._clip_value, self._clip_value)

        return new_h, new_h


class GraphGRU(object):
    """
    Gated Recurrent Unit cell to update Z_t).

    """

    def __init__(self, dc, forget_bias=1.0, weight_scale=1.0,
                 clip_value=np.inf, collections=None):
        """
        Create a GRU object.

        Args:
          forget_bias (optional): Hack to help learning.
          weight_scale (optional): weights are scaled by ws/sqrt(#inputs), with
           ws being the weight scale.
          clip_value (optional): if the recurrent values grow above this value,
            clip them.
          collections (optional): List of additonal collections variables should
            belong to.
        """
        self._dc = dc
        self._forget_bias = forget_bias
        self._weight_scale = weight_scale
        self._clip_value = clip_value
        self._collections = collections

    @property
    def state_multiplier(self):
        return 1

    def output_from_state(self, state):
        """Return the output portion of the state."""
        return state

    def __call__(self, inputs, graph, state, scope=None):
        """
        Gated recurrent unit (GRU) function.

        Args:
          inputs: A 3D batch x num_unit x de tensor of inputs.
          graph: The graphic structure of this graph RNN. It is a 3D binary tensor batch x num_unit x num_unit
          state: The previous state from the last time step. It is a 3D tensor batch x num_unit x dc
          scope (optional): TF variable scope for defined GRU variables.

        Returns:
          A tuple (state, state), where state is the newly computed state at time t.
          It is returned twice to respect an interface that works for LSTMs.
        """

        x = inputs
        h = state
        if inputs is not None:
            xh = tf.concat(axis=2, values=[x, h])
        else:
            xh = h

        with tf.variable_scope(scope or type(self).__name__):  # "GRU"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                # xh = batch x num_unit x (2*de+dc) and [r,u] = batch x num_unit x 2*dc
                r, u = tf.split(axis=2, num_or_size_splits=2, value=linear_map(tf.matmul(graph, xh), 2 * self._dc,
                                                                               alpha=self._weight_scale,
                                                                               name="e_2_ru",
                                                                               collections=self._collections))
                r, u = tf.sigmoid(r), tf.sigmoid(u + self._forget_bias)
            with tf.variable_scope("Candidate"):
                # x = batch x num_unit x 2*de, r = batch x num_unit x dc, h = batch x num_unit x dc
                xrh = tf.concat(axis=2, values=[x, r * h])
                # c = batch x num_unit x dc, xrh = batch x num_unit x (2*de+dc)
                c = tf.tanh(linear_map(tf.matmul(graph, xrh), self._dc, name="xrh_2_c", collections=self._collections))
            new_h = u * h + (1 - u) * c
            new_h = tf.clip_by_value(new_h, -self._clip_value, self._clip_value)

        return new_h, new_h
