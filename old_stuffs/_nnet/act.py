# -*- coding: utf-8 -*-
"""
===============================================================================
Activation functions
===============================================================================
Created on Sat Dec  5 20:32:33 2015

@author: takanori
"""

import theano.tensor as T
import numpy as np


def sigmoid(z):
    ''' sigmoid activation function '''
    return 1. / (1. + T.exp(-1. * z))


def tanh(z):
    ''' hyperbolic tangent activation function '''
    c = T.exp(z)
    c_ = T.exp(-1. * z)
    return (c - c_) / (c + c_)


def softmax(z):
    ''' softmax activation function '''
    max_v = T.max(z, axis=1).dimshuffle(0, 'x')
    log_sum = T.log(T.sum(T.exp(z - max_v), axis=1)).dimshuffle(0, 'x') + max_v
    return T.exp(z - log_sum)


def reLU(z):
    ''' rectified linear activation function '''
    return 0.5 * (z + abs(z))


def np_sigmoid(z):
    ''' numpy sigmoid activation function '''
    return 1. / (1. + np.exp(-1. * z))