# -*- coding: utf-8 -*-
"""
===============================================================================
Keep my neural-net and deep-learning functionality in this script
-------------------------------------------------------------------------------
06/06/2016 - imported codes from: ~/sbia_work/python/analysis/dl/util.py
-------------------------------------------------------------------------------

https://github.com/avasbr/nnet_theano
===============================================================================
Created on Fri Apr  8 21:57:42 2016

@author: takanori
"""
import numpy as np
import pandas as pd
import scipy as sp
import time
import os
import sys
import matplotlib.pyplot as plt
import cPickle as pickle

#from collections import OrderedDict

import theano
import theano.tensor as T

import tak as tw

import lasagne
from lasagne.layers import (
    InputLayer,
    DropoutLayer,
    DenseLayer,
)
#%% ***** lasagne related functions *****
def las_get_objective(network, target_tn, loss, l2=None, l1=None):
    """ Get training objective function for optimization.

    Used with ``las_train_nesterov``

    Parameters
    -----------
    network : lasagne Layer
        Lasagne network
    target_tn : theano.tensor.TensorType (Theano symbolic tensor)
        Theano tensor variable representing target variable
    loss : TensorType(floatX, matrix)
        Tensor scalar expression for the loss function.
        Many popular ones readily available from ``lasagne.objectives``
        (eg, ``binary_crossentropy``, ``squared_error``,``categorical_crossentropy``).
        Can also create your own expression using functions from ``T.nnet``
        (in fact, that's what ``lasagne`` does.
    l2 : float (or None)
        Amount of l2 regularization to apply (optional, default=None)
    l1 : float (or None)
        Amount of l1 regularization to apply (optional, default=None)

    Returns
    -------
    objective : TensorType(floatX, matrix)
        Theano scalar expression for the objective function (loss + penalty)
    params : list
        List of parameter variables

    Warning
    -------
    Remember that the prediction function during training heeds to have
    ``deterministic`` set to False

    History
    --------
    - 04/20/2016 - Code forked from ``get_objective_0418`` from util_dl.py
    """
    # for training, set deterministic to False
    predict_tr_tn = lasagne.layers.get_output(network,deterministic=False)
    params        = lasagne.layers.get_all_params(network, trainable=True)

    # define loss, params, updates, and number of total parameters
    objective = loss(predict_tr_tn,target_tn).mean()

    # === add regularization === #
    if isinstance(l2,int) or isinstance(l2,float):
        objective += l2*las_get_weight_penalty(network,'l2')

    if isinstance(l1,int) or isinstance(l1,float):
        objective += l1*las_get_weight_penalty(network,'l1')

    return objective, params


def las_train_nesterov(X,y,in_var,target,objective,params,
                             lr=1.0,momentum=0.95,
                             num_epochs=100, batch_size=100,
                             disp_freq=10,lr_decay=None):
    """ Train network via Nesterov's method

    Used with ``las_get_objective``

    Parameters
    ----------
    X : ndarray of shape=(num_samples,num_features)
        Input data matrix
    y : ndarray or None
        Target label vector. if ``None``, function assumes we're training
        an auto-encoder (ie, reconstruct X from hidden representation)
    in_var : theano.tensor.TensorType (Theano symbolic tensor)
        Theano symbolic tensor variable representing input variable
    target : theano.tensor.TensorType (Theano symbolic tensor)
        Theano symbolic tensor variable representing target variable
    objective : TensorType(floatX, matrix)
        Theano scalar expression for the objective function
        (loss + penalty; output from ``las_get_objective``)
    params : list
        List of parameter variables (output from ``las_get_objective``)

    Usage
    -----
    >>> in_var = T.matrix('in_var')
    >>> target = T.ivector('target')
    >>> loss = lasagne.objectives.squared_error
    >>> objective,params = util.las_get_objective(network, target, loss=loss)
    >>> errors = util.las_train_nesterov(Xtr,in_var,target,objective,
    >>>                                  params, disp_freq=10,num_epochs=500)

    History
    -------
    - Created 04/19/2016: forked from train_recon_net_nesterov
      (replaced variable name ``loss`` with ``objective``)

    Dev
    ---
    - ``t_419_sdae_greedy4_dropout_mlp.py``
    """
    lr = theano.shared(np.array(lr, dtype=theano.config.floatX),name='lr')
    updates = lasagne.updates.nesterov_momentum(objective, params,lr.get_value(),
                                                momentum)
    # compile training function
    train_fn = theano.function([in_var,target],objective,updates=updates)

    err_=[]
    start_time = time.time()
    if lr_decay is not None:
        #print lr.get_value()
        # need to get access to shared variable for learning rate
        lr_decay = np.asarray(lr_decay,dtype=theano.config.floatX)

    for epoch in range(num_epochs):
        train_err = 0
        try:
            if y is None:
                """ Self-reconstruction """
                for batch in iterate_minibatches(X,X,batch_size,shuffle=True):
                    inputs, _ = batch
                    train_err += train_fn(inputs,inputs)  # <- where model gets updated
            else:
                for batch in iterate_minibatches(X,y,batch_size,shuffle=True):
                    inputs, outputs = batch
                    train_err += train_fn(inputs,outputs) # <- where model gets updated
        except KeyboardInterrupt:
            break
        err_.append(train_err)

        if lr_decay is not None:
            # update learning-rate
            lr_now = lr.get_value()*lr_decay
            lr.set_value(lr_now)
            #print 'lr={:.2e}'.format(float(lr.get_value())),

        if epoch % disp_freq == 0:
            time_ = time.time() - start_time
            if lr_decay is not None: print 'lr={:.2e}'.format(float(lr.get_value())),
            print "epoch = {:3}, err = {:6.5f} ({:5.2f} sec)".format(epoch, train_err,time_)
            sys.stdout.flush()
    return np.asanyarray(err_)



#%% === lasagne "convenience" function ===
def las_get_weight_penalty(network, penalty_type='l2'):
    """ Get theano expression of L1 or L2 penalty on whole network

    Get theano scalar expression in the form of L2 weight decay

    Parameters
    ----------
    network : ``lasagne.layers.Layer`` instance
        Parameters of this layer and all layers below it will be penalized.
    penalty_type : 'l2' or 'l1'
        Type of penalty

    Returns
    -------
    penalty : TensorType(floatX, matrix)
        Tensor scalar expression for the penalty

    Usage
    -----
    >>> network, loss, predict, encoder = dae(input,target,n_hid=500)
    >>> loss = loss + get_weight_penalty(network,'l2')

    References
    -------
    - http://luizgh.github.io/libraries/2015/12/08/getting-started-with-lasagne/
    - http://lasagne.readthedocs.org/en/latest/modules/regularization.html

    History
    -------
    Created 04/13/2016
    """
    if penalty_type.lower() == 'l2':
        pen = lasagne.regularization.l2
    elif penalty_type.lower() == 'l1':
        pen = lasagne.regularization.l1
    else:
        raise ValueError('Supplied penalty "{}" not recognized!'.format(penalty_type))

    penalty = lasagne.regularization.regularize_network_params(network,pen)
    return penalty


def las_get_param_weight(network):
    """ Get weight parameter matrix in ndarray form

    created 04/19/2016
    """
    # so far, all the network with parameter contains [W,b], so just select
    # the first element corresponding to the weight matrix
    return network.get_params()[0].get_value()

def las_get_param_bias(network):
    """ Get bias parameter vector in ndarray form

    created 04/19/2016
    """
    # so far, all the network with parameter contains [W,b], so just select
    # the 2nd element corresponding to the bias vector
    return network.get_params()[1].get_value()


def las_get_input_var(network):
    """Return theano input variable expression of lasagne network (created 4/18/2016)"""
    return lasagne.layers.get_all_layers(network)[0].input_var

def las_get_output_fn(network,in_var,deterministic=False):
    """ Get output function from a network

    Basically the final activation output from the final layer

    Parameters
    ----------
    network : ``lasagne.layers.Layer`` instance
        Parameters of this layer and all layers below it will be penalized.
    in_var : theano.tensor.TensorType
        a symbolic description of the input
    deterministic : bool (default=False)
        If ``True``, will disable dropout/noise effect test time
        (must be set False for training)

    Returns
    -------
    predict_fn : Theano compiled function
        Compiled theano expression (accepts ndarray of dtype=floatX)
    predict_tn : TensorType(float32, matrix)
        Tensor expression for the prediction function

    History
    -------
    - Created 04/14/2016
    - 04/20/2016 - forked from ``output_function`` in util_dl
    """
    # "deterministic=True" to disable Dropout/noise effect at test time
    predict_tn = lasagne.layers.get_output(network,deterministic=deterministic)

    # compile so numpy-array of dtype=floatX can be used as input
    predict_fn = theano.function([in_var],predict_tn)

    return predict_fn, predict_tn

def las_print_layer_names(network):
    """ Print layer names (created 04/12/2016)"""
    print '--- layer names ---'
    for i,layer in enumerate(lasagne.layers.get_all_layers(network)):
        print "Layer{:2}: {}".format(i+1,layer.name)
    print '-------------------'


def las_print_layer_info(network):
    """ Print layer info (created 04/12/2016)

    Last updated 04/14/2016

    Todo: add some try-exception clauses for printing parameter shapes,
          nonlinearity applied, etc
    """
    print '========== layer names =========='
    for i,layer in enumerate(lasagne.layers.get_all_layers(network)):
        print "{:>9}:".format(layer.name),
        if isinstance(layer, InputLayer):
            print 'in.shape =',layer.shape
        elif isinstance(layer, DenseLayer):
            print " W.shape = {:10}".format(layer.get_params()[0].get_value().shape)
            print ' '*11+" b.shape =",layer.get_params()[1].get_value().shape
            print ' '*11+" activat =",layer.nonlinearity.func_name
        elif isinstance(layer, DropoutLayer):
            print " p_drop  =", layer.p
        else:
            print ''
    print '(number of parameters: {})'.format(
        lasagne.layers.count_params(network))
    print '================================='
#%% ***** theano level functions *****
#%% === theano convenience function ===
def get_lr(lr_val=1.0):
    """Simple wrapper for code brevity"""
    lr  = theano.shared(np.array(lr_val, dtype=theano.config.floatX),name='lr')
    return lr
#%% ***** misc *****
def decode_sae(Xfeat, decoder_list):
    """ Apply sequence of decoders from Stacked AE

    Note: to save memory, i just overwrite to the input varible ``Xfeat``
    (so readability somewhat compromised...just be aware of this)

    Parameters
    ----------
    Xencoded : ndarray of shape=(n_samp,n_encoded_feat)
        Encoded feature
    decoder_list : list of decoder function
        list of decoder function

    Returns
    -------
    Xfeat : ndarray of shape=(n_samp,n_feat)
        Reconstruction

    Dev
    ---
    - t_0419_sdae_greedy3.py
    """
    for i,decoder in enumerate(decoder_list[::-1]):
        Xfeat = decoder(Xfeat)
    return Xfeat

def encode_sae(Xfeat, encoder_list):
    """ Apply sequence of encoders from Stacked AE

    Note: to save memory, i just overwrite to the input varible ``Xfeat``
    (so readability somewhat compromised...just be aware of this)

    Parameters
    ----------
    Xfeat : ndarray of shape=(n_samp,n_feat)
        Original features
    encoder_list : list of encoder function
        list of encoder function

    Returns
    -------
    Xfeat : ndarray of shape=(n_samp,n_feat)
        Reconstruction

    Dev
    ---
    - t_0419_sdae_greedy3.py
    """
    for i,decoder in enumerate(encoder_list):
        Xfeat = decoder(Xfeat)
    return Xfeat


def iterate_minibatches(inputs, targets, batchsize=32, shuffle=False):
    """ Helper function for batch iteration

    This is just a simple helper function iterating over training data in
    mini-batches of a particular size, optionally in random order. It assumes
    data is available as numpy arrays. For big datasets, you could load numpy
    arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
    own custom data iteration function. For small datasets, you can also copy
    them to GPU at once for slightly improved performance. This would involve
    several changes in the main program, though, and is not demonstrated here.

    from https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
#%% === functions ===
def get_layer_info():
    """Created 04/11/2016"""
    from nolearn.lasagne import PrintLayerInfo
    layer_info = PrintLayerInfo()
    return layer_info

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def drop(input,p):
    '''The drop regularization

    From /home/takanori/Desktop/cmb-3dcnn-code-v1.0/code/lib/dropout.py

    :type input: numpy.array
    :param input: layer or weight matrix on which dropout resp. dropconnect is applied

    :type p: float or double between 0. and 1.
    :param p: p propobality of dropping out a unit or connection, therefore (1.-p) is the drop rate, typically in range [0.5 0.8]
    '''
    rng = np.random.RandomState(123456)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1,p=1-p,size=input.shape,dtype=theano.config.floatX)
    return input * mask

def get_summary(net):
    train_loss = np.array([row['train_loss'] for row in net.train_history_])
    valid_loss = np.array([row['valid_loss'] for row in net.train_history_])
    ratio = train_loss/valid_loss
    valid_acc  = np.array([row['valid_accuracy'] for row in net.train_history_])
    valid_best  = np.array([row['valid_loss_best'] for row in net.train_history_]).astype(int)

    return pd.DataFrame([train_loss, valid_loss,ratio,valid_acc,valid_best],
                 index=['train_loss','valid_loss','train/valid-loss','valid_acc','valid_best']).T


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.max(x,0)
#%% === MNIST data io ===
mnist_data = os.path.expanduser('~/data/deep_learning/theano/mnist.pkl.gz')
def download_mnist():
    """ Download mnist-data on disk

    Data saved at /home/takanori/data/deep_learning/theano
    """
    dataset=mnist_data
    data_dir, data_file = os.path.split(dataset)

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

def get_mnist():
    """ The dataset is a "pkl.gz"....notice the gz extension...needs gzip...

    Returns
    -------
    train_set, valid_set, test_set

    train_set[0]: ndarray of shape=(50000, 784) of dtype=float32
        Training set features
    train_set[1]: ndarray of shape=(50000,) of dtype=int64
        Training set labels
    valid_set[0]: ndarray of shape=(10000,784) of dtype=float32
        Validation set features
    valid_set[0]: ndarray of shape=(10000,) of dtype=int64
        Validation set labels
    test_set[0]: ndarray of shape=(10000,784) of dtype=float32
        Test set features
    test_set[0]: ndarray of shape=(10000,) of dtype=int64
        Test set labels
    """
    import gzip
    with gzip.open(mnist_data, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    return train_set, valid_set, test_set


def get_mnist_train():
    """ Get training features/labels

    not so efficient (redundant loading), but mnist is small enough

    **Update 04/14/2016** - ensure y is int32, and X is float32

    Returns
    -------
    X_train : ndarray of shape=(50000, 784) of dtype=float32
        Training set features
    y_train: ndarray of shape=(50000,) of dtype=int64
        Training set labels
    """
    X_train, y_train = get_mnist()[0]
    return X_train.astype(np.float32), y_train.astype(np.int32)

def get_mnist_validation():
    """ Get validation features/labels

    not so efficient (redundant loading), but mnist is small enough

    **Update 04/14/2016** - ensure y is int32, and X is float32

    Returns
    -------
    X_valid : ndarray of shape=(10000, 784) of dtype=float32
        Validation set features
    y_valid: ndarray of shape=(10000,) of dtype=int64
        Validation set labels
    """
    X_valid, y_valid = get_mnist()[1]
    return X_valid.astype(np.float32), y_valid.astype(np.int32)

def get_mnist_test():
    """ Get test features/labels

    not so efficient (redundant loading), but mnist is small enough

    **Update 04/14/2016** - ensure y is int32, and X is float32

    Returns
    -------
    X_test : ndarray of shape=(10000, 784) of dtype=float32
        Test set features
    y_test: ndarray of shape=(10000,) of dtype=int64
        Test set labels
    """
    X_test, y_test = get_mnist()[2]
    return X_test.astype(np.float32), y_test.astype(np.int32)

#%% === snippets ===
""" Train if pkl-file doesn't exist
fname = 'dnouri_demo1_net0.pkl'
if os.path.isfile(fname):
    del net0
    with open(fname,'rb') as f:
        net0 = pickle.load(f)
else:
    start=time.time()
    net0.fit(X_in, y_in)
    tw.print_time(start)
    pickle.dump(net0, open(fname,'wb'))
"""
#%% -- nolearn styles --
#%% style 1
"""
def define_net(layers):
    net=nl_las.NeuralNet(
        layers=layers,

        update=las.updates.nesterov_momentum,
        # for the "underscore", see docstr in nesterov_momentum
        update_learning_rate = 0.01,
        update_momentum=0.9,
        verbose=3,
    )
    net.initialize()
    return net

from lasagne.nonlinearities import softmax
layers1 = [
    (las.layers.InputLayer,  dict(name='input', shape=(None,1, 28, 28))),
    (las.layers.DenseLayer,  dict(name='hidden', num_units=100)),
    (las.layers.DenseLayer,  dict(name='output', num_units=10, nonlinearity=softmax)),
]

net1 = define_net(layers1)
nl_las.PrintLayerInfo()(net1)
"""
#%% style 2
#http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
"""
net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=1000,
    verbose=1,
    )
"""
#%% style3
#http://nbviewer.jupyter.org/github/dnouri/nolearn/blob/master/docs/notebooks/CNN_tutorial.ipynb
"""
layers0 = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

    # first stage of our convolutional layers

1
(Conv2DLayer, {'num_filters': 96, 'filter_size': 5}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    # second stage of our convolutional layers
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    # two dense layers with dropout
    (DenseLayer, {'num_units': 64}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 64}),

    # the output layer
    (DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
]
"""