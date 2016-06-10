# -*- coding: utf-8 -*-
"""
===============================================================================
Keep my "utility" function developed for the "transfer learning" project here.

-------------------------------------------------------------------------------
06/06/2016
- Code from: ~/sbia_work/python/analysis/dl/util_dl.py

-------------------------------------------------------------------------------
Update 04/20:
- many of the "legacy" function will be kept here with date appended to the
  function name. once i believe the function is at a matured stage, i'll
  move them to util.py without the date appended.
-------------------------------------------------------------------------------
===============================================================================
Created on Tue Apr 13

@author: takanori
"""
import numpy as np
import time
import sys
import matplotlib.pyplot as plt

from collections import OrderedDict

import theano
import theano.tensor as T

import tak as tw

import lasagne
from lasagne.layers import (
    InputLayer,
    DropoutLayer,
    DenseLayer,
)
#%% === network architectures ===
def dae(input,target,n_vis=784,n_hid=100, p_drop = 0.,get_util=True,
        objective='cross_entropy'):
    """ Created denoising autoencoder with tied weights.

    http://benanne.github.io/2015/11/10/arbitrary-expressions-as-params.html

    Parameters
    ----------
    input : theano.tensor.TensorType
        a symbolic description of the input
    target : theano.tensor.TensorType
        a symbolic description of the target/output
    n_vis : int
        number of visible units (input units)
    n_hid : int
        number of hidden units
    p_drop : float
        Probability of setting an input unit to zero ("masking" noise)
        (note: implemented via ``DropoutLayer``)
    get_util : bool
        Return bunch of utility function (prediction, encoder, decoder function)
    objective : 'cross_entropy' or 'squared_error'
        Reconstruction loss/objective function

    Returns
    -------
    loss : TensorType(floatX, matrix)
        Tensor scalar expression for the loss function

    Usage
    -----
    >>> input = T.matrix('input')
    >>> target = T.matrix('target')
    >>> network, loss, predict, encoder = util_dl.dae(input,target,n_hid=500,p_drop=0.3)
    >>> l2_pen = las.regularization.regularize_network_params(network, las.regularization.l2)
    >>> print "total #params = ", lasagne.layers.count_params(l_output)

    Dev
    ---
    - ``+t_0412_try_dae2.py``
    - ``t_0413_try_dae1.py``

    History
    -------
    Created 04/13/2016
    """
    l_input = InputLayer((None, n_vis),input_var=input,name='input')
    if p_drop != 0:
        # rescale set off then this is not for dropout, but for "masking" input for DAE
        l_input = DropoutLayer(l_input,p=p_drop,rescale=False,name='input_drop')

    # l_hid and l_output share the same weight matrix!
    l_hidden = DenseLayer(l_input,  n_hid, name='hidden',
                          nonlinearity=lasagne.nonlinearities.sigmoid)
    l_output = DenseLayer(l_hidden, n_vis, name='output',W=l_hidden.W.T,
                          nonlinearity=lasagne.nonlinearities.sigmoid)

    # === create prediction tensor and function ===
    #| tensor variable for testing (deterministic=True to disable dropout effect)
    #| (http://lasagne.readthedocs.org/en/latest/user/tutorial.html#run-the-mnist-example)
    predict_t = lasagne.layers.get_output(l_output,deterministic=True)

    # create loss function (binary cross-entropy for now
    if objective == 'cross_entropy':
        loss = lasagne.objectives.binary_crossentropy(predict_t, target)
    elif objective == 'squared_error':
        loss = lasagne.objectives.squared_error(predict_t, target)
    else:
        raise ValueError('Supplied objective "{}" not recognized!'.format(objective))
    loss = loss.mean()

    if not get_util:
        # return just the output layer
        return l_output, loss
    #%% === get extra utility function as output ===
    # compile as theano function (so it'll accept numpy array as input)
    predict_f = theano.function([input],predict_t)

    # === create encoder function ===#
    # Theano tensor for encoder function (deterministic=True to disable Dropout)
    encoder_t = lasagne.layers.get_output(l_hidden,deterministic=True)
    encoder_f = theano.function([input],encoder_t)

    # === create decoder function ===
    #decoder_tensor = lasagne.layers.get_output(l_hidden,deterministic=True)
    return l_output, loss, predict_f, encoder_f, #, decoder,


def sae(input,num_feat,layer_sizes,
        enc_act=lasagne.nonlinearities.rectify,
        dec_act=lasagne.nonlinearities.rectify,
        p_dropout=None):
    """ Create stacked autoencoders with tied weights

    Parameters
    ----------
    input : theano.tensor.TensorType
        a symbolic description of the input
    num_feat : int
        number of input features (number of input units)
    layer_sizes : list of int
        contains list of integers indicating size of the hidden units
    enc_act : list, or a single Theano elementwise function ``T.elemwise.Elemwise``
        Encoder activation functions (default: relu). If list is given, each
        element should be a valid elementwise Theano tensor function, with
        the length of the list matching ``len(layer_size)``
    dec_act : list, or a single Theano elementwise function ``T.elemwise.Elemwise``
        Decoder activation functions (default: relu). Same structure as **enc_act**.
    p_dropout : None, float, or list of floats between 0 and 1
        Probability of setting a value to zero. If a single float value is
        given, the same value will be applied at all layers.
        If a ``list``, than the length must match ``2*len(layer_sizes)``

    Returns
    -------
    ...

    Dev script
    -----------
    - t_0414_try_sae_dropout1.py
    """
    num_enc_layers = len(layer_sizes)

    if isinstance(p_dropout,list):
        assert len(p_dropout) == 2*num_enc_layers

    #| self-note: net.items() contains list of (key,value) pairs.
    #|            so net.items()[i_layer][1] gives the i-th layer object
    net = OrderedDict()
    net['input'] = InputLayer((None,num_feat),input_var=input,name='input')

    # === specify encoding layers ====
    for i,num_units in enumerate(layer_sizes):
        name = 'encoder'+str(i+1)

        # === add dropout layer if drop probability is given === #
        if isinstance(p_dropout,list) or isinstance(p_dropout,float):
            # dropout layer-name
            #drop_name = 'dropout_input' if i==0 else 'dropout'+str(i)
            drop_name = '_drop'+str(i+1)

            # assign dropout probability for this layer
            p_drop = p_dropout if isinstance(p_dropout,float) else p_dropout[i]

            # add dropout layer (the last attached layer becomes the input)
            net[drop_name] = DropoutLayer(net.items()[-1][1],p_drop,name=drop_name)

        # get encoding-layer activation (from a list or single function)
        activ = enc_act[i] if isinstance(enc_act,list) else enc_act

        # add encoding layer (the last attached layer becomes the input)
        net[name]=DenseLayer(net.items()[-1][1],num_units,name=name,nonlinearity=activ)

    # === specify decoding layers (tied weights) ====
    for i,num_units in enumerate(layer_sizes[::-1][1:]+[num_feat]):
        #print i,num_enc_layers
        name = 'decoder'+str(num_enc_layers-i)

        # === add dropout layer if drop probability is given === #
        if isinstance(p_dropout,list) or isinstance(p_dropout,float):
            # dropout layer-name
            #drop_name = 'dropout_input' if i==0 else 'dropout'+str(i)
            drop_name = '_drop'+str(i+1+num_enc_layers)

            # assign dropout probability for this layer
            p_drop = p_dropout if isinstance(p_dropout,float) else p_dropout[i+num_enc_layers]

            # add dropout layer (the last attached layer becomes the input)
            net[drop_name] = DropoutLayer(net.items()[-1][1],p_drop,name=drop_name)

        # get decoding-layer activation (from a list or single function)
        activ = dec_act[i] if isinstance(dec_act,list) else dec_act

        if (i+1)==num_enc_layers: # for final decoding, use sigmoid (overwrite above)
            pass
            #activ = lasagne.nonlinearities.sigmoid

        # corresponding encoder key (to get tied weight)
        encoder_name = 'encoder'+str(num_enc_layers-i)

        # enforce tied weights with corresponding decoder (last attached layer is the input)
        net[name] = DenseLayer(net.items()[-1][1],num_units, name=name,
                        nonlinearity=activ,W=net[encoder_name].W.T)

    # get final output layer
    net_output = net.items()[-1][1]

    # get middle "encoder" layer
    #net_encoder = net.items()[num_enc_layers][1]
    net_encoder = net['encoder'+str(num_enc_layers)]

    #util_dl.print_layer_names(net.items()[-1][1])
    return net_output, net_encoder, net



def sae_0419(in_var,num_feat,layer_sizes,
        enc_act=lasagne.nonlinearities.rectify,
        dec_act=lasagne.nonlinearities.rectify,
        p_dropout=None,
        encoder_only=False):
    """ Create stacked autoencoders with tied weights

    Code forked from ``sae`` on 04/19/2016.

    Update 04/20: change variable name ``input`` to ``in_var`` to avoid name
    clash with built-in function in python

    Added option of "encoder_only".  handy for fine-tuning MLP and such.

    Parameters
    ----------
    in_var : theano.tensor.TensorType
        a symbolic description of the input variable
    num_feat : int
        number of input features (number of input units)
    layer_sizes : list of int
        contains list of integers indicating size of the hidden units
    enc_act : list, or a single Theano elementwise function ``T.elemwise.Elemwise``
        Encoder activation functions (default: relu). If list is given, each
        element should be a valid elementwise Theano tensor function, with
        the length of the list matching ``len(layer_size)``
    dec_act : list, or a single Theano elementwise function ``T.elemwise.Elemwise``
        Decoder activation functions (default: relu). Same structure as **enc_act**.
    p_dropout : None, float, or list of floats between 0 and 1
        Probability of setting a value to zero. If a single float value is
        given, the same value will be applied at all layers.
        If a ``list``, than the length must match ``2*len(layer_sizes)``
    encoder_only : bool
        if true, only return the encoders (no decoding)

    Returns
    -------
    ...

    Dev script
    -----------
    - t_0414_try_sae_dropout1.py
    """
    num_enc_layers = len(layer_sizes)

    if isinstance(p_dropout,list):
        if encoder_only:
            assert len(p_dropout) == num_enc_layers
        else:
            assert len(p_dropout) == 2*num_enc_layers

    #| self-note: net.items() contains list of (key,value) pairs.
    #|            so net.items()[i_layer][1] gives the i-th layer object
    net = OrderedDict()
    net['input'] = InputLayer((None,num_feat),input_var=in_var,name='input')

    # === specify encoding layers ====
    for i,num_units in enumerate(layer_sizes):
        name = 'encoder'+str(i+1)

        # === add dropout layer if drop probability is given === #
        if isinstance(p_dropout,list) or isinstance(p_dropout,float):
            # dropout layer-name
            #drop_name = 'dropout_input' if i==0 else 'dropout'+str(i)
            drop_name = '_drop'+str(i+1)

            # assign dropout probability for this layer
            p_drop = p_dropout if isinstance(p_dropout,float) else p_dropout[i]

            # add dropout layer (the last attached layer becomes the input)
            net[drop_name] = DropoutLayer(net.items()[-1][1],p_drop,name=drop_name)

        # get encoding-layer activation (from a list or single function)
        activ = enc_act[i] if isinstance(enc_act,list) else enc_act

        # add encoding layer (the last attached layer becomes the input)
        net[name]=DenseLayer(net.items()[-1][1],num_units,name=name,nonlinearity=activ)

    # === specify decoding layers (tied weights) ====

    for i,num_units in enumerate(layer_sizes[::-1][1:]+[num_feat]):
        if encoder_only:
            break # don't care about the decoders, break out of loop!

        name = 'decoder'+str(num_enc_layers-i)

        # === add dropout layer if drop probability is given === #
        if isinstance(p_dropout,list) or isinstance(p_dropout,float):
            # dropout layer-name
            #drop_name = 'dropout_input' if i==0 else 'dropout'+str(i)
            drop_name = '_drop'+str(i+1+num_enc_layers)

            # assign dropout probability for this layer
            p_drop = p_dropout if isinstance(p_dropout,float) else p_dropout[i+num_enc_layers]

            # add dropout layer (the last attached layer becomes the input)
            net[drop_name] = DropoutLayer(net.items()[-1][1],p_drop,name=drop_name)

        # get decoding-layer activation (from a list or single function)
        activ = dec_act[i] if isinstance(dec_act,list) else dec_act

        if (i+1)==num_enc_layers: # for final decoding, use sigmoid (overwrite above)
            pass
            #activ = lasagne.nonlinearities.sigmoid

        # corresponding encoder key (to get tied weight)
        encoder_name = 'encoder'+str(num_enc_layers-i)

        # enforce tied weights with corresponding decoder (last attached layer is the input)
        net[name] = DenseLayer(net.items()[-1][1],num_units, name=name,
                        nonlinearity=activ,W=net[encoder_name].W.T)

    # get final output layer
    net_output = net.items()[-1][1]

    # get middle "encoder" layer
    #net_encoder = net.items()[num_enc_layers][1]
    net_encoder = net['encoder'+str(num_enc_layers)]

    #util_dl.print_layer_names(net.items()[-1][1])
    return net_output, net_encoder, net


def dae_0418(input=None,
             n_vis=784,n_hid=100, p_drop = 0.,objective='cross_entropy'):
    """ Created denoising autoencoder with tied weights.

    http://benanne.github.io/2015/11/10/arbitrary-expressions-as-params.html

    Parameters
    ----------
    input : theano.tensor.TensorType (default=None)
        a symbolic description of the input.
        if ``None``, sym-var will be created internally
    target : theano.tensor.TensorType (default=None)
        a symbolic description of the target/output.
        if ``None``, sym-var will be created internally
    n_vis : int
        number of visible units (input units)
    n_hid : int
        number of hidden units
    p_drop : float
        Probability of setting an input unit to zero ("masking" noise)
        (note: implemented via ``DropoutLayer``)
    objective : 'cross_entropy' or 'squared_error'
        Reconstruction loss/objective function

    Returns
    -------
    l_output : ...
        ...
    encoder_fn : ...
        ...

    Dev
    ---
    - ``t_418_sdae_greedy1.py``

    History
    -------
    Created 04/18/2016
    """
    if input is None:
        input = T.matrix('input')

    # input layer
    l_input = InputLayer((None, n_vis),input_var=input,name='input')

    if p_drop != 0:
        # rescale set off then this is not for dropout, but for "masking" input for DAE
        l_input = DropoutLayer(l_input,p=p_drop,rescale=False,name='input_drop')

    # l_hid and l_output share the same weight matrix!
    l_hidden = DenseLayer(l_input,  n_hid, name='hidden',
                          nonlinearity=lasagne.nonlinearities.sigmoid)
    l_output = DenseLayer(l_hidden, n_vis, name='output',W=l_hidden.W.T,
                          nonlinearity=lasagne.nonlinearities.sigmoid)

    # === get deterministic encoder function === #
    # Theano tensor for encoder function (deterministic=True to disable Dropout)
    encoder_tn = lasagne.layers.get_output(l_hidden,deterministic=True)
    encoder_fn = theano.function([input],encoder_tn)

    return l_output, encoder_fn


def dae_0419(input=None,n_vis=784,n_hid=100, p_drop = 0.,
             encoder_nonlin=lasagne.nonlinearities.sigmoid,
             decoder_nonlin=lasagne.nonlinearities.sigmoid):
    """ Created denoising autoencoder with tied weights.

    http://benanne.github.io/2015/11/10/arbitrary-expressions-as-params.html

    >>> network, encoder_fn, decoder_fn, output_fn = dae_0419(...)

    Update 04/26 - allowed optional input to specify encoder and decoder
    nonlinearity

    Parameters
    ----------
    input : theano.tensor.TensorType (default=None)
        a symbolic description of the input.
        if ``None``, sym-var will be created internally
    n_vis : int
        number of visible units (input units)
    n_hid : int
        number of hidden units
    p_drop : float
        Probability of setting an input unit to zero ("masking" noise)
        (note: implemented via ``DropoutLayer``)

    Returns
    -------
    l_output : ...
        ...
    encoder_fn : ...
        ...

    Dev
    ---
    - ``t_0419_decoder_func.py``
    - ``t_0419_decoder_func2.py``

    History
    -------
    Created 04/19/2016...difference from 0418 version: added decoder function
    as output
    """
    if input is None:
        input = T.matrix('input')

    # input layer
    l_input = InputLayer((None, n_vis),input_var=input,name='input')

    if p_drop != 0:
        # rescale set off then this is not for dropout, but for "masking" input for DAE
        l_input = DropoutLayer(l_input,p=p_drop,rescale=False,name='input_drop')

    # l_hid and l_output share the same weight matrix!
    l_hidden = DenseLayer(l_input,  n_hid, name='hidden',
                          nonlinearity=encoder_nonlin)
    l_output = DenseLayer(l_hidden, n_vis, name='output',W=l_hidden.W.T,
                          nonlinearity=decoder_nonlin)

    # === get deterministic encoder function === #
    # Theano tensor for encoder function (deterministic=True to disable Dropout)
    encoder_tn = lasagne.layers.get_output(l_hidden,deterministic=True)
    encoder_fn = theano.function([input],encoder_tn)

    # === get decoder function (new in 04/19 version) === #
    # theano symvar for hidden unit representation
    hid = T.matrix('hid')
    W_out, b_out = l_output.get_params()
    decoder_tn = l_output.nonlinearity(hid.dot(W_out.T) + b_out)
    decoder_fn = theano.function([hid],decoder_tn)

    # === get output function (new in 04/19 version) ===#
    """Note: this outputs the same thing as decoder_fn, but takes as input
       the original feature (so a composition of encoding/decoding operation)"""
    output_tn = lasagne.layers.get_output(l_output,deterministic=True)
    output_fn = theano.function([input], output_tn)

    return l_output, encoder_fn, decoder_fn, output_fn



def make_mlp_from_sae_0421(net_encoder, n_outputs,nonlin_out=None,p_drop=True):
    """

    Assumptions:

    - all dense layers contain [W,b] parameters
    - all dense layers are named 'encoder1', 'encoder2', ... (as created in
      ``sae`` functions i created)

    Parameters
    ----------
    n_outputs : int
        Number of output units (2 for binary, 10 for MNIST)
    nonlin_out : default = None
        Supervised output nonlinearity. If ``None``, defaults to
        ``lasagne.nonlinearities.softmax``
    p_drop : bool
        If True, sets dropout probability of 0.2 in the input, 0.5 elsewhere

    Dev
    ---
    - ``t_0421_sae_dropout_pnc1.py``
    - ``t_419_sdae_greedy6_dropout_mlp_relu.py``
    """
    #=== get parameter matrices as ndarray, and layer size info ===#
    params_list = lasagne.layers.get_all_params(net_encoder)

    # assuming we have [encoder1.W, encoder1.b, encoder22.W, encoder2.b,..] pattern
    n_encoder = len(params_list)/2

    # dropout layers
    if p_drop:
        p_drop_layers = [0.2]+[0.5]*(n_encoder-1)
    else:
        p_drop_layers = None

    layer_size = []

    # input feature size
    n_feat = params_list[0].get_value().shape[0]

    W_list = []
    b_list = []
    for i in range(n_encoder):
        # assuming we have [encoder1.W, encoder1.b, encoder2.W, encoder2.b,..] pattern
        W_list.append(params_list[2*i].get_value())
        #print params_list[2*i]
        b_list.append(params_list[2*i+1].get_value())
        #print params_list[2*i+1]
        layer_size.append(W_list[i].shape[1])

    #=== now construct mlp network ===#
    #| new symbolic input variable for a whole new network system
    in_var_mlp = T.matrix('in_var_mlp')

    #-- for now, assume relu everywhere --#
    nonlin = lasagne.nonlinearities.rectify
    _, _, net_mlp_all = sae_0419(in_var_mlp, n_feat, layer_size, enc_act = nonlin,
                                 p_dropout=p_drop_layers,encoder_only=True)

    # attach supervised output
    if nonlin_out is None:
        nonlin_out = lasagne.nonlinearities.softmax
        #nonlin_out = lasagne.nonlinearities.sigmoid

    net_mlp_all['dropout'] = DropoutLayer(net_mlp_all.items()[-1][1],p=0.5,name='dropout')
    net_mlp_all['output']  = DenseLayer(net_mlp_all['dropout'],num_units=n_outputs,
        nonlinearity=nonlin_out,name='output')

    #=== initialize weight ===#
    for i in range(n_encoder):
        lname = 'encoder'+str(i+1)
        dname = '_drop'+str(i+1)
        #print lname

        # scaled pretraining weights by 1/p_dropout (maybe this isn't needed
        # since they apply the dropout scaling during training)
        scale_weight=True
        if scale_weight:
            net_mlp_all[lname].W.set_value(W_list[i]/net_mlp_all[dname].p)

        net_mlp_all[lname].b.set_value(b_list[i])

    # extract the final output of interest (shared memory so no wasted ram)
    net_mlp = net_mlp_all['output']

    return net_mlp, net_mlp_all, in_var_mlp
#%% === some "convenience" fucntion  ===
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

def get_lr(lr_val=1.0):
    """Simple wrapper for code brevity"""
    lr  = theano.shared(np.array(lr_val, dtype=theano.config.floatX),name='lr')
    return lr

def get_weight_penalty(network, penalty_type='l2'):
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
#%% === training routines ===
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


def train_self_recon_net(X_input,train_fn, num_epochs=100, batch_size=100,
                         disp_freq=10, lr_decay=None):
    """ Train self-reconstruction network (mostly autoencoder-type network)

    I made this function mostly as a handy "template/snippet

    History
    -------
    - Update 04/18/2016: implemented adaptive learning rate option, where
      learning-rate gets updated every epoch by factor ``lr_decay``
    ".
    """
    err_=[]
    start_time = time.time()
    if lr_decay is not None:
        # dig out theano shared-var corresponding to the learning-rate
        lr = [shared for shared in train_fn.get_shared() if shared.name=='lr'][0]
        print lr.get_value()

        # need to get access to shared variable for learning rate
        lr_decay = np.asarray(lr_decay,dtype=theano.config.floatX)

    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        try:
            for batch in iterate_minibatches(X_input,X_input,batch_size,shuffle=True):
                #| here, don't care about second part (we want to reconstruct input
                inputs, _ = batch
                train_err += train_fn(inputs,inputs) # <- where model gets updated
                train_batches += 1
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




def train_recon_net_nesterov(X_input,input,target,loss,params,
                             lr=1.0,momentum=0.95,
                             num_epochs=100, batch_size=100,
                             disp_freq=10,lr_decay=None):
    """ Train self-reconstruction network (AE-type) via Nesterov's method

    I made this function mostly as a handy "template/snippet

    History
    -------
    - Created 04/18/2016: forked from train_self_recon_net
    ".
    """
    lr = theano.shared(np.array(lr, dtype=theano.config.floatX),name='lr')
    updates = lasagne.updates.nesterov_momentum(loss, params,lr.get_value(),
                                                momentum)

    # compile training function
    train_fn = theano.function([input,target],loss,updates=updates)

    err_=[]
    start_time = time.time()
    if lr_decay is not None:
        #print lr.get_value()
        # need to get access to shared variable for learning rate
        lr_decay = np.asarray(lr_decay,dtype=theano.config.floatX)

    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        try:
            for batch in iterate_minibatches(X_input,X_input,batch_size,shuffle=True):
                #| here, don't care about second part (we want to reconstruct input
                inputs, _ = batch
                train_err += train_fn(inputs,inputs) # <- where model gets updated
                train_batches += 1
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


def las_train_nesterov(X_input,y_input,input,target,objective,params,
                             lr=1.0,momentum=0.95,
                             num_epochs=100, batch_size=100,
                             disp_freq=10,lr_decay=None):
    """ Train network via Nesterov's method

    I made this function mostly as a handy "template/snippet

    Best used with ``get_objective_0418``

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
    train_fn = theano.function([input,target],objective,updates=updates)

    err_=[]
    start_time = time.time()
    if lr_decay is not None:
        #print lr.get_value()
        # need to get access to shared variable for learning rate
        lr_decay = np.asarray(lr_decay,dtype=theano.config.floatX)

    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        try:
            for batch in iterate_minibatches(X_input,y_input,batch_size,shuffle=True):
                #| here, don't care about second part (we want to reconstruct input
                inputs, outputs = batch
                train_err += train_fn(inputs,outputs) # <- where model gets updated
                                             #^^^^(modified from ``train_recon_net_nesterov``)
                train_batches += 1
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


def get_objective_0418(network,target,l1=None, l2=None, loss=None):
    """ Get training objective function for optimization.

    Remember that the prediction function during training heeds to have
    ``deterministic`` set to False

    - Created 04/18/2016
    - modified 04/19/2016
    """
    # input tensor variable
    #input = lasagne.layers.get_all_layers(network)[0].input_var

    # for training, set deterministic to False
    predict_tr_tn = lasagne.layers.get_output(network,deterministic=False)
    params        = lasagne.layers.get_all_params(network, trainable=True)

    # define loss, params, updates, and number of total parameters
    if loss is None:
        if network.nonlinearity.func_name == 'sigmoid':
            objective = lasagne.objectives.binary_crossentropy(
                            predict_tr_tn,target).mean()
        else:
            objective = lasagne.objectives.squared_error(predict_tr_tn,target).mean()
    else:
        objective = loss(predict_tr_tn,target).mean() # <- modified 04/19

    # === add regularization === #
    if isinstance(l2,int) or isinstance(l2,float):
        objective += l2*get_weight_penalty(network,'l2')

    if isinstance(l1,int) or isinstance(l1,float):
        objective += l1*get_weight_penalty(network,'l1')

    return objective, params
#%% === helper functions for lasagne layers ========
def output_function(network,input,deterministic=False):
    """ Get output function from a network

    Basically the final activation output from the final layer

    Parameters
    ----------
    network : ``lasagne.layers.Layer`` instance
        Parameters of this layer and all layers below it will be penalized.
    input : theano.tensor.TensorType
        a symbolic description of the input
    deterministic : bool (default=False)
        If ``True``, will disable dropout/noise effect test time
        (must be set False for training)

    Returns
    -------
    predict_f : Theano compiled function
        Compiled theano expression (accepts ndarray of dtype=floatX)
    predict_t : TensorType(float32, matrix)
        Tensor expression for the prediction function

    History
    -------
    Created 04/14/2016
    """
    # "deterministic=True" to disable Dropout/noise effect at test time
    predict_t = lasagne.layers.get_output(network,deterministic=deterministic)

    # compile so numpy-array of dtype=floatX can be used as input
    predict_f = theano.function([input],predict_t)

    return predict_f,predict_t

def print_layer_names(network):
    """ Print layer names (created 04/12/2016)"""
    print '--- layer names ---'
    for i,layer in enumerate(lasagne.layers.get_all_layers(network)):
        print "Layer{:2}: {}".format(i+1,layer.name)
    print '-------------------'

def print_layer_info(network):
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


def get_num_params(network):
    """ Get total number of parameters in a lasagne network"""
    return lasagne.layers.count_params(network)

def get_layer_by_name(network,name):
    """ Get a particular layer from a network by name

    **Note** (04/14/2016): mostly obsolete now that i adopted the habit of using
    ``OrderedDict`` to store all network info (shared memory so no wasted memory)

    Parameters
    ----------
    network : ``lasagne.layers.Layer`` instance
        Network consisting of sequence of lasagne layers
    name : string
        Name of layer

    Return
    ------
    layer : ``lasagne.layers.Layer`` instance
        A particular layer from the input network

    Usage
    -----
    >>> hidden_layer = util_dl.get_layer_by_name(network,'hidden')

    History
    -------
    Created 04/13/2016
    """
    for layer in lasagne.layers.get_all_layers(network):
        if layer.name == name:
            return layer
    print '...layer "{}" not found... (nothing returned)'.format(name)


def get_param_by_name(network,name):
    """ Get a particular parameter from a network by name

    **Note** (04/14/2016): mostly obsolete now that i adopted the habit of using
    ``OrderedDict`` to store all network info (shared memory so no wasted memory)

    Parameters
    ----------
    network : ``lasagne.layers.Layer`` instance
        Network architecture
    name : string
        Name of network, appended by ``.W`` or ``.b`` for weight matrix or bias
        (eg, ``'hidden.W'``, ``hidden.b``)

    Returns
    -------
    param : Theano shared variable
        Theano shared variable of the parameter (use ``param.get_value()``
        to obtain values in numpy array form)

    Usage
    -----
    >>> hidden_layer = util_dl.get_layer_by_name(network,'hidden')

    History
    -------
    Created 04/13/2016
    """
    for param in lasagne.layers.get_all_params(network):
        if param.name == name:
            return param
    print '...param "{}" not found... (nothing returned)'.format(name)
#%% ===

def show_mnist_images(XX,sup_title=None,clim=None,i_shift=0):
    """XX = (n_samp, 784) mnist"""
    tw.figure('f')
    for i in range(64):
        try:
            tmp=np.reshape(XX[i+i_shift],(28,28))
            plt.subplot(8,8,i+1) # argh, always forget subplot is not 0based index
            plt.imshow(tmp)
            plt.imshow(tmp,cmap='gray_r')
            plt.axis('off')
            if clim is not None:
                plt.clim(clim)
        except:
            break

    if isinstance(sup_title,str):
        plt.suptitle(sup_title,fontsize=24)



