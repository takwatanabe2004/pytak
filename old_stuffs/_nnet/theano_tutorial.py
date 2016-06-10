# -*- coding: utf-8 -*-
"""
===============================================================================
Here I intend to keep proto codes for neural net related stuffs
-------------------------------------------------------------------------------
For the moment, mostly stuffs from Theano/DL tutorial
===============================================================================
Created on Mon Dec 14 12:28:39 2015
bhar
@author: takanori
"""
import os
import sys
import numpy as np

import gzip
import cPickle

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
#%% === util functions ===
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def get_pkl_data():
    """ Get data from ``mnist.pkl.gz``


    Returns
    ---------
    ``train_set, valid_set, test_set`` - **format: tuple(input, target)**

    input : ndarray of shape [n_examples, n_features]
        Basically a design matrix
    target : ndarray of shape [n_examples,]
        It should give the target target to the example with the same index in the input.
    """

    pkl_path = '/home/takanori/data/deep_learning/mnist.pkl.gz'

    with gzip.open(pkl_path, "rb") as f:
        train_set, valid_set, test_set = cPickle.load(f)

    return train_set, valid_set, test_set

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(floatX(data_x), borrow=borrow)
    shared_y = theano.shared(floatX(data_y), borrow=borrow)

    """Above, the labels (shared_y) are stored as floatX so the GPU can
    handle them.  This function will return it as int32 via typecasting"""
    return shared_x, T.cast(shared_y, 'int32')


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape,dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array



#%% == from logistic_sgd ===
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        Parameters
        ----------
        input : theano.tensor.TensorType
            Symbolic variable that describes the input of the architecture (one minibatch)

        n_in : int
            number of input units, the dimension of the space in which the data lie

        n_out : int
            number of output units, the dimension of the space in which the labels lie

        Attributes
        ----------
        W : symbolic matrix [n_in, n_out]
            W is a matrix where column-k represent the separation hyperplane for class-k
        b : symbolic vector [n_out]
            b is a vector where element-k represent the free parameter of hyperplane-k
        x : symbolic matrix [n_samples, n_features]
            x is a matrix where row-j represents input training sample-j
        """
        #=== initialize parameters as 0s ===#
        W_init = np.zeros((n_in, n_out))
        b_init = np.zeros(n_out)

        self.W = theano.shared( floatX(W_init), name='W', borrow=True)
        self.b = theano.shared( floatX(b_init), name='b', borrow=True)

        # create symbolic expressionsfor computing matrix of class-membership probabilities
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \\frac{1}{|\\mathcal{D}|} \\mathcal{L} (\\theta=\\{W,b\\}, \\mathcal{D}) =
            \\frac{1}{|\\mathcal{D}|} \\sum_{i=0}^{|\\mathcal{D}|}
                \\log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \\ell (\\theta=\\{W,b\\}, \\mathcal{D})

        Parameters
        ----------
        y : theano.tensor.TensorType
            corresponds to a vector that gives for each example the correct label

        Note
        -----
        we use the mean instead of the sum so that the learning rate is
        less dependent on the batch size
        """
        # y.shape[0] = (symbolically) the number of examples (call it n) in the minibatch
        #
        # T.arange(y.shape[0]) = symbolic vector which will contain # [0,1,2,... n-1]
        #
        # T.log(self.p_y_given_x) = matrix of Log-Probabilities (call it LP) with
        # one row per example and one column per class
        #
        # LP[T.arange(y.shape[0]),y] is a vector v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        #
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is the mean (across minibatch examples)
        # of the elements in v, i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def errors(self, y):
        """ Return a float representing the number of errors in the minibatch

        Computes the zero one loss over the size of the minibatch

        Parameters
        -----------
        y : theano.tensor.TensorType
            corresponds to a vector that gives for each example the correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))

        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
#%% === from denoising AE ====
class dA(object):
    """Denoising Auto-Encoder class (dA)

    1. add corruption: :math:`\\tilde{x} = q_D(\\tilde{x}|x)`
    2. project input onto the latent space: :math:`y = s(W \\tilde{x} + b)`
    3. reconstruct input: :math:`x = s(W' y  + b')`
    4. compute reconstruction error: :math:`L(x,z) = -\\sum_{k=1}^d [x_k \\log z_k + (1-x_k) \\log( 1-z_k)]`

    Parameters
    ----------
    numpy_rng : numpy.random.RandomState
        number random generator used to generate weights

    theano_rng : theano.tensor.shared_randomstreams.RandomStreams
        Theano random generator; if None is given one is generated based
        on a seed drawn from `rng`

    input : theano.tensor.TensorType of shape [n_samples, n_features] (default=None)
        a symbolic description of the input or None for standalone dA

    n_visible : int
        number of visible units

    n_hidden : int
        number of hidden units

    W : theano.tensor.TensorType of shape [n_visible, n_hidden] (default=None)
        Theano variable pointing to a set of weights;
        should be shared belong the dA and another architecture;
        if dA shouldbe standalone set this to ``None``

    bhid : theano.tensor.TensorType of shape [n_hidden,] (default=None)
        Set of biases values (for hidden units);
        should be shared between dA and another architecture;
        if dA should be standalone set this to ``None``

    bvis : theano.tensor.TensorType of shape [n_visible,] (default=None)
        Set of biases values (for visible units);
        should be shared belong dA and another architecture;
        if dA should be standalone set this to ``None``
    """

    def __init__(self,numpy_rng,theano_rng=None,input=None,n_visible=784,
                 n_hidden=500,W=None,bhid=None,bvis=None):
        """ Initialize the dA class by specifying:

        - the number of visible units (the dimension d of the input )
        - the number of hidden units (the dimension d' of the latent space)
        - the corruption level

        The constructor also receives symbolic variables for: **input, weights and bias**.

        Such a symbolic variables are useful when, for example:

        - the input is the result of some computations
        - when weights are shared between the dA and an MLP layer.
        - When dealing with SdAs this always happens, the dA on layer 2 gets
          as input the output of the dA on layer 1, and the weights of the dA
          are used in the second stage of training to construct an MLP.

        How W is initialized (When W = None)
        ------------------------
        W is initialized with ``initial_W`` which is uniformely sampled
        from:

        .. math::

            \\left(
                -4\\sqrt{\\frac{6}
                      {\\text{n_hidden}+\\text{n_visible}}
                      },
                4\\sqrt{\\frac{6}
                      {\\text{n_hidden}+\\text{n_visible}}
                      }
            \\right)

        The output of uniform is converted using asarray to dtype
        theano.config.floatX so that the code is runable on GPU
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # Initialize W (note : W' was written as `W_prime` and b' as `b_prime`)
        if W is None:
            # low/high value to sample from uniform distribution
            low_unif = -4 * np.sqrt(6. / (n_hidden + n_visible))
            hi_unif  = +4 * np.sqrt(6. / (n_hidden + n_visible))

            # sample from uniform distribution
            W_init = numpy_rng.uniform(low=low_unif,high=hi_unif,size=(n_visible, n_hidden))

            # convert dtype theano.config.floatX so the code can be ran on GPU
            initial_W = np.asarray(W_init,dtype=theano.config.floatX)

            # create shared theano graph representing W_init
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if bvis is None:
            bvis_init = np.zeros(n_visible,dtype=theano.config.floatX)
            bvis = theano.shared(value=bvis_init,borrow=True)

        if not bhid:
            bhid_init = np.zeros(n_hidden,dtype=theano.config.floatX)
            bhid = theano.shared(value=bhid_init,name='b',borrow=True)

        self.W = W
        self.b = bhid       # bias of the hidden
        self.b_prime = bvis # bias of the visible

        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T

        self.theano_rng = theano_rng

        if input is None:
            # if no input is given, generate a variable representing the input
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """Corrupt input by zeroing-out randomly selected subset of size ``corruption_level``

        Note about theano.rng.binomial
        ------------------------------
        ``theano.rng.binomial`` will produce an array of 0s and 1s,
        where 1 has a probability of 1 - ``corruption_level`` and
        0 with ``corruption_level``

        The binomial function return int64 data type by default.
        int64 multiplied by the input type(floatX) always return float64.

        To keep all data in floatX when floatX is float32, we set the dtype of
        the binomial to floatX (again, we do this since GPU only supports float32)
        """
        return self.theano_rng.binomial(size=input.shape,
                                        n=1, # the number of trials
                                        p=1 - corruption_level, # probability of success (1)
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer

        Parameters
        -----------
        - input = [n_samples, n_hidden]
        - W = [n_visible, n_hidden]
        - b = [n_hidden]

        Returns
        -------
        hidden = [n_samples, n_hidden]
        """
        hidden = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        return hidden

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the hidden layer

        Parameters
        ----------
        - hidden = [n_samples, n_hidden]
        - W_prime = [n_hidden, n_visible]
        - b_prime = [n_visible]

        Returns
        -------
        recon = [n_samples, n_visible]
        """
        recon = T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
        return recon

    def get_cost_updates(self, corruption_level, learning_rate):
        """ Compute the cost and the updates for one trainng step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        #=== compute cross entropy loss function ====#
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` wrt its parameters
        gparams = T.grad(cost, self.params)

        # generate the list of updates
        updates = [(param, param - learning_rate * gparam)
                    for param, gparam in zip(self.params, gparams)]

        return (cost, updates)

#%% === from mlp.py ===
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,activation=T.tanh):
        """ Typical hidden layer of a MLP: units are fully-connected.

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        Parameters
        ----------
        rng : numpy.random.RandomState
            numpy random number generator used to initialize weights
        input : theano.tensor.dmatrix
            a symbolic tensor of shape (n_examples, n_in)
        n_in : int
            dimensionality of input
        n_out : int
            number of hidden units
        activation : theano.Op or function
            Non-linearity to be applied in the hidden layer

        Attributes
        ----------
        self.input : theano.tensor.dmatrix
            a symbolic tensor of shape (n_examples, n_in)
        self.W : symbolic matrix [n_in, n_out]
            symbolic matrix [n_in, n_out]
        self.b : symbolic vector [n_out]
            symbolic vector [n_out]
        self.output : ...
            output of the hidden node
        self.params :  [self.W, self.b]
            parameters of the model
        """
        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        if W is None:
            low = -np.sqrt(6. / (n_in + n_out))
            high = np.sqrt(6. / (n_in + n_out))
            W_values = rng.uniform(low=low,high=high,size=(n_in, n_out))
            W_values = floatX(W_values) # convert to theano floatX

            # for sigmoid, [Xavier10] suggest to use 4 times initial weights than tanh
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        #--- assign attributes ---#
        self.input = input
        self.W = W
        self.b = b
        self.params = [self.W, self.b] # parameters of the model

        # compute output
        self.output = T.dot(input, self.W) + self.b
        if activation is not None:
            self.output = activation(self.output)


class MLP(object):
    """Multi-Layer Perceptron Class with one hidden layer

    Activation functions
    --------------------
    - Intermediate layer usually have tanh or sigmoid (defined here by a ``HiddenLayer`` class)
      The code uses ``tanh``, but this can be replaced by sigmoid or any other nonlinear function
    - Top (output) layer is a softmax layer (defined here by a ``LogisticRegression`` class).
    """
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        rng : numpy.random.RandomState
            a random number generator used to initialize weights
        input : theano.tensor.TensorType
            symbolic variable that describes the input of the architecture (one minibatch)
        n_in: int
            number of input units, the dimension of the space in which the datapoints lie
        n_hidden : int
            number of hidden units
        n_out : int
            number of output units, the dimension of the space in which the labels lie

        """
        # The single hidden layer
        self.hiddenLayer = HiddenLayer(rng,input,n_in,n_hidden,activation=T.tanh)

        # The output layer is the logistic regression layer
        # (receives as input the output from the hidden layer)
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        # L1 norm regularization
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()

        # L2 norm regularization
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()

        # NLL of the MLP is given by the NLL computed in the output logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        # keep track of model input
        self.input = input


#%% === from sdae ====
class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    - A stacked denoising autoencoder model is obtained by stacking several dAs.
    - The hidden layer of the dA at layer `i` becomes the input of the dA at layer `i+1`.
    - The first layer dA gets as input the input of the SdA, and the hidden layer
      of the last dA represents the output.
    - Note that after pretraining, the SdA is dealt with as a normal MLP,
      the dAs are only used to initialize the weights.
    """
    def __init__(self,numpy_rng,theano_rng=None,n_ins=784,hidden_layers_sizes=[500, 500],
                 n_outs=10,corruption_levels=[0.1, 0.1]):
        """ This class is made to support a variable number of layers.

        Parameters
        ----------
        numpy_rng: numpy.random.RandomState
            numpy random number generator used to draw initial weights
        theano_rng : theano.tensor.shared_randomstreams.RandomStreams
            Theano random generator;
            if None is given one is generated based on a seed drawn from `rng`
        n_ins : int
            dimension of the input to the sdA
        n_layers_sizes : list of ints
            intermediate layers size; must contain at least one value
        n_outs : int
            dimension of the output of the network
        corruption_levels : list of float
            amount of corruption to use for each layer
        """
        self.n_layers = len(hidden_layers_sizes)
        assert self.n_layers > 0

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')   # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels


        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        for i in xrange(self.n_layers):
            """i indexes the layer"""

            if i == 0:
                input_size = n_ins   # first layer
                layer_input = self.x # the input of the SdA
            else:
                # the number of hidden units of the layer below
                input_size = hidden_layers_sizes[i - 1]

                # the input to this layer is the activation of the hidden-layer below
                layer_input = self.sigmoid_layers[-1].output

            # construct the sigmoidal layer
            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAE.
            # the visible biases in the dA are parameters of those dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder with shared weights with this layer
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
        #---- end of looping over layer0 to layer self.n_layers ----#
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)

        #--- construct a function that implements one step of finetunining ---#
        # compute the cost for second phase of training, defined as the NLL
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)


    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        Parameters
        ----------
        train_set_x : theano.tensor.TensorType
             Shared variable that contains all datapoints used for training the dA

        batch_size : int
            size of a [mini]batch

        learning_rate: float
            learning rate used during training for any of the dA layers
        '''
        index = T.lscalar('index')                 # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')             # learning rate to use
        batch_begin = index * batch_size           # begining of a batch, given `index`
        batch_end = batch_begin + batch_size       # ending of a batch given `index`

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,learning_rate)

            # compile the theano function
            fn = theano.function(
                inputs=[index,
                        theano.Param(corruption_level, default=0.2),
                        theano.Param(learning_rate, default=0.1)],
                outputs=cost,
                updates=updates,
                givens={self.x: train_set_x[batch_begin: batch_end]}
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns


    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        Parameters
        -----------
        datasets : list of pairs of theano.tensor.TensorType
            List that contain all the datasets; contain three pairs,
            `train`,`valid`, `test` in this order, where each pair is formed
            of two Theano variables, one for the datapoints, the other for the labels

        batch_size : int
            size of a minibatch

        learning_rate : float
            learning rate used during finetune stage
        '''
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score