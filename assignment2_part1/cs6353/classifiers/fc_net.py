from builtins import range
from builtins import object
import numpy as np

from cs6353.layers import *
from cs6353.layer_utils import *

"""
def softmax_loss(x, y):
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = x.shape[0]
        loss = -np.sum(log_probs[np.arange(N), y]) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx
"""
class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    
    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        #print("updated")
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N = X.shape[0]
        C = X.shape[1]
        X = X.reshape(N, -1)
        # ReLU
        hidden_layer = np.maximum(0, np.dot(X, W1) + b1) 
        # score update 
        scores = np.dot(hidden_layer, W2) + b2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        data_loss, dscores = softmax_loss(scores, y)
        #print("updated")
        # factor of 0.5 introduced
        reg_loss = 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2))
        loss = data_loss + reg_loss
        dhidden = np.dot(dscores, W2.T)
        # second layer
        grads['W2'] = np.dot(hidden_layer.T, dscores)
        grads['W2'] += self.reg * W2
        grads['b2'] = np.sum(dscores, axis=0)
        # ReLU layer for x<0
        dhidden[hidden_layer <= 0] = 0
        # first layer 
        grads['W1'] = np.dot(X.T, dhidden)
        grads['W1'] += self.reg * W1
        grads['b1'] = np.sum(dhidden, axis=0)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deterministic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # initialize rom a normal distribution centered at 0 with standard deviation equal to weight_scale
        self.params['W1'] = np.random.randn(input_dim, hidden_dims[0]) * weight_scale
        self.params['b1'] = np.zeros(shape=(hidden_dims[0],))
        # initialization for batchnorm, not for layernorm
        # scale to ones, shift to zeros
        if self.normalization == "batchnorm":
            self.params['gamma1'] = np.ones(shape=(hidden_dims[0],))
            self.params['beta1'] = np.zeros(shape=(hidden_dims[0],))
        for i in range(1, self.num_layers - 1):
            weight_key = f'W{i+1}'
            bias_key = f'b{i+1}'
            self.params[weight_key] = np.random.randn(hidden_dims[i-1], hidden_dims[i]) * weight_scale
            self.params[bias_key] = np.zeros(shape=(hidden_dims[i],))
            if self.normalization == "batchnorm":
                # storing scale and shift params in gamma2, beta2, ...
                gamma_key = f'gamma{i+1}'
                beta_key = f'beta{i+1}'
                self.params[gamma_key] = np.ones(shape=(hidden_dims[i],))
                self.params[beta_key] = np.zeros(shape=(hidden_dims[i],))
        self.params[f'W{self.num_layers}'] = np.random.randn(hidden_dims[-1], num_classes) * weight_scale
        self.params[f'b{self.num_layers}'] = np.zeros(shape=(num_classes,))
      

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        self.dropout_param = {}
        self.bn_params = []
        # for dropout usage
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
        # for batchnorm
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        # for layernorm
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # Initialize necessary lists and variables
        N = X.shape[0]
        layer_input = X
        cache = {} 
        # forward pass
        for i in range(1, self.num_layers + 1):
            W = self.params['W' + str(i)] 
            b = self.params['b' + str(i)]
            # affine
            out, cache_affine = affine_forward(layer_input, W, b)
            cache['affine' + str(i)] = cache_affine
            if self.normalization == 'batchnorm' and i < self.num_layers:
              # setting the shift and scale parameters for batch norm
                gamma = self.params['gamma' + str(i)]
                beta = self.params['beta' + str(i)]
                out, cache_bn = batchnorm_forward(out, gamma, beta, self.bn_params[i-1])
                cache['batchnorm' + str(i)] = cache_bn
            # ReLU layer
            if i < self.num_layers:
                out, cache_relu = relu_forward(out)
                cache['relu' + str(i)] = cache_relu
            # drop out
            if self.use_dropout and i < self.num_layers:
                out, cache_dropout = dropout_forward(out, self.dropout_param)
                cache['dropout' + str(i)] = cache_dropout
            layer_input = out
        scores = layer_input
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # compute data loss using softmax
        loss, dout = softmax_loss(scores, y)
        grads = {}
        #print("updated")
        for i in reversed(range(1, self.num_layers + 1)):
            # take the caches back
            cache_affine = cache.get('affine' + str(i), None)
            cache_relu = cache.get('relu' + str(i), None)
            cache_bn = cache.get('batchnorm' + str(i), None)
            cache_dropout = cache.get('dropout' + str(i), None)
            # for final layer, no ReLU or dropout is applied
            # if not, we will get the dropout, ReLU, and batchnorm backwards, if they were applied 
            if i == self.num_layers:
                dout, dw, db = affine_backward(dout, cache_affine)
            else:
                if self.use_dropout:
                    dout = dropout_backward(dout, cache_dropout)
                dout = relu_backward(dout, cache_relu)
                if self.normalization == 'batchnorm':
                    dout, dgamma, dbeta = batchnorm_backward(dout, cache_bn)
                    grads['gamma' + str(i)] = dgamma
                    grads['beta' + str(i)] = dbeta
                dout, dw, db = affine_backward(dout, cache_affine)
            # making sure gradient[k] holds the gradients for self.params[k]
            k = 'W' + str(i) 
            grads[k] = dw + self.reg * self.params[k]
            grads['b' + str(i)] = db

            # Update loss with regularization term
            loss += 0.5 * self.reg * np.sum(self.params[k] ** 2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
