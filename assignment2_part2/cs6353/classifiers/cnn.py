from builtins import object
import numpy as np
from cs6353.layers import *
from cs6353.layer_utils import *


class ConvNet(object):
    """
   A simple convolutional network with the following architecture:

    [conv - bn - relu] x M - max_pool - affine - softmax
    
    "[conv - bn - relu] x M" means the "conv-bn-relu" architecture is repeated for
    M times, where M is implicitly defined by the convolution layers' parameters.
    
    For each convolution layer, we do downsampling of factor 2 by setting the stride
    to be 2. So we can have a large receptive field size.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[32], filter_sizes=[7],
                 num_classes=10, weight_scale=1e-3, reg=0.0,use_batch_norm=True, 
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer. It is a
          list whose length defines the number of convolution layers
        - filter_sizes: Width/height of filters to use in the convolutional layer. It
          is a list with the same length with num_filters
        - num_classes: Number of output classes
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - use_batch_norm: A boolean variable indicating whether to use batch normalization
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the simple convolutional         #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params.                                                 #
        #                                                                          #
        # IMPORTANT:                                                               #
        # 1. For this assignment, you can assume that the padding                  #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. You need to         #
        # carefully set the `pad` parameter for the convolution.                   #
        #                                                                          #
        # 2. For each convolution layer, we use stride of 2 to do downsampling.    #
        ############################################################################
        self.num_filters = num_filters
        # get dimension inputs and get the updated height and width of the layer
        C, H, W = input_dim
        HP = (H - 2)//2 + 1 
        WP = (W - 2)//2 + 1
        # looking at the _init_ at fully connected and doing the initializations
        count_conv_layers = len(self.num_filters) # num of convo layers
        self.conv_filters = self.num_filters
        last_filter_count = C
        final_H, final_W = H, W
        for idx in range(count_conv_layers):
            num_f = self.conv_filters[idx]
            filter_h = filter_sizes[idx]
            filter_w = filter_sizes[idx]
            s = 2
            padding = (filter_h - 1) // 2
            # set ther weights and biases keys for convo layer
            conv_w_key, conv_b_key = 'W' + str(idx + 1), 'b' + str(idx + 1)
            self.params[conv_w_key] = np.random.normal(0, weight_scale, (num_f, last_filter_count, filter_h, filter_w))
            self.params[conv_b_key] = np.zeros(num_f)
            # update after convolution layer
            final_H = 1 + (final_H - filter_h + 2 * padding) // s
            final_W = 1 + (final_W - filter_w + 2 * padding) // s
            # initializations for batch normalization 
            if use_batch_norm:
                self.params['gamma' + str(idx + 1)] = np.ones(num_f)
                self.params['beta' + str(idx + 1)] = np.zeros(num_f)
            last_filter_count = num_f
        #print("update")
        # the initializations of weights and biases for the fully connected layer.
        fc_w_key = 'W' + str(count_conv_layers + 1)
        fc_b_key = 'b' + str(count_conv_layers + 1)
        self.params[fc_w_key] = np.random.normal(0, weight_scale, (last_filter_count * (final_H // 2 )* (final_W // 2), num_classes))
        self.params[fc_b_key] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        scores = None
        mode = 'test' if y is None else 'train'
        ############################################################################
        # TODO: Implement the forward pass for the simple convolutional net,       #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # using fast functions to run the code faster. else, replace them with naive functions
        n = len(self.num_filters)
        # for a more comfortable use when working with n+1 in keys
        num = n+1
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        cache = {}
        out = X.copy()  
        # set the stride for the convolutional layer fixed to 2
        conv_param = {'stride': 2}
        # set params for max pool to the usual fixed 2*2 with stride 2
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        # for each filter
        for i in range(1, num):
          # get weights and biases for layer
            W_i = self.params[f'W{i}']
            b_i = self.params[f'b{i}']
            # set convolutional params based on the newly updated weights
            conv_param = {'stride': 2, 'pad': (W_i.shape[2] - 1) // 2}
            # determine batchnorm with gamma, then initialize and run a convolutional layer
            # with batchnorm and relu
            if 'gamma' + str(i) in self.params:
                gamma = self.params[f'gamma{i}'] 
                beta = self.params[f'beta{i}']
                out, cache['conv_bn_relu' + str(i)] = conv_bn_relu_forward(out, W_i, b_i, gamma, beta, conv_param, {'mode': mode})
            # run convolutional for non batchnorm with relu
            else:
                out, cache['conv_relu' + str(i)] = conv_relu_forward(out, W_i, b_i, conv_param)
        # doing the max pool layer
        out, cache['pool'] = max_pool_forward_fast(out, pool_param)
        # do the bias and weights for the last (fc) layer
        W_i, b_i = self.params['W' + str(n + 1)], self.params['b' + str(n + 1)]
        # the last forward step by calling the affine (fc) forward function
        scores, cache['affine'] = affine_forward(out, W_i, b_i)

        ############################################################################                                                                                                            ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the simple convolutional net,      #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # getting softmax loss and gradients of inputs, weights, and biases
        loss, score_grad = softmax_loss(scores, y)
        dout, w_next, b_next = affine_backward(score_grad, cache['affine'])
        # assigning weights and biases to their respective keys in grads
        grads[f'W{num}'], grads[f'b{num}'] = w_next, b_next
        #print("al-update")
        # getting grads of max pooling from the fast library (that I had to copy the files from part1)
        # it can be replaced by the naive one I wrote earlier
        dout = max_pool_backward_fast(dout, cache['pool'])
        # apply regularization
        # adding regularization to the gradient of the last layer's weights
        grads[f'W{num}'] += self.reg * self.params[f'W{num}']
        # for each filter (reversed)
        for idx in reversed(range(1,num)):
            gamma_key = f'gamma{idx}'
            # we have gomma when we have batchnorm
            if gamma_key in self.params:
                # getting back the grads from convolutional layer with batchnorm and relu
                dout, w_, b_, gamma_, beta_ = conv_bn_relu_backward(dout, cache[f'conv_bn_relu{idx}'])
                # assigning the new gradients in place for each filter
                grads[f'W{idx}'], grads[f'b{idx}'], grads[gamma_key], grads[f'beta{idx}'] = w_, b_, gamma_, beta_
            else:
              # same but for nin-batchnorms
                dout, w_, b_ = conv_relu_backward(dout, cache[f'conv_relu{idx}'])
                grads[f'W{idx}'], grads[f'b{idx}'] = w_, b_
            # weight update
            grads[f'W{idx}'] += self.reg * self.params[f'W{idx}']
        # L2 reg loss update 
        reg_loss_term = sum(np.sum(self.params[f'W{i}'] ** 2) for i in range(1, n + 1))
        loss += 0.5 * self.reg * (reg_loss_term + np.sum(self.params[f'W{num}'] ** 2))
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads