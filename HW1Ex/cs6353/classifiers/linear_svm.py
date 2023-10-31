import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on mini batches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a mini batch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] +=X[i]
        dW[:, y[i]] -= X[i]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # we do the same regularization step for dW
  dW /= num_train
  dW += 2*reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  # for loss, everything is ok andwe can return the value. We should only adjust dw
  # adjustments to dw have done above
  # dW w.r.t. w is X[i], because we have w.T*X. 
  # when the margin is bigger than zero, then its not the correct class.
  # for the wrong class, we add X in direction to increase the penalty weight, and for the correct one we subtract X to decrease it.
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # same as above just without loops
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] +=X[i]
        dW[:, y[i]] -= X[i]
  loss /= num_train
  loss += reg * np.sum(W * W)
  """
  # same initializationas before:
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  scores = X.dot(W)
  # the following line of code adds a total of 'num_train' correct answers compared to y from scores in 
  correct_class_scores = scores[np.arange(num_train),y]
  margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1)
  # but when the class is correct, they shouldn't be included in the margins, so we zero them to not contribute to loss. 
  margins[np.arange(num_train),y] = 0.0
  loss = np.sum(margins) / num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  """
  for i in range(num_train):
  scores = X[i].dot(W)
  correct_class_score = scores[y[i]]
  for j in range(num_classes):
    if j == y[i]:
      continue
    margin = scores[j] - correct_class_score + 1 # note delta = 1
    if margin > 0:
      loss += margin
      dW[:, j] +=X[i]
      dW[:, y[i]] -= X[i]
  """
  # check if margin has bigger than 0 values that needs fixing
  W_count = np.sum(margins > 0, axis=1)
  margins[margins > 0] = 1
  margins[np.arange(num_train), y] = -W_count
  dW = X.T.dot(margins) / num_train
  # regularization
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
