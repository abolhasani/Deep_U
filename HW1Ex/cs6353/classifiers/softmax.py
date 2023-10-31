import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  # talked about this in lecture 9
  for i in range(num_train):
    scores = X[i].dot(W)
    #scores -= np.max(scores)
    EXP_scores = np.exp(scores)
    SCORE_SUM = np.sum(EXP_scores)
    #OTHER = np.log(SCORE_SUM)
    loss += -np.log(EXP_scores[y[i]] / SCORE_SUM)
    for j in range(num_classes):
      if j == y[i]:
        dW[:, j] += (EXP_scores[j] / SCORE_SUM - 1) * X[i]
      else:
        dW[:, j] += EXP_scores[j] / SCORE_SUM * X[i]
  # Average loss and gradient
  loss /= num_train
  loss += reg * np.sum(W * W) # FINAL_LOSS

  # Regularization, lambda is 1 
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  """
  num_train = X.shape[0]
  num_classes = W.shape[1]
  # talked about this in lecture 9
  for i in range(num_train):
    scores = X[i].dot(W)
    #scores -= np.max(scores)
    EXP_scores = np.exp(scores)
    SCORE_SUM = np.sum(EXP_scores)
    #OTHER = np.log(SCORE_SUM)
    loss += -np.log(EXP_scores[y[i]] / SCORE_SUM)
    for j in range(num_classes):
      if j == y[i]:
        dW[:, j] += (EXP_scores[j] / SCORE_SUM - 1) * X[i]
      else:
        dW[:, j] += EXP_scores[j] / SCORE_SUM * X[i]
  # Average loss and gradient
  loss /= num_train
  loss += reg * np.sum(W * W) # FINAL_LOSS
  # Regularization, lambda is 1 
  dW /= num_train
  dW += 2 * reg * W
  """
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  EXP_scores = np.exp(scores)
  EXP_sum = np.sum(EXP_scores, axis =1, keepdims=True)
  EXP_frac = EXP_scores / np.sum(EXP_scores, axis=1, keepdims=True)
  loss = np.sum(EXP_frac) / num_train
  # regularization! as always. 
  loss += reg * np.sum(W * W)
  # we talked about this part in lecture 9
  dscores = EXP_scores / np.sum(EXP_scores, axis=1, keepdims=True)
  dscores[np.arange(num_train), y] -= 1
  dscores /= num_train
  dW = X.T.dot(dscores)
  dW += 2*reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

