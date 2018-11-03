"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

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
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    D, C = W.shape
    N, D = X.shape
    
    s = X.dot(W)

    for i in range(N):
        scores = s[i]
        scores_safe = scores - np.max(scores)
        target = y[i]
        
        Syi = scores_safe[target]
        normalizer = np.sum(np.exp(scores_safe))        
        Li = np.negative(np.log(np.exp(Syi)/normalizer))
        
        loss += Li

        for j in range(C):
            softmax_output = np.exp(scores_safe[j])/sum(np.exp(scores_safe))
            if j == target:
               dW[:,j] += (-1 + softmax_output) * X[i] 
            else:
               dW[:,j] += softmax_output * X[i]
            
    loss /= N
    loss +=  0.5 * reg * np.sum(W * W)
    dW = dW / N + reg * W 

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    D, C = W.shape
    N, D = X.shape
    
    scores = X.dot(W)
    
    expScores_stable = np.exp(scores - np.max(scores, axis = 1, keepdims=True))
    
    probs = expScores_stable / np.sum(expScores_stable, axis=1, keepdims=True)
    
    loss = -1.0 * np.sum(np.log(probs[list(range(N)), y])) / N + 0.5 * reg * np.sum(W*W)
    
    dscores = probs
    dscores[list(range(N)), y] -= 1
    dscores /= N
    
    dW = (X.T).dot(dscores) + reg * W

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = np.arange(1e-8, 9e-8, 5e-9)
    regularization_strengths = np.arange(1e4,9e4,5e3)
    
    for learning_rate in learning_rates:
        for reg in regularization_strengths:
            # print('Learning rate: %e, Regularization strength: %e' % (learning_rate, reg))
            print("=",end="")
            softmax = SoftmaxClassifier()
            loss_hist = softmax.train(X_train, y_train, learning_rate=learning_rate, reg=reg, num_iters=800, verbose=False)
            y_train_pred = softmax.predict(X_train)
            training_accuracy = np.mean(y_train_pred == y_train)
            y_val_pred = softmax.predict(X_val)
            validation_accuracy = np.mean(y_val_pred == y_val)
            results[(learning_rate, reg)] = (training_accuracy, validation_accuracy)
            all_classifiers.append((softmax, validation_accuracy, learning_rate, reg))
            
            if validation_accuracy > best_val:
                best_val = validation_accuracy
                best_softmax = softmax            
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
