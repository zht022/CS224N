#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X          -- M x Dx matrix, where each row is a training example x.
    labels     -- M x Dy matrix, where each row is a one-hot vector.
    params     -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### Forward propagation
    z = X.dot(W1) + b1
    h = sigmoid(z)
    theta = h.dot(W2) + b2
    y_hat = softmax(theta)
    cost = -np.sum(labels * np.log(y_hat))

    ### Backward propagation
    # Note: the gradients computed here are w.r.t.weights.
    grad_theta = y_hat - labels
    grad_b2 = np.sum(grad_theta, axis=0, keepdims=True)
    grad_W2 = np.dot(h.T, grad_theta)
    grad_h = np.dot(grad_theta, W2.T)
    grad_sigmoid = grad_h * sigmoid_grad(h)
    grad_b1 = np.sum(grad_sigmoid, axis=0, keepdims=True)
    grad_W1 = np.dot(X.T, grad_sigmoid)

    assert grad_b2.shape == b2.shape
    assert grad_W2.shape == W2.shape
    assert grad_b1.shape == b1.shape
    assert grad_W1.shape == W1.shape

    ### Stack gradients (do not modify)
    grad = np.concatenate((grad_W1.flatten(), grad_b1.flatten(), grad_W2.flatten(), grad_b2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using gradcheck.
    """
    print ("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a data
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
