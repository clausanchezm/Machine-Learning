import numpy as np
import random

def randInitializeWeights(L_in, L_out):
#RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
#incoming connections and L_out outgoing connections
#   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
#   of a layer with L_in incoming connections and L_out outgoing 
#   connections. 
#
#   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
#   the first row of W handles the "bias" terms
#

# ====================== YOUR CODE HERE ======================
# Instructions: Initialize W randomly so that we break the symmetry while
#               training the neural network.
#
# Note: The first row of W corresponds to the parameters for the bias units
#get a row vector of the entire mag of 1's

# You need to return the following variables correctly 
    W = np.zeros((L_out,  L_in))

    one = np.ones((W.shape[0], 1))
    W = np.append(one, W, axis= 1)
    epsilon = 0.12
    W =  2 * np.random.rand(L_out, L_in) * epsilon - epsilon

    return W
