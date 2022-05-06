import numpy as np

from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
# Useful values
    m = np.shape(X)[0]              #number of examples
    # You need to return the following variables correctly 
    p = np.zeros(m);


# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#
    
    #add "bias" term to the X-matrix since it doesnt contain a col of only 1's
    newX = np.append(np.ones((m, 1)),X, axis = 1)
    
    #we must multiply tehta(edge weighs) to the nodes 
    
    
    z2 = np.dot( newX, Theta1.T)
    a2 = sigmoid(z2)
    
    newA2 = np.append(np.ones((m, 1)), a2, axis= 1)
    z3= np.dot( newA2, Theta2.T)
    a3 = sigmoid(z3)
    #thsi retrns the index col with teh max value 
    p = np.argmax(a3, axis= 1)
    #add +1 inorder to keep with teh convention of indexes
    
    return p+1
