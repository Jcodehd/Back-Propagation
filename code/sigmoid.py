import numpy as np

def sigmoid(z):
    result = 1./(1+np.exp(-z));
    return result;