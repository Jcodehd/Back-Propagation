import numpy as np;
import random as rd;

def initializeTheta(put, out):

    epsilon_init = 0.12;

    result = np.random.rand(out, put+1)*2*epsilon_init-epsilon_init;

    return result;
