import re
import numpy as np;
from sigmoid import sigmoid;

def sigmoidGradient(z):

    z = np.array(z);

    result = sigmoid(z)*(1-sigmoid(z));

    return result;