import numpy as np
import math

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    vec=np.array(x)
    
    sig=1/(1+np.exp(-vec))
    return sig# Write code here