import numpy as np 
from scipy.stats import norm
from scipy.stats import uniform

class Heston:
    
    def __init__(self):
        

    def corr_brownian_motion(self, n, T, dim, rho):
        dt = T/n

        dW1 = norm.rvs(size=(dim,n+1) , scale=sqrt(dt))
        dW2 = rho * dW1 + np.sqrt(1 - np.power(rho ,2)) * norm.rvs(size=(dim,n+1) , scale=sqrt(dt))
            
        W1 = np.cumsum(dW1, axis=-1)
        W2 = np.cumsum(dW2, axis=-1)
 
    return W1,W2


if __name__ == '__main__':
    pass