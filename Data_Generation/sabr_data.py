import numpy as np
import tensorflow as tf

from scipy.stats import norm
from scipy.stats import uniform

import scipy
import multiprocessing
from scipy.optimize import brentq

import numpy as np


num_data_points = 1000
num_model_parameters = 3
contract_bounds = np.array([[0.8,1.2],[5,10]]) #bounds for K,T
model_bounds = np.array([[0.01,0.15],[0,1],[-0.9,-0.1]]) #bounds for alpha,beta,rho, make sure alpha>0, beta \in [0,1], rho \in [-1,1]
V0 = 0.3
S0 = 1
r = 0.0
num_strikes = 8
num_maturities = 8

maturities_distance = (contract_bounds[1,1]-contract_bounds[1,0])/(num_maturities) 
strikes_distance = (contract_bounds[0,1]-contract_bounds[0,0])/(num_strikes)

strikes = np.linspace(contract_bounds[0,0],contract_bounds[0,0]+num_strikes*strikes_distance,num_strikes)
maturities = np.linspace(contract_bounds[1,0],contract_bounds[1,0]+num_maturities*maturities_distance,num_maturities)



def corr_brownian_motion(n, T, dim, rho):
    dt = T/n

    dW1 = norm.rvs(size=(dim,n+1) , scale=np.sqrt(dt))
    dW2 = rho * dW1 + np.sqrt(1 - np.power(rho ,2)) * norm.rvs(size=(dim,n+1) , scale=np.sqrt(dt))
        
    W1 = np.cumsum(dW1, axis=-1)
    W2 = np.cumsum(dW2, axis=-1)
 
    return W1,W2

def euler_maruyama(mu,sigma,T,x0,W):
    dim = W.shape[0]
    n = W.shape[1]-1
    Y = np.zeros((dim,n+1))
    dt = T/n
    sqrt_dt = np.sqrt(dt)
    Y[:,0] = x0
    
    for i in range(n):
        Y[:,i+1] = Y[:,i] + np.multiply(mu(Y[:,i]),dt) + sigma(Y[:,i],i)*sqrt_dt*(W[:,i+1]-W[:,i])
    
    return Y

def sabr(alpha,beta,T,W,Z,V0,S0):
#assert(beta>0 and beta<1)

    #def mu2(V,i,k):
    #    return 0.0
    
    #def sigma2(V,i,k):
    #    return np.multiply(alpha,V)
    
    #V = euler_maruyama(mu2,sigma2,T,V0,Z)
    
    def V(k):
        n = W.shape[1]-1
        t = k*T/n
        return V0*np.exp(-alpha*alpha/2*t+alpha*Z[:,k])

    def mu(S):
        return np.zeros(S.shape)
    
    def sigma(S,k):
        return np.multiply(V(k),np.power(np.maximum(0.0,S),beta))
    
    S = euler_maruyama(mu,sigma,T,S0,W)
    
    return S,V

def implied_vol(P,K,T):
    if not P<S0:
        print("P<S0 = ",P<S0,", abitrage!")
        return 0.0
    if not P>S0-K*np.exp(-r*T):
        print("P>S0-K*np.exp(-r*T) = ",P>S0-K*np.exp(-r*T),", abitrage!")
        return 0.0

    def f(sigma):
        dplus = (np.log(S0 / K) + (r  + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        dminus = (np.log(S0 / K) + (r  - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        return S0 * norm.cdf(dplus, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(dminus, 0.0, 1.0) - P
    
    return scipy.optimize.brentq(f, 0.00001, 100000)


def implied_vols_surface(theta):
    #INPUT: theta = (alpha,beta,rho)
    #OUTPUT: implied volatility surface

    IVS = np.zeros((num_maturities,num_strikes))
    n = 500
    dim = 200000
    
    W,Z = corr_brownian_motion(n,maturities[-1],dim,theta[2])
    S,V = sabr(theta[0],theta[1],maturities[-1],W,Z,V0,S0)
    
    for j in range(num_maturities):
        n_current = int(maturities[j]/maturities[-1]*n)
        S_T = S[:,n_current]
        
        for k in range(num_strikes):
            P =  np.mean(np.maximum(S_T-np.ones(dim)*strikes[k],np.zeros(dim)))
            
            IVS[j,k] = implied_vol(P,strikes[k],maturities[j])
    
    return IVS



def reverse_transform_theta(X_scaled):
    X = np.zeros(X_scaled.shape)
    for i in range(num_model_parameters):
        X[:,i] = X_scaled[:,i]*(model_bounds[i][1]-model_bounds[i][0]) + model_bounds[i][0]

    return X

theta = reverse_transform_theta(uniform.rvs(size=(num_data_points,num_model_parameters)))
data = np.zeros((num_data_points,num_model_parameters+num_strikes*num_maturities))
for i in range(num_data_points):
    data[i,:num_model_parameters] = theta[i,:]
    data[i,num_model_parameters:] = implied_vols_surface(theta[i,:]).flatten()
    if i % 10 == 0:
        print(i)

f=open('sabr_data1e6.csv','ab')

np.savetxt(f, data, delimiter=',')

f.close() 