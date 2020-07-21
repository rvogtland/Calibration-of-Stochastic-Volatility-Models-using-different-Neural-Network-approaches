import numpy as np
import tensorflow as tf

from scipy.stats import norm
from scipy.stats import uniform

import scipy
import multiprocessing
from scipy.optimize import brentq

import numpy as np


num_data_points = 10

num_model_parameters = 6
num_strikes = 10
num_maturities = 10


num_input_parameters = 6 + 2
num_output_parameters = 1


#initial values
S0 = 1.0
V0 = 0.05
r = 0.0

contract_bounds = np.array([[0.8*S0,1.2*S0],[5,10]]) #bounds for K,T
model_bounds = np.array([[0.9,1.3],[0.2,0.8],[-0.8,-0.2],[2,5],[0.05,0.1],[0.1,0.3]])  #bounds for alpha,beta,rho,a,b,c, make sure alpha>0,


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

def heston_SLV(alpha,beta,a,b,c,T,W,Z,V0,S0):
   
    if not 2*a*b > c*c:
        print("Error: a= ",a,", b= ",b,", c= ",c,", 2ab>c^2 not fullfilled")

    def mu2(V):
        return np.multiply(a,(b-V))
    
    def sigma2(V,i):
        return np.multiply(c,np.sqrt(np.maximum(np.zeros(V.shape[0]),V)))
    
    V = euler_maruyama(mu2,sigma2,T,V0,Z)
    
    def mu1(S):
        return np.zeros(S.shape)
    
    def sigma1(S,i):
       
        return alpha*np.multiply(np.sqrt(np.maximum(np.zeros(V.shape[0]),V[:,i])),np.power(np.maximum(S,np.zeros(S.shape[0])),1+beta))
    
    S = euler_maruyama(mu1,sigma1,T,S0,W)
    
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

    IVS = np.zeros((num_maturities,num_strikes))
    n = 100
    dim = 10000
    
    W,Z = corr_brownian_motion(n,maturities[-1],dim,theta[2])
    S,V = heston_SLV(theta[0],theta[1],theta[3],theta[4],theta[5],maturities[-1],W,Z,V0,S0)
    
    for j in range(num_maturities):
        n_current = int(maturities[j]/maturities[-1]*n)
        S_T = S[:,n_current]
        
        for k in range(num_strikes):
            P =  np.mean(np.maximum(S_T-np.ones(dim)*strikes[k],np.zeros(dim)))
            
            IVS[j,k] = implied_vol(P,strikes[k],maturities[j])
    
    return IVS

def iv(X):
    ivs = np.zeros((1,1))
    n = 200
    dim = 40000
    W,Z = corr_brownian_motion(n,X[7],dim,X[2])
    S,V = heston_SLV(X[0],X[1],X[3],X[4],X[5],X[7],W,Z,V0,S0)
    
    S_T = S[:,-1]
          
    P =  np.mean(np.maximum(S_T-np.ones(dim)*X[6],np.zeros(dim)))
     
    ivs[0,0] = implied_vol(P,X[6],X[7])
    
    return ivs



def reverse_transform_X(X_scaled):
    X = np.zeros(X_scaled.shape)
    for i in range(num_model_parameters):
        X[:,i] = X_scaled[:,i]*(model_bounds[i][1]-model_bounds[i][0]) + model_bounds[i][0]
    for i in range(2):
        X[:,num_model_parameters + i] = X_scaled[:,i]*(contract_bounds[i][1]-contract_bounds[i][0]) + contract_bounds[i][0]
    return X

X = reverse_transform_X(uniform.rvs(size=(num_data_points,num_input_parameters)))
data = np.zeros((num_data_points,num_input_parameters + num_output_parameters))
for i in range(num_data_points):
    data[i,:num_input_parameters] = X[i,:]
    data[i,-1] = iv(X[i,:])
    if i % 100 == 0:
        print(i)

f=open('hestonLV_data_m2.csv','ab')

np.savetxt(f, data, delimiter=',')

f.close() 