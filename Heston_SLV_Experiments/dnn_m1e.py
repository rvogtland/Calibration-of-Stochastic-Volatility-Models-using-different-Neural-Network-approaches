import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from math import sqrt
from scipy.stats import norm
from scipy.stats import uniform
import cmath #for complex numbers
from scipy.integrate import quad #for numerical integration
from sklearn.preprocessing import MinMaxScaler
import scipy
import time
import multiprocessing
#from py_vollib.black_scholes.implied_volatility import implied_volatility


num_model_parameters = 6
num_strikes = 10
num_maturities = 13


num_input_parameters = 6
num_output_parameters = num_maturities*num_strikes
learning_rate = 0.0001
num_steps = 10
batch_size = 5
num_neurons = 40

#initial values
S0 = 1.0
V0 = 0.05
r = 0.0


contract_bounds = np.array([[0.8*S0,1.2*S0],[5,10]]) #bounds for K,T
model_bounds = np.array([[0.9,1.3],[0.2,0.8],[-1,0],[2,5],[0.05,0.1],[0.1,0.3]])  #bounds for alpha,beta,rho,a,b,c, make sure alpha>0,

"""
Note: The grid of stirkes and maturities is equidistant here put could be choosen differently for real world application.
Note: For the code below to striktly follow the bounds specified above make sure that *_distance x num_* is less than half the distance from the highest to lowest * (* = strikes/maturities). 
"""
maturities_distance = (contract_bounds[1,1]-contract_bounds[1,0])/(num_maturities) 
strikes_distance = (contract_bounds[0,1]-contract_bounds[0,0])/(num_strikes)

strikes = np.linspace(contract_bounds[0,0],contract_bounds[0,0]+num_strikes*strikes_distance,num_strikes)
maturities = np.linspace(contract_bounds[1,0],contract_bounds[1,0]+num_maturities*maturities_distance,num_maturities)

X = tf.placeholder(tf.float32, [None, num_input_parameters])
y = tf.placeholder(tf.float32, [None, num_output_parameters])

def corr_brownian_motion(n, T, dim, rho):
    dt = T/n

    dW1 = norm.rvs(size=(dim,n+1) , scale=sqrt(dt))
    dW2 = rho * dW1 + np.sqrt(1 - np.power(rho ,2)) * norm.rvs(size=(dim,n+1) , scale=sqrt(dt))
        
    W1 = np.cumsum(dW1, axis=-1)
    W2 = np.cumsum(dW2, axis=-1)
 
    return W1,W2

def euler_maruyama(mu,sigma,T,x0,W):
    dim = W.shape[0]
    n = W.shape[1]-1
    Y = np.zeros((dim,n+1))
    dt = T/n
    sqrt_dt = np.sqrt(dt)
    for l in range(dim):
        Y[l,0] = x0
        for i in range(n):
            Y[l,i+1] = Y[l,i] + np.multiply(mu(Y[l,i],l,i),dt) + sigma(Y[l,i],l,i)*sqrt_dt*(W[l,i+1]-W[l,i])
    
    return Y

def heston_SLV(alpha,beta,a,b,c,T,W,Z,V0,S0):
   
    if not 2*a*b > c*c:
        print("Error: a= ",a,", b= ",b,", c= ",c,", 2ab>c^2 not fullfilled")

    def mu2(V,i,k):
        return np.multiply(a,(b-V))
    
    def sigma2(V,i,k):
        return np.multiply(c,np.sqrt(np.maximum(0.0,V)))
    
    V = euler_maruyama(mu2,sigma2,T,V0,Z)
    
    def mu1(S,i,k):
        return np.multiply(r,S)
    
    def sigma1(S,i,k):
       
        return alpha*np.multiply(np.sqrt(np.maximum(0.0,V[i,k])),np.power(S,1+beta))
    
    S = euler_maruyama(mu1,sigma1,T,S0,W)
    
    return S,V

def reverse_transform_X(X_scaled):
    X = np.zeros(X_scaled.shape)
    for i in range(num_input_parameters):
        X[:,i] = X_scaled[:,i]*(model_bounds[i][1]-model_bounds[i][0]) + model_bounds[i][0]
    return X


def implied_vol(P,K,T):

    assert(S0>P)
    assert(P>=S0-K*np.exp(-r*T))

    def f(sigma):
        dplus = (np.log(S0 / K) + (r  + 0.5 * np.power(sigma, 2)) * T) / (sigma * np.sqrt(T))
        dminus = (np.log(S0 / K) + (r  - 0.5 * np.power(sigma, 2)) * T) / (sigma * np.sqrt(T))
        
        return S0 * norm.cdf(dplus, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(dminus, 0.0, 1.0) - P
     
    return scipy.optimize.brentq(f, 0.0001, 1)

def BS_call_price(sigma,K,T):
    dplus = (np.log(S0 / K) + (r  + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    dminus = (np.log(S0 / K) + (r  - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    return S0 * norm.cdf(dplus, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(dminus, 0.0, 1.0)


def next_batch_hestonSLV_EM_train(batch_size,contract_bounds,model_bounds):
    X_scaled = np.zeros((batch_size,num_input_parameters))
    y = np.zeros((batch_size,num_output_parameters))

    X_scaled[:,0] = uniform.rvs(size=batch_size) #alpha
    X_scaled[:,1] = uniform.rvs(size=batch_size) #beta
    X_scaled[:,2] = uniform.rvs(size=batch_size) #rho
    X_scaled[:,3] = uniform.rvs(size=batch_size) #a
    X_scaled[:,4] = uniform.rvs(size=batch_size) #b
    X_scaled[:,5] = uniform.rvs(size=batch_size) #c

    X = reverse_transform_X(X_scaled)

    n = 100
    dim = 10000
    for i in range(batch_size):
        W,Z = corr_brownian_motion(n,maturities[-1],dim,X[i,2])
        S,V = heston_SLV(X[i,0],X[i,1],X[i,3],X[i,4],X[i,5],maturities[-1],W,Z,V0,S0)
        
        for j in range(num_maturities):
            n_current = int(maturities[j]/maturities[-1]*n)
            S_T = S[:,n_current]
            
            for k in range(num_strikes):
                P = np.mean(np.maximum(S_T-np.ones(dim)*strikes[k],np.zeros(dim)))*np.exp(-r*maturities[j])
                
                y[i,j*num_strikes+k] = implied_vol(P,strikes[k],maturities[j])

                #y[i,j*num_strikes+k] = np.exp(-r*maturities[j])*np.mean(np.maximum(S_T-np.ones(dim)*strikes[k],np.zeros(dim)))
    return X_scaled,y

#Layers
hidden1 = fully_connected(X, num_neurons, activation_fn=tf.nn.elu)
bn1 = tf.nn.batch_normalization(hidden1, 0, 1, 0, 1, 0.000001)
hidden2 = fully_connected(bn1, num_neurons, activation_fn=tf.nn.elu)
bn2 = tf.nn.batch_normalization(hidden2, 0, 1, 0, 1, 0.000001)
#hidden3 = fully_connected(bn2, num_neurons, activation_fn=tf.nn.elu)
#bn3 = tf.nn.batch_normalization(hidden3, 0, 1, 0, 1, 0.000001)
hidden4 = fully_connected(bn2, num_neurons, activation_fn=tf.nn.elu)
bn4 = tf.nn.batch_normalization(hidden4, 0, 1, 0, 1, 0.000001)

outputs = fully_connected(bn4, num_output_parameters, activation_fn=None)

#Loss Function
loss = tf.reduce_mean(tf.sqrt(tf.square(outputs - y)))  # MSE

#Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

num_cpu = multiprocessing.cpu_count()
config = tf.ConfigProto(device_count={ "CPU": num_cpu },inter_op_parallelism_threads=num_cpu,intra_op_parallelism_threads=2)

with tf.device('/CPU:0'):
    with tf.Session(config=config) as sess:
        sess.run(init)
        
        for iteration in range(num_steps):
            
            X_batch,Y_batch = next_batch_hestonSLV_EM_train(batch_size,contract_bounds,model_bounds)
            sess.run(train,feed_dict={X: X_batch, y: Y_batch})
            
            if iteration % 1 == 0:
                
                rmse = loss.eval(feed_dict={X: X_batch, y: Y_batch})
                print(iteration, "\tRMSE:", rmse)
        
        saver.save(sess, "./Heston_SLV_Experiments/run/models/hestonSLV_dnn_e")


def prices_grid(theta):
    prices_true = np.zeros((1,num_output_parameters))
    n = 100
    dim = 10000
    W,Z = corr_brownian_motion(n,maturities[-1],dim,theta[2])
    S,V = heston_SLV(theta[0],theta[1],theta[3],theta[4],theta[5],maturities[-1],W,Z,V0,S0)
    
    for i in range(num_maturities):
        n_current = int(maturities[i]/maturities[-1]*n)
        S_T = S[:,n_current]
        for j in range(num_strikes):        
            prices_true[0,i*num_strikes+j] = np.exp(-r*maturities[i])*np.mean(np.maximum(S_T-np.ones(dim)*strikes[j],np.zeros(dim)))
    return prices_true

def predict_theta(prices_true):   
    
    def NNprediction(theta):
        x = np.zeros((1,len(theta)))
        x[0,:] = theta
        return sess.run(outputs,feed_dict={X: x})
    def NNgradientpred(x):
        x = np.asarray(x)
        grad = np.zeros((num_output_parameters,num_input_parameters))
        
        delta = 0.0000000001
        for i in range(num_input_parameters):
            h = np.zeros(x.shape)
            h[0,i] = delta
            
            #two point gradient
            #grad[i] = (sess.run(outputs,feed_dict={X: x+h}) - sess.run(outputs,feed_dict={X: x-h}))/2/delta

            #four point gradient
            grad[:,i] = (-sess.run(outputs,feed_dict={X: x+2*h})+8*sess.run(outputs,feed_dict={X: x+h})-8*sess.run(outputs,feed_dict={X: x-h}) +sess.run(outputs,feed_dict={X: x-2*h}))/12/delta

        return -np.mean(grad,axis=0)

    def CostFuncLS(theta):
        
        return np.mean(np.power((NNprediction(theta)-prices_true.flatten())[0],2),axis=0)


    def JacobianLS(theta):
        x = np.zeros((1,len(theta)))
        x[0,:] = theta
        return NNgradientpred(x).T

    with tf.Session() as sess:                          
        #saver.restore(sess, "./models/hestonSLV")  
        saver.restore(sess, "./Heston_SLV_Experiments/run/models/hestonSLV_dnn_e")    
        
        init = [model_bounds[0,0]+uniform.rvs()*(model_bounds[0,1]-model_bounds[0,0]),model_bounds[1,0]+uniform.rvs()*(model_bounds[1,1]-model_bounds[1,0]),model_bounds[2,0]+uniform.rvs()*(model_bounds[2,1]-model_bounds[2,0]),model_bounds[3,0]+uniform.rvs()*(model_bounds[3,1]-model_bounds[3,0]),model_bounds[4,0]+uniform.rvs()*(model_bounds[4,1]-model_bounds[4,0]),model_bounds[5,0]+uniform.rvs()*(model_bounds[5,1]-model_bounds[5,0])]
        bnds = ([model_bounds[0,0],model_bounds[1,0],model_bounds[2,0],model_bounds[3,0],model_bounds[4,0],model_bounds[5,0]],[model_bounds[0,1],model_bounds[1,1],model_bounds[2,1],model_bounds[3,1],model_bounds[4,1],model_bounds[5,1]])

        
        I=scipy.optimize.least_squares(CostFuncLS,init,JacobianLS,bounds=bnds,gtol=1E-15,xtol=1E-15,ftol=1E-15,verbose=1)

    theta_pred = I.x
    
    return theta_pred

def NNprediction(theta):
    x = np.zeros((1,len(theta)))
    x[0,:] = theta
    return sess.run(outputs,feed_dict={X: x})

N = 2

thetas_true_rand = reverse_transform_X(uniform.rvs(size=(N,num_model_parameters)))

thetas_pred_rand = np.zeros((N,num_model_parameters))
for i in range(N):
    thetas_pred_rand[i,:] = predict_theta(prices_grid(thetas_true_rand[i,:]).flatten())

prices_grid_true_2 = np.zeros((N,num_maturities,num_strikes))
prices_grid_pred_1 = np.zeros((N,num_maturities,num_strikes))
prices_grid_pred_2 = np.zeros((N,num_maturities,num_strikes))
n = 100
dim = 10000


with tf.Session() as sess:         
                       
    saver.restore(sess, "./Heston_SLV_Experiments/run/models/hestonSLV_dnn_e") 

    for i in range(N):
        W,Z = corr_brownian_motion(n,maturities[-1],dim,thetas_true_rand[i,2])

        S1,V1 = heston_SLV(thetas_true_rand[i,0],thetas_true_rand[i,1],thetas_true_rand[i,3],thetas_true_rand[i,4],thetas_true_rand[i,5],maturities[-1],W,Z,V0,S0)
        S2,V2 = heston_SLV(thetas_pred_rand[i,0],thetas_pred_rand[i,1],thetas_pred_rand[i,3],thetas_pred_rand[i,4],thetas_pred_rand[i,5],maturities[-1],W,Z,V0,S0)

        tmp = NNprediction(thetas_pred_rand[i,:])
        for j in range(num_maturities):
            n_current = int(maturities[j]/maturities[-1]*n)
            S_T1 = S1[:,n_current]
            S_T2 = S2[:,n_current]
            for k in range(num_strikes):
                prices_grid_true_2[i,j,k] = np.exp(-r*maturities[j])*np.mean(np.maximum(S_T1-np.ones(dim)*strikes[k],np.zeros(dim)))
                prices_grid_pred_2[i,j,k] = np.exp(-r*maturities[j])*np.mean(np.maximum(S_T2-np.ones(dim)*strikes[k],np.zeros(dim)))
                prices_grid_pred_1[i,j,k] = tmp[0,j*num_strikes+k]

import matplotlib



fig = plt.figure(figsize=(20, 6))

ax1=fig.add_subplot(131)

plt.imshow(np.mean(np.abs((prices_grid_true_2-prices_grid_pred_2)/prices_grid_true_2),axis=0))
plt.title("Average Relative Errors (MC with new theta)")

ax1.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax1.set_yticklabels(np.around(maturities,1))
ax1.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax1.set_xticklabels(np.around(strikes,2))
plt.colorbar()

ax2=fig.add_subplot(132)

plt.imshow(np.max(np.abs((prices_grid_true_2-prices_grid_pred_2)/prices_grid_true_2),axis=0))
plt.title("Max Relative Errors (MC with new theta)")

ax2.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax2.set_yticklabels(np.around(maturities,1))
ax2.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax2.set_xticklabels(np.around(strikes,2))

plt.colorbar()

ax1=fig.add_subplot(133)

plt.imshow(np.mean(np.abs((prices_grid_true_2-prices_grid_pred_1)/prices_grid_true_2),axis=0))
plt.title("Average Relative Errors NN")

ax1.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax1.set_yticklabels(np.around(maturities,1))
ax1.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax1.set_xticklabels(np.around(strikes,2))
plt.colorbar()


plt.savefig('images/errors_dnn_m1_euler_hestonSLV.pdf')