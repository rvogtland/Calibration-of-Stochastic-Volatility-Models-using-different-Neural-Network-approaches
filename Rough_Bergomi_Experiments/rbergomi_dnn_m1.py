import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from math import sqrt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.preprocessing import MinMaxScaler
import scipy
import time
import multiprocessing
from scipy.optimize import brentq

import os
#change this to your own path
os.chdir('/cluster/home/robinvo/rv_bachelor_thesis/Rough_Bergomi_Experiments/rbergomi')

import numpy as np
from matplotlib import pyplot as plt
from rbergomi import rBergomi 


""" Helper Functions from utils.py """
def g(x, a):
    """
    TBSS kernel applicable to the rBergomi variance process.
    """
    return x**a

def b(k, a):
    """
    Optimal discretisation of TBSS process for minimising hybrid scheme error.
    """
    return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)

def cov(a, n):
    """
    Covariance matrix for given alpha and n, assuming kappa = 1 for
    tractability.
    """
    cov = np.array([[0.,0.],[0.,0.]])
    cov[0,0] = 1./n
    cov[0,1] = 1./((1.*a+1) * n**(1.*a+1))
    cov[1,1] = 1./((2.*a+1) * n**(2.*a+1))
    cov[1,0] = cov[0,1]
    return cov

def bs(F, K, V, o = 'call'):
    """
    Returns the Black call price for given forward, strike and integrated
    variance.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    sv = np.sqrt(V)
    d1 = np.log(F/K) / sv + 0.5 * sv
    d2 = d1 - sv
    P = w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)
    return P

def bsinv(P, F, K, t, o = 'call'):
    """
    Returns implied Black vol from given call price, forward, strike and time
    to maturity.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    # Ensure at least instrinsic value
    P = np.maximum(P, np.maximum(w * (F - K), 0))

    def error(s):
        return bs(F, K, s**2 * t, o) - P
    s = brentq(error, 1e-9, 1e+9)
    return s

vec_bsinv = np.vectorize(bsinv)


""" Definiton of some Hyper Parameters """

num_forward_var = 1
num_model_parameters = 3 + num_forward_var
num_strikes = 8
num_maturities = 8

num_input_parameters = num_model_parameters
num_output_parameters = num_maturities*num_strikes
learning_rate = 0.00001
num_steps = 50
batch_size = 32
num_neurons = 40

#initial values
S0 = 1.0
r = 0.00


contract_bounds = np.array([[0.8*S0,1.2*S0],[0.1,2]]) #bounds for K,T
model_bounds = np.array([[0.1,0.5],[0.5,3],[-0.9,-0.1],[0.01,0.15]]) #bounds for H,eta,rho,lambdas


#Note: The grid of stirkes and maturities is equidistant here put could be choosen differently for real world application.
#Note: For the code below to striktly follow the bounds specified above make sure that *_distance x num_* is less than half the distance from the highest to lowest * (* = strikes/maturities). 

maturities_distance = (contract_bounds[1,1]-contract_bounds[1,0])/(2*num_maturities) 
strikes_distance = (contract_bounds[0,1]-contract_bounds[0,0])/(2*num_strikes)

strikes = np.linspace(contract_bounds[0,0],contract_bounds[0,0]+num_strikes*strikes_distance,num_strikes)
maturities = np.linspace(contract_bounds[1,0],contract_bounds[1,0]+num_maturities*maturities_distance,num_maturities)

np.random.seed(42)

""" Helper functions """

def reverse_transform_X(X_scaled):
    X = np.zeros(X_scaled.shape)
    for i in range(3):
        X[:,i] = X_scaled[:,i]*(model_bounds[i][1]-model_bounds[i][0]) + model_bounds[i][0]
    
    X[:,3] = X_scaled[:,3]*(model_bounds[3][1]-model_bounds[3][0]) + model_bounds[3][0]
    
    return X

def implied_vols_surface(theta):
    #INPUT: theta = (H,eta,rho,lambda)
    #OUTPUT: implied volatility surface

    IVS = np.zeros((num_maturities,num_strikes))

    for i in range(num_maturities):
        rB = rBergomi.rBergomi(n = 100, N = 30000, T = maturities[i], a = theta[0]-0.5)

        dW1 = rB.dW1()
        dW2 = rB.dW2()

        Y = rB.Y(dW1)

        dB = rB.dB(dW1, dW2, rho = theta[2])

        V = rB.V(Y, xi = theta[3], eta = theta[1])

        S = rB.S(V, dB) 

        ST = S[:,-1][:,np.newaxis]
        call_payoffs = np.maximum(ST - strikes,0)
        
        call_prices = np.mean(call_payoffs, axis = 0)[:,np.newaxis]
        K = strikes[np.newaxis,:]
        implied_vols = vec_bsinv(call_prices, S0, np.transpose(K), maturities[i])
      
        IVS[i,:] = implied_vols[:,0]
    
    return IVS

def next_batch_rBergomi(batch_size,contract_bounds,model_bounds):
    #INPUT: batch size, bounds for contract and model parameters, NO CHECKS IF BOUNDS FULLFILL NO-ABITRAGE CONDITION
    #OUTPUT: random model parameters (scaled!) and corresponding implied volatility surface

    X_scaled = np.zeros((batch_size,num_input_parameters))
    y = np.zeros((batch_size,num_output_parameters))

    X_scaled[:,0] = uniform.rvs(size=batch_size) #H
    X_scaled[:,1] = uniform.rvs(size=batch_size) #eta
    X_scaled[:,2] = uniform.rvs(size=batch_size) #rho
    lambdas = uniform.rvs(size=(batch_size,num_forward_var))
    for i in range(num_forward_var):
        X_scaled[:,i+3] = lambdas[:,i]

    X = reverse_transform_X(X_scaled)

    for i in range(batch_size):
        for j in range(num_maturities):
            rB = rBergomi.rBergomi(n = 100, N = 30000, T = maturities[j], a = X[i,0]-0.5)

            dW1 = rB.dW1()
            dW2 = rB.dW2()

            Y = rB.Y(dW1)

            dB = rB.dB(dW1, dW2, rho = X[i,2])

            V = rB.V(Y, xi = X[i,3], eta = X[i,1])

            S = rB.S(V, dB) 

            ST = S[:,-1][:,np.newaxis]
            call_payoffs = np.maximum(ST - strikes,0)
            
            call_prices = np.mean(call_payoffs, axis = 0)[:,np.newaxis]
            K = strikes[np.newaxis,:]
            implied_vols = vec_bsinv(call_prices, S0, np.transpose(K), maturities[j])
            
            y[i,j*num_maturities:j*num_maturities+num_strikes] = implied_vols[:,0]
    
    return X_scaled,y

""" Design of DNN """

X = tf.placeholder(tf.float32, [None, num_input_parameters])
y = tf.placeholder(tf.float32, [None, num_output_parameters])

#Layers
bn0 = tf.nn.batch_normalization(X, 0, 1, 0, 1, 0.000001)
hidden1 = fully_connected(bn0, num_neurons, activation_fn=tf.nn.elu)
bn1 = tf.nn.batch_normalization(hidden1, 0, 1, 0, 1, 0.000001)
hidden2 = fully_connected(bn1, num_neurons, activation_fn=tf.nn.elu)
bn2 = tf.nn.batch_normalization(hidden2, 0, 1, 0, 1, 0.000001)
hidden3 = fully_connected(bn2, num_neurons, activation_fn=tf.nn.elu)
bn3 = tf.nn.batch_normalization(hidden3, 0, 1, 0, 1, 0.000001)
hidden4 = fully_connected(hidden3, num_neurons, activation_fn=tf.nn.elu)
bn4 = tf.nn.batch_normalization(hidden4, 0, 1, 0, 1, 0.000001)

outputs = fully_connected(bn4, num_output_parameters, activation_fn=None)

#Loss Function
loss = tf.reduce_mean(tf.sqrt(tf.square(outputs - y)))  # MSE

#Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

""" Train the network """

num_cpu = multiprocessing.cpu_count()
config = tf.ConfigProto(device_count={ "CPU": num_cpu },inter_op_parallelism_threads=num_cpu,intra_op_parallelism_threads=2)

with tf.device('/CPU:0'):
    with tf.Session(config=config) as sess:
        sess.run(init)
        
        for iteration in range(num_steps):
            
            X_batch,Y_batch = next_batch_rBergomi(batch_size,contract_bounds,model_bounds)
            sess.run(train,feed_dict={X: X_batch, y: Y_batch})
            
            if iteration % 1 == 0:
                
                rmse = loss.eval(feed_dict={X: X_batch, y: Y_batch})
                print(iteration, "\tRMSE:", rmse)
        
        saver.save(sess, "./models/rBergomi_dnn_m1")


""" Optimize """ 

def predict_theta(implied_vols_true):   
    
    def NNprediction(theta):
        x = np.zeros((1,len(theta)))
        x[0,:] = theta
        return sess.run(outputs,feed_dict={X: x})
    def NNgradientpred(x):
        x = np.asarray(x)
        grad = np.zeros((num_output_parameters,num_input_parameters))
        
        delta = 1e-9
        for i in range(num_input_parameters):
            h = np.zeros(x.shape)
            h[0,i] = delta
            
            #two point gradient
            #grad[i] = (sess.run(outputs,feed_dict={X: x+h}) - sess.run(outputs,feed_dict={X: x-h}))/2/delta

            #four point gradient
            grad[:,i] = (-sess.run(outputs,feed_dict={X: x+2*h})+8*sess.run(outputs,feed_dict={X: x+h})-8*sess.run(outputs,feed_dict={X: x-h}) +sess.run(outputs,feed_dict={X: x-2*h}))/12/delta

        return -np.mean(grad,axis=0)

    def CostFuncLS(theta):
        
        return np.mean(np.power((NNprediction(theta)-implied_vols_true.flatten())[0],2),axis=0)


    def JacobianLS(theta):
        x = np.zeros((1,len(theta)))
        x[0,:] = theta
        return NNgradientpred(x).T

    with tf.Session() as sess:                          
         
        saver.restore(sess, "./models/rBergomi_dnn_m1")    
        
        init = [model_bounds[0,0]+uniform.rvs()*(model_bounds[0,1]-model_bounds[0,0]),model_bounds[1,0]+uniform.rvs()*(model_bounds[1,1]-model_bounds[1,0]),model_bounds[2,0]+uniform.rvs()*(model_bounds[2,1]-model_bounds[2,0]),model_bounds[3,0]+uniform.rvs()*(model_bounds[3,1]-model_bounds[3,0])]
        bnds = ([model_bounds[0,0],model_bounds[1,0],model_bounds[2,0],model_bounds[3,0]],[model_bounds[0,1],model_bounds[1,1],model_bounds[2,1],model_bounds[3,1]])

        I=scipy.optimize.least_squares(CostFuncLS,init,JacobianLS,bounds=bnds,gtol=1E-15,xtol=1E-15,ftol=1E-15,verbose=1)

    theta_pred = I.x
  
    return theta_pred



""" Test the Performance and Plot """

N = 5 #number of test thetas 

thetas_true = reverse_transform_X(uniform.rvs(size=(N,num_model_parameters)))

thetas_pred = np.zeros((N,num_model_parameters))
for i in range(N):
    thetas_pred[i,:] = predict_theta(implied_vols_surface(thetas_true[i,:]).flatten())

iv_surface_true = np.zeros((N,num_maturities,num_strikes))
iv_surface_pred = np.zeros((N,num_maturities,num_strikes))
iv_surface_pred_NN = np.zeros((N,num_maturities,num_strikes))


for i in range(N):
    iv_surface_true[i,:,:] = implied_vols_surface(thetas_true[i,:])
    iv_surface_pred[i,:,:] = implied_vols_surface(thetas_pred[i,:])

with tf.Session() as sess:                          
         
    saver.restore(sess, "./models/rBergomi_dnn_m2")
    x = np.zeros((N,num_input_parameters))
    for i in range(N):
        iv_surface_pred_NN[i,:,:] = sess.run(outputs,feed_dict={X: x})


""" Plot """
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()

fig = plt.figure(figsize=(20,6))

ax1=fig.add_subplot(121)

plt.imshow(np.mean(np.abs((iv_surface_true-iv_surface_pred)/iv_surface_true),axis=0))
plt.title("Average Relative Errors in Implied Volatilities")

ax1.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax1.set_yticklabels(np.around(maturities,1))
ax1.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax1.set_xticklabels(np.around(strikes,2))
plt.colorbar()
ax2=fig.add_subplot(122)

plt.imshow(np.max(np.abs((iv_surface_true-iv_surface_pred)/iv_surface_true),axis=0))
plt.title("Max Relative Errors Implied Volatilities")

ax2.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax2.set_yticklabels(np.around(maturities,1))
ax2.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax2.set_xticklabels(np.around(strikes,2))
plt.colorbar()

ax3=fig.add_subplot(133)

plt.imshow(np.mean(np.abs((iv_surface_true-iv_surface_pred_NN)/iv_surface_true),axis=0))
plt.title("Average Relative Errors in Implied Volatilities NN")

ax1.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax1.set_yticklabels(np.around(maturities,1))
ax1.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax1.set_xticklabels(np.around(strikes,2))
plt.colorbar()


#plt.show()

plt.savefig('rel_errors_dnn_m1_rBergomi.pdf') 


print("Number of trainable Parameters: ",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
