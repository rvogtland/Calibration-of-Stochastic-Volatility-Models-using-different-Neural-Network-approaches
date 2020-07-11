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
os.chdir('/Users/robinvogtland/Documents/RV_ETH_CSE_Bachelor/3_Jahr/FS/Bachelor_Thesis/rv_bachelor_thesis/Rough_Bergomi_Experiments/rbergomi')

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
num_strikes = 16
num_maturities = 16

only_iv = True #decides if only IVs are input to CNN (True) or also strikes and maturities (False)

num_input_parameters = num_maturities*num_strikes
num_output_parameters = num_model_parameters
learning_rate = 0.00001
num_steps = 20
batch_size = 3

#initial values
S0 = 1.0
r = 0.0


contract_bounds = np.array([[0.8*S0,1.2*S0],[1,10]]) #bounds for K,T
model_bounds = np.array([[0.1,0.5],[0.5,3],[-0.9,-0.1],[0.01,0.15]]) #bounds for H,eta,rho,lambdas


#Note: The grid of stirkes and maturities is equidistant here put could be choosen differently for real world application.
#Note: For the code below to striktly follow the bounds specified above make sure that *_distance x num_* is less than half the distance from the highest to lowest * (* = strikes/maturities). 

maturities_distance = (contract_bounds[1,1]-contract_bounds[1,0])/(2*num_maturities) 
strikes_distance = (contract_bounds[0,1]-contract_bounds[0,0])/(2*num_strikes)

strikes = np.linspace(contract_bounds[0,0],contract_bounds[0,0]+num_strikes*strikes_distance,num_strikes)
maturities = np.linspace(contract_bounds[1,0],contract_bounds[1,0]+num_maturities*maturities_distance,num_maturities)


""" Helper functions """

def reverse_transform_X(X_scaled):
    X = np.zeros(X_scaled.shape)
    for i in range(X_scaled.shape[-1]-1):
        X[:,:,:,i] = X_scaled[:,:,:,i]*(contract_bounds[i][1]-contract_bounds[i][0]) + contract_bounds[i][0]
    return X

def reverse_transform_y(y_scaled):
    y = np.zeros(y_scaled.shape)
    for i in range(4):
        y[:,i] = y_scaled[:,i]*(model_bounds[i][1]-model_bounds[i][0]) + model_bounds[i][0]

    return y

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

def next_batch_rBergomi(batch_size,contract_bounds,model_bounds,only_iv=True):
    #INPUT: batch size, bounds for contract and model parameters, NO CHECKS IF BOUNDS FULLFILL NO-ABITRAGE CONDITION
    #OUTPUT: random model parameters (scaled!) and corresponding implied volatility surface

    X = np.zeros((batch_size,num_maturities,num_strikes,3))
    X_scaled = np.zeros((batch_size,num_maturities,num_strikes,3))
    y = np.zeros((batch_size,num_model_parameters))
    y_scaled = np.zeros((batch_size,num_model_parameters))

    X_scaled[:,:,0,0] = 0.5*uniform.rvs(size=(batch_size,1)) * np.ones((1,num_maturities))
    X_scaled[:,0,:,1] = 0.5*uniform.rvs(size=(batch_size,1)) * np.ones((1,num_strikes))
    
    for i in range(num_strikes):
        if i == 0:
            pass
        X_scaled[:,:,i,0] = X_scaled[:,:,0,0] + i*strikes_distance/(contract_bounds[0][1]-contract_bounds[0][0])
    for i in range(num_maturities):
        if i == 0:
            pass
        X_scaled[:,i,:,1] = X_scaled[:,0,:,1] + i*maturities_distance/(contract_bounds[1][1]-contract_bounds[1][0])

    y_scaled = uniform.rvs(size=(batch_size,num_model_parameters)) #H,eta,rho,lambdas in each row

    X = reverse_transform_X(X_scaled)
    y = reverse_transform_y(y_scaled)
    
    for i in range(batch_size): 
        for j in range(num_maturities):
            rB = rBergomi.rBergomi(n = 100, N = 30000, T = maturities[j], a = y[i,0]-0.5)

            dW1 = rB.dW1()
            dW2 = rB.dW2()

            Y = rB.Y(dW1)

            dB = rB.dB(dW1, dW2, rho = y[i,2])

            V = rB.V(Y, xi = y[i,3], eta = y[i,1])

            S = rB.S(V, dB) 

            ST = S[:,-1][:,np.newaxis]
            call_payoffs = np.maximum(ST - strikes,0)
            
            call_prices = np.mean(call_payoffs, axis = 0)[:,np.newaxis]
            K = strikes[np.newaxis,:]
            implied_vols = vec_bsinv(call_prices, S0, np.transpose(K), maturities[j])
            
            X[i,j,:,2] = implied_vols[:,0]
    
    if only_iv:
        return X[:,:,:,2:],y_scaled
    
    return X,y_scaled

""" Helper functions for CNN """
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

"""
Create a 2D convolution using builtin conv2d from TF. From those docs:
Computes a 2-D convolution given 4-D input and filter tensors.
Given an input tensor of shape [batch_size, len(T), len(K), #params] and a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels], this op performs the following:
Flattens the filter to a 2-D matrix with shape [filter_height * filter_width * in_channels, output_channels].
Extracts image patches from the input tensor to form a virtual tensor of shape [batch_size, len(T), len()K, filter_height * filter_width * #params].
For each patch, right-multiplies the filter matrix and the image patch vector.
"""
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

"""
Args:
  value: A 4-D `Tensor` with shape `[batch_size, len(T), len(K), #params]` and
    type `tf.float32`.
  ksize: A list of ints that has length >= 4.  The size of the window for
    each dimension of the input tensor.
  strides: A list of ints that has length >= 4.  The stride of the sliding
    window for each dimension of the input tensor.
  padding: A string, either `'VALID'` or `'SAME'`. 
"""
def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2by2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

""" Design of CNN """
if only_iv:
    X = tf.placeholder(tf.float32,shape=[None,num_maturities,num_strikes,1])
else:
    X = tf.placeholder(tf.float32,shape=[None,num_maturities,num_strikes,3])
y = tf.placeholder(tf.float32, shape=[None,num_model_parameters])

filter_size = 4
"""
4x4 Filter
1 `ìmages` as input
8 outputs
"""
convo_1 = convolutional_layer(X,shape=[filter_size,filter_size,1,8]) 
convo_1_pooling = avg_pool_2by2(convo_1)

"""
4x4 Filter
8 inputs
32 outputs
"""
convo_2 = convolutional_layer(convo_1_pooling,shape=[filter_size,filter_size,8,32])
convo_2_pooling = avg_pool_2by2(convo_2)

convo_3 = convolutional_layer(convo_2_pooling,shape=[filter_size,filter_size,32,128])
convo_3_pooling = avg_pool_2by2(convo_3)


convo_3_flat = tf.reshape(convo_3_pooling,[-1,128*int(num_maturities*num_strikes/(filter_size**3))])
full_layer_one = tf.nn.elu(normal_full_layer(convo_3_flat,1024))

outputs = fully_connected(full_layer_one, num_model_parameters, activation_fn=None)


#Loss Function
loss = tf.reduce_mean(tf.sqrt(tf.square(outputs - y)))  # MSE

#Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

num_cpu = multiprocessing.cpu_count()
config = tf.ConfigProto(device_count={"CPU": num_cpu})

with tf.Session(config=config) as sess:
    sess.run(init)
    step = []
    rmse = []
    rmse_val = []
    for iteration in range(num_steps):
        
        X_batch,Y_batch = next_batch_rBergomi(batch_size,contract_bounds,model_bounds,only_iv=True)
        
        sess.run(train,feed_dict={X: X_batch, y: Y_batch})
        
        """
        step.append(iteration)
        rmse.append(loss.eval(feed_dict={X: X_batch, y: Y_batch}))
        X_batch_val,Y_batch_val = next_batch_rBergomi(batch_size,contract_bounds,model_bounds,only_prices=True)
        rmse_val.append(loss.eval(feed_dict={X: X_batch_val, y: Y_batch_val}))
        
        plt.figure(figsize=(7,5))
        plt.yscale("log")
        plt.xlabel("step")
        plt.ylabel("RMSE")
        plt.grid()
        plt.plot(step,rmse,"*-",label="rmse")
        plt.plot(step,rmse_val,"*-",label="validation rmse")
        clear_output(wait=True)
        plt.legend()
        plt.show()
        """

        if iteration % 1 == 0:
            
            rmse = loss.eval(feed_dict={X: X_batch, y: Y_batch})
            print(iteration, "\tRMSE:", rmse)
            
    saver.save(sess, "./models/rBergomi_cnn")


""" Test the Performance and Plot """

N = 1 #number of test thetas 

thetas_true = reverse_transform_y(uniform.rvs(size=(N,num_model_parameters)))

iv_surface_true = np.zeros((N,num_maturities,num_strikes,1))
iv_surface_pred = np.zeros((N,num_maturities,num_strikes,1))

for i in range(N):
    iv_surface_true[i,:,:,0] = implied_vols_surface(thetas_true[i,:])
    
with tf.Session() as sess:                          
    saver.restore(sess, "./models/rBergomi_cnn")    
    thetas_pred_scaled = np.zeros((N,num_model_parameters))
    
    theta_pred_scaled = sess.run(outputs,feed_dict={X: iv_surface_true})

thetas_pred = reverse_transform_y(theta_pred_scaled)

for i in range(N):
    iv_surface_pred[i,:,:,0] = implied_vols_surface(thetas_pred[i,:])


""" Plot """
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20,6))

ax1=fig.add_subplot(121)
plt.imshow(np.mean(np.abs((iv_surface_true-iv_surface_pred)/iv_surface_true),axis=0)[:,:,0])
plt.title("Average Relative Errors in Implied Volatilities")

ax1.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax1.set_yticklabels(np.around(maturities,1))
ax1.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax1.set_xticklabels(np.around(strikes,2))
plt.colorbar()
ax2=fig.add_subplot(122)

plt.imshow(np.max(np.abs((iv_surface_true-iv_surface_pred)/iv_surface_true),axis=0)[:,:,0])
plt.title("Max Relative Errors Implied Volatilities")

ax2.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax2.set_yticklabels(np.around(maturities,1))
ax2.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax2.set_xticklabels(np.around(strikes,2))


plt.colorbar()
plt.show()

plt.savefig('rel_errors_cnn_rBergomi.pdf') 


print("Number of trainable Parameters: ",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))