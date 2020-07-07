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
from IPython.display import clear_output
import multiprocessing
import time

num_model_parameters = 6
num_strikes = 16
num_maturities = 16


num_input_parameters = num_strikes * num_maturities * 3
num_output_parameters = num_model_parameters
learning_rate = 0.00001

num_steps = 2
batch_size = 1

#num_neurons = 30

#initial values
S0 = 1.0
V0 = 0.2
r = 0.05


contract_bounds = np.array([[0.8*S0,1.2*S0],[1,10]]) #bounds for K,T
model_bounds = np.array([[0.01,0.15],[0.2,0.8],[-1,0],[1,3],[0.1,0.6],[0.01,0.1]]) #bounds for alpha,beta,rho,a,b,c, make sure alpha>0,


"""
Note: The grid of stirkes and maturities is equidistant here put could be choosen differently for real world application.
Note: For the code below to striktly follow the bounds specified above make sure that *_distance x num_* is less than half the distance from the highest to lowest * (* = strikes/maturities). 
"""
maturities_distance = (contract_bounds[1,1]-contract_bounds[1,0])/(2*num_maturities) 
strikes_distance = (contract_bounds[0,1]-contract_bounds[0,0])/(2*num_strikes)


strikes = np.linspace(contract_bounds[0,0],contract_bounds[0,0]+num_strikes*strikes_distance,num_strikes)
maturities = np.linspace(contract_bounds[1,0],contract_bounds[1,0]+num_maturities*maturities_distance,num_maturities)

def corr_brownian_motion(n, T, dim, rho):
    if rho > 1:
        rho = 1
    if rho < -1:
        rho = -1
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
   
    #assert(2*a*b > c*c)

    def mu2(V,i,k):
        return np.multiply(a,(b-V))
    
    def sigma2(V,i,k):
        return np.multiply(c,np.sqrt(np.maximum(0.0,V)))
    
    V = euler_maruyama(mu2,sigma2,T,V0,Z)
    
    def mu1(S,i,k):
        return 0.0
    
    def sigma1(S,i,k):
       
        return alpha*np.multiply(np.sqrt(np.maximum(0.0,V[i,k])),np.power(S,1+beta))
    
    S = euler_maruyama(mu1,sigma1,T,S0,W)
    
    return S,V

def reverse_transform_X(X_scaled):
    X = np.zeros(X_scaled.shape)
    for i in range(X_scaled.shape[-1]-1):
        X[:,:,:,i] = X_scaled[:,:,:,i]*(contract_bounds[i][1]-contract_bounds[i][0]) + contract_bounds[i][0]
    return X

def reverse_transform_y(y_scaled):
    y = np.zeros_like(y_scaled)

    for i in range(y_scaled.shape[-1]):
        y[:,i] = y_scaled[:,i]*(model_bounds[i][1]-model_bounds[i][0]) + model_bounds[i][0]
    return y

def price_pred(alpha,beta,rho,a,b,c,n,dim,T,K,V0,S0):
    W,Z = corr_brownian_motion(n,T,dim,rho)
    S,V = heston_SLV(alpha,beta,a,b,c,T,W,Z,V0,S0)
    S_T = S[:,n]
    P = np.exp(-r*T) * np.mean(np.maximum(S_T-K,np.zeros(dim)))
    
    return P

def implied_vol(P,K,T):
    #Find root using Newtons method
    sigma_new = 0.25
    sigma_old = sigma_new-0.1
    n = 0
    while np.abs(sigma_old-sigma_new) > 0.000001:
        sigma_old = sigma_new
        dplus = (np.log(S0 / K) + (r  + 0.5 * sigma_old ** 2) * T) / (sigma_old * np.sqrt(T))
        dminus = (np.log(S0 / K) + (r  - 0.5 * sigma_old ** 2) * T) / (sigma_old * np.sqrt(T))

        f = S0 * norm.cdf(dplus, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(dminus, 0.0, 1.0) - P
        df = (1 / np.sqrt(2 * np.pi)) * S0 * np.sqrt(T) * np.exp(-(norm.cdf(dplus, 0.0, 1.0) ** 2) * 0.5)

        sigma_new = sigma_old - f/df
        n += 1
        if n > 100:
            #print("not converged")
            return sigma_new
    return np.abs(sigma_new)

def BS_call_price(sigma,K,T):
    dplus = (np.log(S0 / K) + (r  + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    dminus = (np.log(S0 / K) + (r  - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    return S0 * norm.cdf(dplus, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(dminus, 0.0, 1.0)

def next_batch_hestonSLV_EM_train(batch_size,contract_bounds,model_bounds,only_prices=True):
    X = np.zeros((batch_size,num_maturities,num_strikes,3))
    X_scaled = np.zeros((batch_size,num_maturities,num_strikes,3))
    y = np.zeros(num_model_parameters)
    y_scaled = np.zeros(num_model_parameters)

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

    y_scaled = uniform.rvs(size=(batch_size,num_model_parameters))
     
    X = reverse_transform_X(X_scaled)
    y = reverse_transform_y(y_scaled)
    
    
    n = 100
    dim = 10000
    for batch in range(batch_size):  
        W,Z = corr_brownian_motion(n,X[batch,-1,0,1],dim,y[batch,2])
        S,V = heston_SLV(y[batch,0],y[batch,1],y[batch,3],y[batch,4],y[batch,5],X[batch,-1,0,1],W,Z,V0,S0)
        for i in range(num_maturities):
            n_current = int(X[batch,i,0,1]/X[batch,-1,0,1]*n)
            S_T = S[:,n_current]
            
            for j in range(num_strikes):
                #P = np.exp(-r*X[batch,i,0,1])*np.mean(np.maximum(S_T-np.log(X[batch,0,j,0]),np.zeros(dim)))
                
                #X[batch,i,j,2] = implied_vol(P,X[batch,0,j,0],X[batch,i,0,1])
                X[batch,i,j,2] = np.exp(-r*X[batch,i,0,1])*np.mean(np.maximum(S_T-np.log(X[batch,0,j,0]),np.zeros(dim)))
                X_scaled[batch,i,j,2] = X[batch,i,j,2]

    if only_prices:
        return X_scaled[:,:,:,2:],y_scaled
    return X_scaled,y_scaled


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


X = tf.placeholder(tf.float32,shape=[None,num_maturities,num_strikes,1])
y = tf.placeholder(tf.float32, shape=[None,num_model_parameters])


filter_size = 4
"""
3x3 Filter
3 `ìmages` as input
32 outputs
"""
convo_1 = convolutional_layer(X,shape=[filter_size,filter_size,1,128]) 
convo_1_pooling = avg_pool_2by2(convo_1)

"""
3x3 Filter
32 inputs
32 outputs
"""
convo_2 = convolutional_layer(convo_1_pooling,shape=[filter_size,filter_size,128,128])
convo_2_pooling = avg_pool_2by2(convo_2)

convo_3 = convolutional_layer(convo_2_pooling,shape=[filter_size,filter_size,128,64])
convo_3_pooling = avg_pool_2by2(convo_3)


convo_3_flat = tf.reshape(convo_3_pooling,[-1,64*int(np.power(np.maximum(num_maturities,num_strikes),2)/filter_size/filter_size/filter_size)])
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
        
        X_batch,Y_batch = next_batch_hestonSLV_EM_train(batch_size,contract_bounds,model_bounds,only_prices=True)
        
        sess.run(train,feed_dict={X: X_batch, y: Y_batch})
        
        
        #uncomment for performance
        """ 
        step.append(iteration)
        rmse.append(loss.eval(feed_dict={X: X_batch, y: Y_batch}))
        X_batch_val,Y_batch_val = next_batch_sabr_EM_train(batch_size,contract_bounds,model_bounds,only_prices=True)
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
            
    saver.save(sess, "./Heston_SLV_Experiments/run/models/hestonSLV_cnn_e")





num_thetas = 10

def reverse_transform_theta(theta_scaled):
    X = np.zeros(theta_scaled.shape)
    for i in range(num_model_parameters):
        X[:,i] = theta_scaled[:,i]*(model_bounds[i][1]-model_bounds[i][0]) + model_bounds[i][0]
    return X

thetas_true = reverse_transform_theta(uniform.rvs(size=(num_thetas,num_model_parameters)))

price_grids_true = np.zeros((num_thetas,1,num_strikes*num_maturities))
price_grids_true_ = np.zeros((num_thetas,1,num_strikes,num_maturities,1))

n = 100
dim = 10000

for i in range(num_thetas):
    W,Z = corr_brownian_motion(n,maturities[-1],dim,thetas_true[i,2])
    S,V = heston_SLV(thetas_true[i,0],thetas_true[i,1],thetas_true[i,3],thetas_true[i,4],thetas_true[i,5],maturities[-1],W,Z,V0,S0)
    for j in range(num_maturities):
        n_current = int(maturities[j]/maturities[-1]*n)
        S_T = S[:,n_current]
        for k in range(num_strikes):
            price_grids_true[i,0,j*num_strikes+k] = np.exp(-r*maturities[j])*np.mean(np.maximum(S_T-np.ones(dim)*strikes[k],np.zeros(dim)))
            price_grids_true_[i,0,j,k,0] = price_grids_true[i,0,j*num_strikes+k] 

thetas_pred = np.zeros((num_thetas,num_model_parameters))
with tf.Session() as sess:                          
    saver.restore(sess, "Heston_SLV_Experiments/run/models/hestonSLV_cnn_e")    
    theta_pred_scaled = np.zeros((1,num_model_parameters))
    for i in range(num_thetas):
        theta_pred_scaled[0,:] = sess.run(outputs,feed_dict={X: price_grids_true_[i,:,:,:,:]})[0]
 
        thetas_pred[i,:] = reverse_transform_y(theta_pred_scaled)[0,:]




prices_grid_true_2 = np.zeros((num_thetas,num_maturities,num_strikes))
prices_grid_pred_2 = np.zeros((num_thetas,num_maturities,num_strikes))
for i in range(num_thetas):
    W,Z = corr_brownian_motion(n,maturities[-1],dim,thetas_pred[i,2])
    S,V = heston_SLV(thetas_pred[i,0],thetas_pred[i,1],thetas_pred[i,3],thetas_pred[i,4],thetas_pred[i,5],maturities[-1],W,Z,V0,S0)
    for j in range(num_maturities):
        n_current = int(maturities[j]/maturities[-1]*n)
        S_T = S[:,n_current]
        for k in range(num_strikes):
            prices_grid_true_2[i,j,k] = price_grids_true[i,0,j*num_strikes+k]
            prices_grid_pred_2[i,j,k] = np.exp(-r*maturities[j])*np.mean(np.maximum(S_T-np.ones(dim)*strikes[k],np.zeros(dim)))

fig = plt.figure(figsize=(20, 6))

ax1=fig.add_subplot(121)

plt.imshow(np.mean(np.abs((prices_grid_true_2-prices_grid_pred_2)/prices_grid_true_2),axis=0))
plt.title("Average Relative Errors in Prices (CNN)")

ax1.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax1.set_yticklabels(np.around(maturities,1))
ax1.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax1.set_xticklabels(np.around(strikes,2))
plt.colorbar()
ax2=fig.add_subplot(122)

plt.imshow(np.max(np.abs((prices_grid_true_2-prices_grid_pred_2)/prices_grid_true_2),axis=0))
plt.title("Max Relative Errors in Prices (CNN)")

ax2.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax2.set_yticklabels(np.around(maturities,1))
ax2.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax2.set_xticklabels(np.around(strikes,2))

plt.colorbar()
plt.savefig('errors_cnn_e_hestonSLV.pdf') 