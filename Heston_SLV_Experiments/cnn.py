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

use_data = True

num_input_parameters = num_strikes * num_maturities * 3
num_output_parameters = num_model_parameters
learning_rate = 0.00001

num_steps = 2000
batch_size = 20

#num_neurons = 30

#initial values
S0 = 1.0
V0 = 0.05
r = 0.0

contract_bounds = np.array([[0.8*S0,1.2*S0],[2,5]]) #bounds for K,T
model_bounds = np.array([[0.9,1.3],[0.2,0.8],[-0.8,-0.2],[3,5],[0.1,0.2],[0.1,0.3]])  #bounds for alpha,beta,rho,a,b,c, make sure alpha>0,


"""
Note: The grid of stirkes and maturities is equidistant here put could be choosen differently for real world application.
Note: For the code below to striktly follow the bounds specified above make sure that *_distance x num_* is less than half the distance from the highest to lowest * (* = strikes/maturities). 
"""
maturities_distance = (contract_bounds[1,1]-contract_bounds[1,0])/(2*num_maturities) 
strikes_distance = (contract_bounds[0,1]-contract_bounds[0,0])/(2*num_strikes)


strikes = np.linspace(contract_bounds[0,0],contract_bounds[0,0]+num_strikes*strikes_distance,num_strikes)
maturities = np.linspace(contract_bounds[1,0],contract_bounds[1,0]+num_maturities*maturities_distance,num_maturities)


if use_data==True:
    data = np.genfromtxt('Data_Generation/hestonLV_data_m3_4e4.csv', delimiter=',')
    x_train = data[:,:16*16]
    y_train = data[:,16*16:]

print("Number training data points: ", x_train.shape[0])

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
        return 0.01*np.ones(S.shape)
    
    def sigma1(S,i):
       
        return alpha*np.multiply(np.sqrt(np.maximum(np.zeros(V.shape[0]),V[:,i])),np.power(np.maximum(S,np.zeros(S.shape[0])),1+beta))
    
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

def next_batch_hestonSLV_data(batch_size):
    n = np.array(uniform.rvs(size=batch_size)*x_train.shape[0]).astype(int)
    
    return x_train[n,:].reshape(batch_size,16,16)[:,:,:,np.newaxis],y_train[n,:]

def next_batch_hestonSLV_EM_train(batch_size,contract_bounds,model_bounds,only_prices=True):
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
    return tf.nn.sigmoid(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


X = tf.placeholder(tf.float32,shape=[None,num_maturities,num_strikes,1])
y = tf.placeholder(tf.float32, shape=[None,num_model_parameters])


filter_size = 8

convo_1 = convolutional_layer(X,shape=[filter_size,filter_size,1,16]) 
convo_1_pooling = avg_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling,shape=[filter_size,filter_size,16,64])
convo_2_pooling = avg_pool_2by2(convo_2)

#convo_3 = convolutional_layer(convo_2_pooling,shape=[filter_size,filter_size,8,32])
#convo_3_pooling = avg_pool_2by2(convo_3)


convo_3_flat = tf.reshape(convo_2_pooling,[-1,64*int(num_strikes*num_maturities/(4**2))])
full_layer_one = tf.nn.elu(normal_full_layer(convo_3_flat,128))

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
        
        if use_data==True:
            X_batch,Y_batch = next_batch_hestonSLV_data(batch_size)
        else:
            X_batch,Y_batch = next_batch_hestonSLV_EM_train(batch_size,contract_bounds,model_bounds,only_prices=True)
        
        sess.run(train,feed_dict={X: X_batch, y: Y_batch})
        
        
        
        if iteration % 100 == 0:
            
            rmse = loss.eval(feed_dict={X: X_batch, y: Y_batch})
            print(iteration, "\tRMSE:", rmse)
            
    saver.save(sess, "./Heston_SLV_Experiments/run/models/hestonSLV_cnn_e")




"""
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
"""

def avg_rmse_2d(x,y):
    rmse = 0.0
    n = x.shape[0]
    for i in range(n):
        rmse += np.sqrt(np.mean(np.mean(np.power((x[i,:,:]-y[i,:,:]),2),axis=0),axis=0))
    return (rmse/n)

def implied_vols_surface(theta):
    ivs = np.zeros((1,num_strikes*num_maturities))
    n = 100
    dim = 10000
    W,Z = corr_brownian_motion(n,maturities[-1],dim,theta[2])
    S,V = heston_SLV(theta[0],theta[1],theta[3],theta[4],theta[5],maturities[-1],W,Z,V0,S0)
    
    for i in range(num_maturities):
        n_current = int(maturities[i]/maturities[-1]*n)
        S_T = S[:,n_current]
        for j in range(num_strikes):   
            P =  np.mean(np.maximum(S_T-np.ones(dim)*strikes[j],np.zeros(dim)))
     
            ivs[0,i*num_strikes+j] = implied_vol(P,strikes[j],maturities[i])
    return ivs

""" Test the Performance and Plot """

N = 1 #number of test thetas 

thetas_true = reverse_transform_y(uniform.rvs(size=(N,num_model_parameters)))

iv_surface_true = np.zeros((N,num_maturities,num_strikes,1))
iv_surface_pred = np.zeros((N,num_maturities,num_strikes,1))

for i in range(N):
    iv_surface_true[i,:,:,0] = implied_vols_surface(thetas_true[i,:]).reshape(16,16)
    
implied_vols_surface(thetas_true[0,:]).reshape(16,16)

with tf.Session() as sess:                          
    saver.restore(sess, "./Heston_SLV_Experiments/run/models/hestonSLV_cnn_e")    
    thetas_pred= np.zeros((N,num_model_parameters))
    
    thetas_pred = sess.run(outputs,feed_dict={X: iv_surface_true})

num_strikes = 10
num_maturities = 10

maturities_distance = (contract_bounds[1,1]-contract_bounds[1,0])/(num_maturities) 
strikes_distance = (contract_bounds[0,1]-contract_bounds[0,0])/(num_strikes)

strikes = np.linspace(contract_bounds[0,0],contract_bounds[0,0]+num_strikes*strikes_distance,num_strikes)
maturities = np.linspace(contract_bounds[1,0],contract_bounds[1,0]+num_maturities*maturities_distance,num_maturities)
    
iv_surface_pred = np.zeros((N,num_maturities,num_strikes,1))
iv_surface_true_plot = np.zeros((N,num_maturities,num_strikes,1))
iv_surface_true_plot2 = np.zeros((N,num_maturities,num_strikes,1))

print(thetas_true)
print(thetas_pred)

for i in range(N):
    iv_surface_pred[i,:,:,0] = implied_vols_surface(thetas_pred[i,:]).reshape(10,10)
    iv_surface_true_plot[i,:,:,0] = implied_vols_surface(thetas_true[i,:]).reshape(10,10)
    iv_surface_true_plot2[i,:,:,0] = implied_vols_surface(thetas_true[i,:]).reshape(10,10)


""" Plot """
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

fig = plt.figure(figsize=(22,6))

ax1=fig.add_subplot(132)
plt.imshow(100*np.mean(np.abs((iv_surface_true_plot-iv_surface_pred)/iv_surface_true_plot),axis=0)[:,:,0])
plt.title("Average Relative Errors in \n Implied Volatilities using \n MC, theta predicted")

ax1.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax1.set_yticklabels(np.around(maturities,1))
ax1.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax1.set_xticklabels(np.around(strikes,2),rotation = (45), fontsize = 10)
plt.colorbar(format=mtick.PercentFormatter())
plt.xlabel("Strike",fontsize=12,labelpad=5)
plt.ylabel("Maturity",fontsize=12,labelpad=5)

"""
ax2=fig.add_subplot(122)

plt.imshow(np.max(np.abs((iv_surface_true-iv_surface_pred)/iv_surface_true),axis=0)[:,:,0])
plt.title("Max Relative Errors Implied Volatilities")

ax2.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax2.set_yticklabels(np.around(maturities,1))
ax2.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax2.set_xticklabels(np.around(strikes,2))
plt.xlabel("Strike",fontsize=12,labelpad=5)
plt.ylabel("Maturity",fontsize=12,labelpad=5)

plt.colorbar()
"""

plt.show()

plt.savefig('rel_errors_cnn_rBergomi.pdf') 

print("Number of trainable Parameters: ",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

print("Relative error in Thetas: ", np.mean(np.abs((thetas_true-thetas_pred)/thetas_true),axis=0))
print("RMSE: ",avg_rmse_2d(iv_surface_true_plot,iv_surface_pred)) 
print("MC rel Error in percent",100*np.mean(np.mean(np.mean(np.abs((iv_surface_true_plot-iv_surface_true_plot2)/iv_surface_true_plot),axis=0),axis=0),axis=0))
