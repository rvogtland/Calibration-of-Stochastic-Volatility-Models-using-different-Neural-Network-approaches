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


num_model_parameters = 3
num_strikes = 16
num_maturities = 16


num_input_parameters = 5
num_output_parameters = 1
learning_rate = 0.0001

num_steps = 1
batch_size = 10

num_neurons = 100

#initial values
S0 = 1.0
V0 = 0.1
r = 0.05


contract_bounds = np.array([[0.6*S0,1.2*S0],[1,10]]) #bounds for K,T
model_bounds = np.array([[0.01,0.15],[0,1],[-1,0]]) #bounds for alpha,beta,rho, make sure alpha>0, beta,rho \in [0,1]

"""
Note: The grid of stirkes and maturities is equidistant here put could be choosen differently for real world application.
Note: For the code below to striktly follow the bounds specified above make sure that *_distance x num_* is less than half the distance from the highest to lowest * (* = strikes/maturities). 
"""
maturities_distance = (contract_bounds[1,1]-contract_bounds[1,0])/(2*num_maturities) 
strikes_distance = (contract_bounds[0,1]-contract_bounds[0,0])/(2*num_strikes)

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

def sabr(alpha,beta,T,W,Z,V0,S0):
    #print(beta)
    #assert(beta>0 and beta<1)

    def mu2(V,i,k):
        return 0.0
    
    def sigma2(V,i,k):
        return np.multiply(alpha,V)
    
    V = euler_maruyama(mu2,sigma2,T,V0,Z)
    
    def mu1(S,i,k):
        return 0.0
    
    def sigma1(S,i,k):
        return np.multiply(V[i,k],np.power(np.maximum(0.0,S),beta))
    
    S = euler_maruyama(mu1,sigma1,T,S0,W)
    
    return S,V

def reverse_transform_X(X_scaled):
    X = np.zeros(X_scaled.shape)
    for i in range(num_model_parameters):
        X[:,i] = X_scaled[:,i]*(model_bounds[i][1]-model_bounds[i][0]) + model_bounds[i][0]
    for i in range(2):
        X[:,num_model_parameters+i] = X_scaled[:,num_model_parameters+i]*(contract_bounds[i][1]-contract_bounds[i][0]) + contract_bounds[i][0]
    return X

def price_pred(alpha,beta,rho,n,dim,T,K,V0,S0):
    W,Z = corr_brownian_motion(n,T,dim,rho)
    S,V = sabr(alpha,beta,T,W,Z,V0,S0)
    S_T = S[:,n]
    P = np.exp(-r*T) * np.mean(np.maximum(S_T-K,np.zeros(dim)))
    
    return P

def next_batch_sabr_EM_train(batch_size,contract_bounds,model_bounds):
    X_scaled = np.zeros((batch_size,num_input_parameters))
    y = np.zeros((batch_size,num_output_parameters))

    X_scaled[:,0] = uniform.rvs(size=batch_size) #alpha
    X_scaled[:,1] = uniform.rvs(size=batch_size) #beta
    X_scaled[:,2] = uniform.rvs(size=batch_size) #rho
    X_scaled[:,3] = uniform.rvs(size=batch_size) #maturities
    X_scaled[:,4] = uniform.rvs(size=batch_size) #strikes

    X = reverse_transform_X(X_scaled)

    n = 100
    dim = 6000
    for i in range(batch_size):
        y[i,0] = price_pred(X[i,0],X[i,1],X[i,2],n,dim,X[i,3],X[i,4],V0,S0)

    return X_scaled,y


    #Layers
hidden1 = fully_connected(X, num_neurons, activation_fn=tf.nn.softplus)
bn1 = tf.nn.batch_normalization(hidden1, 0, 1, 0, 1, 0.000001)
hidden2 = fully_connected(bn1, num_neurons, activation_fn=tf.nn.softplus)
bn2 = tf.nn.batch_normalization(hidden2, 0, 1, 0, 1, 0.000001)
hidden3 = fully_connected(bn2, num_neurons, activation_fn=tf.nn.softplus)
bn3 = tf.nn.batch_normalization(hidden3, 0, 1, 0, 1, 0.000001)

outputs = fully_connected(bn3, num_output_parameters, activation_fn=None)


#Loss Function
loss = tf.reduce_mean(tf.sqrt(tf.square(outputs - y)))  # MSE

#Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()


num_cpu = multiprocessing.cpu_count()
config = tf.ConfigProto(device_count={ "CPU": num_cpu },
                                        inter_op_parallelism_threads=num_cpu,
                                        intra_op_parallelism_threads=2,
                                        )
with tf.device('/CPU:0'):
    with tf.Session(config=config) as sess:
        sess.run(init)
        
        for iteration in range(num_steps):
            
            X_batch,Y_batch = next_batch_sabr_EM_train(batch_size,contract_bounds,model_bounds)
            sess.run(train,feed_dict={X: X_batch, y: Y_batch})
            
            if iteration % 2 == 0:
                
                rmse = loss.eval(feed_dict={X: X_batch, y: Y_batch})
                print(iteration, "\tRMSE:", rmse)
        
        saver.save(sess, "./models/sabr_dnn_e_m2_soft100")

def predict_theta(prices_true):   
    
    def NNprediction(theta,maturities,strikes):
        res = np.zeros((1,num_maturities,num_strikes))
        x = np.zeros((1,num_input_parameters))
        
        x[0,0:3] = theta
       
        for i in range(num_maturities):
            for j in range(num_strikes):
                x[0,3] = maturities[i]
                x[0,4] = strikes[j]
                res[0,i,j] = sess.run(outputs,feed_dict={X: x})
        return res
    def NNgradientpred(theta,maturities,strikes):
        x = np.zeros((1,num_input_parameters))
        
        x[0,0:3] = theta
       
        grad = np.zeros((num_model_parameters,num_maturities,num_strikes))
        
        delta = 0.000001
        for i in range(num_model_parameters):
            h = np.zeros((1,num_input_parameters))
            h[0,i] = delta
            for j in range(num_maturities):
                for k in range(num_strikes):
                    x[0,3] = maturities[j]
                    x[0,4] = strikes[k]
                    #two point gradient
                    #grad[i] = (sess.run(outputs,feed_dict={X: x+h}) - sess.run(outputs,feed_dict={X: x-h}))/2/delta

                    #four point gradient
                    grad[i,j,k] = (-sess.run(outputs,feed_dict={X: x+2*h})+8*sess.run(outputs,feed_dict={X: x+h})-8*sess.run(outputs,feed_dict={X: x-h}) +sess.run(outputs,feed_dict={X: x-2*h}))/12/delta

        return np.mean(np.mean(grad,axis=1),axis=1)

    def CostFuncLS(theta,maturities,strikes):
        
        return np.sqrt(np.mean(np.mean(np.power((NNprediction(theta,maturities,strikes)-prices_true)[0],2),axis=0),axis=0))


    def JacobianLS(theta,maturities,strikes):

        return NNgradientpred(theta,maturities,strikes)

    with tf.Session() as sess:                          
        saver.restore(sess, "./models/sabr_dnn_e_m2_soft100")      
        
        init = [model_bounds[0,0]+uniform.rvs()*(model_bounds[0,1]-model_bounds[0,0]),model_bounds[1,0]+uniform.rvs()*(model_bounds[1,1]-model_bounds[1,0]),model_bounds[2,0]+uniform.rvs()*(model_bounds[2,1]-model_bounds[2,0])]
        bnds = ([model_bounds[0,0],model_bounds[1,0],model_bounds[2,0]],[model_bounds[0,1],model_bounds[1,1],model_bounds[2,1]])


        I=scipy.optimize.least_squares(CostFuncLS,init,JacobianLS,bounds=bnds,gtol=1E-25,xtol=1E-25,verbose=1,args=(maturities,strikes))

    theta_pred = I.x
    
    return theta_pred

def prices_grid(theta):
    prices_true = np.zeros((1,num_maturities,num_strikes))
    n = 100
    dim = 200
    for i in range(num_maturities):
        for j in range(num_strikes):        
            prices_true[0,i,j] = price_pred(theta[0],theta[1],theta[2],n,dim,maturities[i],strikes[j],V0,S0)
    return prices_true

N = 50
num_thetas = 50

def reverse_transform_theta(theta_scaled):
    X = np.zeros(theta_scaled.shape)
    for i in range(num_model_parameters):
        X[:,i] = theta_scaled[:,i]*(model_bounds[i][1]-model_bounds[i][0]) + model_bounds[i][0]
    return X

thetas_true = reverse_transform_theta(uniform.rvs(size=(num_thetas,num_model_parameters)))

thetas_pred = np.zeros((N,num_model_parameters))
for i in range(N):
    thetas_pred[i,:] = predict_theta(prices_grid(thetas_true[i,:]).flatten())


prices_grid_true_2 = np.zeros((N,num_maturities,num_strikes))
prices_grid_pred_2 = np.zeros((N,num_maturities,num_strikes))
n = 100
dim = 200
for i in range(N):
    for j in range(num_maturities):
        for k in range(num_strikes):
            prices_grid_true_2[i,j,k] = price_pred(thetas_true[i,0],thetas_true[i,1],thetas_true[i,2],n,dim,maturities[i],strikes[j],V0,S0)
            prices_grid_pred_2[i,j,k] = price_pred(thetas_pred[i,0],thetas_pred[i,1],thetas_pred[i,2],n,dim,maturities[i],strikes[j],V0,S0)


fig = plt.figure(figsize=(18, 6))

ax1=fig.add_subplot(121)

plt.imshow(np.mean(np.abs((prices_grid_true_2-prices_grid_pred_2)/prices_grid_true_2),axis=0))
plt.title("Average Relative Errors in Prices")

ax1.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax1.set_yticklabels(np.around(maturities,1))
ax1.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax1.set_xticklabels(np.around(strikes,2))
plt.colorbar()
ax2=fig.add_subplot(122)

plt.imshow(np.max(np.abs((prices_grid_true_2-prices_grid_pred_2)/prices_grid_true_2),axis=0))
plt.title("Max Relative Errors in Prices")

ax2.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax2.set_yticklabels(np.around(maturities,1))
ax2.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax2.set_xticklabels(np.around(strikes,2))


plt.colorbar()
plt.savefig('errors_dnn_m2_euler.pdf') 
plt.show()
