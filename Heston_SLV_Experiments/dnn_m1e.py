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


use_data = True

num_model_parameters = 6
num_strikes = 10
num_maturities = 10

np.random.seed(1)

num_input_parameters = 6
num_output_parameters = num_maturities*num_strikes
learning_rate = 0.0001
num_steps = 5000
batch_size = 50
num_neurons = 80


model = "./Heston_SLV_Experiments/run/models/hestonSLV_dnn_e"

S0 = 1.0
V0 = 0.05
r = 0.0

contract_bounds = np.array([[0.8*S0,1.2*S0],[2,5]]) #bounds for K,T
model_bounds = np.array([[0.9,1.3],[0.2,0.8],[-0.8,-0.2],[3,5],[0.1,0.2],[0.1,0.3]])  #bounds for alpha,beta,rho,a,b,c, make sure alpha>0,

"""
Note: The grid of stirkes and maturities is equidistant here put could be choosen differently for real world application.
Note: For the code below to striktly follow the bounds specified above make sure that *_distance x num_* is less than half the distance from the highest to lowest * (* = strikes/maturities). 
"""
maturities_distance = (contract_bounds[1,1]-contract_bounds[1,0])/(num_maturities) 
strikes_distance = (contract_bounds[0,1]-contract_bounds[0,0])/(num_strikes)

strikes = np.linspace(contract_bounds[0,0],contract_bounds[0,0]+num_strikes*strikes_distance,num_strikes)
maturities = np.linspace(contract_bounds[1,0],contract_bounds[1,0]+num_maturities*maturities_distance,num_maturities)

if use_data==True:
    data = np.genfromtxt('Data_Generation/hestonLV_data_m1_4e4.csv', delimiter=',')
    x_train = data[:,:num_model_parameters]
    y_train = data[:,num_model_parameters:]
    print(data.shape)
    print("Number training data points: ", x_train.shape[0] )

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
    for i in range(num_input_parameters):
        X[:,i] = X_scaled[:,i]*(model_bounds[i][1]-model_bounds[i][0]) + model_bounds[i][0]
    return X


def implied_vol(P,K,T):

    if not P<S0:
        print("P<S0 = ",P<S0,", abitrage!")
        return 0.0
    if not P>S0-K*np.exp(-r*T):
        print("P>S0-K*np.exp(-r*T) = ",P>S0-K*np.exp(-r*T),", abitrage!")
        return 0.0

    def f(sigma):
        dplus = (np.log(S0 / K) + (r  + 0.5 * np.power(sigma, 2)) * T) / (sigma * np.sqrt(T))
        dminus = (np.log(S0 / K) + (r  - 0.5 * np.power(sigma, 2)) * T) / (sigma * np.sqrt(T))
        
        return S0 * norm.cdf(dplus, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(dminus, 0.0, 1.0) - P
     
    return scipy.optimize.brentq(f, 0.001, 1)

def BS_call_price(sigma,K,T):
    dplus = (np.log(S0 / K) + (r  + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    dminus = (np.log(S0 / K) + (r  - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    return S0 * norm.cdf(dplus, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(dminus, 0.0, 1.0)

def next_batch_hestonLV_data(batch_size):
    n = np.array(uniform.rvs(size=batch_size)*x_train.shape[0]).astype(int)
    
    return x_train[n,:],y_train[n,:]

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

    n = 200
    dim = 40000
    for i in range(batch_size):
        W,Z = corr_brownian_motion(n,maturities[-1],dim,X[i,2])
        S,V = heston_SLV(X[i,0],X[i,1],X[i,3],X[i,4],X[i,5],maturities[-1],W,Z,V0,S0)
        
        for j in range(num_maturities):
            n_current = int(maturities[j]/maturities[-1]*n)
            S_T = S[:,n_current]
            
            for k in range(num_strikes):
                P = np.mean(np.maximum(S_T-np.ones(dim)*strikes[k],np.zeros(dim)))*np.exp(-r*maturities[j])
                
                y[i,j*num_strikes+k] = implied_vol(P,strikes[k],maturities[j])

                #y[i,j*num_strikes+k] = np.mean(np.maximum(S_T-np.ones(dim)*strikes[k],np.zeros(dim)))

    return X_scaled,y

#Layers
act_func = tf.nn.elu

bn0 = tf.nn.batch_normalization(X, 0, 1, 0, 1, 1e-9)
hidden1 = fully_connected(bn0, num_neurons, activation_fn=act_func)
#bn1 = tf.nn.batch_normalization(hidden1, 0, 1, 0, 1, 1e-9)
hidden2 = fully_connected(hidden1, num_neurons, activation_fn=act_func)
#bn2 = tf.nn.batch_normalization(hidden2, 0, 1, 0, 1, 1e-9)
hidden3 = fully_connected(hidden2, num_neurons, activation_fn=act_func)
#bn3 = tf.nn.batch_normalization(hidden3, 0, 1, 0, 1, 1e-9)
hidden4 = fully_connected(hidden3, num_neurons, activation_fn=tf.nn.elu)
##bn4 = tf.nn.batch_normalization(hidden4, 0, 1, 0, 1, 1e-9)
hidden5 = fully_connected(hidden4, num_neurons, activation_fn=tf.nn.elu)


outputs = fully_connected(hidden4, num_output_parameters, activation_fn=None)

#Loss Function
loss = tf.sqrt(tf.reduce_mean(tf.square(outputs - y)))  # MSE

#Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

num_cpu = multiprocessing.cpu_count()
config = tf.ConfigProto(device_count={ "CPU": num_cpu },inter_op_parallelism_threads=num_cpu,intra_op_parallelism_threads=2)
"""
with tf.device('/CPU:0'):
    with tf.Session(config=config) as sess:
        sess.run(init)
        
        for iteration in range(num_steps):
            
            if use_data==True:
                X_batch,Y_batch = next_batch_hestonLV_data(batch_size)
            else:
                X_batch,Y_batch = next_batch_hestonSLV_EM_train(batch_size,contract_bounds,model_bounds)

            sess.run(train,feed_dict={X: X_batch, y: Y_batch})
            
            if iteration % 100 == 0:
                
                rmse = loss.eval(feed_dict={X: X_batch, y: Y_batch})
                print(iteration, "\tRMSE:", rmse)
        
        saver.save(sess, model)
"""

def iv_surface(theta):
    ivs = np.zeros((1,num_output_parameters))
    n = 200
    dim = 40000
    W,Z = corr_brownian_motion(n,maturities[-1],dim,theta[2])
    S,V = heston_SLV(theta[0],theta[1],theta[3],theta[4],theta[5],maturities[-1],W,Z,V0,S0)
    
    for i in range(num_maturities):
        n_current = int(maturities[i]/maturities[-1]*n)
        S_T = S[:,n_current]
        for j in range(num_strikes):   
            P =  np.mean(np.maximum(S_T-np.ones(dim)*strikes[j],np.zeros(dim)))
     
            ivs[0,i*num_strikes+j] = implied_vol(P,strikes[j],maturities[i])
    return ivs

def price_surface(theta):
    ps = np.zeros((1,num_output_parameters))
    n = 200
    dim = 40000
    W,Z = corr_brownian_motion(n,maturities[-1],dim,theta[2])
    S,V = heston_SLV(theta[0],theta[1],theta[3],theta[4],theta[5],maturities[-1],W,Z,V0,S0)
    
    for i in range(num_maturities):
        n_current = int(maturities[i]/maturities[-1]*n)
        S_T = S[:,n_current]
        for j in range(num_strikes):   
            ps[0,i*num_strikes+j] = np.mean(np.maximum(S_T-np.ones(dim)*strikes[j],np.zeros(dim)))
    return ps

def predict_theta(iv_surface):   
    
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
        
        return np.mean(np.mean(np.power((NNprediction(theta)-iv_surface.flatten()),2),axis=0),axis=0)


    def JacobianLS(theta):
        x = np.zeros((1,len(theta)))
        x[0,:] = theta
        return NNgradientpred(x).T

    with tf.Session() as sess:                          
        #saver.restore(sess, "./models/hestonSLV")  
        saver.restore(sess, model)    
        tmp1 = 1000  
        init = [model_bounds[0,0]+uniform.rvs()*(model_bounds[0,1]-model_bounds[0,0]),model_bounds[1,0]+uniform.rvs()*(model_bounds[1,1]-model_bounds[1,0]),model_bounds[2,0]+uniform.rvs()*(model_bounds[2,1]-model_bounds[2,0]),model_bounds[3,0]+uniform.rvs()*(model_bounds[3,1]-model_bounds[3,0]),model_bounds[4,0]+uniform.rvs()*(model_bounds[4,1]-model_bounds[4,0]),model_bounds[5,0]+uniform.rvs()*(model_bounds[5,1]-model_bounds[5,0])]
  
        for i in range(100):
            init_tmp = [model_bounds[0,0]+uniform.rvs()*(model_bounds[0,1]-model_bounds[0,0]),model_bounds[1,0]+uniform.rvs()*(model_bounds[1,1]-model_bounds[1,0]),model_bounds[2,0]+uniform.rvs()*(model_bounds[2,1]-model_bounds[2,0]),model_bounds[3,0]+uniform.rvs()*(model_bounds[3,1]-model_bounds[3,0]),model_bounds[4,0]+uniform.rvs()*(model_bounds[4,1]-model_bounds[4,0]),model_bounds[5,0]+uniform.rvs()*(model_bounds[5,1]-model_bounds[5,0])]
            tmp2 = CostFuncLS(init_tmp)
            if tmp2 < tmp1:
                init = init_tmp
                tmp1 = tmp2
        #bnds = ([model_bounds[0,0],model_bounds[1,0],model_bounds[2,0],model_bounds[3,0],model_bounds[4,0],model_bounds[5,0]],[model_bounds[0,1],model_bounds[1,1],model_bounds[2,1],model_bounds[3,1],model_bounds[4,1],model_bounds[5,1]])
        bnds = [[model_bounds[0,0],model_bounds[0,1]],[model_bounds[1,0],model_bounds[1,1]],[model_bounds[2,0],model_bounds[2,1]],[model_bounds[3,0],model_bounds[3,1]],[model_bounds[4,0],model_bounds[4,1]],[model_bounds[5,0],model_bounds[5,1]]]

        out = scipy.optimize.fmin_slsqp(CostFuncLS,init,bounds=bnds)
        #I = scipy.optimize.least_squares(CostFuncLS,init,bounds=bnds,gtol=1E-15,xtol=1E-15,ftol=1E-15,verbose=1)

    theta_pred = out
    
    return theta_pred

def NNprediction(theta):
    x = np.zeros((1,len(theta)))
    x[0,:] = theta
    return sess.run(outputs,feed_dict={X: x})

def avg_rmse_2d(x,y):
    rmse = 0.0
    n = x.shape[0]
    for i in range(n):
        rmse += np.sqrt(np.mean(np.mean(np.power((x[i,:,:]-y[i,:,:]),2),axis=0),axis=0))
    return (rmse/n)

N = 10

thetas_true = reverse_transform_X(uniform.rvs(size=(N,num_model_parameters)))
thetas_pred = np.zeros((N,num_model_parameters))
iv_surface_true = np.zeros((N,num_maturities,num_strikes))
iv_surface_true_NN = np.zeros((N,num_maturities,num_strikes))
iv_surface_pred = np.zeros((N,num_maturities,num_strikes))
iv_surface_pred_NN = np.zeros((N,num_maturities,num_strikes))

for i in range(N):
    iv_surface_true[i,:,:] = iv_surface(thetas_true[i,:]).reshape(num_maturities,num_strikes)
    thetas_pred[i,:] = predict_theta(iv_surface_true[i,:,:]).flatten()
    iv_surface_pred[i,:,:] = iv_surface(thetas_pred[i,:]).reshape(num_maturities,num_strikes)


with tf.Session() as sess:         
                       
    saver.restore(sess, model) 

    iv_surface_true_NN = sess.run(outputs,feed_dict={X: thetas_true}).reshape(N,num_maturities,num_strikes)
    iv_surface_pred_NN = sess.run(outputs,feed_dict={X: thetas_pred}).reshape(N,num_maturities,num_strikes)
  

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

#print(iv_surface_true[0,:,:])
#print(iv_surface_pred[0,:,:])


fig = plt.figure(figsize=(21, 6))

ax1=fig.add_subplot(131)

plt.imshow(100*np.mean(np.abs((iv_surface_true-iv_surface_true_NN)/iv_surface_true),axis=0))
plt.title("Average Relative Errors in \n Implied Volatilities using \n DNN1, theta true")

ax1.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax1.set_yticklabels(np.around(maturities,1))
ax1.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax1.set_xticklabels(np.around(strikes,2),rotation = (45), fontsize = 10)
plt.colorbar(format=mtick.PercentFormatter())
plt.xlabel("Strike",fontsize=12,labelpad=5)
plt.ylabel("Maturity",fontsize=12,labelpad=5)

ax2=fig.add_subplot(132)

plt.imshow(100*np.mean(np.abs((iv_surface_true-iv_surface_pred)/iv_surface_true),axis=0))
plt.title("Average Relative Errors in \n Implied Volatilities using \n MC, theta predicted")

ax2.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax2.set_yticklabels(np.around(maturities,1))
ax2.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax2.set_xticklabels(np.around(strikes,2),rotation = (45), fontsize = 10)
plt.colorbar(format=mtick.PercentFormatter())
plt.xlabel("Strike",fontsize=12,labelpad=5)
plt.ylabel("Maturity",fontsize=12,labelpad=5)

ax1=fig.add_subplot(133)

plt.imshow(100*np.mean(np.abs((iv_surface_true-iv_surface_pred_NN)/iv_surface_true),axis=0))
plt.title("Average Relative Errors in \n Implied Volatilities using \n DNN1, theta predicted")

ax1.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax1.set_yticklabels(np.around(maturities,1))
ax1.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax1.set_xticklabels(np.around(strikes,2),rotation = (45), fontsize = 10)
plt.colorbar(format=mtick.PercentFormatter())
plt.xlabel("Strike",fontsize=12,labelpad=5)
plt.ylabel("Maturity",fontsize=12,labelpad=5)

plt.tight_layout(pad=2.0, w_pad=5.0, h_pad=20.0)
plt.show()

plt.savefig('images/errors_dnn_m1_euler_hestonSLV.pdf')

print("RMSE: ",avg_rmse_2d(iv_surface_true,iv_surface_true_NN)) 
print("RMSE: ",avg_rmse_2d(iv_surface_true,iv_surface_pred)) 
print("RMSE: ",avg_rmse_2d(iv_surface_true,iv_surface_pred_NN)) 

print("Number of trainable Parameters: ",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
print("Relative error in Thetas: ", np.mean(np.abs((thetas_true-thetas_pred)/thetas_true),axis=0))