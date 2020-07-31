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


#initial values
use_data = False

num_model_parameters = 3
num_strikes = 8
num_maturities = 8


num_input_parameters = 3
num_output_parameters = num_maturities*num_strikes
learning_rate = 0.0001
num_steps = 5000
batch_size = 20
num_neurons = 40

#initial values
S0 = 1.0
V0 = 0.3
r = 0.0

contract_bounds = np.array([[0.8*S0,1.2*S0],[0.2,2]]) #bounds for K,T
model_bounds = np.array([[0.05,0.2],[0.2,0.8],[-0.5,-0.1]]) #bounds for alpha,beta,rho, make sure alpha>0, beta,rho \in [0,1]

"""
Note: The grid of stirkes and maturities is equidistant here put could be choosen differently for real world application.
Note: For the code below to striktly follow the bounds specified above make sure that *_distance x num_* is less than half the distance from the highest to lowest * (* = strikes/maturities). 
"""
maturities_distance = (contract_bounds[1,1]-contract_bounds[1,0])/(num_maturities) 
strikes_distance = (contract_bounds[0,1]-contract_bounds[0,0])/(num_strikes)

strikes = np.linspace(contract_bounds[0,0],contract_bounds[0,0]+num_strikes*strikes_distance,num_strikes)
maturities = np.linspace(contract_bounds[1,0],contract_bounds[1,0]+num_maturities*maturities_distance,num_maturities)


if use_data==True:
    data = np.genfromtxt('sabr_data__.csv', delimiter=',')
    x_train = data[:,:num_model_parameters]
    y_train = data[:,num_model_parameters:]
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

def sabr_iv(alpha,beta,rho,T,K,S0,V0):
    zeta = alpha/(V0*(1-beta))*(np.power(S0,1-beta)-np.power(K,1-beta))
    S_mid = (S0 + K)/2
    gamma1 = beta/S_mid
    gamma2 = -beta*(1-beta)/S_mid/S_mid
    D = np.log((np.sqrt(1-2*rho*zeta+zeta*zeta)+zeta-rho)/(1-rho))
    eps = T*alpha*alpha

    return alpha*(S0-K)/D*(1+eps*((2*gamma2-gamma1*gamma1)/24*np.power((V0*S_mid**beta/alpha),2)+rho*gamma1/4*V0*np.power(S_mid,beta)/alpha+(2-3*rho*rho)/24))


def reverse_transform_X(X_scaled):
    X = np.zeros(X_scaled.shape)
    for i in range(num_input_parameters):
        X[:,i] = X_scaled[:,i]*(model_bounds[i][1]-model_bounds[i][0]) + model_bounds[i][0]
    return X

def price_pred(alpha,beta,rho,n,dim,T,K,V0,S0,W,Z):
    S,V = sabr(alpha,beta,T,W,Z,V0,S0)
    S_T = S[:,n]
    P = np.exp(-r*T) * np.mean(np.maximum(S_T-K,np.zeros(dim)))
    
    return P

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
    #return implied_volatility(P, S0, K, T, r, 'c')

def BS_call_price(sigma,K,T):
    dplus = (np.log(S0 / K) + (r  + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    dminus = (np.log(S0 / K) + (r  - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    return S0 * norm.cdf(dplus, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(dminus, 0.0, 1.0)

def sabr_ivs(theta):
    #INPUT: theta = (alpha,beta,rho)
    #OUTPUT: implied volatility surface

    IVS = np.zeros((num_maturities,num_strikes))

    for j in range(num_maturities):
        for k in range(num_strikes):
            IVS[j,k] = sabr_iv(theta[0],theta[1],theta[2],maturities[j],strikes[k],S0,V0)
    return IVS

def next_batch_sabr_data(batch_size):
    n = np.array(uniform.rvs(size=batch_size)*x_train.shape[0]).astype(int)
    
    return x_train[n,:],y_train[n,:]

def next_batch_sabr_iv(batch_size):
    theta = reverse_transform_X(uniform.rvs(size=(batch_size,num_model_parameters)))
    iv = np.zeros((batch_size,num_output_parameters))
    for i in range(batch_size):
        iv[i,:] =  sabr_ivs(theta[i,:]).flatten()

    return theta,iv


def next_batch_sabr_EM_train(batch_size,contract_bounds,model_bounds):
    X_scaled = np.zeros((batch_size,num_input_parameters))
    y = np.zeros((batch_size,num_output_parameters))

    X_scaled[:,0] = uniform.rvs(size=batch_size) #alpha
    X_scaled[:,1] = uniform.rvs(size=batch_size) #beta
    X_scaled[:,2] = uniform.rvs(size=batch_size) #rho

    X = reverse_transform_X(X_scaled)

    n = 200
    dim = 40000
    for i in range(batch_size):
        W,Z = corr_brownian_motion(n,maturities[-1],dim,X[i,2])
        S,V = sabr(X[i,0],X[i,1],maturities[-1],W,Z,V0,S0)
        
        for j in range(num_maturities):
            n_current = int(maturities[j]/maturities[-1]*n)
            S_T = S[:,n_current]
            
            for k in range(num_strikes):
                P =  np.exp(-r*maturities[j])*np.mean(np.maximum(S_T-np.ones(dim)*strikes[k],np.zeros(dim)))
                
                y[i,j*num_strikes+k] = implied_vol(P,strikes[k],maturities[j])

                #y[i,j*num_strikes+k] = np.exp(-r*maturities[j])*np.mean(np.maximum(S_T-np.ones(dim)*strikes[k],np.zeros(dim)))
    return X_scaled,y

#Layers
bn0 = tf.nn.batch_normalization(X, 0, 1, 0, 1, 0.000001)
hidden1 = fully_connected(bn0, num_neurons, activation_fn=tf.nn.elu)
bn1 = tf.nn.dropout(tf.nn.batch_normalization(hidden1, 0, 1, 0, 1, 0.000001),keep_prob=1.0)
hidden2 = fully_connected(bn1, num_neurons, activation_fn=tf.nn.elu)
bn2 = tf.nn.dropout(tf.nn.batch_normalization(hidden2, 0, 1, 0, 1, 0.000001),keep_prob=1.0)
hidden3 = fully_connected(bn2, num_neurons, activation_fn=tf.nn.elu)
bn3 = tf.nn.dropout(tf.nn.batch_normalization(hidden3, 0, 1, 0, 1, 0.000001),keep_prob=1.0)
#hidden4 = fully_connected(hidden3, num_neurons, activation_fn=tf.nn.elu)
#bn4 = tf.nn.batch_normalization(hidden4, 0, 1, 0, 1, 0.000001)

outputs = fully_connected(hidden3, num_output_parameters, activation_fn=None)

#Loss Function

#weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

loss = tf.sqrt(tf.reduce_mean(tf.square(outputs - y))) # MSE
#for i in range(8):
#    loss += tf.nn.l2_loss(weights[i])

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
                X_batch,Y_batch = next_batch_sabr_data(batch_size)
            else:
                X_batch,Y_batch = next_batch_sabr_iv(batch_size)

            sess.run(train,feed_dict={X: X_batch, y: Y_batch})
            
            if iteration % 100 == 0:
                
                rmse = loss.eval(feed_dict={X: X_batch, y: Y_batch})
                print(iteration, "\tRMSE:", rmse)
        
        saver.save(sess, "./models/sabr_dnn_m1")
"""
def iv_surface(theta):
    ivs = np.zeros((1,num_output_parameters))
    n = 200
    dim = 40000
    W,Z = corr_brownian_motion(n,maturities[-1],dim,theta[2])
    S,V = sabr(theta[0],theta[1],maturities[-1],W,Z,V0,S0)
    
    for i in range(num_maturities):
        n_current = int(maturities[i]/maturities[-1]*n)
        S_T = S[:,n_current]
        for j in range(num_strikes):   
            P =  np.mean(np.maximum(S_T-np.ones(dim)*strikes[j],np.zeros(dim)))
     
            ivs[0,i*num_strikes+j] = implied_vol(P,strikes[j],maturities[i])
    return ivs



def predict_theta(ivs_true):   
    
    def NNprediction(theta):
        x = np.zeros((1,len(theta)))
        x[0,:] = theta
        return sess.run(outputs,feed_dict={X: x})
    def NNgradientpred(x):
        x = np.asarray(x)
        grad = np.zeros((num_output_parameters,num_input_parameters))
        
        delta = 1e-8
        for i in range(num_input_parameters):
            h = np.zeros(x.shape)
            h[0,i] = delta
            
            #two point gradient
            grad[:,i] = (sess.run(outputs,feed_dict={X: x+h}) - sess.run(outputs,feed_dict={X: x-h}))/2/delta

            #four point gradient
            #grad[:,i] = (-sess.run(outputs,feed_dict={X: x+2*h})+8*sess.run(outputs,feed_dict={X: x+h})-8*sess.run(outputs,feed_dict={X: x-h}) +sess.run(outputs,feed_dict={X: x-2*h}))/12/delta

        return np.mean(grad,axis=0)

    def CostFuncLS(theta):
        
        return np.mean(np.power((NNprediction(theta)-ivs_true.flatten())[0],2),axis=0)


    def JacobianLS(theta):
        x = np.zeros((1,len(theta)))
        x[0,:] = theta
        return NNgradientpred(x).T

    with tf.Session() as sess:                          
        #saver.restore(sess, "./models/sabr_dnn")  
        saver.restore(sess, "./models/sabr_dnn_m1")    
        tmp1 = 1000  
        init = [model_bounds[0,0]+uniform.rvs()*(model_bounds[0,1]-model_bounds[0,0]),model_bounds[1,0]+uniform.rvs()*(model_bounds[1,1]-model_bounds[1,0]),model_bounds[2,0]+uniform.rvs()*(model_bounds[2,1]-model_bounds[2,0])]
  
        for i in range(100):
            init_tmp = [model_bounds[0,0]+uniform.rvs()*(model_bounds[0,1]-model_bounds[0,0]),model_bounds[1,0]+uniform.rvs()*(model_bounds[1,1]-model_bounds[1,0]),model_bounds[2,0]+uniform.rvs()*(model_bounds[2,1]-model_bounds[2,0])]
            tmp2 = CostFuncLS(init_tmp)
            if tmp2 < tmp1:
                init = init_tmp
                tmp1 = tmp2
        bnds = ([model_bounds[0,0],model_bounds[1,0],model_bounds[2,0]],[model_bounds[0,1],model_bounds[1,1],model_bounds[2,1]])

        
        I_=scipy.optimize.least_squares(CostFuncLS,init,JacobianLS,bounds=bnds,gtol=1E-15,xtol=1E-15,ftol=1E-15,verbose=0)
        I=scipy.optimize.least_squares(CostFuncLS,I_.x,bounds=bnds,gtol=1E-15,xtol=1E-15,ftol=1E-15,verbose=0)
    theta_pred = I.x
  
    return theta_pred

def avg_rmse_2d(x,y):
    rmse = 0.0
    n = x.shape[0]
    for i in range(n):
        rmse += np.sqrt(np.mean(np.mean(np.power((x[i,:,:]-y[i,:,:]),2),axis=0),axis=0))
    return (rmse/n)

N = 100

iv_surface_true = np.zeros((N,num_maturities,num_strikes))
iv_surface_pred = np.zeros((N,num_maturities,num_strikes))
iv_surface_true_NN = np.zeros((N,num_maturities,num_strikes))
iv_surface_pred_NN = np.zeros((N,num_maturities,num_strikes))

thetas_true = reverse_transform_X(uniform.rvs(size=(N,num_model_parameters)))
thetas_pred = np.zeros((N,num_model_parameters))

for i in range(N):
    iv_surface_true[i,:,:] = sabr_ivs(thetas_true[i,:])
    thetas_pred[i,:] = predict_theta(iv_surface_true[i,:,:]).flatten()
    iv_surface_pred[i,:,:] = sabr_ivs(thetas_pred[i,:])

with tf.Session() as sess:                          
    saver.restore(sess, "./models/sabr_dnn_m1")

    iv_surface_true_NN = sess.run(outputs,feed_dict={X: thetas_true}).reshape(N,num_maturities,num_strikes)
    iv_surface_pred_NN = sess.run(outputs,feed_dict={X: thetas_pred}).reshape(N,num_maturities,num_strikes)

#print(iv_surface_true[0,:,:])
#print(iv_surface_true_NN[0,:,:])
#print(iv_surface_pred[0,:,:])
#print(iv_surface_true_NN[1,:,:])

""" Plot """


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

fig = plt.figure(figsize=(21, 6))
fig.tight_layout()
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

ax3=fig.add_subplot(133)

plt.imshow(100*np.mean(np.abs((iv_surface_true-iv_surface_pred_NN)/iv_surface_true),axis=0))
plt.title("Average Relative Errors in \n Implied Volatilities using \n DNN1, theta predicted")

ax3.set_yticks(np.linspace(0,num_maturities-1,num_maturities))
ax3.set_yticklabels(np.around(maturities,1))
ax3.set_xticks(np.linspace(0,num_strikes-1,num_strikes))
ax3.set_xticklabels(np.around(strikes,2),rotation = (45), fontsize = 10)
plt.colorbar(format=mtick.PercentFormatter())
plt.xlabel("Strike",fontsize=12,labelpad=5)
plt.ylabel("Maturity",fontsize=12,labelpad=5)
plt.tight_layout(pad=2.0, w_pad=5.0, h_pad=20.0)
plt.show()

plt.savefig('rel_errors_dnn_m2_sabr.pdf') 

print("Number of trainable Parameters: ",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
print("Relative error in Thetas: ", np.mean(np.abs((thetas_true-thetas_pred)/thetas_true),axis=0))

print("RMSE: ",avg_rmse_2d(iv_surface_true,iv_surface_true_NN)) 
print("RMSE: ",avg_rmse_2d(iv_surface_true,iv_surface_pred)) 
print("RMSE: ",avg_rmse_2d(iv_surface_true,iv_surface_pred_NN)) 


import timeit

with tf.Session() as sess:                          
    saver.restore(sess, "./models/sabr_dnn_m1")
    start = timeit.timeit()
    
    sess.run(outputs,feed_dict={X: thetas_true})
    
    end = timeit.timeit()
print("Time: ",(end-start))
"""
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
fig = plt.figure(figsize=(30,6))
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X=strikes, Y=maturities, Z=iv_surface_true[0,:,:], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax1.zaxis.set_major_locator(LinearLocator(10))
ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax1.set_xlabel('T')
ax1.set_ylabel('K')
ax1.set_zlabel('Implied Volatility')

plt.show()
"""