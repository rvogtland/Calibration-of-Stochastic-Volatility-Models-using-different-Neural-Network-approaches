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

np.random.seed(42)
model = "./models/sabr_m2"
use_data = False

num_model_parameters = 3
num_strikes = 8
num_maturities = 8


num_input_parameters = 5
num_output_parameters = 1
learning_rate = 0.00002

num_steps = 5000
batch_size = 60

num_neurons = 40

#initial values
S0 = 1.0
V0 = 0.3
r = 0.0

np.random.seed(42)

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
#assert(beta>0 and beta<1)

    #def mu2(V,i,k):
    #    return 0.0
    
    #def sigma2(V,i,k):
    #    return np.multiply(alpha,V)
    
    #V = euler_maruyama(mu2,sigma2,T,V0,Z)
    
    def V(i,k):
        n = W.shape[1]-1
        t = k*T/n
        return V0*np.exp(-alpha*alpha/2*t+alpha*Z[i,k])

    def mu1(S,i,k):
        return np.multiply(r,S)
    
    def sigma1(S,i,k):
        return np.multiply(V(i,k),np.power(np.maximum(0.0,S),beta))
    
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
    #return implied_volatility(P, S0, K, T, r, 'c')

def reverse_transform_X(X_scaled):
    X = np.zeros(X_scaled.shape)
    for i in range(num_model_parameters):
        X[:,i] = X_scaled[:,i]*(model_bounds[i][1]-model_bounds[i][0]) + model_bounds[i][0]
    for i in range(2):
        X[:,num_model_parameters+i] = X_scaled[:,num_model_parameters+i]*(contract_bounds[i][1]-contract_bounds[i][0]) + contract_bounds[i][0]
    return X

def transform_X(X):
    X_scaled = np.zeros(X.shape)
    for i in range(num_model_parameters):
        X_scaled[:,i] = (X[:,i] - model_bounds[i][0])/(model_bounds[i][1]-model_bounds[i][0])
    for i in range(2):
        X_scaled[:,num_model_parameters+i] = (X[:,num_model_parameters+i] - contract_bounds[i][0])/(contract_bounds[i][1]-contract_bounds[i][0])
    return X_scaled

def transform_theta(X):
    X_scaled = np.zeros(num_model_parameters)
    for i in range(num_model_parameters):
        X_scaled[i] = (X[i] - model_bounds[i][0])/(model_bounds[i][1]-model_bounds[i][0])
    return X_scaled

def reverse_transform_theta(X_scaled):
    X = np.zeros(X_scaled.shape)
    for i in range(num_model_parameters):
        X[:,i] = X_scaled[:,i]*(model_bounds[i][1]-model_bounds[i][0]) + model_bounds[i][0]
    return X

def price_pred(alpha,beta,rho,n,dim,T,K,V0,S0):
    W,Z = corr_brownian_motion(n,T,dim,rho)
    S,V = sabr(alpha,beta,T,W,Z,V0,S0)
    S_T = S[:,n]
    P = np.exp(-r*T) * np.mean(np.maximum(S_T-K,np.zeros(dim)))
    
    return P

def sabr_iv(alpha,beta,rho,T,K,S0,V0):
    
    zeta = alpha/(V0*(1-beta))*(np.power(S0,1-beta)-np.power(K,1-beta))
    S_mid = (S0 + K)/2
    gamma1 = beta/S_mid
    gamma2 = -beta*(1-beta)/S_mid/S_mid
    D = np.log((np.sqrt(1-2*rho*zeta+zeta*zeta)+zeta-rho)/(1-rho))
    eps = T*alpha*alpha
    tmp1 = alpha*(S0-K)/D
    tmp2 = (2*gamma2-gamma1*gamma1)/24*np.power((V0*np.power(S_mid,beta)/alpha),2)
    tmp3 = rho*gamma1/4*V0*np.power(S_mid,beta)/alpha
    tmp4 = (2-3*rho*rho)/24

    return tmp1*(1+eps*(tmp2+tmp3+tmp4))
    

    

def next_batch_sabr_iv(batch_size):
    x_scaled = uniform.rvs(size=(batch_size,num_input_parameters))
    x = reverse_transform_X(x_scaled)
  
    y = np.zeros((batch_size,num_output_parameters))
    for i in range(batch_size):
        y[i,0] = sabr_iv(x[i,0],x[i,1],x[i,2],x[i,4],x[i,3],S0,V0)
    return x_scaled,y


def next_batch_sabr_EM_train(batch_size,contract_bounds,model_bounds):
    X_scaled = np.zeros((batch_size,num_input_parameters))
    y = np.zeros((batch_size,num_output_parameters))

    X_scaled[:,0] = uniform.rvs(size=batch_size) #alpha
    X_scaled[:,1] = uniform.rvs(size=batch_size) #beta
    X_scaled[:,2] = uniform.rvs(size=batch_size) #rho
    X_scaled[:,3] = uniform.rvs(size=batch_size) #strikes
    X_scaled[:,4] = uniform.rvs(size=batch_size) #maturities

    X = reverse_transform_X(X_scaled)

    n = 100
    dim = 10000
    for i in range(batch_size):
        P =  price_pred(X[i,0],X[i,1],X[i,2],n,dim,X[i,3],X[i,4],V0,S0)
                
        y[i,0] = implied_vol(P,X[i,4],X[i,3])
        
        #y[i,0] = price_pred(X[i,0],X[i,1],X[i,2],n,dim,X[i,3],X[i,4],V0,S0)

    return X_scaled,y

act_func = tf.nn.elu
#Layers
bn0 = tf.nn.batch_normalization(X, 0, 1, 0, 1, 0.000001)
hidden1 = fully_connected(bn0, num_neurons, activation_fn=act_func)
bn1 = tf.nn.batch_normalization(hidden1, 0, 1, 0, 1, 0.000001)
hidden2 = fully_connected(bn1, num_neurons, activation_fn=act_func)
bn2 = tf.nn.batch_normalization(hidden2, 0, 1, 0, 1, 0.000001)
hidden3 = fully_connected(bn2, num_neurons, activation_fn=act_func)
bn3 = tf.nn.batch_normalization(hidden3, 0, 1, 0, 1, 0.000001)
hidden4 = fully_connected(bn3, num_neurons, activation_fn=act_func)
bn4 = tf.nn.batch_normalization(hidden4, 0, 1, 0, 1, 0.000001)
hidden5 = fully_connected(bn4, num_neurons, activation_fn=act_func)
bn5 = tf.nn.batch_normalization(hidden5, 0, 1, 0, 1, 0.000001)

outputs = fully_connected(bn3, num_output_parameters, activation_fn=act_func)


#Loss Function
loss = tf.sqrt(tf.reduce_mean(tf.square(outputs - y)))  # RMSE

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

"""
with tf.device('/CPU:0'):
    with tf.Session(config=config) as sess:
        sess.run(init)
        
        for iteration in range(num_steps):
            
            X_batch,Y_batch = next_batch_sabr_iv(batch_size)

            sess.run(train,feed_dict={X: X_batch, y: Y_batch})
            
            if iteration % 100 == 0:
                
                rmse = loss.eval(feed_dict={X: X_batch, y: Y_batch})
                print(iteration, "\tRMSE:", rmse)
        
        
        saver.save(sess, model)
"""

def predict_theta(prices_true):   
    
    def NNprediction(theta,maturities,strikes):
        res = np.zeros((1,num_maturities,num_strikes))
        x = np.zeros((1,num_input_parameters))
        
        x[0,0:3] = theta
       
        for i in range(num_maturities):
            for j in range(num_strikes):
                x[0,4] = maturities[i]
                x[0,3] = strikes[j]
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

        return -np.mean(np.mean(grad,axis=1),axis=1)

    def CostFuncLS(theta,maturities,strikes):
        
        return np.sqrt(np.mean(np.mean(np.power((NNprediction(theta,maturities,strikes)-prices_true)[0],2),axis=0),axis=0))


    def JacobianLS(theta,maturities,strikes):

        return NNgradientpred(theta,maturities,strikes)

    with tf.Session() as sess:         
                       
        saver.restore(sess, model)      
        
        #init = [model_bounds[0,0]+uniform.rvs()*(model_bounds[0,1]-model_bounds[0,0]),model_bounds[1,0]+uniform.rvs()*(model_bounds[1,1]-model_bounds[1,0]),model_bounds[2,0]+uniform.rvs()*(model_bounds[2,1]-model_bounds[2,0])]
        #bnds = ([model_bounds[0,0],model_bounds[1,0],model_bounds[2,0]],[model_bounds[0,1],model_bounds[1,1],model_bounds[2,1]])
        tmp1 = 1000  
        init = [model_bounds[0,0]+uniform.rvs()*(model_bounds[0,1]-model_bounds[0,0]),model_bounds[1,0]+uniform.rvs()*(model_bounds[1,1]-model_bounds[1,0]),model_bounds[2,0]+uniform.rvs()*(model_bounds[2,1]-model_bounds[2,0])]
  
        for i in range(100):
            init_tmp = [model_bounds[0,0]+uniform.rvs()*(model_bounds[0,1]-model_bounds[0,0]),model_bounds[1,0]+uniform.rvs()*(model_bounds[1,1]-model_bounds[1,0]),model_bounds[2,0]+uniform.rvs()*(model_bounds[2,1]-model_bounds[2,0])]
            tmp2 = CostFuncLS(init_tmp,maturities,strikes)
            if tmp2 < tmp1:
                init = init_tmp
                tmp1 = tmp2
        bnds = ([0,0,0],[1,1,1])

        I=scipy.optimize.least_squares(CostFuncLS,transform_theta(init),bounds=bnds,gtol=1E-15,xtol=1E-15,verbose=1,args=(maturities,strikes))

    theta_pred = I.x
    
    return theta_pred

def prices_grid(theta):
    prices_true = np.zeros((1,num_maturities,num_strikes))
    n = 100
    dim = 10000
    W,Z = corr_brownian_motion(n,maturities[-1],dim,theta[2])
    S,V = sabr(theta[0],theta[1],maturities[-1],W,Z,V0,S0)
    
    for i in range(num_maturities):
        n_current = int(maturities[i]/maturities[-1]*n)
        S_T = S[:,n_current]
        for j in range(num_strikes):        
            prices_true[0,i,j] = np.exp(-r*maturities[i])*np.mean(np.maximum(S_T-np.ones(dim)*strikes[j],np.zeros(dim)))
    return prices_true

def avg_rmse_2d(x,y):
    rmse = 0.0
    n = x.shape[0]
    for i in range(n):
        rmse += np.sqrt(np.mean(np.mean(np.power((x[i,:,:]-y[i,:,:]),2),axis=0),axis=0))
    return (rmse/n)

def sabr_ivs(theta):
    #INPUT: theta = (alpha,beta,rho)
    #OUTPUT: implied volatility surface

    IVS = np.zeros((num_maturities,num_strikes))

    for j in range(num_maturities):
        for k in range(num_strikes):
            IVS[j,k] = sabr_iv(theta[0],theta[1],theta[2],maturities[j],strikes[k],S0,V0)
    return IVS

N = 100

iv_surface_true = np.zeros((N,num_maturities,num_strikes))
iv_surface_pred = np.zeros((N,num_maturities,num_strikes))
iv_surface_true_NN = np.zeros((N,num_maturities,num_strikes))
iv_surface_pred_NN = np.zeros((N,num_maturities,num_strikes))

thetas_true_scaled = uniform.rvs(size=(N,num_model_parameters))
thetas_true = reverse_transform_theta(thetas_true_scaled)
thetas_pred = np.zeros((N,num_model_parameters))
thetas_pred_scaled = np.zeros((N,num_model_parameters))


for i in range(N):
    iv_surface_true[i,:,:] = sabr_ivs(thetas_true[i,:])
    thetas_pred_scaled[i,:] = predict_theta(iv_surface_true[i,:,:]).flatten()
    thetas_pred[i,:] = reverse_transform_theta(thetas_pred_scaled)[i,:]
    iv_surface_pred[i,:,:] = sabr_ivs(thetas_pred[i,:])



with tf.Session() as sess:                          
    saver.restore(sess, model)
    x_true = np.zeros((N,num_input_parameters)) 
    x_pred = np.zeros((N,num_input_parameters))
    x_true[:,:num_model_parameters] = thetas_true
    x_pred[:,:num_model_parameters] = thetas_pred
    for i in range(num_maturities):
        x_true[:,num_model_parameters+1] = maturities[i]
        x_pred[:,num_model_parameters+1] = maturities[i]
        for j in range(num_strikes):
            x_true[:,num_model_parameters] = strikes[j]
            x_pred[:,num_model_parameters] = strikes[j]
            iv_surface_true_NN[:,i,j] = sess.run(outputs,feed_dict={X: transform_X(x_true)})[:,0]
            iv_surface_pred_NN[:,i,j] = sess.run(outputs,feed_dict={X: transform_X(x_pred)})[:,0]

#print(iv_surface_true[0,:,:])
#print(iv_surface_true_NN[0,:,:])
#print(iv_surface_pred[0,:,:])
#print(iv_surface_true_NN[1,:,:])

""" Plot """
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

fig = plt.figure(figsize=(21,6))

ax1=fig.add_subplot(131)

plt.imshow(100*np.mean(np.abs((iv_surface_true-iv_surface_true_NN)/iv_surface_true),axis=0))
plt.title("Average Relative Errors in \n Implied Volatilities using \n DNN2, theta true")

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
plt.title("Average Relative Errors in \n Implied Volatilities using \n DNN2, theta predicted")

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
print("Avg theta true",np.mean(thetas_true,axis=0))
print("Avg theta pred",np.mean(thetas_pred,axis=0))

print("RMSE: ",avg_rmse_2d(iv_surface_true,iv_surface_true_NN)) 
print("RMSE: ",avg_rmse_2d(iv_surface_true,iv_surface_pred)) 
print("RMSE: ",avg_rmse_2d(iv_surface_true,iv_surface_pred_NN)) 
