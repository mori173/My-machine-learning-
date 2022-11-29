# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math as mh

def find_best_split(X_i, Y, weight):##找分界
    b = [None]*(X_i.shape[0] - 1)
    sign = [None]*(X_i.shape[0] - 1)
    e = [None]*(X_i.shape[0] - 1)
    Y_hat = np.zeros_like(Y)
    
    X_sorted = np.sort(X_i)
    for i in range(X_i.shape[0] - 1):
        b[i] = (X_sorted[i] + X_sorted[i+1]) / 2
        idx = (X_i>b[i])
        keys, counts = np.unique(Y[idx], return_counts=True)
        sign[i] = keys[np.argmax(counts)]
        Y_hat[idx] = sign[i]
        Y_hat[~idx] = -sign[i]
        e[i] = np.sum(weight[Y_hat != Y])
        
    idx = np.argmin(e)
    return b[idx], sign[idx], e[idx]

def train_weak_learner(X, Y, weight):##训练弱学习器
    b = [None]*X.shape[1]
    sign = [None]*X.shape[1]
    e = [None]*X.shape[1]
    
    for i in range(X.shape[1]):
        b[i], sign[i], e[i] = find_best_split(X[:,i], Y, weight)
        
    idx = np.argmin(e)
    W = np.zeros([X.shape[1], 1])
    W[idx] = sign[idx]
    b = -sign[idx]*b[idx]
    
    return W, b, e[idx]

def ada_boost(X, Y, n, T):  #主函数，把上面的方法放进来用
    weight = np.full([n,1], 1/n)#初始权值，1/n
    #e = [None]
    #sign = [None]*T
    alpha = [None]*T
    W = [None]*T
    b = [None]*T

    ###### Calculate all alpha_i, W_i and b_i

    for i in range(T):
        W[i],b[i],e = train_weak_learner(X, Y, weight)

        #alpha[i] = 1/2 * (np.log((1 - e[i])/e[i]))
        alpha[i] = -np.log((1-e)/e)/2
        #weight = (weight*mh.exp(-alpha[i] * Y * train_weak_learner(X, Y, weight)))\
        #         /np.sum(mh.exp(-alpha[i] * Y * train_weak_learner(X, Y, weight)))
        for j in range(n):
            if(Y[j] * (X[j] @ W[i] + b[i])>0):
                weight[j] = weight[j]/(2 * (1-e))
            else:
                weight[j] = weight[j]/(2 * e)


    return alpha, W, b

def error(alphas, Ws, bs, X, Y):
    z = 0
    for alpha, W, b in zip(alphas, Ws, bs):
        z += alpha*np.sign(X@W + b)
    Y_hat = np.sign(z)
    
    return (1-np.mean(np.equal(Y_hat, Y)))

data = np.loadtxt('AB2.txt', delimiter=',')

n = data.shape[0]
X = data[:,0:2]
Y = np.expand_dims(data[:, 2], axis=1)

T = 3###### Determine how many weak learners do you need
alphas, Ws, bs = ada_boost(X, Y, n, T)
print(error(alphas, Ws, bs, X, Y))

#Draw figure
idx0 = (data[:, 2]==-1)
idx1 = (data[:, 2]==1)

plt.figure()
plt.ylim(-10,10)
plt.plot(data[idx0,0], data[idx0,1],'bx')
plt.plot(data[idx1,0], data[idx1,1],'ro')

x1 = np.arange(-10,10,0.1)
for W, b in zip(Ws, bs):
    y1 = (b + W[0]*x1) / (-W[1] + 1e-8)
    plt.plot(x1, y1)
plt.show()