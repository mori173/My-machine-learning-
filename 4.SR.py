# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
    
def cost_gradient(W, X, Y, n):
    Z = X @ W
    Y_hat = np.exp(Z)
    ###### Output Y_hat by the trained model
    for i in range(n):
        Y_hat[i,:] = Y_hat[i,:] / np.sum(Y_hat[i,:])
    G = X.T @ (Y_hat - Y)  ###### Gradient
    temp = np.log(Y_hat)
    j =  -(temp @ Y.T )
    j = np.sum(j)   ###### cost with respect to current W
    return (j, G)

def train(W, X, Y, n, lr, iterations):
    J = np.zeros([iterations, 1])

    for i in range(iterations):
        (J[i], G) = cost_gradient(W, X, Y, n)
        W = W - lr*G

    return (W,J)

def error(W, X, Y):
    Z = X @ W
    Y_hat = np.exp(Z)
    ###### Output Y_hat by the trained model
    for i in range(1500):
        Y_hat[i, :] = Y_hat[i, :] / np.sum(Y_hat[i, :])
    pred = np.argmax(Y_hat, axis=1)
    label = np.argmax(Y, axis=1)

    return (1-np.mean(np.equal(pred, label)))

iterations = 300###### Training loops
lr = 0.005###### Learning rate

data = np.loadtxt('SR.txt', delimiter=',')

n = data.shape[0]
X = np.concatenate([np.ones([n, 1]),
                    np.expand_dims(data[:,0], axis=1),
                    np.expand_dims(data[:,1], axis=1),
                    np.expand_dims(data[:,2], axis=1)],
                   axis=1)
Y = data[:, 3].astype(np.int32)
c = np.max(Y)+1
Y = np.eye(c)[Y]

W = np.random.random([X.shape[1], c])

(W,J) = train(W, X, Y, n, lr, iterations)

plt.figure()
plt.plot(range(iterations), J)
plt.show()
print(error(W,X,Y))