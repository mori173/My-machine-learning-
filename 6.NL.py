# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#根据几次实验，softmax,sigmoid单独拿出来做一个函数，在后续方便
def softmax(Z):
      row_max = np.max(Z, axis = 1) #axis=1按行，=0按列,求取Z每行的最大值
      row_max = row_max.reshape(-1, 1)#reshape(-1,1)转换成1列，每行为一个元素
      Z = Z - row_max#把最大值去掉
      Z_exp = np.exp(Z)
      Z_exp_sum = np.sum(Z_exp, axis = 1, keepdims = True)
      #当 keepidms=True,保持其二维或者三维的特性,这里结果仍为列

      return Z_exp/Z_exp_sum

def sigmoid(x):
      return 1./(1 + np.exp(-x))

#Utilities
def onehotEncoder(Y, ny):
      return np.eye(ny)[Y]
      ##np.eye返回的是一个二维的数组(ny,M)，M由ny决定，对角线的地方为1，其余的地方为0.[Y]则将矩阵转化为one-hot形式

#Xavier Initialization
def initWeights(M): ##初始化，不用修改,这里是重点，M只有3层，代表了输入层、隐藏层、输出层
      l = len(M) #M = [400, 25, 10]
      W = []
      B = []
      
      for i in range(1, l): ##随机生成w、b权值
            W.append(np.random.randn(M[i-1], M[i]))
            B.append(np.zeros([1, M[i]]))

      return W, B # W = [(400,25),(25,10)]  B = [(1,25),(1,10)]

#Forward propagation，“正向传播”求损失
def networkForward(X, W, B):
      l = len(W)
      A = [None for i in range(l+1)]
      A[0] = X #输入层
      A[1] = sigmoid(np.dot(A[0], W[0]) + B[0]) #隐藏层
      A[2] = softmax(np.dot(A[1], W[1]) + B[1]) #输出层
      ##### Calculate the output of every layer A[i], where i = 0, 1, 2, ..., l
      return A ##理解A是什么？A是用来计算每一层神经元的激活值，即a = σ（z）
#--------------------------

#Backward propagation，“反向传播”回传误差。神经网络每层的每个神经元都可以根据误差信号修正每层的权重
def networkBackward(Y, A, W):
      l = len(W)

      dZ = [None for i in range(l)]
      dZ[1] = A[2] - Y
      dZ[0] = A[1] * (1 - A[1]) * np.dot(dZ[1], W[1].T)

      dW = [None for i in range(l)]
      dW[1] = A[1].T @ dZ[1]/n
      dW[0] = A[0].T @ dZ[0]/n
      # for i in range(l):
      #       dW[i+1] = (A[i+1] - Y) @ A[i]

      dB = [None for i in range(l)]
      dB[1] = 1 / n * np.sum(dZ[1], axis = 0)
      dB[0] = 1 / n * np.sum(dZ[0], axis = 0)
      # for i in range(l):
      #       dB[i] = (np.sum(A[i] - Y))/l


      ##### Calculate the partial derivatives of all w and b in each layer dW[i] and dB[i], where i = 1, 2, ..., l

      return dW, dB
#--------------------------

#Update weights by gradient descent
def updateWeights(W, B, dW, dB, lr):
      l = len(W)

      for i in range(l):
            W[i] = W[i] - lr*dW[i]
            B[i] = B[i] - lr*dB[i]

      return W, B

#Compute regularized cost function
def cost(A_l, Y, W):
      n = Y.shape[0]
      c = -np.sum(Y*np.log(A_l)) / n

      return c

def train(X, Y, M, lr = 0.1, iterations = 3000):
      costs = []
      W, B = initWeights(M)

      for i in range(iterations):
            A = networkForward(X, W, B)
            c = cost(A[-1], Y, W)
            dW, dB = networkBackward(Y, A, W)
            W, B = updateWeights(W, B, dW, dB, lr)

            if i % 100 == 0:
                  print("Cost after iteration %i: %f" %(i, c))
                  costs.append(c)

      return W, B, costs

def predict(X, W, B, Y):
      Y_out = np.zeros([X.shape[0], Y.shape[1]])
      
      A = networkForward(X, W, B)
      idx = np.argmax(A[-1], axis=1)
      Y_out[range(Y.shape[0]),idx] = 1
      
      return Y_out

def test(Y, X, W, B):
      Y_out = predict(X, W, B, Y)
      acc = np.sum(Y_out*Y) / Y.shape[0]
      print("Training accuracy is: %f" %(acc))
      
      return acc

iterations = 3000 ###### Training loops
lr = 0.7 ###### Learning rate

data = np.load("data.npy")

X = data[:,:-1]
Y = data[:,-1].astype(np.int32)
(n, m) = X.shape
Y = onehotEncoder(Y, 10)

M = [400, 25, 10]
W, B, costs = train(X, Y, M, lr, iterations)

plt.figure()
plt.plot(range(len(costs)), costs)

test(Y, X, W, B)
plt.show()
