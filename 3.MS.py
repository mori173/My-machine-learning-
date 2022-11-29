# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def read_data(addr):
    data = np.loadtxt(addr, delimiter=',')
    n = data.shape[0]  # 1000
    # [1 x1 x2 x3 x4 x5 x6]
    x1 = data[:, 0:1]
    x2 = data[:, 1:2]
    x3 = data[:, 2:3]
    x4 = data[:, 3:4]
    x5 = data[:, 4:5]
    x6 = data[:, 5:6]
    X = np.concatenate(
        [np.ones([n, 1]),
         np.power(x1, 4).reshape([1000, 1]),  # 4
         np.power(x2, 2).reshape([1000, 1]),  # 2
         np.power(x3, 3).reshape([1000, 1]),  # 3
         np.power(x4, 1).reshape([1000, 1]),  # 1
         np.power(x5, 5).reshape([1000, 1]),  # 5
         np.power(x6, 6).reshape([1000, 1])],  # 4
        axis=1)
    Y = np.expand_dims(data[:, 6], axis=1) # 真实值 (1000,1)

    return (X,Y,n)

def cost_gradient(W, X, Y, n):
    Z = X @ W  # [100,1]
    Y_hat = 1 / (1 + np.exp(-Z))
    G = X.T @ (Y_hat - Y)  # 不要除以n!!!梯度没有除以n 是W = W - lr/n * G 的时候
    j = 1 / n * np.sum(Y * np.log(1 + np.exp(-Z)) + (1 - Y) * np.log(1 + np.exp(Z)))  # 变形提出负号消掉罢了
    return (j, G)

def train(W, X, Y, lr, n, iterations):
    J = np.zeros([iterations, 1])
    E_trn = np.zeros([iterations, 1])  # 训练err
    E_val = np.zeros([iterations, 1])  # 验证err

    W_nf = []  # 存放每一折的参数
    J_nf = []  # 存放每一折的损失
    E_trn_nf = []  # 存放每一折的训练误差
    E_val_nf = []  # 存放每一折的验证误差
    n_fold = 10
    step = n/n_fold  # 数据集切割比例因子，此处数据类型为float下面需要转int
    for i in range(n_fold):
        line = i * int(step)
        X_trn = np.concatenate([X[:line], X[line+int(step):]], axis=0)
        X_val = X[line:line+int(step)]
        Y_trn = np.concatenate([Y[:line], Y[line+int(step):]], axis=0)
        Y_val = Y[line:line + int(step)]

        W_tmp = W  # 恢复传入的随机初始化参数W
        for i in range(iterations):
            (J[i], G) = cost_gradient(W_tmp, X_trn, Y_trn, (n_fold-1)/n_fold * n)
            W_tmp = W_tmp - lr*G
            E_trn[i] = error(W_tmp, X_trn, Y_trn)
            E_val[i] = error(W_tmp, X_val, Y_val)

        W_nf.append(W_tmp)
        J_nf.append(J)
        E_trn_nf.append(E_trn)
        E_val_nf.append(E_val)

    return (W_nf, J_nf, E_trn_nf, E_val_nf, n_fold)
    # line = int(0.9*n)  # 10折交叉 9份train 1份cv
    # X_trn = X[:line]
    # Y_trn = Y[:line]
    # X_val = X[line:]
    # Y_val = Y[line:]
def error(W, X, Y):
    Y_hat = 1 / (1 + np.exp(-X @ W))
    Y_hat[Y_hat < 0.5] = 0
    Y_hat[Y_hat > 0.5] = 1

    return (1-np.mean(np.equal(Y_hat, Y)))  # err的平均/bias

# def test():
#     W = np.loadtxt("weights.txt", delimiter=',')
#     W = np.expand_dims(W, axis=1)
#     (X, Y, _) = read_data("test.txt")
#
#     return error(W, X, Y)

iterations = 10000  ###### 10000
lr = 0.0001 ###### 0.0001

(X, Y, n) = read_data("train.txt")
W = np.zeros([X.shape[1], 1])  # (13,1)

###### You may modify this section to do 10-fold validation
# (W, J, E_trn, E_val) = train(W, X, Y, lr, n, iterations)
(W_nf, J_nf, E_trn_nf, E_val_nf, n_fold) = train(W, X, Y, lr, n, iterations)
for i in range(n_fold):
    plt.figure()
    plt.ylim(0, 0.3)
    print("第{}折:".format(i+1))
    print('val error:', (E_val_nf[i])[-1])
    print('train error:', (E_trn_nf[i])[-1])
    print(W_nf[i])
    plt.plot(range(iterations), E_trn_nf[i], "b", label='train error')
    plt.plot(range(iterations), E_val_nf[i], "r", label='val error')
    plt.legend()
    plt.show()

# plt.figure()
# plt.title('Loss')
# plt.plot(range(iterations), J)
# plt.figure()
# plt.ylim(0,0.3)
# print('val:', E_val[-1])
# print('tra:', E_trn[-1])
# plt.plot(range(iterations), E_trn, "b", label='train error')
# plt.plot(range(iterations), E_val, "r", label='val error')
# plt.legend()
# plt.show()
###### You may modify this section to do 10-fold validation

np.savetxt("weights.txt", W, delimiter=',')

# print(test())
