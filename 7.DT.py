# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter

def entropy(Y):
    '''计算信息熵，既可以计算Y也可以计算Xi'''
    Y = list(map(int, Y))
    count = Counter(Y)
    # l_y = len(Y)
    ent = 0
    for key in count.keys():
        ent -= count[key]/len(Y) * np.log(count[key]/len(Y) + 1e-5)
    return ent

def conditional_entropy(Y, Xi):
    '''计算条件熵'''
    # 原始数据读进来元素是字符串类型
    Y = list(map(int, Y))
    Xi = list(map(int, Xi))

    count_Xi = Counter(Xi)  # 一个字典，键key是特征值，值value是对应值的出现次数        {0:5,1:5,2:6}
    count_Y = Counter(Y) # 计算每种决策的频数

    # print(count)
    # print(Xi)

    count_decision = {}  # 嵌套字典，把决策值当作键key，把这个决策的特征所有取值的频数，也是一个字典作为值value
    for dece in count_Y.keys():
        count_decision[dece] = {}  # 创建空的子字典
        for key in count_Xi:
            (count_decision[dece])[key] = 0

    for idx in range(len(Xi)):  # 对于特征xi的每一种取值，把取值作为键，出现次数作为值
        (count_decision[Y[idx]])[Xi[idx]] += 1

    # for dece in count_decision.keys():
    # print(count_decision)
    # print()

    # 计算每种决策中，每个特征的每个取值出现的次数
    # count_1 = {}
    # count_0 = {}
    # for key in count_Xi:
    #     count_1[key] = 0
    #     count_0[key] = 0
    #
    # for idx in range(len(Xi)):
    #     if Y[idx] == 1:
    #         count_1[Xi[idx]] += 1
    #     else:
    #         count_0[Xi[idx]] += 1

    # print(count_1)
    # print(count_0)

    P_Vi = {}
    for k in count_Xi:
        P_Vi[k] = count_Xi[k]/len(Xi)  # 1/16

    # print(P_Vi)

    entropy_Y_Xi = 0
    for k in count_Xi.keys():
        sum = 0
        for dece in count_decision.keys():
            sum += (count_decision[dece])[k] / count_Xi[k] * np.log((count_decision[dece])[k] / count_Xi[k] + 1e-5)
        entropy_Y_Xi -= P_Vi[k] * sum
        # entropy_Y_Xi -= P_Vi[k] * ((count_1[k]/count_Xi[k]) * np.log(count_1[k]/count_Xi[k] + 1e-5) + (count_0[k]/count_Xi[k]) * np.log(count_0[k]/count_Xi[k] + 1e-5))
    # 养成习惯 遇到log就加上1e-5
    return entropy_Y_Xi


def select_feature(Y, X):
    d = X.shape[1]  # 一共有d个特征
    CEs = [None for i in range(d)]  # 计算每一个特征的信息熵

    entropy_Y = entropy(Y)
    for i in range(d):
        CEs[i] = entropy_Y - conditional_entropy(Y,X[:,i])  # 信息增益IG
        #CEs[i] = (entropy_Y - conditional_entropy(Y, X[:, i])) / entropy(X[:, i])  # 信息增益比

    return np.argmax(CEs)

def delete_feature(X, features, x_i):
    X[:,x_i] = X[:,-1]
    features[x_i] = features[-1]
    
    return X[:,:-1], features[:-1]

# 主线程不能做耗时大的任务，分支线程不能更新界面

def ID3(Y, X, features):
    keys, counts = np.unique(Y, return_counts=True)
    # 边界条件
    if keys.shape[0] == 1:# 只有一个键，纯净了
        return keys[0]
    elif X.shape[1] == 1:# 所有特征都用光了，还是不纯净，那就把占比多的作为决策
        return keys[np.argmax(counts)]
    # 选择一个特征作为根节点
    x_i = select_feature(Y, X)
    x_feature = features[x_i]
    tree = {x_feature: {}}
    # 递归
    idx = []
    keys = np.unique(X[:,x_i])
    for k in keys:
        idx.append(X[:,x_i]==k)
        
    X, features = delete_feature(X, features, x_i)
    for k, i in zip(keys, idx):
        tree[x_feature][k] = ID3(Y[i], X[i], features)
        
    return tree

def predict(X, features, tree):
    feature = list(tree.keys())[0]
    tree = tree[feature]
    
    idx = np.where(features == feature)[0]
    value = X[idx].item()
    tree = tree[value]
    
    if isinstance(tree, dict):
        return predict(X, features, tree)
    else:
        return tree

def DTtest(Y, X, features, tree):
    Y_hat = np.zeros(X.shape[0], dtype="str")
    
    for i in range(X.shape[0]):
        Y_hat[i] = predict(X[i], features, tree)
    
    return np.mean(Y_hat == Y)

data = np.loadtxt('ID3.txt', dtype="str", delimiter=',')
features = data[0,:-1]
X = data[1:,:-1]  # [x1 x2 x3 x4]
Y = data[1:,-1]   # [y]

tree = ID3(Y, X, features)
print(tree)

data = np.loadtxt('ID3.txt', dtype="str", delimiter=',')
features = data[0,:-1]
X = data[1:,:-1]
Y = data[1:,-1]

print(DTtest(Y, X, features, tree))

