import numpy as np
import matplotlib.pyplot as plt
import math
x = np.arange(-3,3,0.3)
y = np.arange(-3,3,0.3)
x,y = np.meshgrid(x,y)
levels = 24

# 3*(1-x)^2*exp(-(x^2)-(y+1)^2)-10*(x/5-x^3-y^5)*exp(-x^2-y^2)-1/3*exp(-(x+1)^2-y^2)
z = 3*(1-x)**2*np.exp(-x**2-(y+1)**2)-10*(x/5-x**3-y**5)*np.exp(-x**2-y**2)-1/3*np.exp(-(x+1)**2-y**2)

fig = plt.figure(figsize=(8,5))
plt.tick_params(labelsize=18)
plt.xlabel("$x$", fontsize=24)
plt.ylabel("$y$", fontsize=24)

plt.contourf(x,y,z,levels=levels,cmap="rainbow")
line = plt.contour(x,y,z,levels=levels,colors="k")

# x = np.random.uniform(-2,2)
# y = np.random.uniform(-2,2)
x = -0.15
y = 1.2

iterations = 30###### Training loops
lr = 0.025###### Learning rate

dx = dy = 0

α = 0.9 #平滑常数
α1 = 0.9 #用于平滑m,Adam
α2 =  0.99#用于平滑v,Adam
v = 0 #累计梯度
β = 0.000001#AdaGrad是10^-7,RMS\Adam是10^-6

mx = my = 0 #累计梯度，Adam
rx = ry = 0 #累计梯度的平方,初始化很重要

for  i in range(iterations):
    #梯度计算
    pdx = (-6*x**3+12*x**2-6)*np.exp(-x**2-(y+1)**2)-(20*x*y**5+20*x**4-34*x**2+2)*np.exp(-x**2-y**2)+2/3*(x+1)*np.exp(-(x+1)**2-y**2)
    pdy = ((-6*x**2+12*x-6)*y-6*x**2+12*x-6)*np.exp(-x**2-(y+1)**2)-(20*y**6-50*y**4+20*x**3*y-4*x*y)*np.exp(-x**2-y**2)+2/3*y*np.exp(-(x+1)**2-y**2)
    
    ###### Revise the code and use different GD algorithm to reach the global optimum

    #2022-10-18,测试GDM算法
    mx = α * mx + (1 - α) * pdx
    my = α * my + (1 - α) + pdy
    dx = -lr * mx
    dy = -lr * my
    #次数很少就可以到达，参数：lr = 0.025，次数30

    #AdaGrad
    # rx = rx + pdx * pdx
    # ry = ry + pdy * pdy
    # dx = -lr * pdx/(np.sqrt(rx+β))
    # dy = -lr * pdy/(np.sqrt(ry+β))
    # 跑到左边的峰去了

    #RMSProp算法，ppt里的w实际上就是要更新的x，y。ppt里的St就是梯度的平方rx,ry
    # rx = α * rx + (1 - α) * (pdx * pdx)
    # ry = α * ry + (1 - α) * (pdy * pdy)
    # dx = -lr * pdx/(np.sqrt(rx) + β)
    # dy = -lr * pdy/(np.sqrt(ry) + β)#dx、dy其实只是拿来方便运算。把-lr分开来看即可
    #跑到左边的峰去了

    #Adam
    # mx = α1 * mx + (1 - α1) * pdx
    # my = α1 * my + (1 - α1) * pdy
    # rx = α2 * rx + (1 - α2) * (pdx * pdx)
    # ry = α2 * ry + (1 - α2) * (pdy * pdy)
    # mx_hat = mx/(1 - pow(α1,i+1))
    # my_hat = my/(1 - pow(α1,i+1))
    # rx_hat = rx/(1 - pow(α2,i+1))
    # ry_hat = ry/(1 - pow(α2,i+1))
    # dx = -lr * mx_hat/(np.sqrt(rx_hat)+ β)
    # dy = -lr * my_hat/(np.sqrt(ry_hat)+ β)
    #跑到左边的峰去了


    #梯度下降算法例子
    # dx = -lr*pdx
    # dy = -lr*pdy
    ###### Revise the code and use different GD algorithm to reach the global optimum
    
    plt.arrow(x, y, dx, dy, length_includes_head=False, head_width=0.1, fc='r', ec='k')
    x += dx
    #x = x + dx
    y += dy
    #y = y + dy
plt.show()
