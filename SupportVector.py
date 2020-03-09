import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit

C=10
learning_rate = 0.00000001

def Loss(X, Y, w, bias):
    global C
    loss= (1-Y*(X @ w.T + bias)).clip(min=0)
    Reg = np.power(w, 2)
    return (np.sum(Reg) / 2) + C * max(0,np.sum(loss))

def preprocess():
    my_data = pd.read_csv('data.txt',sep='\t')
    X = my_data.iloc[:, 0:8].values
    Y = my_data.iloc[:, 8:9].values
    w = np.zeros([1, 8])
    bias=0
    return X, Y, w, bias


def gradientweights(X, Y, w, bias, j):
    cal = Y*(X @ w.T + bias)
    n = np.zeros([760, 1])
    Xj = X[:, j]
    Xj = Xj.reshape(len(X), 1)
    cal = np.where(cal >= 1, n, -1 * Y * Xj)
    return cal


def gradientbias(X, Y, w, bias):
    cal = Y*(X @ w.T + bias)
    n = np.zeros([760, 1])
    cal = np.where(cal >= 1, n, -1 * Y)
    return cal


def Mbatchweight(X,Y,w,bias,j,batch_size,l):

   cal = gradientweights(X,Y,w,bias,j)
   batch_sum = np.sum(cal[int(l*batch_size+1):int(min(len(X),((l+1)*batch_size)))])
   return w[0,j]+C*batch_sum

def Mbatchbias(X,Y,w,bias,batch_size,l):

    cal = gradientbias(X,Y,w,bias)
    batch_sum = np.sum(cal[int(l*batch_size+1):int(min(len(X),((l+1)*batch_size)))])
    return bias + C*batch_sum


def Stocasticweight(X,Y,w,bias,j,i):
    cal = gradientweights(X,Y,w,bias,j)
    return w[0,j] + C*cal[i,0]

def Stocasticbias(X,Y,w,bias,i):
    cal = gradientbias(X,Y,w,bias)
    return bias + C*cal[i,0]

def Batchweights(X,Y,w,bias,j):
    global C
    return w[0,j]+C*np.sum(gradientweights(X,Y,w,bias,j))

def Batchbias(X,Y,w,bias):
    global C
    return bias + C*np.sum(gradientbias(X,Y,w,bias))


def Convergence(cost,k):
    if(k != 0):
        return (abs(cost[k-1]-cost[k])*100)/cost[k-1]
    else:
        return 1


def Stocastic(X, Y, w, bias, epsilon):
    global learning_rate
    k = 0
    i = 0
    cost = []
    costk = [10]
    m=len(w[0])
    while (costk[k] > epsilon):

        for j in range(m):
            w[0][j] = w[0][j] - learning_rate * Stocasticweight(X, Y, w, bias, j, i)
            bias = bias - learning_rate * Stocasticbias(X,Y, w, bias, i)

        cost.append(Loss(X, Y, w, bias))
        tempCost = Convergence(cost, k)
        costk.append(0.5 * costk[k - 1] + 0.5 * tempCost)

        i = (i + 1) % len(X)
        k += 1
    return w, cost, k, bias


def Mbatch(X, Y, w, bias,epsilon):
    global learning_rate, C
    learning_rate =0.00000001
    l = 0
    k = 0
    batch_size = 4
    cost = []
    costk = [10]
    m=len(w[0])

    while (costk[k] > epsilon):

        for j in range(m):
            w[0][j] = w[0][j] - learning_rate * Mbatchweight(X, Y, w, bias, j, batch_size, l)
            bias = bias - learning_rate * Mbatchbias(X, Y, w, bias, batch_size, l)

        cost.append(Loss(X, Y, w, bias))
        tempCost = Convergence(cost, k)
        costk.append(0.5 * costk[k - 1] + 0.5 * tempCost)

        l = (l + 1) % ((len(X) + batch_size - 1) / batch_size)
        k = k + 1

    return w, cost, k, bias


def Batch(X, Y, w, bias, epsilon):
    global C, learning_rate
    learning_rate =0.000000001
    k = 0
    cost = []
    tempCost=10
    while (tempCost > epsilon):

        for j in range(len(w[0])):
            w[0][j] = w[0][j] - learning_rate * Batchweights(X, Y, w, bias, j)
            bias = bias - learning_rate * Batchbias(X, Y, w, bias)

        cost.append(Loss(X, Y, w, bias))
        tempCost = Convergence(cost, k)
        k += 1
    return w, cost, k, bias

def main():
    global learning_rate, C
    X, Y, w, bias = preprocess()
    bias = 0

    print("\n\n")
    print("Stochastic-gradient\n")

    X, Y, w, bias = preprocess()
    epsilon=0.0003
    begin = timeit.timeit()
    W, stochcost, k1, B = Stocastic(X, Y, w, bias, epsilon)
    end = timeit.timeit()
    print("Converged at cost:", Loss(X, Y, W, B))
    print("Time to converge:", abs((1000 * (begin - end))))
    print("Converged in: ", k1, " iterations")

    print("\n\n")

    print("Batch-gradient\n")


    X, Y, w, bias = preprocess()
    epsilon = 0.004
    begin1 = timeit.timeit()
    W ,batchcost,k2, B = Batch(X,Y,w,bias,epsilon)
    end1 = timeit.timeit()
    print("Converged at cost:", Loss(X, Y, W, B))
    print("Time to converge:", abs((1000 * (begin1 - end1))))
    print("Converged in: ", k2, " iterations")

    print("\n\n")

    print("MiniBatch-gradient\n")


    X, Y, w, bias = preprocess()
    print(Loss(X, Y, w, bias))
    epsilon = 0.004
    begin2 = timeit.timeit()
    W, Mbcost, k3, B = Mbatch(X, Y, w, bias,epsilon)
    end2 = timeit.timeit()
    print("Converged at cost:", Loss(X, Y, W, B))
    print("Time to converge:", abs((1000 * (begin2 - end2))))
    print("Converged in: ", k3, " iterations")

    print("\n\n")


    fig, ax = plt.subplots()
    ax.plot(np.arange(k1), stochcost, color='r', label='Stocastic')
    ax.plot(np.arange(k2), batchcost, color='b', label='Batch')
    ax.plot(np.arange(k3), Mbcost, color='g', label='Mini Batch')

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    plt.legend(loc='best')
    plt.show()



main()