import numpy as np

def sigmoid(A):
    return 1 / (1 + np.exp(-A))

#def softmax(A):
#    expA = np.exp(A)
#    return expA / expA.sum(axis=1, keepdims=True)

def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

#def cost(T, Y):
#    return -(T*np.log(Y)).sum()

#def cost2(T, Y):
    # same as cost(), just uses the targets to index Y
    # instead of multiplying by a large indicator matrix with mostly 0s
#    N = len(T)
#    return -np.log(Y[np.arange(N), T]).mean()


def error_rate(targets, predictions):
    return np.mean(targets != predictions)
