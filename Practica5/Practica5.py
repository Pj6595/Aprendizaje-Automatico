import numpy as np
from numpy.lib.function_base import gradient
import operator
import matplotlib.pyplot as plt
from numpy.lib.npyio import load
import scipy.optimize as opt
from scipy.io import loadmat

data = loadmat('ex5data1.mat')

X = data['X']
y = data['y']
X_orig = X

X_test = data['Xtest']
X_test_orig = X_test
y_test = data['ytest']

X_val = data['Xval']
X_val_orig = X_val
y_val = data['yval']

x = np.hstack([np.ones([X.shape[0],1]),X])
X_val=np.hstack([np.ones([X_val.shape[0],1]),X_val])
X_test=np.hstack([np.ones([X_test.shape[0],1]),X_test])

thetas = np.ones(X.shape[1])
thetas_orig = np.zeros(X.shape[1])

# def cost(thetas, X, y, reg = 0):
#     m = X.shape[0]
#     H = np.dot(X, thetas)
#     cost = (1/(2*m)) * (np.sum(np.power(H - np.transpose(y), 2)))
#     cost2 = (reg/(2*m)) * np.sum(thetas**2)
#     return cost + cost2

def cost(thetas, X, Y, reg=0):
    m = X.shape[0]
    H = np.dot(X, thetas)
    cost = (1/(2*m)) * np.sum((H-Y.T)**2) + ( reg / (2 * m) ) * np.sum(thetas[1:]**2)
    return cost

print(cost(thetas, X, y, 1))