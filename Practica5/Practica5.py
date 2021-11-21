import numpy as np
from numpy.lib.function_base import gradient
import operator
import matplotlib.pyplot as plt
from numpy.lib.npyio import load
import scipy.optimize as opt
from scipy.io import loadmat

data = loadmat('ex5data1.mat')

X = data['X']
X_orig = X
y = data['y']

X_test = data['Xtest']
y_test = data['ytest']

X_val = data['Xval']
y_val = data['yval']

X = np.hstack([np.ones([X.shape[0],1]),X])
X_val=np.hstack([np.ones([X_val.shape[0],1]),X_val])
X_test=np.hstack([np.ones([X_test.shape[0],1]),X_test])

thetas = np.ones(X.shape[1])

def cost(thetas, X, Y, reg=0):
    m = X.shape[0]
    H = np.dot(X, thetas)
    cost = (1/(2*m)) * np.sum(np.square((H-Y.T))) + (reg/(2*m)) * np.sum(np.square(thetas[1:]))
    return cost

def gradient(thetas, X, Y, reg=0):
    aux = np.hstack(([0], thetas[1:]))
    m = X.shape[0]
    H = np.dot(X, thetas)
    grad = (1/m) * np.dot((H-Y.T), X) + (reg/m) * aux
    return grad

def apartado_1():
    print("Cost: " + str(cost(thetas, X, y, 1)))
    print("Gradient: " + str(gradient(thetas, X, y, 1)))

    reg = 0
    thetas_opt = opt.minimize(fun= cost, x0= thetas, args= (X, y, reg)).x

    plt.figure()
    plt.scatter(X[:,1], y, marker= "x", color="red")
    Y_pred = np.dot(X, thetas_opt)
    plt.plot(X[:,1], Y_pred)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.savefig("apartado1.png")

def apartado_2():
    m = X.shape[0]
    reg = 0
    train_errors = []
    val_errors = []

    for i in range(1, m + 1):
        thetas_opt = opt.minimize(fun= cost, x0= thetas, args= (X[:i], y[:i], reg)).x
        train_errors.append(cost(thetas_opt, X[:i], y[:i]))
        val_errors.append(cost(thetas_opt, X_val, y_val))

    plt.figure()
    plt.plot(range(1, m+1), train_errors)
    plt.plot(range(1, m+1), val_errors, c='orange')
    plt.legend(("Train", "Cross Validation"))
    plt.xlabel("Number of trainig examples")
    plt.ylabel("Error")
    plt.savefig("apartado2.png")

def polinomial(X, p):
    pol = np.empty((X.shape[0],p))
    for i in range(p):
        pol[:,i] = (X**(i+1)).ravel()
    return pol

def normalizar(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

def apartado_3():
    p = 8
    reg = 0
    
    X_pol, mu, sigma = normalizar(polinomial(X_orig, p))
    X_pol = np.hstack([np.ones((X_pol.shape[0], 1)), X_pol])

    thetas = np.zeros(X_pol.shape[1])
    
    thetas_opt = opt.minimize(fun = cost, x0 = thetas, args = (X_pol, y, reg)).x
    plt.figure()

    X_test = np.arange(np.min(X),np.max(X),0.05)
    X_test = polinomial(X_test,8)
    X_test = (X_test - mu) / sigma
    X_test =np.hstack([np.ones([X_test.shape[0],1]),X_test])
    Y_pred = np.dot(X_test, thetas_opt)
    plt.plot(np.arange(np.min(X),np.max(X),0.05),Y_pred)
    plt.scatter(X_orig,y,marker="X", color="red")
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.savefig("apartado_3.png")
    plt.show()

apartado_1()
apartado_2()
apartado_3()