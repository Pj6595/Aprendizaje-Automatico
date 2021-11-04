import operator
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import load
import scipy.optimize as opt
from scipy.io import loadmat

def oneVsAll(X, y, num_etiquetas, reg):
    clasificadores = np.zeros(shape=(10, 400))
    
    for i in range (1, num_etiquetas + 1):
        filtrados = (y==i) * 1
        thetas = np.zeros(np.shape(X)[1])
        clasificadores[i - 1] = opt.fmin_tnc(func=costeReg, x0=thetas, fprime=gradienteReg, args=(X, filtrados, reg))[0]
        
    return clasificadores

def prediccion(X, Y, clasificadores):
    predicciones = {}
    Y_pred = []
    for imagen in range(np.shape(X)[0]):
        for i in range(clasificadores.shape[0]):
            theta_opt = clasificadores[i]
            etiqueta = i + 1
            prediccion = sigmoide(np.matmul(np.transpose(theta_opt), X[imagen]))

            predicciones[etiqueta] = prediccion
        Y_pred.append(max(predicciones.items(), key=operator.itemgetter(1))[0])
    return np.sum((Y == np.array(Y_pred)))/np.shape(X)[0] * 100

def sigmoide(target):
    result = 1 / (1 + np.exp(-target))
    return result

def costeReg(thetas, x, y, lamb):
    sigXT = sigmoide(np.matmul(x, thetas))
    return (-1/np.shape(x)[0]) * (np.matmul(np.transpose(np.log(sigXT)), y) + np.matmul(np.transpose(np.log(1-sigXT)), (1-y))) + ((lamb/(2*np.shape(x)[0])) * sum(thetas ** 2))

def gradienteReg(thetas, x, y, lamb):
    sigXT = sigmoide(np.matmul(x, thetas))
    return ((1/np.shape(x)[0]) * np.matmul(np.transpose(x), (sigXT - y))) + ((lamb/np.shape(x)[0]) * thetas)

def apartado1():
    data = loadmat('ex3data1.mat')
    y = data['y'].ravel()
    X = data['X']

    sample = np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1, 20).T)
    plt.axis('off')
    plt.savefig('numeros.png')

    clasificadores = oneVsAll(X, y, 10, 0.1)
    print(prediccion(X, y , clasificadores))

def propagar(X, theta1, theta2):
    m = np.shape(X)[0]

    a1 = np.hstack([np.ones([m,1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoide(z2)])
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoide(z3)
    return a3

def prediccion_neuronal(X, a3):
    return [(np.argmax(a3[imagen]) + 1) for imagen in range(X.shape[0])]

def apartado2():
    weights = loadmat('ex3weights.mat')
    data = loadmat('ex3data1.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    X = data['X']
    y = data['y'].ravel()
    h = propagar(X, theta1, theta2)
    Y_pred = prediccion_neuronal(X, h)
    precision = np.sum((y == np.array(Y_pred))) / np.shape(X)[0]
    print("La precisión de la red neuronal es de " , precision * 100)

apartado2()
