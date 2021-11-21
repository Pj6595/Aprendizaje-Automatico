import numpy as np
from numpy.lib.function_base import gradient
import checkNNGradients as cnn
from displayData import displayData
import operator
import matplotlib.pyplot as plt
from numpy.lib.npyio import load
import scipy.optimize as opt
from scipy.io import loadmat

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def sigmoide_prima(x):
    return sigmoide(x) / (1 - sigmoide(x))

def random_weights(min, max, epsilon = 0.0001):
    return np.random.random((min, max)) * 2 * epsilon - epsilon

def propagar(X, theta1, theta2):
    m = np.shape(X)[0]

    a1 = np.hstack([np.ones([m,1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoide(z2)])
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoide(z3)
    return a1, a2, a3

def coste_neuronal(X, y, theta1, theta2, reg):
    a1, a2, h = propagar(X, theta1, theta2)
    m = X.shape[0]

    J = 0
    for i in range(m):
        J += np.sum(-y[i]*np.log(h[i]) - (1 - y[i])*np.log(1-h[i]))
    J = J/m

    sum_theta1 = np.sum(np.square(theta1[:, 1:]))
    sum_theta2 = np.sum(np.square(theta2[:, 1:]))
    
    reg_term = (sum_theta1 + sum_theta2) * reg / (2*m)

    return J + reg_term

def gradiente(X, y, Theta1, Theta2, reg):
    m = X.shape[0]

    delta1 = np.zeros(Theta1.shape)
    delta2 = np.zeros(Theta2.shape)

    a1, a2, h = propagar(X, Theta1, Theta2)

    for t in range(m):
        a1t = a1[t, :]
        a2t = a2[t, :]
        ht = h[t, :]
        yt = y[t]
        d3t = ht - yt
        d2t = np.dot(Theta2.T, d3t) * (a2t * (1 - a2t))

        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta1[:, 1:] += Theta1[:, 1:] * reg/m
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])
        delta2[:, 1:] += Theta2[:, 1:] * reg/m

    return np.concatenate((np.ravel(delta1/m), np.ravel(delta2/m)))


def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    Theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    Theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas+1)))
    return coste_neuronal(X, y, Theta1, Theta2, reg), gradiente(X, y, Theta1, Theta2, reg)

def random_weights(L_in, L_out):
    epsilon = np.sqrt(6)/np.sqrt(L_in + L_out)
    return np.random.random((L_in, L_out)) * epsilon - epsilon/2

def predict_nn(X, h):
    return [(np.argmax(h[image])) for image in range(X.shape[0])]

weights = loadmat('ex4weights.mat')
data = loadmat('ex4data1.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']
X = data['X']
y = data['y'].ravel()

sample = np.random.choice(X.shape[0], 100)
imgs = displayData(X[sample, :])
plt.savefig('numbers')

m = len(y)
input_size = X.shape[1]
num_labels = 10
num_ocultas = 25

y = y -1
y_onehot = np.zeros((m, num_labels))

for i in range(m):
    y_onehot[i][y[i]] = 1
a1, a2, h = propagar(X, theta1, theta2)

#params_rn = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
theta1 = random_weights(theta1.shape[0], theta1.shape[1])
theta2 = random_weights(theta2.shape[0], theta2.shape[1])
params_rn = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
reg_param = 1

cost, grad = backprop(params_rn, input_size, num_ocultas, num_labels, X, y_onehot, reg_param)
a = cnn.checkNNGradients(backprop, 1)
print(a)

fmin = opt.minimize(fun=backprop, x0=params_rn, args=(input_size, num_ocultas, num_labels, X, y_onehot, reg_param), method='TNC', jac=True, options={'maxiter': 70})

theta1_opt = np.reshape(fmin.x[:num_ocultas * (input_size + 1)], (num_ocultas, (input_size + 1)))
theta2_opt = np.reshape(fmin.x[num_ocultas * (input_size + 1):], (num_labels, (num_ocultas + 1)))

a1, a2, h = propagar(X, theta1_opt, theta2_opt)

print("El porcentaje de acierto del modelo es: {}%".format(np.sum((y == predict_nn(X, h)))/X.shape[0] * 100))

lambdas = np.linspace(0, 1, 10)

accuracy = []

for lamb in lambdas:
    reg_param = lamb
    fmin = opt.minimize(fun=backprop, x0=params_rn, args=(input_size, num_ocultas, num_labels, X, y_onehot, reg_param), method='TNC', jac=True, options={'maxiter': 70})

    theta1_opt = np.reshape(fmin.x[:num_ocultas * (input_size + 1)], (num_ocultas, (input_size + 1)))
    theta2_opt = np.reshape(fmin.x[num_ocultas * (input_size + 1):], (num_labels, (num_ocultas + 1)))

    a1, a2, h = propagar(X, theta1_opt, theta2_opt)
    accuracy.append((np.sum((y == predict_nn(X, h)))/X.shape[0] * 100))

plt.plot(lambdas, accuracy)
plt.savefig("accuracychart")

lambdas = np.linspace(10, 70, 7)

accuracy = []

for lamb in lambdas:
    reg_param = lamb
    fmin = opt.minimize(fun=backprop, x0=params_rn, args=(input_size, num_ocultas, num_labels, X, y_onehot, reg_param), method='TNC', jac=True, options={'maxiter': int(lamb)})

    theta1_opt = np.reshape(fmin.x[:num_ocultas * (input_size + 1)], (num_ocultas, (input_size + 1)))
    theta2_opt = np.reshape(fmin.x[num_ocultas * (input_size + 1):], (num_labels, (num_ocultas + 1)))

    a1, a2, h = propagar(X, theta1_opt, theta2_opt)
    accuracy.append((np.sum((y == predict_nn(X, h)))/X.shape[0] * 100))

plt.plot(lambdas, accuracy)
plt.savefig("accuracychart2")
#print(a)