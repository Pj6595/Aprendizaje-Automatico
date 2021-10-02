import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def carga_csv(file_name):
    return read_csv(file_name, header = None).to_numpy().astype(float)

def make_data(t0_range, t1_range, X, Y):
    step = 0.1
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)
    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)
    Coste = np.empty_like(Theta0)
    for ix, iy in np.ndindex(Theta0.shape):
        Coste[ix, iy] = coste(X, Y, [Theta0[ix, iy], Theta1[ix, iy]])
    return [Theta0, Theta1, Coste]

def coste(X, Y, Theta):
    m = len(X)
    sumatorio = 0
    for i in range(m):
        sumatorio += ((Theta[0] + Theta[1] * X[i]) - Y[i]) ** 2
    return sumatorio / (2 * len(X))

def dibuja_coste(Theta0, Theta1, Coste):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(Theta0, Theta1, Coste, cmap=cm.rainbow, linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    fig2 = plt.figure()
    plt.contour(Theta0, Theta1, Coste, np.logspace(-2,3,20))
    plt.show()

def descenso_gradiente(X,Y):
    m = len(X)
    alpha = 0.01
    theta_0 = theta_1 = 0
    for _ in range(1500):
        sum_0 = sum_1 = 0
        for i in range(m):
            sum_0 += (theta_0 + theta_1 * X[i]) - Y[i]
            sum_1 += ((theta_0 + theta_1 * X[i]) - Y[i]) * X[i]
        theta_0 = theta_0 - (alpha/m) * sum_0
        theta_1 = theta_1 - (alpha/m) * sum_1
    min_x = min(X)
    max_x = max(X)
    min_y = theta_0 + theta_1 * min_x
    max_y = theta_0 + theta_1 * max_x

    Coste = coste(X, Y, (theta_0, theta_1))

    #Dibujamos el resultado
    plt.plot(X, Y, "x")
    plt.plot([min_x, max_x], [min_y, max_y])
    plt.savefig("descenso_gradiente.pdf")

    return (theta_0, theta_1), Coste 

def apartado_1():
    data = carga_csv('ex1data1.csv')
    X = data[:,0]
    Y = data[:,1]
    descenso_gradiente(X, Y)

    A, B, C = make_data([-10,10], [-1,4], X, Y)
    dibuja_coste(A, B, C)

def normalizar(X):
    mu = np.average(X)
    sigma = np.std(X)
    X_norm = X

    for i in range(len(X)):
        X_norm[i] = (X[i] - mu) / sigma

    return X_norm, mu, sigma
    

def apa():
    data = carga_csv("ex1data2.csv")
    X = data[:,0]
    Y = data[:,1]
    Z = data[:,2]
    X_norm, mux, sigmax = normalizar(X)
    Y_norm, muy, sigmay = normalizar(Y)
    Z_norm, muz, sigmaz = normalizar(Z)
    print(X_norm)
    print(mux)
    print(sigmax)

def descenso_gradiente_vec(X, Y, m, Theta):
    J = (np.transpose((X * Theta) - Y) * ((X * Theta) - Y)) / 2*m
    return J

def apartado_2_1():
    data = carga_csv("ex1data2.csv")
    X = data[:, :-1]
    Y = data[:, -1]
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    X = np.hstack([np.ones([m, 1]), X])

def ecuacion_normal(X,Y):
    Theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), Y)
    return Theta

def apartado_2_2():
    data = carga_csv('ex1data2.csv')
    X = data[:, :-1]
    Y = data[:, -1]

    m = np.shape(X)[0]
    X = np.hstack([np.ones([m, 1]), X])
    theta_normal = ecuacion_normal(X,Y)

    print('Theta de ecuación normal: ', theta_normal, '\n')

#main()
apartado_2_2()
