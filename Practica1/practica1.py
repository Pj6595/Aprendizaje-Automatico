import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def carga_csv(file_name):
    return read_csv(file_name, header=None).to_numpy().astype(float)


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
    plt.contour(Theta0, Theta1, Coste, np.logspace(-2, 3, 20))
    plt.show()


def descenso_gradiente(X, Y):
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

    # Dibujamos el resultado
    plt.plot(X, Y, "x")
    plt.plot([min_x, max_x], [min_y, max_y])
    plt.savefig("descenso_gradiente.png")

    return (theta_0, theta_1), Coste


def apartado_1():
    data = carga_csv('ex1data1.csv')
    X = data[:, 0]
    Y = data[:, 1]
    descenso_gradiente(X, Y)

    A, B, C = make_data([-10, 10], [-1, 4], X, Y)
    dibuja_coste(A, B, C)


def normalizar(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


def coste_vectorizado(X, Y, Theta):
    H = np.dot(X, Theta)
    Aux = (H - Y) ** 2
    return Aux.sum() / (2*len(X))


def ecuacion_normal(X, Y):
    Theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), Y)
    return Theta


def descenso_gradiente_vec(X, Y, alpha):
    Theta = np.zeros(np.shape(X)[1])
    iteraciones = 1500
    costes = np.zeros(iteraciones)
    for i in range(iteraciones):
        costes[i] = coste_vectorizado(X, Y, Theta)
        Theta = gradiente_vec(X, Y, Theta, alpha)

    return Theta, costes


def gradiente_vec(X, Y, Theta, alpha):
    NuevaTheta = Theta
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    H = np.dot(X, Theta)
    Aux = (H - Y)
    for i in range(n):
        Aux_i = Aux * X[:, i]
        NuevaTheta[i] -= (alpha / m) * Aux_i.sum()
    return NuevaTheta


def apartado_2_1():
    data = carga_csv("ex1data2.csv")
    X = data[:, :-1]
    Y = data[:, -1]
    m = np.shape(X)[0]

    X, mu, sigma = normalizar(X)
    X = np.hstack([np.ones([m, 1]), X])

    alphas = [0.3, 0.1, 0.03, 0.01]
    colors = ['indigo', 'darkviolet', 'mediumorchid', 'plum']

    plt.figure()

    for i in range(len(alphas)):
        Theta, costes = descenso_gradiente_vec(X, Y, alphas[i])
        plt.scatter(np.arange(np.shape(costes)[0]), costes, c=colors[i], label='alpha = ' + str(alphas[i]))
        
    plt.legend()
    plt.savefig("pjbobo.png")

    print('Theta de gradiente vectorizado: ', Theta, '\n')


def apartado_2_2():
    data = carga_csv('ex1data2.csv')
    X = data[:, :-1]
    Y = data[:, -1]
    m = np.shape(X)[0]

    X_norm, mu, sigma = normalizar(X)
    X_norm = np.hstack([np.ones([m, 1]), X_norm])

    theta_vec, costecitos = descenso_gradiente_vec(X_norm, Y, 0.01)

    X = np.hstack([np.ones([m, 1]), X])
    theta_normal = ecuacion_normal(X, Y)

    pred_normal = theta_normal[0] + theta_normal[1] * 1650 + theta_normal[2] * 3
    pred_gradient = theta_vec[0] + theta_vec[1] * ((1650 - mu[0]) / sigma[0]) + theta_vec[2] * ((3 - (mu[1]) / sigma[1]))

    print('Theta de ecuación normal: ', pred_normal, '\n')
    print('Theta de gradiente vectorizado: ', pred_gradient, '\n')

apartado_2_2()
