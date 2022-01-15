import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from practica2 import sigmoide, costeReg, gradienteReg, efectosReg, graficaLimite
from sklearn.preprocessing import PolynomialFeatures
import scipy.optimize as opt

def carga_csv(file_name):
    return read_csv(file_name)
    
def logistic_regression(X, y, pos, neg):

    # Apartado 2.1
    poly = PolynomialFeatures(6)
    Xp = poly.fit_transform(X)

    # Apartado 2.2
    thetas = np.zeros(np.shape(Xp)[1])
    costeR = costeReg(thetas, Xp, Y, 1)
    gradR = gradienteReg(thetas, Xp, Y, 1)
    print(costeR)

    # Apartado 2.3
    result = opt.fmin_tnc(func=costeReg, x0=thetas,
                          fprime=gradienteReg, args=(Xp, Y, 1))
    theta_opt = result[0]
    print(theta_opt)
    graficaLimite(Xp, Y, theta_opt, poly)

    # Apartado 2.4
    efectosReg(Xp, Y, thetas)
    
def neural_network():
    pass

def svm():
    pass

data = carga_csv('data.csv')

columns = [1, 5, 11, 12, 29, 33, 56, 64, 74]

#X = data.to_numpy()[:, columns]
Y = data["Bankrupt?"].values
# Obtiene un vector con los índices de los ejemplos positivos
pos = data[Y == 1]
neg = data[Y == 0]
# Cogemos solo 800 ejemplos negativos ya que hay demasiados
neg = neg[:800]

X = np.concatenate((pos.to_numpy()[:, columns], neg.to_numpy()[:, columns]))
Y = np.concatenate((pos.to_numpy()[:, 0], neg.to_numpy()[:, 0]))

features = data.columns.to_numpy()[columns]

# Dibuja los ejemplos positivos
fig = plt.figure()
plt.suptitle("Features distribution")

for i, f in enumerate(features):
    i += 1
    ax = fig.add_subplot(3, 3, i)

    # Plot corresponding histogram
    ax.hist(neg[f], label="Not Bankrupt", stacked=True, alpha=0.5, color="g")
    ax.hist(pos[f], label="Bankrupt", stacked=True, alpha=0.5, color="r")
    ax.set_title(f)

plt.tight_layout()
plt.legend()
plt.savefig('features.png')
#plt.show()

logistic_regression(X, Y, pos, neg)