import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import PolynomialFeatures
import scipy.optimize as opt
import operator
import checkNNGradients as cnn

def carga_csv(file_name):
    return read_csv(file_name)

def normalizar(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

def oneVsAll(X, y, num_etiquetas, reg):
    clasificadores = np.zeros(shape=(num_etiquetas, X.shape[1]))
	
    for i in range(num_etiquetas):
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
            prediccion = sigmoide(
                np.matmul(np.transpose(theta_opt), X[imagen]))

            predicciones[etiqueta] = prediccion
        Y_pred.append(max(predicciones.items(), key=operator.itemgetter(1))[0])
    return np.sum((Y == np.array(Y_pred)))/np.shape(X)[0] * 100

def sigmoide(target):
    result = 1 / (1 + np.exp(-target))
    return result

def costeReg(thetas, x, y, lamb):
	sigXT = sigmoide(np.dot(x, thetas))
	a1 = (-1/np.shape(x)[0])
	a2 = np.dot(np.log(sigXT+1e-7), y)
	a3 = np.dot(np.log(1-sigXT+1e-7), 1-y)
	a = a1 * (a2+a3)
	b = ((lamb/(2*np.shape(x)[0])) * np.sum(thetas ** 2))
	return a + b

def gradienteReg(thetas, x, y, lamb):
	sigXT = sigmoide(np.matmul(x, thetas))
	a = ((1/np.shape(x)[0]) * np.matmul(np.transpose(x), (sigXT - y))) + ((lamb/np.shape(x)[0]) * thetas)
	return a
	
def logistic_regression(X_train, y_train, X_test, y_test):
	#sample = np.random.choice(X.shape[0], 9)
	#plt.imshow(X[sample, :].reshape(-1, 20).T)
	#plt.axis('off')
	#plt.savefig('numeros.png')

	clasificadores = oneVsAll(X_train, y_train, 2, 0.1)
	# aux = clasificadores[0]
	# clasificadores[0] = clasificadores[1]
	# clasificadores[1] = aux
	clasificadores *= 1e5
	print(prediccion(X_test, y_test, clasificadores))
	print("hola")

def sigmoide(x):
	return 1 / (1 + np.exp(-x))

def sigmoide_prima(x):
    return sigmoide(x) / (1 - sigmoide(x))


def random_weights(min, max, epsilon=0.0001):
    return np.random.random((min, max)) * 2 * epsilon - epsilon


def propagar(X, theta1, theta2):
    m = np.shape(X)[0]

    a1 = np.hstack([np.ones([m, 1]), X])
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
    Theta1 = np.reshape(
        params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    Theta2 = np.reshape(
        params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas+1)))
    return coste_neuronal(X, y, Theta1, Theta2, reg), gradiente(X, y, Theta1, Theta2, reg)


def random_weights(L_in, L_out):
    epsilon = np.sqrt(6)/np.sqrt(L_in + L_out)
    return np.random.random((L_in, L_out)) * epsilon - epsilon/2


def predict_nn(X, h):
    return [(np.argmax(h[image])) for image in range(X.shape[0])]
	
def neural_network(X, y):
    # weights = loadmat('ex4weights.mat')
    # data = loadmat('ex4data1.mat')
    # theta1, theta2 = weights['Theta1'], weights['Theta2']
    # X = data['X']
    # y = data['y'].ravel()

    # sample = np.random.choice(X.shape[0], 100)
    # imgs = displayData(X[sample, :])
    # plt.savefig('numbers')

    m = len(y)
    input_size = X.shape[1]
    num_labels = 10
    num_ocultas = 25

    #y = y - 1
    y_onehot = np.zeros((m, num_labels))

    for i in range(m):
        y_onehot[i][int(y[i])] = 1
    # a1, a2, h = propagar(X, theta1, theta2)

    #params_rn = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
    theta1 = random_weights(num_ocultas, input_size + 1)
    theta2 = random_weights(num_labels, num_ocultas +1)
    params_rn = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
    reg_param = 1

    cost, grad = backprop(params_rn, input_size, num_ocultas,
                        num_labels, X, y_onehot, reg_param)
    a = cnn.checkNNGradients(backprop, 1)
    print(a)

    fmin = opt.minimize(fun=backprop, x0=params_rn, args=(input_size, num_ocultas, num_labels,
                        X, y_onehot, reg_param), method='TNC', jac=True, options={'maxiter': 70})

    theta1_opt = np.reshape(
        fmin.x[:num_ocultas * (input_size + 1)], (num_ocultas, (input_size + 1)))
    theta2_opt = np.reshape(
        fmin.x[num_ocultas * (input_size + 1):], (num_labels, (num_ocultas + 1)))

    a1, a2, h = propagar(X, theta1_opt, theta2_opt)

    print("El porcentaje de acierto del modelo es: {}%".format(
        np.sum((y == predict_nn(X, h)))/X.shape[0] * 100))

    lambdas = np.linspace(0, 1, 10)

    accuracy = []

    for lamb in lambdas:
        reg_param = lamb
        fmin = opt.minimize(fun=backprop, x0=params_rn, args=(input_size, num_ocultas, num_labels,
                            X, y_onehot, reg_param), method='TNC', jac=True, options={'maxiter': 70})

        theta1_opt = np.reshape(
            fmin.x[:num_ocultas * (input_size + 1)], (num_ocultas, (input_size + 1)))
        theta2_opt = np.reshape(
            fmin.x[num_ocultas * (input_size + 1):], (num_labels, (num_ocultas + 1)))

        a1, a2, h = propagar(X, theta1_opt, theta2_opt)
        accuracy.append((np.sum((y == predict_nn(X, h)))/X.shape[0] * 100))

    plt.plot(lambdas, accuracy)
    plt.savefig("accuracychart")

    lambdas = np.linspace(10, 70, 7)

    accuracy = []

    for lamb in lambdas:
        reg_param = lamb
        fmin = opt.minimize(fun=backprop, x0=params_rn, args=(input_size, num_ocultas, num_labels,
                            X, y_onehot, reg_param), method='TNC', jac=True, options={'maxiter': int(lamb)})

        theta1_opt = np.reshape(
            fmin.x[:num_ocultas * (input_size + 1)], (num_ocultas, (input_size + 1)))
        theta2_opt = np.reshape(
            fmin.x[num_ocultas * (input_size + 1):], (num_labels, (num_ocultas + 1)))

        a1, a2, h = propagar(X, theta1_opt, theta2_opt)
        accuracy.append((np.sum((y == predict_nn(X, h)))/X.shape[0] * 100))

    plt.plot(lambdas, accuracy)
    plt.savefig("accuracychart2")
    #print(a)
    pass

def svm():
    pass


def main():
    data = carga_csv('data.csv')

    columns = [1, 5, 11, 12, 29, 33, 56, 64, 74]
    #columns = [0, 3, 4, 5, 7, 9, 11]

    #X = data.to_numpy()[:, columns]
    Y = data["Bankrupt?"].values
    # Obtiene un vector con los índices de los ejemplos positivos
    pos = data[Y == 1]
    neg = data[Y == 0]
    # Cogemos solo 800 ejemplos negativos ya que hay demasiados
    neg = neg[:220]

    X = np.concatenate((pos.to_numpy()[:, 1:], neg.to_numpy()[:, 1:]))
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

    #X = normalizar(X)[0]

    logistic_regression(X, Y)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression

df = pd.read_csv('data.csv')

#Take sample to balance the data
bankrupt_sample = df[df['Bankrupt?'] == 0][0:220]
non_bankrupt_sample = df[df['Bankrupt?'] == 1]
#create new data frame
new_df = pd.concat([bankrupt_sample,non_bankrupt_sample],axis = 0)
new_df.head()

X = new_df.drop('Bankrupt?', axis=1)
y = new_df['Bankrupt?']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=777
)

# lr_model = LogisticRegression()
# lr_model.fit(X_train,y_train)
# accuracy_lr = lr_model.score(X_test,y_test)
# print("Logistic Regression accuracy is :",accuracy_lr)
# lr_pred= lr_model.predict(X_test)
# report = classification_report(y_test,lr_pred)
# print("-"*100)
# print(report)
# cm = confusion_matrix(y_test,lr_model.predict(X_test))
# sns.heatmap(cm,annot=True,fmt="d",cmap = "Blues")

X_train = normalizar(X_train.to_numpy())[0]
X_test = normalizar(X_test.to_numpy())[0]
logistic_regression(X_train, y_train.to_numpy(), X_test, y_test.to_numpy())

#logistic_regression(X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy())