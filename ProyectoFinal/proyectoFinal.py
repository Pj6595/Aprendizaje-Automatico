import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import scipy.optimize as opt
import operator
import checkNNGradients as cnn
from sklearn.model_selection import train_test_split
import pandas as pd
import scipy.special as special
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.svm import SVC
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
		clasificadores[i] = opt.fmin_tnc(func=costeReg, x0=thetas, fprime=gradienteReg, args=(X, filtrados, reg), messages=0)[0]
		
	return clasificadores

def prediccion(X, Y, clasificadores):
	predicciones = {}
	Y_pred = []
	for imagen in range(np.shape(X)[0]):
		for i in range(clasificadores.shape[0]):
			theta_opt = clasificadores[i]
			prediccion = sigmoide(
				np.matmul(np.transpose(theta_opt), X[imagen]))

			predicciones[i] = prediccion
		Y_pred.append(max(predicciones.items(), key=operator.itemgetter(1))[0])
	return Y_pred, np.sum((Y == np.array(Y_pred)))/np.shape(X)[0] * 100

# def sigmoide(target):
# 	result = 1 / (1 + np.exp(-target))
# 	return result

def costeReg(thetas, x, y, lamb):
	sigXT = sigmoide(np.dot(x, thetas))
	a1 = (-1/np.shape(x)[0])
	a2 = np.dot(np.log(sigXT), y)
	a3 = np.dot(np.log(1-sigXT+1e-6), 1-y)
	a = a1 * (a2+a3)
	b = ((lamb/(2*np.shape(x)[0])) * np.sum(thetas ** 2))
	return a + b

def gradienteReg(thetas, x, y, lamb):
	sigXT = sigmoide(np.matmul(x, thetas))
	a = ((1/np.shape(x)[0]) * np.matmul(np.transpose(x), (sigXT - y))) + ((lamb/np.shape(x)[0]) * thetas)
	return a
	
def logistic_regression(X_train, y_train, X_test, y_test, fileName):

	lambdas = np.linspace(1, 3, 20)

	accuracy = []
	y_preds = []

	for i in lambdas:
		clasificadores = oneVsAll(X_train, y_train, 2, i)
		y_pred, acc = prediccion(X_test, y_test, clasificadores)
		accuracy.append(acc)
		y_preds.append(y_pred)
		print(acc)

	y_pred = y_preds[np.argmax(accuracy)]
	cm = confusion_matrix(y_test, y_pred)
	plt.figure()
	fig = sns.heatmap(cm, annot=True,fmt="",cmap='Blues').get_figure()
	fig.savefig(fileName + "Logistic", dpi=400)

	plt.figure()
	plt.plot(lambdas, accuracy)
	plt.savefig(fileName + "LambdaAccuracyLogistic")

def sigmoide(x):
	return special.expit(x)

def sigmoide_prima(x):
	return sigmoide(x) / (1 - sigmoide(x))

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
		J += np.sum(-y[i]*np.log(h[i]) - (1 - y[i])*np.log(1-h[i]+1e-9))
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
	
def neural_network(X_train, y_train, X_test, y_test, fileName):

	X = X_train
	y = y_train
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

	y_pred = predict_nn(X_test, h)

	print("El porcentaje de acierto del modelo es: {}%".format(
		np.sum((y_test == y_pred))/X_test.shape[0] * 100))

	cm = confusion_matrix(y_test, y_pred)
	plt.figure()
	fig = sns.heatmap(cm, annot=True,fmt="",cmap='Blues').get_figure()
	fig.savefig(fileName + "Neural", dpi=400)

	lambdas = np.linspace(0, 1, 10)

	accuracy = []
	y_preds = []

	for lamb in lambdas:
		print("Voy por ", lamb, " de ", len(lambdas))
		reg_param = lamb
		fmin = opt.minimize(fun=backprop, x0=params_rn, args=(input_size, num_ocultas, num_labels,
							X, y_onehot, reg_param), method='TNC', jac=True, options={'maxiter': 70})

		theta1_opt = np.reshape(
			fmin.x[:num_ocultas * (input_size + 1)], (num_ocultas, (input_size + 1)))
		theta2_opt = np.reshape(
			fmin.x[num_ocultas * (input_size + 1):], (num_labels, (num_ocultas + 1)))

		a1, a2, h = propagar(X, theta1_opt, theta2_opt)
		y_pred = predict_nn(X, h)
		y_preds.append(y_pred)
		accuracy.append((np.sum((y == y_pred))/X.shape[0] * 100))

	plt.plot(lambdas, accuracy)
	plt.savefig(fileName + "LambdaAccuracyNeural")

	y_pred = y_preds[np.argmax(accuracy)]
	cm = confusion_matrix(y, y_pred)
	plt.figure()
	fig = sns.heatmap(cm, annot=True,fmt="",cmap='Blues').get_figure()
	fig.savefig(fileName + "NeuralBestLambda", dpi=400)

	iters = np.linspace(10, 70, 7)

	accuracy = []
	y_preds = []

	reg_param = 1

	for iter in iters:
		fmin = opt.minimize(fun=backprop, x0=params_rn, args=(input_size, num_ocultas, num_labels,
							X, y_onehot, reg_param), method='TNC', jac=True, options={'maxiter': int(iter)})

		theta1_opt = np.reshape(
			fmin.x[:num_ocultas * (input_size + 1)], (num_ocultas, (input_size + 1)))
		theta2_opt = np.reshape(
			fmin.x[num_ocultas * (input_size + 1):], (num_labels, (num_ocultas + 1)))

		a1, a2, h = propagar(X, theta1_opt, theta2_opt)
		y_pred = predict_nn(X, h)
		y_preds.append(y_pred)
		accuracy.append((np.sum((y == y_pred))/X.shape[0] * 100))

	y_pred = y_preds[np.argmax(accuracy)]
	cm = confusion_matrix(y, y_pred)
	plt.figure()
	fig = sns.heatmap(cm, annot=True,fmt="",cmap='Blues').get_figure()
	fig.savefig(fileName + "NeuralBestIter", dpi=400)

	plt.figure()
	plt.plot(iters, accuracy)
	plt.savefig(fileName + "IterationAccuracyNeural")

def svm(X_train, y_train, X_test, y_test, fileName):
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 
	
	values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
	n = len(values)
	scores = np.zeros((n, n))

	startTime = time.process_time()

	for i in range(n):
		C = values[i]
		print("Voy por el i ", i, "de " , n) 
		for j in range(n):
			print("Voy por el j ", j, "de " , n) 
			sigma = values[j]
			svm = SVC(kernel='rbf', C = C, gamma= 1 / (2 * sigma **2))
			svm.fit(X_train, y_train)
			scores[i, j] = svm.score(X_val, y_val)

	print("Error mínimo: {}".format(1 - scores.max())) 
	C_opt = values[scores.argmax()//n]
	sigma_opt = values[scores.argmax()%n]
	print("C óptimo: {}, sigma óptimo: {}".format(C_opt, sigma_opt))

	svm = SVC(kernel='rbf', C= C_opt, gamma=1 / (2 * sigma_opt)**2)
	svm.fit(X_train, y_train)
	y_pred = svm.predict(X_test)
	endTime = time.process_time()

	score = svm.score(X_test, y_test)
	totalTime = endTime - startTime

	print('Precisión: {:.3f}%'.format(score*100))
	print('Tiempo de ejecución: {}'.format(totalTime))
	print('Matriz de confusión: ')
	cm = confusion_matrix(y_test, y_pred)
	fig = sns.heatmap(cm, annot=True,fmt="",cmap='Blues').get_figure()
	fig.savefig(fileName + "SVM", dpi=400)

##  separate between fractional columns and non-fraction columns
def get_fraction_valued_columns(df):
    my_columns  = []
    for col in df.columns:
        if (df[col].max()<=1) & (df[col].min() >= 0):
            my_columns.append(col)
    return(my_columns)

class data:
	X = None
	y = None
	X_train = None
	y_train = None
	X_test = None
	y_test = None
	name = ""
	def __init__(self, x, y, x_train, y_train, x_test, y_test, name):
		self.X = x
		self.y = y
		self.X_train = x_train
		self.y_train = y_train
		self.X_test = x_test
		self.y_test = y_test
		self.name = name
	def to_numpy(self):
		self.X = self.X.to_numpy()
		self.y = self.y.to_numpy()
		self.X_train = self.X_train.to_numpy()
		self.y_train = self.y_train.to_numpy()
		self.X_test = self.X_test.to_numpy()
		self.y_test = self.y_test.to_numpy()

df = pd.read_csv('data.csv')

#Take sample to balance the data
cname = 'Bankrupt?'
columns = np.arange(df.shape[1]-1)
#columns = [1, 5, 11, 12, 29, 33, 56, 64, 74]

non_bankrupt_sample = df[df[cname] == 0]
# size = non_bankrupt_sample.shape[0] * 0.5
# non_bankrupt_sample_smote = non_bankrupt_sample[:int(size)]
non_bankrupt_sample_cut = non_bankrupt_sample[:220]

bankrupt_sample = df[df[cname] == 1]
#create new data frame
# smote_df = pd.concat([bankrupt_sample,non_bankrupt_sample_smote],axis = 0)
# smote_df.head()
cut_df = pd.concat([bankrupt_sample,non_bankrupt_sample_cut],axis = 0)
cut_df.head()

X_cut = cut_df.drop(cname, axis=1)
y_cut = cut_df[cname]

X_orig = df.drop(cname, axis=1)
y_orig = df[cname]

fractional_columns = get_fraction_valued_columns(df=df.drop([cname],axis=1))
non_fractional_columns = df.drop([cname],axis=1).columns.difference(fractional_columns)
norm = preprocessing.normalize(df[non_fractional_columns])
normalized_df = pd.DataFrame(norm, columns=df[non_fractional_columns].columns)

scaled_data = pd.concat([df.drop(non_fractional_columns,axis=1),normalized_df],axis = 1)

X = scaled_data.drop(cname, axis = 1)
y = scaled_data[cname]

sm = SMOTE(random_state=123)
X_sm , y_sm = sm.fit_resample(X,y)

print(f'''Shape of X before SMOTE:{X.shape}
Shape of X after SMOTE:{X_sm.shape}''',"\n\n")

print(f'''Target Class distributuion before SMOTE:\n{y.value_counts(normalize=True)}
Target Class distributuion after SMOTE :\n{y_sm.value_counts(normalize=True)}''')

# pos = smote_df[y_sm == 1]
# neg = smote_df[y_sm == 0][:220]

# features = smote_df.columns.to_numpy()[columns]

# Dibuja los ejemplos positivos
# fig = plt.figure()
# plt.suptitle("Features distribution")

# for i, f in enumerate(features):
# 	i += 1
# 	ax = fig.add_subplot(10, 10, i)

# 	# Plot corresponding histogram
# 	ax.hist(neg[f], label="Not Bankrupt", stacked=True, alpha=0.5, color="g")
# 	ax.hist(pos[f], label="Bankrupt", stacked=True, alpha=0.5, color="r")
# 	#ax.set_title(f)

# plt.tight_layout()
# plt.legend()
# plt.savefig('features.png')
# plt.show()

X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(
	X_sm, y_sm, test_size=0.2, random_state=777
)

X_train_cut, X_test_cut, y_train_cut, y_test_cut = train_test_split(
	X_cut, y_cut, test_size=0.2, random_state=777
)

X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
	X_orig, y_orig, test_size=0.2, random_state=777
)

size = bankrupt_sample.shape[0] * 0.8
bs = bankrupt_sample[int(size):]
nbs = non_bankrupt_sample[(non_bankrupt_sample.shape[0] - bs.shape[0]):]

ex_df = pd.concat([bs, nbs],axis = 0)
ex_df.head()

X_test_orig = ex_df.drop(cname, axis = 1)
y_test_orig = ex_df[cname]

datasets = [	
	data(X_orig, y_orig, X_train_orig, y_train_orig, X_test_orig, y_test_orig, 'Original'),
	data(X_cut, y_cut, X_train_cut, y_train_cut, X_test_cut, y_test_cut, 'Cut'),
	data(X_sm, y_sm, X_train_sm, y_train_sm, X_test_sm, y_test_sm, 'Smote')
]

for d in datasets:
	d.to_numpy()
	logistic_regression(d.X_train, d.y_train, d.X_test, d.y_test, d.name)
	neural_network(d.X_train, d.y_train, d.X_test, d.y_test, d.name)
	svm(d.X_train, d.y_train, d.X_test, d.y_test, d.name)