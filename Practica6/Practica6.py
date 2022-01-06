import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
from process_email import email2TokenList
import codecs
from get_vocab_dict import getVocabDict
import glob
import sklearn.model_selection as ms
from sklearn.metrics import confusion_matrix
import time

def visualize_boundary(X, y, svm, file_name):
    svm.fit(X, y.ravel())
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
    plt.contour(x1, x2, yp)
    plt.savefig(file_name)
    plt.show()
    plt.close()

def apartado1_1():
    data = loadmat('ex6data1.mat')
    X, y = data['X'], data['y'].ravel() 
    svm = SVC(kernel= 'linear', C = 1)
    visualize_boundary(X, y, svm, "apartado1_1_C1")
    svm = SVC(kernel= 'linear', C = 100)
    visualize_boundary(X, y, svm, "apartado1_1_C100")

def apartado1_2():
    data = loadmat('ex6data2.mat')
    X, y = data['X'], data['y'].ravel() 
    C = 1
    sigma = 0.1 
    svm = SVC(kernel= 'rbf', C = C, gamma=1 / (2 * sigma**2))
    visualize_boundary(X, y, svm, "apartado1_2")

def apartado1_3():
    data = loadmat('ex6data3.mat')
    X = data['X']
    y = data['y'].ravel()
    Xval = data['Xval']
    yval = data['yval'].ravel()
    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    n = len(values)
    scores = np.zeros((n, n))

    for i in range(n):
        C = values[i]
        for j in range(n):
            sigma = values[j]
            svm = SVC(kernel='rbf', C = C, gamma= 1 / (2 * sigma **2))
            svm.fit(X, y.ravel())
            scores[i, j] = svm.score(Xval, yval)

    print("Error mínimo: {}".format(1 - scores.max())) 
    C_opt = values[scores.argmax()//n]
    sigma_opt = values[scores.argmax()%n]
    print("C óptimo: {}, sigma óptimo: {}".format(C_opt, sigma_opt))

    svm = SVC(kernel= 'rbf', C = C_opt, gamma= 1 / (2 * sigma_opt ** 2))
    visualize_boundary(X, y, svm, "apartado1_3")

def palabras_comunes(X, fracción):
    num_palabras = int(np.round(len(X[0])*fracción))
    frecuencia = np.sort(X.sum(0))[-num_palabras]
    atributos = np.where(X.sum(0) >= frecuencia)[0]
    return X[:, atributos]

def read_directory(name, vocab):
    files = glob.glob(name)
    X = np.zeros((len(files), len(vocab)))
    y = np.ones(len(files))
    i = 0
    for f in files:
        email_contents = codecs.open(f, 'r', encoding='utf-8', errors='ignore').read()
        tokens = email2TokenList(email_contents)
        words = filter(None,[vocab.get(x) for x in tokens])
        for w in words:
            X[i, w-1] = 1
        i +=1
    return X, y

def score_emails(X, y):
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = ms.train_test_split(X_train, y_train, test_size=0.25, random_state=1) 

    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
    n = len(values)
    scores = np.zeros((n, n))

    startTime = time.process_time()

    for i in range(n):
        C = values[i]
        for j in range(n):
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
    print(confusion_matrix(y_test, y_pred))

def apartado2():
    vocab = getVocabDict()
    files = glob.glob('spam/*.txt')
    Xspam, yspam = read_directory('spam/*.txt', vocab)

    Xeasy_ham, yeasy_ham = read_directory('easy_ham/*.txt', vocab)

    Xhard_ham, yhard_ham = read_directory('hard_ham/*.txt', vocab)

    X = np.vstack((Xspam, Xeasy_ham, Xhard_ham))

    yspam = [1]*len(Xspam)
    yeasy_ham = [0]*len(Xeasy_ham)
    yhard_ham = [0]*len(Xhard_ham)
    y = np.r_[yspam, yeasy_ham, yhard_ham]

    # con reducción de atributos
    X_freq = palabras_comunes(X, 0.1)

    score_emails(X_freq, y)

    # sin reducción de atributos
    
    score_emails(X, y)

#apartado1_1()
#apartado1_2()
#apartado1_3()
apartado2()