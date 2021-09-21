import time
import numpy as np
import random
import matplotlib.pyplot as plt

#version iterativa
def integra_mc_iter(fun, a, b, num_puntos= 1000):
    tic = time.process_time()
    max = cont = 0
    for i in range(1000):
        x = a + i*(b - a)/1000
        if fun(x) > max:
            max = fun(x)
    for i in range(num_puntos):
        x = random.random() * b - a
        y = random.random() * max
        if fun(x) > y: 
            cont += 1
    res = cont/num_puntos * (b - a) * max
    toc = time.process_time()
    return 1000 * (toc - tic)
    
#version con arrays
def integra_mc_vec(fun, a, b, num_puntos = 1000):
    tic = time.process_time()
    arr = np.arange(num_puntos)
    arr = arr *(b-a)/num_puntos + a
    far = fun(arr)
    max = np.max(far)
    ran = np.random.random(num_puntos) * max
    cont = np.sum(ran < far)
    res = cont/num_puntos * (b - a) * max
    toc = time.process_time()
    return 1000 * (toc - tic)

def fun(x):
    return -2*x + 5
    #return -x*x + 4

def compara_tiempos():
    sizes = np.linspace(100,1000000, 20,dtype='int')
    times_iter = []
    times_vec = []
    for size in sizes:
        x1 = 0
        x2 = 1
        times_iter += [integra_mc_iter(fun, x1, x2, size)]
        times_vec += [integra_mc_vec(fun, x1, x2, size)]

    plt.figure()
    plt.scatter(sizes, times_iter, c='red', label='bucle')
    plt.scatter(sizes, times_vec, c='blue', label='vector')
    plt.legend()
    plt.savefig('time.png')

compara_tiempos()