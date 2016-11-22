#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
from builtins import super

from random import random
import numpy as np
import matplotlib.pyplot as plt

def newton(f, x0, fprime, maxiter=50):
    f0 = f(x0)
    for iter in range(maxiter):
        f1 = fprime(x0)
        if f1==0:
            return None, maxiter
        x0 = x0 - f0 / f1
        f0 = f(x0)
        if abs(f0) < 1e-3:
            return x0, iter
    return x0, iter


X = 4*2**7
Y = 3*X//4

x_min = -5
x_max = -x_min
y_min = x_min
y_max = x_max

maxiter = 100
ndigits = 1

#f = lambda x: np.power(x,3) - 1; f1 = lambda x: 3*np.power(x,2)
f = lambda x: np.power(x,5) + np.power(x,3)*7 - np.power(x,2)*3 -1
f1 = lambda x: 5*np.power(x,4) + 21*np.power(x,2) - 6*x

pic = np.zeros((Y, X), dtype=np.int)
iters = np.zeros((Y, X))
zeros = ['', None,]

for xi in range(X):
    for yi in range(Y):
        x = x_min + (x_max-x_min)/X*xi
        y = y_min + (y_max-y_min)/Y*yi
        z, iter = newton(f, x + 1.j*y, fprime=f1, maxiter=maxiter)
        if iter < maxiter:
            z = round(z.real, ndigits) + 1j * round(z.imag, ndigits)
        else:
            z = None
        try:
            i = zeros.index(z)
        except ValueError:
            zeros.append(z)
            i = len(zeros)
        pic[yi,xi] = 0 if z is None else -i if abs(x+1.j*y - z) > .1**ndigits else i
        iters[yi,xi] = iter

plt.imshow(pic, interpolation='None', cmap='Set1')
plt.contour(np.log(iters))
plt.figure()
plt.imshow(np.log(iters), interpolation='None', cmap='gray')
plt.show()
