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


X = 4*2**6
Y = 3*X//4

x_min = -2
x_max = 2
y_min = -2
y_max = 2

maxiter = 100
ndigits = 1

f = lambda x: np.power(x,3) - 1; f1 = lambda x: 3*np.power(x,2)

pic = np.zeros((Y, X), dtype=np.int)
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
        pic[yi,xi] = 0 if z is None else i if abs(x+1.j*y - z) > .1**ndigits else -i

plt.imshow(pic, interpolation='None', cmap='Set1')
plt.show()
