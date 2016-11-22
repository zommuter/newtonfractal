#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
from builtins import super

from random import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

X = 4*2**6
Y = 3*X//4

x_min = -5
x_max = 5
y_min = -2
y_max = 2

maxiter = 100
ndigits = 3

f = lambda x: np.power(x,3) - 1
f1 = lambda x: 3*np.power(x,2)

pic = np.zeros((Y, X), dtype=np.int)
zeros = [None,]

for xi in range(X):
    for yi in range(Y):
        x = x_min + (x_max-x_min)/X*xi
        y = y_min + (y_max-y_min)/Y*yi
        try:
            z = newton(f, x + 1.j*y, fprime=f1, maxiter=maxiter)
            z = round(z.real, ndigits) + 1j * round(z.imag, ndigits)
        except RuntimeError:
            z = None
        try:
            i = zeros.index(z)
        except ValueError:
            zeros.append(z)
            i = len(zeros)
        pic[yi,xi] = i

plt.imshow(pic, interpolation='None', cmap='Set1')
plt.show()
