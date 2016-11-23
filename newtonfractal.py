#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
from builtins import super

from random import random
import numpy as np
import matplotlib.pyplot as plt

def newton(f, x0, fprime, dx, dy=None, maxiter=50):
    x0 = np.asarray(x0)
    iters = np.zeros(np.shape(x0), dtype=np.int)
    if dy is None:
        dy = dx
    f0 = np.asarray(f(x0))
    f1 = np.asarray(fprime(x0))
    wip = abs(f0) >= 1e-3
    for iter in range(maxiter):
        #print(iter, f0, f1, done)
        f1[wip] = fprime(x0[wip])
        xf1 = f1!=0
        x0[~xf1] += .5*dx + .5j*dy  # TODO: Randomize?
        #if f1==0:
        #    x0 = x0 + (2*random()-1)*dx + 1.j*(2*random()-1)*dy
        #    continue
        x0[xf1] = x0[xf1] - f0[xf1] / f1[xf1]
        f0[wip] = f(x0[wip])  # TODO: Lookup if new x0's Pixel has already been checked
        wip = abs(f0) >= 1e-3
        iters[wip] += 1
        if (~wip).all():
            break
    return x0, iters


X = 4*2**5
Y = 3*X//4

x_min = -5
x_max = -x_min
y_min = x_min*Y/X
y_max = x_max*Y/X

dx = (x_max - x_min) / X
dy = (y_max - y_min) / Y

maxiter = 100
ndigits = 1

f = lambda x: np.power(x,3) - 1; f1 = lambda x: 3*np.power(x,2)
#f = lambda x: np.power(x,5) + np.power(x,3)*7 - np.power(x,2)*3 -1
#f1 = lambda x: 5*np.power(x,4) + 21*np.power(x,2) - 6*x

pic = np.zeros((Y, X), dtype=np.int)  # TODO: WHY (Y,X) and not (X,Y)???
#zeros = ['', None,]

xs = np.linspace(x_min, x_max, X)
ys = np.linspace(y_min, y_max, Y)
xs, ys = np.meshgrid(xs, ys)

zeros, iters = newton(f, xs + 1.j * ys, dx=dx, dy=dy, fprime=f1, maxiter=maxiter)
converged = iters<maxiter  # TODO: Use np.where?
zeros[converged] = np.round(zeros[converged], ndigits)
zeros[~converged] = np.nan


unique_zeros = np.unique(zeros.flatten())
for i, zero in enumerate(unique_zeros):
    pic[zeros==zero] = i
    basin = np.where((zeros == zero) & (abs(xs+1.j*ys - zero) <= .1*ndigits))
    pic[basin] = -i-1

plt.imshow(pic, interpolation='None', cmap='Set1', origin='lower', extent=[x_min, x_max, y_min, y_max])
#plt.colorbar()
plt.contour(xs, ys, np.log(iters))
plt.figure()
plt.imshow(np.log(iters), interpolation='None', cmap='gray', origin='lower', extent=[x_min, x_max, y_min, y_max])
plt.show()
