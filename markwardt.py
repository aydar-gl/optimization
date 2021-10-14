import numpy as np
import sympy
from numpy import linalg
from sympy import *
from scipy import linalg as lng
import copy
import time

def Gradient(xk):
    x1, x2 = symbols('x1 x2')
    func = 4 * np.square(x1 - 5) + np.square(x2 - 6)
    grad = [diff(func, x1), diff(func, x2)]
    for i in range(0, len(grad)):
        grad[i] = grad[i].subs([(x1, xk[0]), (x2, xk[1])])
    grad = np.array(grad).astype(np.float64)
    return grad

def H():
    x1, x2 = symbols('x1 x2')
    func = 4 * np.square(x1 - 5) + np.square(x2 - 6)
    grad = [diff(func, x1), diff(func, x2)]
    H = [[grad[0].coeff(x1), grad[0].coeff(x2)], [grad[1].coeff(x1), grad[1].coeff(x2)]]
    H = np.array(H).astype(np.float64)
    return H

def F_target(xk):
    return 4 * np.square(xk[0] - 5) + np.square(xk[1] - 6)

xk = []
xk.append([8, 9])
e1 = 0.1
M = 10
k = 0
mu = [20]

gradF = []
H = H()
d = []
E = np.eye(2)

while true:
    gradF.append([Gradient(xk[k])])
    norma = np.linalg.norm(gradF[k])
    if norma <= e1 or k >= M:
        break
    while true:
        Hmu = H + mu[k] * E
        Hmu_min1 = np.linalg.matrix_power(Hmu, -1)
        d.append(np.dot(np.dot(-1, Hmu_min1), gradF[k][0]))
        xk.append(xk[k] + d[k])
        xk[k + 1] = list(xk[k + 1])
        if F_target(xk[k + 1]) < F_target(xk[k]):
            mu.append(mu[k] / 2)
            k += 1
            break
        else:
            mu[k] *= 2
print('Минимум функции равен:', np.round(F_target(xk[k]), 6), ', при x1, x2 =', xk[k], ', k = ', k)
