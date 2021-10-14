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
e2 = 0.1
M = 10
k = 0

gradF = []
H = H()
H_min1 = np.linalg.matrix_power(H, -1)
d = []

while true:
    gradF.append([Gradient(xk[k])])
    norma = np.linalg.norm(gradF[k])
    if norma <= e1 or k >= M:
        break
    if H_min1[0,0] > 0 and np.linalg.det(H_min1) > 0:
        d.append(np.dot(-H_min1, gradF[k][0]))
        xk.append([])
        xk[k] = np.array(xk[k]).astype(np.float32)
        xk[k + 1] = np.array(xk[k + 1]).astype(np.float32)
        d[k] = np.array(d[k]).astype(np.float32)
        xk[k+1] = xk[k] + d[k]
    if np.linalg.norm(xk[k + 1] - xk[k]) < e2 and np.absolute(F_target(xk[k+1]) - F_target(xk[k])) < e2:
        break
    else:
        k += 1
    xk[k - 1] = list(xk[k - 1])
    xk[k] = list(xk[k])
    d[k - 1] = list(d[k - 1])
print('Минимум функции равен:', F_target(xk[k]), ', при x1, x2 =', xk[k], ', k = ', k)
