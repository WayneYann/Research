#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:47:34 2016

@author: andrewalferman
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as fig


def functionone(x, parameters):
    '''Function to evaluate analytical solution, returns the Jacobian matrix
    and the local coordinates of y1 and y2.  Equation 6.1 of 1985 Shampine.'''
    y, y2p = np.zeros(2), np.zeros(2)
    a, n, b = parameters
    y[0] = np.exp(-a*n*x)
    y[1] = np.exp(-a*x)
    #y1p[0] = -(b + a*n)*y[0] + b*y[1]**n
    #y1p[1] = y[0] - a*y[1] - y[1]**n
    A = [[-(b + a*n), b*(y[1]**(n-1))],[1, -a - (y[1]**(n-1))]]
    y2p[0] = a**2 * n**2 * np.exp(-1 * a * n * x)
    y2p[1] = a**2 * np.exp(-1 * a * x)
    return A, y2p, y


def weightednorm(matrix, weights):
    """Weighted average norm function as defined in 1985 Shampine.  Takes a
    matrix and 2 weights and returns the maximum value (divided by wi) of the
    sum of each value in each row multiplied by wj."""
    wi, wj = weights
    ivalues = []
    matrix = np.array(matrix)
    try:
        num_rows, num_columns = matrix.shape
        dimensions = 2
    except ValueError:
        dimensions = 1
        try:
            num_rows = 1
            num_columns = matrix.shape[0]
        except IndexError:
            num_rows = matrix.shape[1]
            num_columns = 1
    for i in range(num_columns):
        columnsum = 0.
        for j in range(num_rows):
            if dimensions == 2:
                columnsum += np.abs(matrix[j][i]) * wj
            else:
                columnsum += np.abs(matrix[i]) * wj
        ivalues.append(columnsum / wi)
    return np.max(ivalues)


def stiffnessindex(sp, jacobian, derivativevals, normweights):
    '''Function that uses stiffness parameters (sp), the local Jacobian matrix,
    and a vector of the local function values to determine the local stiffness
    index as defined in 1985 Shampine'''
    # Method 1 uses the weighted norm of the Jacobian, Method 2 uses the
    # spectral radius of the Jacobian.
    method = 2
    tolerance, order, xi, gamma = sp
    if method == 1:
        index = tolerance**(1./(order + 1.)) *\
            weightednorm(jacobian, normweights) *\
             weightednorm(derivativevals, normweights)**\
                 (-1. / (order + 1.)) *\
             ((np.abs(xi)**(-1./(order + 1.))) / np.abs(gamma))
    else:
        index = tolerance**(1./(order + 1.)) *\
            np.max(np.abs(np.linalg.eigvals(jacobian))) *\
             weightednorm(derivativevals, normweights)**\
                 (-1. / (order + 1.)) *\
             ((np.abs(xi)**(-1./(order + 1.))) / np.abs(gamma))
    return index

# Function one parameters (funcparams)
funca = 1.
funcn = 4.

# Stiffness index parameter values (stiffnessparams)
gamma = 1.
xi = 1.
order = 1
tolerance = 1.
stiffnessparams = tolerance, order, xi, gamma

# Weighted norm parameters (normweights)
wi = 1.
wj = 1.
normweights = wi, wj

# Lists of x and b values to plot for the first function
b_list = [0., 100., 10000.]
x1_list = np.arange(0, 5.00001, 0.5)

# Iterate across the range of 0<x<5 for each of the b values using a
# nested for loop for the first function
indexvalues = []
for b in b_list:
    indexvalrow = []
    funcparams = funca, funcn, b
    for x in x1_list:
        jacobian, dblprime, localvals = functionone(x, funcparams)
        indexvalrow.append(stiffnessindex(stiffnessparams, jacobian,
                                          dblprime, normweights))
    indexvalues.append(indexvalrow)


for i in range(len(b_list)):
    plt.plot(x1_list, indexvalues[i], label='b value: {}'.format(b_list[i]))
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.title('IA-Stiffness Index', fontsize=16)
plt.xlabel('X range')
plt.ylabel('Stiffness Index Value')
plt.yscale('log')
plt.xlim(0, 5)
fig.Figure(dpi=900)
plt.grid(b=True, which='both')
plt.show()
