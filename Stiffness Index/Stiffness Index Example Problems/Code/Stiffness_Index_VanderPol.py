#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 23:54:30 2016

@author: andrewalferman
"""

import numpy as np
import matplotlib.pyplot as plt





def backwardeuler(yprev, eta, dx, tolerance):
    '''A backward (implicit) Euler scheme for the van der Pol equation.'''
#    ynext = np.zeros(2)
    yprimenext = np.zeros(2)
    ynext = SOR(yprev, tolerance, eta, dx)
#    ynextcomp = newtons_method(yprev, tolerance, eta)
#    ynext[0] = yprev[0] + dx * ynextcomp[1]
#    ynext[1] = (yprev[1] - dx * ynextcomp[0]) /\
#                (1 + (dx/eta)*(-1 + ynextcomp[0]**2))
    yprimenext[0] = ynext[1]
    yprimenext[1] = ynext[1]/eta - ynext[0] -\
                (ynext[1]*ynext[0]**2)/eta
    return ynext, yprimenext


def SOR(yvals, tolerance, eta, dx):
    '''Simple function that uses Newton's method to find the next value in the
    implicit scheme.  Configured for the van der Pol equation specifically.'''
    deviation = 99999.
    weight = 0.95
    y1, y2 = yvals[0], yvals[1]
    y1n = y1
    y2n = y2
    y1old = y1
    y2old = y2
    while deviation > tolerance:
        fy1 = y1old + weight * ((y1 + dx*y2n) - y1old)
        fy2 = y2old + weight *\
            (((y2 - dx*y1n) / (1 + (dx/eta)*(-1 + y1n**2))) - y2old)
        y1old = y1n
        y2old = y2n
        y1n = fy1
        y2n = fy2
        deviation = max(abs(y1n-y1old), abs(y2n-y2old))
    return y1n, y2n


#def functiontwo(x, eta):
#    '''Function to evaluate the semi-analytical solution to the Van der Pol
#    equation.   Equation 6.3 of 1985 Shampine.'''
#    y, y1p = np.zeros(2), np.zeros(2)
#    y[0] = x*y[1] + 2
#    y[1] = x*y[1]/eta - x*y[0] - x*(y[0]**2)*y[1]/eta
##    y1p[0] = y[1]
##    y1p[1] = y[1]*eta**-1 - y[0] - y[0]*y[1]*eta**-1
#    return y, y1p


def jacobiantwo(y, eta):
    '''Find the local Jacobian matrix of the Van der Pol equation.'''
    return [[0, 1], [-1 - y[0]*y[1]*eta**-1, eta**-1]]


def factorial(number):
    '''Simple function that returns the factorial of a given integer.  Prints
    an error message and returns None if the value given is not a positive
    integer.'''
    if isinstance(number, int) and number > 0:
        result = 1
        for i in range(number):
            result *= number
            number -= 1
        return result
    else:
        print('Factorial Error: Number must be a positive integer')
        return None


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

# Stepsize and other important variables that affect accuracy
dx = 0.0001
newtontolerance = 0.0000001

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

# List of all the eta values to plot for the second function
eta_list = [1/1.e3]
x_list = np.arange(0, 3000 + .5*dx, dx)

# Iterate across the range of 0<x<3000 for each of the eta values for the
# second function
indexvalues = []
funcvals = []
for eta in eta_list:
    indexvalrow = []
    y = [2., 0.]
    for x in x_list:
        yprev = y
        y, y1p = backwardeuler(yprev, eta, dx, newtontolerance)
        funcvals.append(y[0])
#        derivativevals, localvals = functiontwo(x, eta, order)
#        derivativevals =
        jacobian = jacobiantwo(y, eta)
#                                          derivativevals, normweights))
#    indexvalues.append(indexvalrow)

# Plot all of the values calculated
for i in range(1):
    plt.figure(i)
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    plt.xlabel('X range')
    plt.grid(b=True, which='both')
plt.figure(0)
plt.plot(x_list, funcvals, label='Eta value: {}'.format(eta_list[0]))
plt.title('Numerical Solution')
plt.ylabel('Y1 value')
#plt.figure(1)
#plt.title('IA-Stiffness Index')
#plt.ylabel('Stiffness Index Value')
#plt.yscale('log')
#plt.plot(x2_list, indexvalues[i], label='Eta value: {}'.format(eta_list[i]))
#plt.show()
