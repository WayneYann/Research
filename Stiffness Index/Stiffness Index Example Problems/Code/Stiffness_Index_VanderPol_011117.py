#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 23:54:30 2016

@author: andrewalferman
"""

import numpy as np
import matplotlib.pyplot as plt



def backwardeuler(yold, newtontolerance, eta, dx):
    '''A backward (implicit) Euler scheme for the van der Pol equation.'''
    ynext = np.zeros(2)
    yest = newtons_method(yold, newtontolerance, eta, dx)
#    jacobianterm = dx * np.array(jacobiantwo(ynew, eta))
#    ynext = np.dot(np.linalg.inv(np.eye(2) - jacobianterm), yold)
    ynext[1] = (yold[1] - dx*yest[0]) / (1 + dx*eta*(-1 + yest[0]**2))
    ynext[0] = yold[0] + dx * yest[1]
    return ynext


def forwardeuler(currentvalue, currentslope, dx):
    return currentvalue + currentslope * dx


def newtons_method(yvals, tolerance, eta, dx):
    '''Simple function that uses Newton's method to find the next value in the
    implicit scheme.  Configured for the van der Pol equation specifically.'''
    print('FUNCTION CALLED')
    deviation = tolerance + 1
    ynew, yold = yvals[:], yvals[:]
    while deviation > tolerance:
        nextvalue = forwardeuler(ynew[0], ynew[1], dx)
        print('NEXTVALUE: {}'.format(nextvalue))
        jacobianterm = np.dot(jacobiantwo(ynew, eta), ynew)
        print('JACOBIANTERM: {}'.format(jacobianterm))
        ynew[0] = ynew[0] - nextvalue / jacobianterm[0]
#        functionvalue = functionvalue - backresult[0] / jacobianterm[0]
#        print('FUNCTION VALUE ESTIMATE: {}'.format(functionvalue))
        deviation = np.abs(ynew[0] - yold[0])
        print('DEVIATION: {}'.format(deviation))
        yold = ynew
#        ynew = [functionvalue, jacobianterm[1]]
    return ynew


def jacobiantwo(y, eta):
    '''Find the local Jacobian matrix of the Van der Pol equation.'''
    return [[0, 1], [-1 - y[0]*y[1]*eta, eta]]


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
dx = 1.
newtontolerance = 1.e-6

# Stiffness index parameter values (stiffnessparams)
#gamma = 1.
#xi = 1.
#order = 1
#tolerance = 1.
#stiffnessparams = tolerance, order, xi, gamma

# Weighted norm parameters (normweights)
#wi = 1.
#wj = 1.
#normweights = wi, wj

# List of all the eta values to plot for the second function
eta_list = [1.e3]
x_list = np.arange(0, 3 + .5*dx, dx)

# Iterate across the range of 0<x<3000 for each of the eta values for the
# second function
#indexvalues = []
funcvals = []
for eta in eta_list:
#    indexvalrow = []
    y = [2., 0.]
    for x in x_list:
        funcvals.append(y[0])
        yprev = y
        print(y)
        y = backwardeuler(y, newtontolerance, eta, dx)
#        derivativevals, localvals = functiontwo(x, eta, order)
#        jacobian = jacobiantwo(x, eta, order)
#        indexvalrow.append(stiffnessindex(stiffnessparams, jacobian,
#                                          derivativevals, normweights))
#    indexvalues.append(indexvalrow)

# Plot all of the values calculated
plt.figure(0)
plt.plot(x_list, funcvals, label='Eta value: {}'.format(eta_list[0]))
#    plt.figure(1)
#    plt.plot(x2_list, indexvalues[i], label='Eta value: {}'.format(eta_list[i]))
for i in range(1):
    plt.figure(i)
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    plt.xlabel('X range')
    plt.grid(b=True, which='both')
plt.figure(0)
plt.title('Numerical Solution')
plt.ylabel('Y1 value')
#plt.figure(1)
#plt.title('IA-Stiffness Index')
#plt.ylabel('Stiffness Index Value')
#plt.yscale('log')
plt.show()
