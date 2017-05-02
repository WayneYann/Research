#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:53:14 2017

@author: alfermaa
"""

import numpy as np
import scipy as sci
import pylab as pyl


def dydx(y, x, eta):
    '''Equations that describe a coupled system of first order ODE's that
    make up the van der Pol equation.'''
    # Unpack the y vector
    y1, y2 = y

    # Create dydx vector (y1', y2')
    f = [y2, eta*y2 - y1 - eta*y2*y1**2]
    return f


def weightednorm(matrix, weights):
    """Weighted average norm function as defined in 1985 Shampine.  Takes a
    matrix and 2 weights and returns the maximum value (divided by wi) of the
    sum of each value in each row multiplied by wj."""
    wi, wj = weights
    ivalues = []
    matrix = np.array(matrix)

    # The shape of the matrix that is fed to this function will change
    #depending on the problem, so these try statements figure out the number
    #of rows and columns of the matrix.
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

    # A for loop is used to sum the value of each row and apply the weights.
    for i in range(num_columns):
        columnsum = 0.
        for j in range(num_rows):
            if dimensions == 2:
                columnsum += np.abs(matrix[j][i]) * wj
            else:
                columnsum += np.abs(matrix[i]) * wj
        ivalues.append(columnsum / wi)
    return np.max(ivalues)


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


def jacobiantwo(y, eta):
    '''Find the local Jacobian matrix of the Van der Pol equation.'''
    return [[0, 1], [-1 - y[0]*y[1]*eta, eta]]

# Define the range of the computation
xstart = 0.
xstop = 1000.
dx = 1.0e-2

# Equation parameters
eta = 1.e3

# Initial conditions
y1zero = 2
y2zero = 0

# ODE Solver parameters
abserr = 1.0e-8
relerr = 1.0e-6

# Weighted norm parameters
wi = 1.
wj = 1.
normweights = wi, wj

# Stiffness index parameter values
gamma = 1.
xi = 1.
order = 1
tolerance = 1.
stiffnessparams = tolerance, order, xi, gamma

# Create the range of points along x to integrate
x_list = np.arange(xstart, xstop + 0.5*dx, dx)

# Pack up the initial conditions to send to the integrator
y = [y1zero, y2zero]

# Call the integrator
solution = sci.integrate.odeint(dydx, y, x_list, args=(eta,),
                                atol=abserr, rtol=relerr)
derivvals = []
for i in range(len(solution)):
    derivvals.append(dydx(solution[i], x_list[i], eta))

secondderivvals = np.gradient(derivvals)

# Generate the solution values
y1sol, y2sol, indexvals = [], [], []

for i in range(len(solution)):
    y1s = solution[i][0]
    y2s = solution[i][1]
    ys = [y1s, y2s]
    y1sol.append(y1s)
    y2sol.append(y2s)
    indexvals.append(stiffnessindex(stiffnessparams, jacobiantwo(ys, eta),
                                    secondderivvals, normweights))
    y1sold = y1s
    y2sold = y2s

# Plot the solution
for p in range(1, 4):
    pyl.figure(p, figsize=(6, 4.5), dpi=400)
    pyl.xlabel('x Value')
    pyl.grid(True)
    pyl.hold(True)
    pyl.xlim(805,810)

lw = 1

pyl.figure(1)
pyl.ylabel('y1 Value')
pyl.plot(x_list, y1sol, 'b', linewidth=lw)
pyl.title('Solution Component')

pyl.figure(2)
pyl.ylabel('y2 Value')
pyl.plot(x_list, y2sol, 'b', linewidth=lw)
pyl.title('Derivative Value')

pyl.figure(3)
pyl.ylabel('Stiffness Index Value')
pyl.plot(x_list, indexvals, 'b', linewidth=lw)
# pyl.yscale('log')
pyl.title('IA-Stiffness Index')

pyl.show()
