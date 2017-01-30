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

# Define the range of the computation
xstart = 0
xstop = 3000
dx = 1.0e-0

# Equation parameters
eta = 1.e3

# Initial conditions
y1zero = 2
y2zero = 0

# ODE Solver parameters
abserr = 1.0e-8
relerr = 1.0e-6

# Create the range of points along x to integrate
x_list = np.arange(xstart, xstop + 0.5*dx, dx)

# Pack up the parameters to send to the integrator
y = [y1zero, y2zero]

# Call the integrator
solution = sci.integrate.odeint(dydx, y, x_list, args=(eta,),
                                atol=abserr, rtol=relerr)

# Generate the solution values
y1sol = []
y2sol = []
for i in range(len(solution)):
    y1sol.append(solution[i][0])
    y2sol.append(solution[i][1])

# Plot the solution
for p in range(1, 3):
    pyl.figure(p, figsize=(6, 4.5), dpi=400)
    pyl.xlabel('x Value')
    pyl.grid(True)
    pyl.hold(True)
#    pyl.xlim(807.05,807.07)

lw = 1

pyl.figure(1)
pyl.ylabel('y1 Value')
pyl.plot(x_list, y1sol, 'b', linewidth=lw)
pyl.title('Solution Component')

pyl.figure(2)
pyl.ylabel('y2 Value')
pyl.plot(x_list, y2sol, 'b', linewidth=lw)
pyl.title('Derivative Value')

pyl.show()
