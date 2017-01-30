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
    '''Finds the local vector of the first derivative of the Van der Pol
    equation.'''
    # Unpack the y vector
    y1, y2 = y

    # Create dydx vector (y1', y2')
    f = [y2, eta*y2 - y1 - eta*y2*y1**2.]
    return f


def d2ydx2(y, x, eta):
    '''Finds the local vector of the second derivative of the Van der Pol
    equation.'''
    # Unpack the y vector
    y1, y2 = y

    # Create vector of the second derivative
    y2prime = eta*y2 - y1 - eta*y2*y1**2.
    f = [y2prime, eta*y2prime - y2 - 2*eta*y1*y2 - eta*y2prime*y1**2]
    return f


def jacobian(y, eta):
    '''Find the local Jacobian matrix of the Van der Pol equation.'''
    return [[0., 1.], [-1. - 2*y[0]*y[1]*eta, eta-eta*y[0]**2]]


def weightednorm(matrix, weights):
    """Weighted average norm function as defined in 1985 Shampine.  Takes a
    matrix and 2 weights and returns the maximum value (divided by wi) of the
    sum of each value in each row multiplied by wj."""
    wi, wj = weights
    ivalues = []
    matrix = np.array(matrix)
    # A few try statements are used to figure out the shape of the matrix
    # Try statements are used because the code would otherwise return an
    # exception if the matrix is one dimensional.  The shape of the matrix is
    # needed to iterate across the rows and columns later.
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
    # Sums up the values across each of the columns and applies the weights,
    # then finds the maximum value (the weighted matrix norm) after the weights
    # have been applied.
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


def deriv1cd4():
    '''Finds the first derivative using the fourth order central difference
    formula.'''
    pass


def deriv2cd4():
    '''Finds the second derivative using the fourth order central difference
    formula'''
    pass


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

# Stiffness index parameter values to be sent to the stiffness index function
gamma = 1.
xi = 1.
order = 1
tolerance = 1.
stiffnessparams = tolerance, order, xi, gamma

# Weighted norm parameters to be sent to the weighted norm function
wi = 1.
wj = 1.
normweights = wi, wj

# Create the range of points along x to integrate
x_list = np.arange(xstart, xstop + 0.5*dx, dx)

# Pack up the parameters to send to the integrator
y = [y1zero, y2zero]

# Call the integrator
solution = sci.integrate.odeint(dydx, y, x_list, args=(eta,),
                                atol=abserr, rtol=relerr)

indexvalues = []
#firstderivals, indexvalues, jacobianlist, y1primelist, y2primelist = [], [],\
#    [], [], []
#y1doubleprimes, y2doubleprimes, y1tripleprimes = [], [], []
#oldy2 = solution[0][1]
for i in range(len(solution)):
    localjac = jacobian(solution[i], eta)
#    jacobianlist.append(localjac)
    firstderiv = dydx(solution[i], 0, eta)
    secondderiv = d2ydx2(solution[i], 0, eta)
#    firstderivals.append(firstderiv)
#    y1primelist.append(firstderiv[0])
#    y2primelist.append(firstderiv[1])
#    try:
#        solutionm2 = dydx(solution[i-2], 0, eta)
#        solutionm1 = dydx(solution[i-1], 0, eta)
#        solutionp1 = dydx(solution[i+1], 0, eta)
#        solutionp2 = dydx(solution[i+2], 0, eta)
#        y1doubleprime = (-solutionp2[0] + 8*solutionp1[0] - 8*solutionm1[0] +\
#                        solutionm2[0]) / (12*dx)
#        y2doubleprime = (-solutionp2[1] + 8*solutionp1[1] - 8*solutionm1[1] +\
#                        solutionm2[1]) / (12*dx)
##        (nextsolution[0] - prevsolution[0]) / (2*dx)
##        y2doubleprime = (nextsolution[1] - prevsolution[1]) / (2*dx)
##        y1doubleprime = (solution[i-1][0] - 2*solution[i][0] +
##                         solution[i+1][0]) / (dx**2)
##        y2doubleprime = (solution[i-1][1] - 2*solution[i][1] +
##                         solution[i+1][1]) / (dx**2)
#    except IndexError:
#        try:
#            nextsolution = dydx(solution[i+1], 0, eta)
#            y1doubleprime = (nextsolution[0] - firstderiv[0]) / dx
#            y2doubleprime = (nextsolution[1] - firstderiv[1]) / dx
#        except IndexError:
#            prevsolution = dydx(solution[i-1], 0, eta)
#            y1doubleprime = (firstderiv[0] - prevsolution[0]) / dx
#            y2doubleprime = (firstderiv[1] - prevsolution[0]) / dx
#    y1doubleprimes.append(y1doubleprime)
#    y2doubleprimes.append(y2doubleprime)

#    y1dblprime = firstderiv[1]
#    y2dblprime = np.abs(oldy2 - firstderiv[1]) / dx
#    ydblprime = [y1dblprime, y2dblprime]
    localstiffness = stiffnessindex(stiffnessparams, localjac,
                                    secondderiv, normweights)
    indexvalues.append(localstiffness)
#for i in range(len(y1doubleprimes)):
#    try:
#        y1tripleprime = (-y1doubleprimes[i+2] + 8*y1doubleprimes[i+1] -
#                         8*y1doubleprimes[i-1] + y1doubleprimes[i-2]) / (12*dx)
#    except IndexError:
#        try:
#            y1tripleprime = (y1doubleprimes[i+1] - y1doubleprimes[i]) / dx
#        except:
#            y1tripleprime = (y1doubleprimes[i] - y1doubleprimes[i-1]) / dx
#    y1tripleprimes.append(y1tripleprime)



#secondderivals = np.gradient(firstderivals, dx)
#y1doubleprimes = np.gradient(y1primelist, dx)
#y2doubleprimes = np.gradient(y2primelist, dx)
#y1tripleprimes = np.gradient(y1doubleprimes, dx)
#y2tripleprimes = np.gradient(y2doubleprimes, dx)
#
##print(y1doubleprimes)
##print(secondderivals)
#derivvals = np.zeros(2)
#for i in range(len(solution)):
###    secondderiv = np.gradient(firstderivals[i])
###    print(i)
#    derivvals[0] = y1doubleprimes[i]
#    derivvals[1] = y1tripleprimes[i]
###    print(y22p)
###    print(secondderiv)
###    print(secondderivals[0][i])
##    derivvals = [y12p, y22p]
##    print(derivvals)
###    print(derivvals)
#    localstiffness = stiffnessindex(stiffnessparams, jacobianlist[i],
#                                    derivvals, normweights)
#    indexvalues.append(localstiffness)

y1sol = []
y2sol = []
for i in range(len(solution)):
    y1sol.append(solution[i][0])
    y2sol.append(solution[i][1])

# Plot the solution
for p in range(1, 4):
    pyl.figure(p, figsize=(6, 4.5), dpi=400)
    pyl.xlabel('x Value')
    pyl.grid(True)
    pyl.hold(True)
    pyl.xlim(1, 3000)

lw = 1

pyl.figure(1)
pyl.ylabel('y1 Value')
pyl.plot(x_list, y1sol, 'b', linewidth=lw)
pyl.title('Solution Component')
#pyl.ylim(1, 2.1)

pyl.figure(2)
pyl.ylabel('y2 Value')
pyl.plot(x_list, y2sol, 'b', linewidth=lw)
pyl.title('y2 Value')
pyl.ylim(-0.01,0.01)

pyl.figure(3)
pyl.ylabel('Index Value')
pyl.plot(x_list, indexvalues, 'b', linewidth=lw)
pyl.title('IA-Stiffness Index')
pyl.yscale('log')

#pyl.figure(4)
#pyl.ylabel('First Derivative Value')
#pyl.plot(x_list, y1primelist, 'b', linewidth = lw)
#pyl.title('Y1 First Derivative Values')
#pyl.ylim(-0.01, 0.01)
#
#pyl.figure(5)
#pyl.ylabel('First Derivative Value')
#pyl.plot(x_list, y2primelist, 'b', linewidth = lw)
#pyl.title('Y2 First Derivative Values')
#pyl.ylim(-0.0001, 0.0001)
#
#pyl.figure(6)
#pyl.ylabel('Second Derivative Value')
#pyl.plot(x_list, y1doubleprimes, 'b', linewidth = lw)
#pyl.title('Y1 Second Derivative Values')
#pyl.ylim(-0.000001, 0.000001)
#
#pyl.figure(7)
#pyl.ylabel('Second Derivative Value')
#pyl.plot(x_list, y2doubleprimes, 'b', linewidth = lw)
#pyl.title('Y2 Second Derivative Values')
#pyl.ylim(-0.00001, 0.00001)
#
#pyl.figure(8)
#pyl.ylabel('Third Derivative Value')
#pyl.plot(x_list, y1tripleprimes, 'b', linewidth = lw)
#pyl.title('y1 Third Derivative Values')
#pyl.ylim(-0.00001, 0.00001)

pyl.show()
