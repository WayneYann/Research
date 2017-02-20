#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:54:13 2017

@author: andrewalferman
"""

import numpy as np
import pyjac as pyj
import pyjacob as pyjacob
import pylab as pyl
import scipy as sci
import os as os

filedir = os.path.dirname(os.path.realpath(__file__))
workingdir = os.getcwd()

def get_dydt(self, dydt):
    """Evaluate derivative

    Parameters
    ----------
    dydt : ``numpy.array``
        Derivative of state vector

    Returns
    -------
    None

    """
    dydt[:] = self.test_dydt[self.index, :-1]


def derivcd4(vals, dx):
    """Given a list of values at equally spaced points, returns the first
    derivative using the fourth order central difference formula, or forward/
    backward differencing at the boundaries."""
    deriv = []
    for i in range(len(vals)):
        try:
            deriv.append((-1*vals[i-2] + 8*vals[i-1] + 8*vals[i+1] -\
                          vals[i+2]) / 12*dx)
        except IndexError:
            try:
                deriv.append((-3*vals[i] + 4*vals[i+1] - vals[i+2]) / 2*dx)
            except IndexError:
                deriv.append((3*vals[i] - 4*vals[i-1] + vals[i-2]) / 2*dx)
    return deriv


def weightednorm(matrix, weights):
    """Weighted average norm function as defined in 1985 Shampine.  Takes a
    matrix and 2 weights and returns the maximum value (divided by wi) of the
    sum of each value in each row multiplied by wj."""
    # Unpack the parameters
    wi, wj = weights

    # Initialize a list that will be called later to obtain the maximum value
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

    # Unpack the parameters
    tolerance, order, xi, gamma = sp

    # The second derivative normally will come in one row for this program,
    # however we want to be taking the weighted norm of the second derivative
    # values in one column instead.  The derivative values must then be
    # transposed.  Should try to make this smarter by checking the number of
    # rows/columns before transposing.
    np.asarray(derivativevals).T.tolist()

    if method == 1:
        exponent = 1./(order + 1)
        index = tolerance**exponent *\
            weightednorm(jacobian, normweights) *\
             weightednorm(derivativevals, normweights)**exponent *\
             ((np.abs(xi)**exponent) / np.abs(gamma))
    else:
        exponent = 1./(order + 1)
        index = tolerance**exponent *\
            np.max(np.abs(np.linalg.eigvals(jacobian))) *\
             weightednorm(derivativevals, normweights)**exponent *\
             ((np.abs(xi)**exponent) / np.abs(gamma))
    return index

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

## Define the range of the computation
tstart = 0
tstop = 0.01
dt = 1.e-3
tlist = np.arange(tstart, tstop+dt, dt)

t0 = 0.050

# ODE Solver parameters
abserr = 1.0e-10
relerr = 1.0e-8

# Load the initial conditions from the pasr file
ic = np.load('/Users/andrewalferman/Desktop/Research/pasr_out_h2-co_0.npy')

# Call the integrator
#solution = sci.integrate.ode(pyjacob.py_dydt, pyjacob.py_eval_jacobian).\
#                            set_integrator('zvode', method='bdf')
#solution.set_initial_value(ic[500,0,:], t0)
#t1 = 0.100
#while solution.successful() and solution.t < t1:
#    print(solution.t + dt, solution.integrate(solution.t+dt))
solution = sci.integrate.odeint(pyjacob.py_dydt, # Call the dydt function
                                # Pass it initial conditions
                                ic[500,0,:],
                                # Pass it time steps to be evaluated
                                tlist,
                                # Pass whatever additional information is needed
                                args=(ic[500,0,0],get_dydt(pyjacob.py_dydt)),
                                # Pass it the Jacobian (not sure if needed)
                                Dfun=pyjacob.py_eval_jacobian,
                                # Pass it the absolute and relative tolerances
                                atol=abserr, rtol=relerr,
                                # Print a message stating if it worked or not
                                printmessg=1
                                )





#print(pyjacob.py_dydt(0.05,2418.195016625938,ic[500,0,:],solution))

## Obtain the derivative values
#for i in solution:
#    firstderiv = pyjacob.py_dydt(initconditions)
#    jacobian = pyjacob.py_eval_jacobian(initconditions)
#
#secondderiv = derivcd4(firstderiv)
#
## Find the stiffness index across the range of the solution by using the above
## functions to get the Jacobian matrix and second derivative
#indexvalues = []
#for i in solution:
#    localstiffness = stiffnessindex(stiffnessparams, jacobian[i],
#                                    secondderiv[i], normweights)
#    indexvalues.append(localstiffness)
#
## Plot the solution.  This loop just sets up some of the parameters that we
## want to modify in all of the plots.
#for p in range(1, 3):
#    pyl.figure(p, figsize=(6, 4.5), dpi=400)
#    pyl.xlabel('x Value')
#    pyl.grid(True)
#    pyl.hold(True)
#
##Set the linewidth to make plotting look nicer
#lw = 1
#
## Set all of the parameters that we want to apply to each plot specifically.
#pyl.figure(1)
#pyl.ylabel('y1 Value')
#pyl.plot(tlist, solution[:,0], 'b', linewidth=lw)
#pyl.title('Temperature Graph')
#
#pyl.figure(2)
#pyl.ylabel('Index Value')
#pyl.plot(tlist, indexvalues, 'b', linewidth=lw)
#pyl.title('IA-Stiffness Index, Order = {}'.format(order))
#pyl.yscale('log')
#
#pyl.show()
