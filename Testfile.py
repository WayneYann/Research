#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:54:13 2017

@author: andrewalferman
"""

import numpy as np
import pylab as pyl
import scipy as sci
import datetime


def firstderiv(y, x, eta):
    '''Equations that describe a coupled system of first order ODE's that
    make up the van der Pol equation.'''
    # Create dydx vector (y1', y2')
    f = [y[1], eta*y[1] - y[0] - eta*y[1]*y[0]**2.]
    return np.array(f)


def d2ydx2(y, x, eta):
    '''Finds the local vector of the second derivative of the Van der Pol
    equation.'''
    # Create vector of the second derivative
    y2prime = eta*y[1] - y[0] - eta*y[1]*y[0]**2.
    f = [y2prime, eta*y2prime - y[1] - 2*eta*y[0]*y[1] - eta*y2prime*y[0]**2]
    return np.array(f)


def jacobval(y, x, eta):
    '''Find the local Jacobian matrix of the Van der Pol equation.'''
    return np.array([[0., 1.], [-1. - 2*y[0]*y[1]*eta, eta-eta*y[0]**2]])


def derivcd4(vals, dx):
    """Given a list of values at equally spaced points, returns the first
    derivative using the fourth order central difference formula, or forward/
    backward differencing at the boundaries."""
    deriv = []
    for i in range(2):
        deriv.append((-3*vals[i] + 4*vals[i+1] - vals[i+2]) / (2*dx))
    for i in range(2, len(vals) - 2):
        deriv.append((-1*vals[i-2] + 8*vals[i-1] + 8*vals[i+1] -\
                          vals[i+2]) / (12*dx))
        # Note that due to the fact that this function has been set up this
        # way, this will not output a value at 5000000
        if i % 500000 == 0:
            print('Derivative list: {}'.format(i))
    for i in range((len(vals) - 2), len(vals)):
        deriv.append((-3*vals[i-2] + 4*vals[i-1] - vals[i]) / (2*dx))
    return deriv


def weightednorm(matrix, weights):
    """Weighted average norm function as defined in 1985 Shampine.  Takes a
    matrix and 2 weights and returns the maximum value (divided by wi) of the
    sum of each value in each row multiplied by wj.  Needs to be passed either
    a matrix of m x n dimensions where m,n > 1, or a column vector."""
    # Unpack the parameters
    wi, wj = weights

    # Initialize a list that will be called later to obtain the maximum value
    ivalues = []

    try:
        num_rows,num_columns = matrix.shape
        for i in range(num_columns):
            matrixcol = [np.abs(j)*wj for j in matrix[:,i]]
            columnsum = np.sum(matrixcol)
            ivalues.append(columnsum)
        return np.max(ivalues) / wi
    except ValueError:
        matrixcol = [np.abs(j)*wj for j in matrix]
        return np.sum(matrixcol) / wi


def stiffnessindex(sp, normweights, xlist, dx, solution, eta):
    '''Function that uses stiffness parameters (sp), the local Jacobian matrix,
    and a vector of the local function values to determine the local stiffness
    index as defined in 1985 Shampine.

    Future optimizations:
        1.  Make it smarter regarding the shape of the derivative values, etc.
        2.  Use a different integrator that saves the values of the derivative
        as it goes so that we don't need to calculate each derivative twice.
        Same goes with the Jacobian, it would be much faster to evaluate
        it on the fly.
        3.  Make this a class.
        4.  Implement this index in the integrator itself so that it makes all
        of the values as the integration goes.  This would eliminate the need
        to save the dydx list beyond a few variables that would be needed to
        compute the higher level derivatives.
    '''
    # Method 1 uses the weighted norm of the Jacobian, Method 2 uses the
    # spectral radius of the Jacobian.
    method = 2

    # Unpack the parameters
    tolerance, order, xi, gamma = sp

    # Obtain the derivative values for the derivative of order p
    dydxlist = []
    d2ydx2list = []
    print('Finding first derivative values...')
    for i in range(len(solution)):
        #dydxlist.append(firstderiv(solution[i,:],xlist[i],eta))
        d2ydx2list.append(d2ydx2(solution[i,:],xlist[i],eta))
        if i % 500000 == 0:
            print('dydxlist: {}'.format(i))
    # Raise the derivative to the order we need it
#    for i in range(order):
#        print('Finding derivative of order {}...'.format(i+2))
#        dydxlist = derivcd4(dydxlist, dx)
#    dydxlist = np.array(dydxlist)

    # Create a list to return for all the index values in a function
    indexlist = []

    # The weighted norm needs to be a single column, not a single row, so it
    # needs to be transposed.
    print('Transposing dydx vector...')
    d2ydx2list = np.transpose(d2ydx2list)

    # Figure out some of the values that will be multiplied many times, so that
    # each computation only needs to happen once.
    exponent = 1./(order + 1)
    xiterm = ((np.abs(xi)**(-1 * exponent)) / np.abs(gamma))
    toleranceterm = tolerance**exponent

    # Obtain the shape of the jacobian and the dydx vector.  This will speed
    # up the weightednorm function (see function for details).  May not be
    # needed if a better method can be found with numpy arrays.  Note that
    # these two values are only dummy values thrown in to get the shape, they
    # are not included in the solution.
    for i in range(len(solution)):
        jacobian = jacobval(solution[i,:],xlist[i],eta)
        if method == 1:
            index = toleranceterm *\
                    weightednorm(jacobian, normweights) *\
                    weightednorm(d2ydx2list[:,i], normweights)**(-1 * exponent) *\
                    xiterm
        else:
            index = toleranceterm *\
                    np.max(np.abs(np.linalg.eigvals(jacobian))) *\
                    weightednorm(d2ydx2list[:,i], normweights)**(-1 * exponent) *\
                    xiterm
        indexlist.append(index)
        if i % 500000 == 0:
            print('Index list: {}'.format(i))
    indexlist = np.array(indexlist)
    return indexlist

# Finding the current time to time how long the simulation takes
starttime = datetime.datetime.now()
print('Start time: {}'.format(starttime))

savedata = 0

# Define the range of the computation
xstart = 0
xstop = 810.
xswitch = 807.01
xswitch2 = 807.3
dx1 = 1.0e-1
dx2 = 1.0e-8

# Create the range of points along x to integrate
xlist1 = np.arange(xstart, xswitch, dx1)
xlist2 = np.arange(xswitch, xswitch2, dx2)
xlist3 = np.arange(xswitch2, xstop+dx1*.5, dx1)
x_list = np.concatenate((xlist1, xlist2, xlist3))

# Equation parameters
eta = 1.e3

# Initial conditions
y1zero = 2
y2zero = 0

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

# Pack up the initial conditions to send to the integrator
y = np.array([y1zero, y2zero])

# ODE Solver parameters
abserr = 1.0e-12
relerr = 1.0e-10

# Call the integrator
solution = sci.integrate.odeint(firstderiv, # Call the dydt function
                                # Pass it initial conditions
                                y,
                                # Pass it time steps to be evaluated
                                x_list,
                                # Pass whatever additional information is needed
                                args=(eta,),
                                # Pass it the Jacobian (not sure if needed)
#                                Dfun=jacobval,
                                # Pass it the absolute and relative tolerances
                                atol=abserr, rtol=relerr,
                                # Print a message stating if it worked or not
                                printmessg=1
                                )

print('Code progress:')

# Find the stiffness index across the range of the solution
indexvalues = stiffnessindex(stiffnessparams, normweights, x_list, dx1, solution,
                             eta)

# Plot the solution.  This loop just sets up some of the parameters that we
# want to modify in all of the plots.
for p in range(1,3):
    pyl.figure(p, figsize=(6, 4.5), dpi=600)
    pyl.xlabel('x Value', fontsize=14)
    pyl.xlim(xstart,xstop)
    pyl.grid(True)
    pyl.hold(True)

#Set the linewidth to make plotting look nicer
lw = 1

# Set all of the parameters that we want to apply to each plot specifically.
pyl.figure(1)
pyl.ylabel('Solution Value', fontsize=14)
pyl.plot(x_list, solution[:,0], 'b', linewidth=lw)
#pyl.title('Temperature Graph, Time={}, Particle={}'.format(
#            pasrtimestep,particle), fontsize=16)
#pyl.xlim(0,0.005)

pyl.figure(2)
pyl.ylabel('Index Value', fontsize=14)
pyl.plot(x_list, indexvalues, 'b', linewidth=lw)
#pyl.title('IA-Stiffness Index, Order = {}'.format(order), fontsize=16)
pyl.yscale('log')
#pyl.text(0.1,0.001,'dt = {}, Abs Error = {}, Rel Error = {}'.format(
#            dt,abserr,relerr))

pyl.show()

#if savedata == 1:
#    indexvalues = np.array(indexvalues)
#    dataneeded = [indexvalues,tlist, solution, pasrtimestep, particle, order,
#                  dt, abserr, relerr]
#    print('Saving data...')
#    np.save('IndexVals_Autoignition_{}'.format(dt), dataneeded)

finishtime = datetime.datetime.now()
print('Finish time: {}'.format(finishtime))
