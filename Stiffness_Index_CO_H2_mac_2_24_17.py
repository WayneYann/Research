#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:54:13 2017

@author: andrewalferman
"""

import os as os
import numpy as np
import pyjacob as pyjacob
import pylab as pyl
import scipy as sci
import datetime


def firstderiv(state, time, press):
    """This function forces the integrator to use the right arguments."""
    dy = np.zeros_like(state)
    pyjacob.py_dydt(time, press, state, dy)
    return dy


def jacobval(state, time, press):
    """This function forces the integrator to use the right arguments."""
    a = len(state)
    jacobian = np.zeros(a**2)
    pyjacob.py_eval_jacobian(time, press, state, jacobian)
    jacobian = np.reshape(jacobian, (a,a))
    return jacobian


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
        deriv.append((3*vals[i] - 4*vals[i-1] + vals[i-2]) / (2*dx))
    return deriv


def weightednorm(matrix, weights, dims):
    """Weighted average norm function as defined in 1985 Shampine.  Takes a
    matrix and 2 weights and returns the maximum value (divided by wi) of the
    sum of each value in each row multiplied by wj."""
    # Unpack the parameters
    wi, wj = weights

    # Initialize a list that will be called later to obtain the maximum value
    ivalues = []

    # Retrieve the number of rows and columns
    num_rows, num_columns, dimensions = dims

    # There's an optimization to be had in here somewhere with numpy arrays
    # Here's the failed attempt at it.  Will try again later.
#    for i in matrix[:,0]:
#        print(i)
#        print(np.shape(i))
#        columnsum = np.sum(i * wj)
#        ivalues.append(columnsum / wi)
#    # Sums up the values across each of the columns and applies the weights,
#    # then finds the maximum value (the weighted matrix norm) after the weights
#    # have been applied.
    for i in range(num_columns):
        columnsum = 0.
        for j in range(num_rows):
            if dimensions == 2:
                columnsum += np.abs(matrix[j,i]) * wj
            else:
                columnsum += np.abs(matrix[i]) * wj
        ivalues.append(columnsum / wi)
    return np.max(ivalues)


def getshape(matrix):
    ''' A few try statements are used to figure out the shape of the matrix
    Try statements are used because the code would otherwise return an
    exception if the matrix is one dimensional.  The shape of the matrix is
    needed to iterate across the rows and columns later.'''
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
    return (num_rows,num_columns, dimensions)


def stiffnessindex(sp, normweights, xlist, dx, solution):
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
    print('Finding first derivative values...')
    for i in range(len(solution)):
        dydxlist.append(firstderiv(solution[i,:],tlist[i],Y_press))
        if i % 500000 == 0:
            print('dydxlist: {}'.format(i))
    # Raise the derivative to the order we need it
    for i in range(order):
        print('Finding derivative of order {}...'.format(i+2))
        dydxlist = derivcd4(dydxlist, dt)
    dydxlist = np.array(dydxlist)

    # Create a list to return for all the index values in a function
    indexlist = []

    # The weighted norm needs to be a single column, not a single row, so it
    # needs to be transposed.
    print('Transposing dydx vector...')
    dydxlist = np.asarray(dydxlist).T

    # Figure out some of the values that will be multiplied many times, so that
    # each computation only needs to happen once.
    exponent = 1./(order + 1)
    xiterm = ((np.abs(xi)**exponent) / np.abs(gamma))
    toleranceterm = tolerance**exponent

    # Obtain the shape of the jacobian and the dydx vector.  This will speed
    # up the weightednorm function (see function for details).  May not be
    # needed if a better method can be found with numpy arrays.  Note that
    # these two values are only dummy values thrown in to get the shape, they
    # are not included in the solution.
    jacshape = getshape(jacobval(solution[0,:],xlist[0],101300))
    dshape = getshape(dydxlist[:,0])

    print('Computing stiffness index...')

    for i in range(len(solution)):
        jacobian = jacobval(solution[i,:],tlist[i],Y_press)
        if method == 1:
            index = toleranceterm *\
                    weightednorm(jacobian, normweights, jacshape) *\
                    weightednorm(dydxlist[:,i], normweights, dshape)**exponent *\
                    xiterm
        else:
            index = toleranceterm *\
                    np.max(np.abs(np.linalg.eigvals(jacobian))) *\
                    weightednorm(dydxlist[:,i], normweights, dshape)**exponent *\
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

# Define the range of the computation
tstart = 0
tstop = 5.e-0
dt = 1.e-5
tlist = np.arange(tstart, tstop+0.5*dt, dt)

t0 = 0.050

# ODE Solver parameters
abserr = 1.0e-15
relerr = 1.0e-13

# Load the initial conditions from the PaSR file
filepath = os.path.join(os.getcwd(),'pasr_out_h2-co_0.npy')
pasr = np.load(filepath)

# Select a point out of the PaSR file as an initial condition
pasrtimestep = 4
particle = 92

# Get the initial condition.
Y = pasr[pasrtimestep, particle, :].copy()

# Rearrange the array for the initial condition
N2_pos = 11
press_pos = 2
temp_pos = 1
arraylen = len(Y)
Y_N2 = Y[N2_pos]
Y_x = Y[arraylen - 1]
Y[arraylen - 1] = Y_N2
Y[N2_pos] = Y_x
Y_press = Y[press_pos]
Y_temp = Y[temp_pos]
Y_species = Y[3:]
Ys = np.hstack((Y_temp,Y_species))

# Call the integrator
solution = sci.integrate.odeint(firstderiv, # Call the dydt function
                                # Pass it initial conditions
                                Ys,
                                # Pass it time steps to be evaluated
                                tlist,
                                # Pass whatever additional information is needed
                                args=(Y_press,),
                                # Pass it the Jacobian (not sure if needed)
#                                Dfun=jacobval,
                                # Pass it the absolute and relative tolerances
                                atol=abserr, rtol=relerr,
                                # Print a message stating if it worked or not
                                printmessg=1
                                )

print('Code progress:')

# Find the stiffness index across the range of the solution
indexvalues = stiffnessindex(stiffnessparams, normweights, tlist, dt, solution)

# Plot the solution.  This loop just sets up some of the parameters that we
# want to modify in all of the plots.
for p in range(1,3):
    pyl.figure(p, figsize=(6, 4.5), dpi=600)
    pyl.xlabel('Time (s)', fontsize=14)
    pyl.xlim(tstart,tstop)
    pyl.grid(True)
    pyl.hold(True)

#Set the linewidth to make plotting look nicer
lw = 1

# Set all of the parameters that we want to apply to each plot specifically.
pyl.figure(1)
pyl.ylabel('Temperature', fontsize=14)
pyl.plot(tlist, solution[:,0], 'b', linewidth=lw)
#pyl.title('Temperature Graph, Time={}, Particle={}'.format(
#            pasrtimestep,particle), fontsize=16)
pyl.xlim(0,0.005)

pyl.figure(2)
pyl.ylabel('Index Value', fontsize=14)
pyl.plot(tlist, indexvalues, 'b', linewidth=lw)
#pyl.title('IA-Stiffness Index, Order = {}'.format(order), fontsize=16)
pyl.yscale('log')
#pyl.text(0.1,0.001,'dt = {}, Abs Error = {}, Rel Error = {}'.format(
#            dt,abserr,relerr))

pyl.show()

if savedata == 1:
    indexvalues = np.array(indexvalues)
    dataneeded = [indexvalues,tlist, solution, pasrtimestep, particle, order,
                  dt, abserr, relerr]
    np.save('IndexVals_Order5_{}'.format(dt), dataneeded)

finishtime = datetime.datetime.now()
print('Finish time: {}'.format(finishtime))
