#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:54:13 2017

@author: andrewalferman
"""

import numpy as np
import pyjacob as pyjacob
import pylab as pyl
import scipy as sci


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
    n = len(vals)
    for i in range(n):
        if (i > 2 and i < n - 2):
            deriv.append((-1*vals[i-2] + 8*vals[i-1] + 8*vals[i+1] -\
                          vals[i+2]) / (12*dx))
        elif i < 2:
            deriv.append((-3*vals[i] + 4*vals[i+1] - vals[i+2]) / (2*dx))
        else:
            deriv.append((3*vals[i] - 4*vals[i-1] + vals[i-2]) / (2*dx))
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

# Define the range of the computation
tstart = 0
tstop = 5.e-3
dt = 1.e-8
tlist = np.arange(tstart, tstop+dt, dt)

t0 = 0.050

# ODE Solver parameters
abserr = 1.0e-12
relerr = 1.0e-10

# Load the initial conditions from the PaSR file
ic = np.load('/Users/andrewalferman/Desktop/Research/pasr_out_h2-co_0.npy')

# Select a point out of the PaSR file as an initial condition
timestep = 896
particle = 50

# Get the initial condition
Y = ic[timestep, particle, :].copy()

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

# Obtain the derivative values
dydtlist = []
jaclist = []
for i in range(len(solution)):
    dydtlist.append(firstderiv(solution[i,:],tlist[i],Y_press))
    jaclist.append(jacobval(solution[i,:],tlist[i],Y_press))
dydtlist = np.array(dydtlist)
jaclist = np.array(jaclist)
d2list = derivcd4(dydtlist, dt)
d2list = np.array(d2list)

# Find the stiffness index across the range of the solution by using the above
# functions to get the Jacobian matrix and second derivative
indexvalues = []
for i in range(len(solution)):
    localstiffness = stiffnessindex(stiffnessparams, jaclist[i],
                                    d2list[i], normweights)
    indexvalues.append(localstiffness)

# Plot the solution.  This loop just sets up some of the parameters that we
# want to modify in all of the plots.
for p in range(1,3):
    pyl.figure(p, figsize=(6, 4.5), dpi=400)
    pyl.xlabel('Time (s)')
    pyl.xlim(tstart,tstop)
    pyl.grid(True)
    pyl.hold(True)

#Set the linewidth to make plotting look nicer
lw = 1

# Set all of the parameters that we want to apply to each plot specifically.
pyl.figure(1)
pyl.ylabel('Temperature')
pyl.plot(tlist, solution[:,0], 'b', linewidth=lw)
pyl.title('Temperature Graph, Time={}, Particle={}'.format(timestep,particle))

pyl.figure(2)
pyl.ylabel('Index Value')
pyl.plot(tlist, indexvalues, 'b', linewidth=lw)
pyl.title('IA-Stiffness Index, Order = {}'.format(order))
pyl.yscale('log')

pyl.show()
