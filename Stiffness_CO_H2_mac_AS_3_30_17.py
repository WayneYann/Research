#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:54:13 2017

@author: andrewalferman
"""

# filenum = '0'

import os as os
import numpy as np
import pyjacob as pyjacob
import pylab as pyl
import scipy as sci
import datetime
import time as timer


def firstderiv(state, time, press):
    """This function forces the integrator to use the right arguments."""
    # Need to make sure that N2 is at the end of the state array
    dy = np.zeros_like(state)
    pyjacob.py_dydt(time, press, state, dy)
    return dy


def jacobval(state, time, press):
    """This function forces the integrator to use the right arguments."""
    # Need to get rid of N2 because PyJac doesn't compute it.
    new = state[:-1]
    a = len(new)
    jacobian = np.zeros(a**2)
    pyjacob.py_eval_jacobian(time, press, new, jacobian)
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
    for i in range((len(vals) - 2), len(vals)):
        deriv.append((vals[i] - vals[i-1]) / dx)
    return deriv


def weightednorm(matrix, weights):
    """Weighted average norm function as defined in 1985 Shampine.  Takes a
    matrix and 2 weights and returns the maximum value (divided by wi) of the
    sum of each value in each row multiplied by wj.  Needs to be passed either
    a matrix of m x n dimensions where m,n > 1, or a column vector."""
    # Unpack the parameters
    wi, wj = weights

    # Initialize a list that will be called later to obtain the maximum value
    colsums = np.zeros(len(matrix))

    try:
        for i in range(len(matrix)):
            colsums += wj * np.abs(matrix[i])
        return np.max(colsums) / wi
    except TypeError:
        matrixcol = wj * np.abs(matrix)
        return np.sum(matrixcol)/wi


def stiffnessindex(sp, normweights, xlist, dx, solution, press):
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
    method = 1

    # Unpack the parameters
    tolerance, order, xi, gamma = sp

    # Obtain the derivative values for the derivative of order p
    dydxlist = []
    for i in range(len(solution)):
        dydxlist.append(firstderiv(solution[i,:],xlist[i],press))
    # Raise the derivative to the order we need it
    for i in range(order):
        dydxlist = derivcd4(dydxlist, dx)
    dydxlist = np.array(dydxlist)

    # Create a list to return for all the index values in a function
    indexlist = []

    # The weighted norm needs to be a single column, not a single row, so it
    # needs to be transposed.
    dydxlist = np.transpose(dydxlist)

    # Figure out some of the values that will be multiplied many times, so that
    # each computation only needs to happen once.
    exponent = 1./(order + 1)
    xiterm = ((np.abs(xi)**(-1 * exponent)) / np.abs(gamma))
    toleranceterm = tolerance**exponent

    # Actual computation of the stiffness index for the method specified.
    for i in range(len(solution)):
        jacobian = jacobval(solution[i,:],xlist[i],Y_press)
        if method == 1:
            index = toleranceterm *\
                    weightednorm(jacobian, normweights) *\
                    weightednorm(dydxlist[:,i], normweights)**(-1 * exponent) *\
                    xiterm
        else:
            index = toleranceterm *\
                    np.max(np.abs(np.linalg.eigvals(jacobian))) *\
                    weightednorm(dydxlist[:,i], normweights)**(-1 * exponent) *\
                    xiterm
        indexlist.append(index)
    indexlist = np.array(indexlist)
    return indexlist, dydxlist

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
dt = 1.e-7
tstart = 0
tstop = 5*dt
tlist = np.arange(tstart, tstop+0.5*dt, dt)

# ODE Solver parameters
abserr = 1.0e-15
relerr = 1.0e-13

# Load the initial conditions from the PaSR files
pasrarrays = []
print('Loading data...')
for i in range(9):
    filepath = os.path.join(os.getcwd(),'pasr_out_h2-co_' + str(i) + '.npy')
    filearray = np.load(filepath)
    pasrarrays.append(filearray)

pasr = np.concatenate(pasrarrays, 1)

# Initialize the array of stiffness index values
numparticles = len(pasr[0,:,0])
#numparticles = 50
#numtsteps = len(pasr[:,0,0])
numtsteps = 1
# Cheated a little here to make coding faster
numparams = 15
pasrstiffnesses = np.zeros((numtsteps,numparticles,numparams))
pasrstiffnesses2 = np.zeros((numtsteps,numparticles,numparams))

# Create vectors for that time how long it takes to compute stiffness index and
# the solution itself
solutiontimes, stiffcomptimes = [], []

# Loop through the PaSR file for initial conditions
print('Code progress:')
for particle in range(numparticles):
    print(particle)
    for tstep in range(numtsteps):
#        print(tstep)
        # Get the initial condition.
        #Y = pasr[tstep, particle, :].copy()
        Y = pasr[50, particle, :].copy()

        # Rearrange the array for the initial condition
        press_pos = 2
        temp_pos = 1
        arraylen = len(Y)

        Y_press = Y[press_pos]
        Y_temp = Y[temp_pos]
        Y_species = Y[3:arraylen]
        Ys = np.hstack((Y_temp,Y_species))

        # Put N2 to the last value of the mass species
        N2_pos = 9
        newarlen = len(Ys)
        Y_N2 = Ys[N2_pos]
        Y_x = Ys[newarlen - 1]
        for i in range(N2_pos,newarlen-1):
            Ys[i] = Ys[i+1]
        Ys[newarlen - 1] = Y_N2

        # Call the integrator and time it
        time0 = timer.time()
        solution = sci.integrate.odeint(firstderiv, # Call the dydt function
                                        # Pass it initial conditions
                                        Ys,
                                        # Pass it time steps to be evaluated
                                        tlist,
                                        # Pass any additional information needed
                                        args=(Y_press,),
                                        # Pass it the Jacobian (not sure if needed)
        #                                Dfun=jacobval,
                                        # Pass it the absolute and relative tolerances
                                        atol=abserr, rtol=relerr,
                                        # Print a message stating if it worked or not
                                        printmessg=0
                                        )
        time1 = timer.time()


        # Find the stiffness index across the range of the solution and time it
        indexvalues, derivatives = stiffnessindex(stiffnessparams, normweights,
                                                  tlist, dt, solution, Y_press)
        time2 = timer.time()

        solutiontimes.append(time1 - time0)
        stiffcomptimes.append(time2 - time1)

#        pasrstiffnesses[tstep,particle] = np.log(indexvalues[2])
        pasrstiffnesses[tstep,particle,:] = np.hstack((solution[2],indexvalues[2]))
        pasrstiffnesses2[tstep,particle,:] = np.hstack(
                (np.transpose(derivatives[:,2]),indexvalues[2]))

speciesnames = ['H', 'H$_2$', 'O', 'OH', 'H$_2$O', 'O$_2$', 'HO$_2$',
                'H$_2$O$_2$', 'Ar', 'He', 'CO', 'CO$_2$', 'N$_2$']


pyl.figure(0,figsize=(6, 4.5), dpi=400)
pyl.xlabel('Temperature (K)')
pyl.ylabel('Stiffness Index')
for i in range(1,14):
    pyl.figure(i,figsize=(6, 4.5), dpi=400)
    pyl.ylabel('Stiffness Index', fontsize = 12)
    pyl.xlabel(speciesnames[i-1] + ' Mass Fraction, 2nd derivative',
               fontsize = 12)

for i in range(14):
    for j in range(len(pasrstiffnesses[0,:,0])):
        pyl.figure(i)
        plot = pyl.hexbin(abs(pasrstiffnesses2[:,j,i]),pasrstiffnesses2[:,j,14],
                          yscale='log',
                          xscale='log',
                          bins='log',
                          cmap='Blues',
                          gridsize=75,
                          mincnt=1
                          )
    cb = pyl.colorbar(plot)
    label = cb.set_label('log$_{10}$ |count + 1|', fontsize=12)

if savedata == 1:
    for i in range(14):
        pyl.figure(i)
        pyl.savefig('COH2_PaSR_SI_p' + str(i) + '.pdf')

finishtime = datetime.datetime.now()
print('Finish time: {}'.format(finishtime))
"""

ratios = []
for i in range(len(solutiontimes)):
    ratios.append(stiffcomptimes[i] / solutiontimes[i])

pyl.figure(0, figsize=(6,4.5), dpi=400)
pyl.xlim(0,numparticles)
#pyl.ylim(0,)
pyl.xlabel('Particle Number')
pyl.ylabel('Integration Time')
pyl.scatter(range(numparticles), solutiontimes)

pyl.figure(1, figsize=(6,4.5), dpi=400)
pyl.xlim(0,numparticles)
#pyl.ylim(0,)
pyl.xlabel('Particle Number')
pyl.ylabel('Stiffness Index Computation Time')
pyl.scatter(range(numparticles), stiffcomptimes)

pyl.figure(2, figsize=(6,4.5), dpi=400)
pyl.xlim(0,numparticles)
#pyl.ylim(0,)
pyl.xlabel('Particle Number')
pyl.ylabel('Ratio')
pyl.scatter(range(numparticles), ratios)

pyl.show()
"""