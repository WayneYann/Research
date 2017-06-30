#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:54:13 2017

@author: andrewalferman
"""

filenum = '0'

import os as os
import numpy as np
import pyjacob as pyjacob
import pylab as pyl
import scipy as sci
import datetime


def firstderiv(state, time, press):
    """This function forces the integrator to use the right arguments."""
    '''PyJac requires the state to be given to it to have zeros for all the
    mass fractions that are considered to be part of the reaction but don't
    do anything.  So I'm putting in a workaround where those zeros are added
    back onto the state, then removed before returning the values back to the
    program.  This is extremely gimmicky and a bit of a waste, but it might be
    best to fix the problem with PyJac instead.
    '''
    # Need to make sure that N2 is at the end of the state array
    state = np.hstack((state[:-1],np.zeros(4),state[-1]))

    dy = np.zeros_like(state)
    pyjacob.py_dydt(time, press, state, dy)
    # When returning the derivative values, need to get rid of the zero values
    return np.hstack((dy[:-5],dy[-1]))


def jacobval(state, time, press):
    """This function forces the integrator to use the right arguments."""
    '''PyJac requires the state to be given to it to have zeros for all the
    mass fractions that are considered to be part of the reaction but don't
    do anything.  So I'm putting in a workaround where those zeros are added
    back onto the state, then removed before returning the values back to the
    program.  This is extremely gimmicky and a bit of a waste, but it might be
    best to fix the problem with PyJac instead.  This all is needed so that
    only nonzero eigenvalues of the matrix are returned.
    '''
    olen = len(state)

    new = np.hstack((state[:-1],np.zeros(4),state[-1])).copy()

    a = len(new) - 1
    jacobian = np.zeros(a**2)
    pyjacob.py_eval_jacobian(time, press, new, jacobian)
    #print(len(jacobian))
    jacobian = np.reshape(jacobian, (a,a))

#    return np.delete(np.delete(jacobian,range(olen,a),0),range(olen,a),1)
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
    method = 2

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

    # Obtain the shape of the jacobian and the dydx vector.  This will speed
    # up the weightednorm function (see function for details).  May not be
    # needed if a better method can be found with numpy arrays.  Note that
    # these two values are only dummy values thrown in to get the shape, they
    # are not included in the solution.
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
dt = 1.e-7
tstart = 0
tstop = 5*dt
tlist = np.arange(tstart, tstop+0.5*dt, dt)

# ODE Solver parameters
abserr = 1.0e-15
relerr = 1.0e-13

# Load the initial conditions from the PaSR files
pasrarrays = []
for i in range(9):
    filepath = os.path.join(os.getcwd(),'pasr_out_h2-co_' + filenum + '.npy')
    filearray = np.load(filepath)
    pasrarrays.append(filearray)

pasr = np.concatenate(pasrarrays, 1)

# Initialize the array of stiffness index values
numparticles = len(pasr[0,:,0])
numtsteps = len(pasr[:,0,0])
# Cheated a little here to make coding faster
numparams = 11
pasrstiffnesses = np.zeros((numtsteps,numparticles,numparams))
#pasrstiffnesses = np.zeros((numtsteps,numparticles))

# Loop through the PaSR file for initial conditions
for particle in range(numparticles):
    print(particle)
    for tstep in range(numtsteps):
#        print(tstep)
        # Get the initial condition.
        Y = pasr[tstep, particle, :].copy()

        # Rearrange the array for the initial condition
        press_pos = 2
        temp_pos = 1
        arraylen = len(Y)

        Y_press = Y[press_pos]
        Y_temp = Y[temp_pos]
        Y_species = Y[3:arraylen-4]
        Ys = np.hstack((Y_temp,Y_species))

        # Call the integrator
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

        #print('Code progress:')

        # Find the stiffness index across the range of the solution
        indexvalues = stiffnessindex(stiffnessparams, normweights,
                                     tlist, dt, solution, Y_press)

#        pasrstiffnesses[tstep,particle] = np.log(indexvalues[2])
        pasrstiffnesses[tstep,particle,:] = np.hstack((solution[2],indexvalues[2]))


# Create a mesh to plot on
#xcoords = np.arange(numparticles)
#ycoords = np.arange(numtsteps)
#xmesh, ymesh = np.meshgrid(xcoords,ycoords)
#
#for i in range(len(pasrstiffnesses[0,0,:])):
#    pyl.figure(i, figsize=(6, 4.5), dpi=600)
#    pyl.xlabel('Particle')
#    pyl.ylabel('Timestep')
#    if i == 10:
#        plot = pyl.contourf(xmesh,ymesh,np.log(pasrstiffnesses[:,:,i]),50)
#    else:
#        plot = pyl.contourf(xmesh,ymesh,pasrstiffnesses[:,:,i],50)
#    pyl.grid(b=True, which='both')
#    pyl.colorbar(plot)

pyl.figure(0,figsize=(6, 4.5), dpi=600)
pyl.xlabel('Temperature (K)')
pyl.ylabel('Stiffness Index')
pyl.yscale('log')
#pyl.xscale('log')
for i in range(1,10):
    pyl.figure(i,figsize=(6, 4.5), dpi=600)
    pyl.xlabel('Mass Fraction')
    pyl.ylabel('Stiffness Index')
    pyl.yscale('log')
#    pyl.xscale('log')
#pyl.grid(b=True, which='both')
for i in range(len(pasrstiffnesses[:,0,0])):
    for j in range(10):
        pyl.figure(j)
        pyl.scatter(pasrstiffnesses[i,:,j],
                    pasrstiffnesses[i,:,10], 1, 'B', '.')

for i in range(11):
    pyl.savefig('COH2_PaSR_Stiffness_Index_'+ str(filenum) +
                '_p' + str(i) + '.pdf')

finishtime = datetime.datetime.now()
print('Finish time: {}'.format(finishtime))
