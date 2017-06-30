#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:54:13 2017

@author: andrewalferman
"""

filenum = '_1'


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


def sqmatrixreader(matrix):
    """Prints the values of a square matrix"""
    dim = len(matrix[0])
    dimsused = 0
    for i in range(dim):
        print('New row')
        for j in range(dim):
            print(matrix[i,j])
            if round(matrix[i,j], 15) != 0:
                dimsused += 1
    return dimsused


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
#    print('Finding first derivative values...')
    for i in range(len(solution)):
        dydxlist.append(firstderiv(solution[i,:],xlist[i],press))
#        if i % 500000 == 0:
#            print('dydxlist: {}'.format(i))
    # Raise the derivative to the order we need it
    for i in range(order):
#        print('Finding derivative of order {}...'.format(i+2))
        dydxlist = derivcd4(dydxlist, dx)
    dydxlist = np.array(dydxlist)

    # Create a list to return for all the index values in a function
    indexlist = []

    # The weighted norm needs to be a single column, not a single row, so it
    # needs to be transposed.
#    print('Transposing dydx vector...')
    dydxlist = np.transpose(dydxlist)

    # Figure out some of the values that will be multiplied many times, so that
    # each computation only needs to happen once.
    exponent = 1./(order + 1)
    xiterm = ((np.abs(xi)**(-1 * exponent)) / np.abs(gamma))
    toleranceterm = tolerance**exponent

    # Create a list to return for all the ratio values in a function
#    stiffratios = []

    # A few diagnostic lines so that we can play with the Jacobian without
    # running the whole code.
#    jacone = jacobval(solution[0,:],xlist[0],Y_press)
#    print([np.abs(j.real) for j in np.linalg.eigvals(jacone)
#                    if np.abs(j.real) != 0])

    # Obtain the shape of the jacobian and the dydx vector.  This will speed
    # up the weightednorm function (see function for details).  May not be
    # needed if a better method can be found with numpy arrays.  Note that
    # these two values are only dummy values thrown in to get the shape, they
    # are not included in the solution.
    for i in range(len(solution)):
        jacobian = jacobval(solution[i,:],xlist[i],Y_press)
#        eigvalsr = [np.abs(j.real) for j in np.linalg.eigvals(jacobian)
#                    if np.abs(j.real) != 0]
#        stiffratios.append(max(eigvalsr)/min(eigvalsr))
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
#        if i % 500000 == 0:
#            print('Index list: {}'.format(i))
    indexlist = np.array(indexlist)
#    return indexlist, stiffratios, jacone
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

# Load the initial conditions from the PaSR file
filepath = os.path.join(os.getcwd(),'pasr_out_h2-co' + filenum + '.npy')
pasr = np.load(filepath)

# Initialize the array of stiffness index values
numparticles = len(pasr[0,:,0])
numtsteps = len(pasr[:,0,0])
# Cheated a little here to make coding faster
numparams = 11
pasrstiffnesses = np.zeros((numtsteps,numparticles,numparams))

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

        N2_pos = 9
        newarlen = len(Ys)
        Y_N2 = Ys[N2_pos]
        Y_x = Ys[newarlen - 1]
        Ys[newarlen - 1] = Y_N2
        Ys[N2_pos] = Y_x

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

        pasrstiffnesses[tstep,particle,:] = np.hstack((solution[3],indexvalues[3]))


        ## Plot the solution.  This loop just sets up some of the parameters that we
        ## want to modify in all of the plots.
        #for p in range(1,4):
        #    pyl.figure(p, figsize=(6, 4.5), dpi=600)
        #    pyl.xlabel('Time (s)', fontsize=14)
        #    pyl.xlim(tstart,tstop)
        #    pyl.grid(True)
        #    pyl.hold(True)

        ##Set the linewidth to make plotting look nicer
        #lw = 1
        #
        ## Set all of the parameters that we want to apply to each plot specifically.
        #pyl.figure(1)
        #pyl.ylabel('Temperature', fontsize=14)
        #pyl.plot(tlist, solution[:,0], 'b', linewidth=lw)
        ##pyl.title('Temperature Graph, Time={}, Particle={}'.format(
        ##            pasrtimestep,particle), fontsize=16)
        ##pyl.xlim(0,0.005)
        #
        #pyl.figure(2)
        #pyl.ylabel('Index Value', fontsize=14)
        #pyl.plot(tlist, indexvalues, 'b', linewidth=lw)
        ##pyl.title('IA-Stiffness Index, Order = {}'.format(order), fontsize=16)
        #pyl.yscale('log')
        ##pyl.text(0.1,0.001,'dt = {}, Abs Error = {}, Rel Error = {}'.format(
        ##            dt,abserr,relerr))
        #
        #pyl.figure(3)
        #pyl.ylabel('Stiffness Ratio')
        #pyl.plot(tlist, stiffratio, 'b', linewidth=lw)
        #pyl.yscale('log')
        #
        #pyl.show()
        #
        #if savedata == 1:
        #    indexvalues = np.array(indexvalues)
        #    dataneeded = [indexvalues,tlist, solution, pasrtimestep, particle, order,
        #                  dt, abserr, relerr]
        #    np.save('IndexVals_Order5_{}'.format(dt), dataneeded)

# Create a mesh to plot on
xcoords = np.arange(numparticles)
ycoords = np.arange(numtsteps)
xmesh, ymesh = np.meshgrid(xcoords,ycoords)

pyl.figure(0, figsize=(6, 4.5), dpi=600)
pyl.xlabel('Particle')
pyl.ylabel('Timestep')
plot = pyl.contourf(xmesh,ymesh,pasrstiffnesses[:,:,10],50)
pyl.grid(b=True, which='both')
pyl.colorbar(plot)

#pyl.savefig('COH2_PaSR_Stiffness_Index'+ filenum + '.png')






#pyl.show()

finishtime = datetime.datetime.now()
print('Finish time: {}'.format(finishtime))
