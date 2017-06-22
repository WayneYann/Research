#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:54:13 2017

@author: andrewalferman
"""

import matplotlib
matplotlib.use('Agg')
import os as os
import numpy as np
import pyjacob as pyjacob
import pylab as pyl
# import scipy as sci
import datetime
import time as timer

# from scipy.integrate import odeint
from scipy.integrate import ode

pyl.ioff()


def firstderiv(state, time, press):
    """Force the integrator to use the right arguments."""
    # Need to make sure that N2 is at the end of the state array
    dy = np.zeros_like(state)
    pyjacob.py_dydt(time, press, state, dy)
    return dy


def jacobval(state, time, press):
    """Force the integrator to use the right arguments."""
    # Need to get rid of N2 because PyJac doesn't compute it.
    new = state[:-1]
    a = len(new)
    jacobian = np.zeros(a**2)
    pyjacob.py_eval_jacobian(time, press, new, jacobian)
    jacobian = np.reshape(jacobian, (a, a))
    return jacobian


def derivcd4(vals, dx):
    """Take the derivative of a series using 4th order central differencing."""
    """Given a list of values at equally spaced points, returns the first
    derivative using the fourth order central difference formula, or forward/
    backward differencing at the boundaries."""
    deriv = []
    for i in range(2):
        deriv.append((-3 * vals[i] + 4 * vals[i + 1] - vals[i + 2]) / (2 * dx))
    for i in range(2, len(vals) - 2):
        deriv.append((-1 * vals[i + 2] + 8 * vals[i + 1] - 8 * vals[i - 1] +
                      vals[i - 2]) / (12 * dx))
    for i in range((len(vals) - 2), len(vals)):
        deriv.append((3 * vals[i] - 4 * vals[i - 1] + vals[i - 2]) / 2 * dx)
    return deriv


def weightednorm(matrix, weights):
    """Weighted average norm function as defined in 1985 Shampine."""
    """Takes a matrix and 2 weights and returns the maximum value (divided by
    wi) of the sum of each value in each row multiplied by wj.  Needs to be
    passed either a matrix of m x n dimensions where m,n > 1, or a column
    vector."""
    # Unpack the parameters
    wi, wj = weights

    # Initialize a list that will be called later to obtain the maximum value
    colsums = np.zeros(len(matrix))

    # Try loop used because numpy didn't seem to like 1D arrays for the
    # weighted norm
    try:
        for i in range(len(matrix)):
            colsums += wj * np.abs(matrix[i])
        return np.max(colsums) / wi
    except TypeError:
        matrixcol = wj * np.abs(matrix)
        return np.sum(matrixcol) / wi


def stiffnessindex(sp, normweights, xlist, dx, solution, press):
    """Determine the local stiffness index."""
    '''Function that uses stiffness parameters (sp), the local Jacobian matrix,
    and a vector of the local function values to determine the local stiffness
    index as defined in 1985 Shampine.

    Future optimizations:
        1.  Use a different integrator that saves the values of the derivative
        as it goes so that we don't need to calculate each derivative twice.
        Same goes with the Jacobian, it would be much faster to evaluate
        it on the fly.
        2.  Implement this index in the integrator itself so that it makes all
        of the values as the integration goes.  This would eliminate the need
        to save the dydx list beyond a few variables that would be needed to
        compute the higher level derivatives.
    '''
    # Method 2 uses the weighted norm of the Jacobian, Method 1 uses the
    # spectral radius of the Jacobian.
    method = 1

    # Unpack the parameters
    tolerance, order, xi, gamma = sp

    # Obtain the derivative values for the derivative of order p
    dydxlist = []
    for i in range(len(solution)):
        dydxlist.append(firstderiv(solution[i, :], xlist[i], press))
    # Raise the derivative to the order we need it
    for i in range(order):
        dydxlist = derivcd4(dydxlist, dx)
    dydxlist = np.array(dydxlist)

    # Create a list to return for all the index values in a function
    indexlist = []

    # Figure out some of the values that will be multiplied many times, so that
    # each computation only needs to happen once.
    exponent = 1. / (order + 1)
    xiterm = ((np.abs(xi)**(-1 * exponent)) / np.abs(gamma))
    toleranceterm = tolerance**exponent

    # Actual computation of the stiffness index for the method specified.
    for i in range(len(solution)):
        jacobian = jacobval(solution[i, :], xlist[i], Y_press)
        if method == 1:
            eigenvalues = np.linalg.eigvals(jacobian)
            index = toleranceterm *\
                np.max(np.abs(eigenvalues)) *\
                weightednorm(dydxlist[i, :], normweights)**(-1 * exponent) *\
                xiterm
        else:
            index = toleranceterm *\
                weightednorm(jacobian, normweights) *\
                weightednorm(dydxlist[i, :], normweights)**(-1 * exponent) *\
                xiterm
        indexlist.append(index)
    indexlist = np.array(indexlist)
    return indexlist, dydxlist


# Finding the current time to time how long the simulation takes
starttime = datetime.datetime.now()
print('Start time: {}'.format(starttime))

savedata = 0
savefigures = 1
figformat = 'png'

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
tstop = 0.05
tlist = np.arange(tstart, tstop + 0.5 * dt, dt)

# ODE Solver parameters
abserr = 1.0e-15
relerr = 1.0e-13

# Load the initial conditions from the PaSR files
pasrarrays = []
print('Loading data...')
for i in range(9):
    filepath = os.path.join(os.getcwd(), 'pasr_out_h2-co_' + str(i) + '.npy')
    filearray = np.load(filepath)
    pasrarrays.append(filearray)

pasr = np.concatenate(pasrarrays, 1)

# Initialize the array of stiffness index values
numparticles = len(pasr[0, :, 0])
numtsteps = len(pasr[:, 0, 0])

# Cheated a little here and entered the number of variables to code faster
numparams = 15
# pasrstiffnesses = np.zeros((numtsteps, numparticles))
pasrstiffnesses2 = np.zeros((numtsteps, numparticles, numparams))

# Create vectors for that time how long it takes to compute stiffness index and
# the solution itself
solutiontimes, stiffcomptimes = [], []
stiffvals = []

expeigs = np.zeros((numtsteps, numparticles))

# Loop through the PaSR file for initial conditions
print('Code progress:')
for particle in [92]:
    # print(particle)
    for tstep in [4]:
        #        print(tstep)
        # Get the initial condition.
        Y = pasr[tstep, particle, :].copy()

        # Rearrange the array for the initial condition
        press_pos = 2
        temp_pos = 1
        arraylen = len(Y)

        Y_press = Y[press_pos]
        Y_temp = Y[temp_pos]
        Y_species = Y[3:arraylen]
        Ys = np.hstack((Y_temp, Y_species))

        # Put N2 to the last value of the mass species
        N2_pos = 9
        newarlen = len(Ys)
        Y_N2 = Ys[N2_pos]
        Y_x = Ys[newarlen - 1]
        for i in range(N2_pos, newarlen - 1):
            Ys[i] = Ys[i + 1]
        Ys[newarlen - 1] = Y_N2

        # Call the integrator and time it
        solution = []
        curstate = Ys[:]
        currentt = tstart

        # Specify the integrator
        solver = ode(firstderiv  # ,
                     # jac=jacobval
                     ).set_integrator('vode',
                                      method='bdf')

        # intrange = np.arange(currentt, currentt + dt, dt)

        # Integrate the ODE across all steps
        while solver.successful() and solver.t <= tstop:
            time0 = timer.time()
            solver.set_initial_value(curstate,
                                     solver.t
                                     ).set_f_params(curstate, solver.t, Y_press)
            solver.integrate(solver.t + dt)
            curstate = solver.y
            time1 = timer.time()
            solution.append(curstate)
            solutiontimes.append(time1 - time0)

        # Convert the solution to an array for ease of use.  Maybe just using
        # numpy function to begin with would be faster?
        solution = np.array(solution)
        # Find the stiffness index across the range of the solution and time it
        time2 = timer.time()
        indexvalues, derivatives = stiffnessindex(stiffnessparams, normweights,
                                                  tlist, dt, solution, Y_press)
        time3 = timer.time()
        # This statement intended to cut back on the amount of data processed
        # derivatives = derivatives[2]

        stiffcomptimes.append(time3 - time2)

        # stiffvals.append(indexvalues[2])

        # Commented old code for the maximum eigenvalue or CEMA analysis
        # expeigs[tstep,particle] = np.log10(maxeig)
        # Commented old code for just figuring out the PaSR stiffness values
        # pasrstiffnesses[tstep, particle] = np.log10(indexvalues[2])
        # pasrstiffnesses[tstep,particle,:] = np.hstack((solution[2],
        #                                               indexvalues[2]))
        # This variable includes the values of the derivatives
        # pasrstiffnesses2[tstep, particle, :] = np.hstack(
        #    (derivatives, indexvalues[2]))

speciesnames = ['H', 'H$_2$', 'O', 'OH', 'H$_2$O', 'O$_2$', 'HO$_2$',
                'H$_2$O$_2$', 'Ar', 'He', 'CO', 'CO$_2$', 'N$_2$']

# Plot the results

# Clear all previous figures and close them all
for i in range(15):
    pyl.figure(i)
    pyl.clf()
pyl.close('all')

# print('Solution[:, 0] shape:')
# print(np.shape(solution[:, 0]))
# print('tlist shape:')
# print(np.shape(tlist[1: len(solution[:, 0])]))

# Plot the solution of the temperature
pyl.figure(0)
pyl.xlabel('Time (sec)')
pyl.ylabel('Temperature (K)')
pyl.plot(tlist[: len(solution[:, 0])], solution[:, 0])
if savefigures == 1:
    pyl.savefig('Autoignition_Temperature.' + figformat)

# Plot the time per integration
pyl.figure(1)
pyl.xlabel('Time (sec)')
pyl.ylabel('Integration time (sec)')
pyl.plot(tlist[: len(solution[:, 0])], solutiontimes)
if savefigures == 1:
    pyl.savefig('Autoignition_Integration_Times.' + figformat)

# Plot the stiffness index vs. time
# Plot the time per integration
pyl.figure(1)
pyl.xlabel('Time (sec)')
pyl.ylabel('Stiffness Index')
pyl.yscale('log')
pyl.plot(tlist[: len(solution[:-3, 0])], indexvalues[:-3])
if savefigures == 1:
    pyl.savefig('Autoignition_Stiffness_Indes.' + figformat)

"""
# Plot all of the 2nd derivatives vs stiffness index
for i in range(14):
    for j in range(len(pasrstiffnesses2[0, :, 0])):
        pyl.figure(i)
        # Label all of the x axes
        if i == 0:
            pyl.xlabel('Temperature, 2nd derivative')
            pyl.ylabel('Stiffness Index')
        else:
            pyl.ylabel('Stiffness Index', fontsize=12)
            pyl.xlabel(speciesnames[i - 1] + ' Mass Fraction, 2nd derivative',
                       fontsize=12)

        # Some of the derivatives may come out to zero, so we need to mask it
        for k in range(len(pasrstiffnesses2[:, j, i])):
            if pasrstiffnesses2[k, j, i] == 0:
                # Set values to None so that the plotter skips over them
                pasrstiffnesses2[k, j, i] = None
                pasrstiffnesses2[k, j, 14] = None

        # Hexbin plot used for sake of ease and making the files smaller
        plot = pyl.hexbin(abs(pasrstiffnesses2[:, j, i]),
                          pasrstiffnesses2[:, j, 14],
                          yscale='log',
                          xscale='log',
                          bins='log',
                          cmap='Blues',
                          gridsize=50,
                          mincnt=1
                          )
    cb = pyl.colorbar(plot)
    label = cb.set_label('log$_{10}$ |count + 1|', fontsize=12)

if savefigures == 1:
    for i in range(14):
        pyl.figure(i)
        pyl.savefig('COH2_PaSR_SI_p' + str(i) + figformat)

finishtime = datetime.datetime.now()
print('Finish time: {}'.format(finishtime))
"""

"""
# Ratios of the stiffness computation times to integration times
ratios = []
for i in range(len(solutiontimes)):
    ratios.append(stiffcomptimes[i] / solutiontimes[i])

# Average stiffness computation and solution times
datanum = len(solutiontimes)
solavg = 0.0
stiffavg = 0.0
for i in range(len(solutiontimes)):
    solavg += solutiontimes[i]
    stiffavg += stiffcomptimes[i]
solavg = 1000. * (solavg / datanum)
stiffavg = 1000. * (stiffavg / datanum)
print("Average integration time (ms): {:.7f}".format(solavg))
print("Average SI computation time (ms): {:.7f}".format(stiffavg))
print("Maximum integration time (ms): {:.7f}".format(
    max(solutiontimes) * 1000.))
print("Maximum SI computation time (ms): {:.7f}".format(max(stiffcomptimes)
                                                        * 1000.))

# Plot of integration times vs. computation number
pyl.figure(0)
pyl.xlim(0, datanum)
pyl.ylim(0, max(solutiontimes))
pyl.xlabel('Computation Number')
pyl.ylabel('Integration Time')
pyl.scatter(range(datanum), solutiontimes, 0.1)
if savefigures == 1:
    pyl.savefig('Integration_Times.' + figformat)

# Plot of stiffness computation times vs. computation number
pyl.figure(1)
pyl.xlim(0, datanum)
pyl.ylim(0, max(stiffcomptimes))
pyl.xlabel('Computation Number')
pyl.ylabel('Stiffness Index Computation Time')
pyl.scatter(range(datanum), stiffcomptimes, 0.1)
if savefigures == 1:
    pyl.savefig('Stiff_Comp_Times.' + figformat)

# Plot of ratio of stiffness computation times vs. integration times
pyl.figure(2)
pyl.xlim(0, datanum)
# pyl.ylim(0,)
pyl.xlabel('Particle Number')
pyl.ylabel('Ratio')
pyl.scatter(range(datanum), ratios, 0.1)
if savefigures == 1:
    pyl.savefig('Stiff_Comp_Ratios.' + figformat)

# Plot of stiffness computation times vs. stiffness index
pyl.figure(3)
pyl.ylabel('Stiffness Index ')
pyl.xlabel('Stiffness Index Computation Time')
pyl.xlim(0., max(solutiontimes))
pyl.ylim(0., max(stiffcomptimes))
pyl.scatter(stiffcomptimes, stiffvals, 0.1)
if savefigures == 1:
    pyl.savefig('Stiffcomp_Stiffvals.' + figformat)

# Plot of stiffness computation times vs. stiffness index
pyl.figure(4)
pyl.ylabel('Stiffness Index ')
pyl.xlabel('Integration Time')
pyl.xlim(0., max(solutiontimes))
pyl.ylim(0., max(stiffvals))
pyl.scatter(solutiontimes, stiffvals, 0.1)
if savefigures == 1:
    pyl.savefig('Int_Stiffvals1.' + figformat)
"""

"""
# Plot the stiffness at every point in the PaSR simulation
# Create a mesh to plot on
xcoords = np.arange(numparticles)
ycoords = np.arange(numtsteps)
xmesh, ymesh = np.meshgrid(xcoords, ycoords)

pyl.figure(3)
# pyl.xlim(0,numparticles)
# pyl.ylim(0,numtsteps)
pyl.xlabel('Particle Number')
pyl.ylabel('PaSR Timestep')
plot = pyl.contourf(xmesh, ymesh, pasrstiffnesses, 50)
pyl.grid(b=True, which='both')
cb = pyl.colorbar(plot)
label = cb.set_label('log$_{10}$ (Stiffness Index)')
if savefigures == 1:
    pyl.savefig('PaSR_Range_Stiffness_Index.' + figformat)
"""
# pyl.close('all')
# pyl.show()
