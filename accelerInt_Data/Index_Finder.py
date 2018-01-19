#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 2017

@author: andrewalferman
"""

import os as os
import numpy as np
import csv as csv
import pyjacob as pyjacob
from scipy.integrate import ode


def firstderiv(time, state, press):
    """Force the integrator to use the right arguments."""
    # Need to make sure that N2 is at the end of the state array
    dy = np.zeros_like(state)
    pyjacob.py_dydt(time, press, state, dy)
    return dy


def jacobval(time, state, press):
    """Force the integrator to use the right arguments."""
    # Need to get rid of N2 because PyJac doesn't compute it.
    # new = state[:-1]
    # print('Jacobian function called.')
    a = len(state)
    jacobian = np.zeros(a**2)
    # Obtain the jacobian from pyJac
    pyjacob.py_eval_jacobian(time, press, state, jacobian)
    jacobian = np.reshape(jacobian, (a, a))
    # Re-add the zeros back in
    # jacobian = np.insert(jacobian, a, np.zeros(a), axis=1)
    # jacobian = np.vstack((jacobian, np.zeros(a+1)))
    return jacobian


def loadpasrdata():
    """Load the initial conditions from the full PaSR file."""
    print('Loading data...')
    filepath = '/Users/andrewalferman/Desktop/Research/GRI_Mech_3/ch4_full_pasr_data.npy'
    return np.load(filepath)


def weightednorm(matrix, weights):
    """Weighted average norm function as defined in 1985 Shampine.

    Takes a matrix and 2 weights and returns the maximum value (divided by
    wi) of the sum of each value in each row multiplied by wj.  Needs to be
    passed either a matrix of m x n dimensions where m,n > 1, or a column
    vector.
    """
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


def stiffnessindex(xlist, solution, dfun, jfun, *args, **kwargs):
    """Determine the stiffness index across a solution vector.

    Function that uses stiffness parameters, the local Jacobian matrix,
    and a vector of the local function values to determine the local stiffness
    index as defined in 1985 Shampine.

    This function has been modified to only use 3 solution values.
    """
    SIparams = {'method': 2,
                'gamma': 1,
                'xi': 1,
                'order': 1,
                'tolerance': 1,
                'wi': 1,
                'wj': 1
                }

    for key, value in kwargs.items():
        SIparams[key] = value

    funcparams = []
    for arg in args:
        funcparams.append(arg)

    # Method 2 uses the weighted norm of the Jacobian, Method 1 uses the
    # spectral radius of the Jacobian.
    method = SIparams['method']
    # Stiffness index parameter values
    gamma = SIparams['gamma']
    xi = SIparams['xi']
    order = SIparams['order']
    tolerance = SIparams['tolerance']
    # Weighted norm parameters
    wi = SIparams['wi']
    wj = SIparams['wj']

    normweights = wi, wj

    # Obtain the derivative values for the derivative of order p
    dx = xlist[1] - xlist[0]
    dydxlist = []
    for i in range(len(xlist)):
        dydxlist.append(dfun(xlist[i], solution[i, :], funcparams[0]))
    # Raise the derivative to the order we need it
    for i in range(order):
        dydxlist = derivbdf2(dydxlist, dx)
    dydxlist = np.array(dydxlist)

    # Create a list to return for all the index values in a function
    indexlist = []

    # Figure out some of the values that will be multiplied many times, so that
    # each computation only needs to happen once.
    exponent = 1. / (order + 1)
    xiterm = ((np.abs(xi)**(-1 * exponent)) / np.abs(gamma))
    toleranceterm = tolerance**exponent

    # Actual computation of the stiffness index for the method specified.
    for i in range(len(xlist)):
        jacobian = jfun(xlist[i], solution[i, :], funcparams[0])
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
    return indexlist


def derivbdf2(vals, dx):
    """Find the derivative using the 2nd order forward difference formula."""
    deriv = []
    deriv.append((-3 * vals[0] + 4 * vals[1] - vals[2]) / (2 * dx))
    deriv.append((-1 * vals[0] + vals[2]) / (2 * dx))
    deriv.append((3 * vals[2] - 4 * vals[1] + vals[0]) / (2 * dx))
    return deriv


def rearrangepasr(Y, N2_pos, useN2):
    """Rearrange the PaSR data so it works with pyJac."""
    press_pos = 2
    temp_pos = 1
    arraylen = len(Y)

    Y_press = Y[press_pos]
    Y_temp = Y[temp_pos]
    Y_species = Y[3:arraylen]
    Ys = np.hstack((Y_temp, Y_species))

    # Put N2 to the last value of the mass species
    newarlen = len(Ys)
    Y_N2 = Ys[N2_pos]
    # Y_x = Ys[newarlen - 1]
    for i in range(N2_pos, newarlen - 1):
        Ys[i] = Ys[i + 1]
    Ys[newarlen - 1] = Y_N2
    if useN2:
        initcond = Ys
    else:
        initcond = Ys[:-1]
    return initcond, Y_press


def CSPtimescales(jacobian):
    """Find the timescales used in a computational singular perturbation."""
    ts = []
    for i in range(np.shape(jacobian)[0]):
        # Only append if the value of the jacobian is greater than the
        # machine epsilon for double floating point precision
        if jacobian[i, i] > 2.22045e-16:
            ts.append(1/jacobian[i, i])
        else:
            # These values assumed to be equal to zero
            ts.append(-1)
    ts = np.array(ts)
    return ts


# Set up some of the integration parameters
savevalues = True
usejac = False
intj = None
intmode = 'vode'
RHSfunction = firstderiv
EQjac = jacobval

# Define the range of the computation
dt = 1e-10
tstop = 2*dt
tstart = 0.0

# Create the list of times to compute
tlist = np.arange(tstart, tstop + 0.5 * dt, dt)

# Load the initial conditions from the PaSR files
pasr = loadpasrdata()
numparticles = len(pasr[:, 0])
particlelist = range(numparticles)
# particlelist = range(10)

# Array for capturing the stiffness index at t=0 for each particle
indexes, CSPs = [], []

solver = ode(RHSfunction, jac=intj ).set_integrator(intmode,
                                                    method='bdf',
                                                    nsteps=1e10,
                                                    # atol=abserr,
                                                    # rtol=relerr,
                                                    with_jacobian=usejac,
                                                    first_step=dt,
                                                    # min_step=dt,
                                                    max_step=dt
                                                    )

# Loop through all particles in the PaSR
print('Looping through particles...')
for particle in particlelist:
    if particle % 1000 == 0:
        print(particle)
    # Get the initial condition and rearrange it for pyJac/integration
    Y = pasr[particle, :].copy()
    initcond, RHSparam = rearrangepasr(Y, 50, True)
    # Initialize the solver
    solver.set_initial_value(initcond, tstart)
    solver.set_f_params(RHSparam)
    solver.set_jac_params(RHSparam)
    solver._integrator.iwork[2] = -1

    # Array for capturing the solution values for stiffness index computation
    solutions = [initcond]

    timescales = CSPtimescales(jacobval(solver.t, solver.y, RHSparam))
    CSP = min([i for i in timescales if i > 0]) / max(timescales)
    if CSP <= 0.0:
        print(min([i for i in timescales if i > 0]))
        print(max(timescales))
        raise Exception('Something dun fucked up.')
    CSPs.append(CSP)


    # Integrate the ODE across all steps
    while solver.successful() and solver.t <= tstop:
        tnext = solver.t + dt
        solver.integrate(tnext)
        solutions.append(solver.y)
    solutions = np.array(solutions)
    indices_local = stiffnessindex(tlist, solutions, RHSfunction, EQjac,
                                   RHSparam)
    indexes.append(indices_local[0])

if savevalues:
    print('Writing values to spreadsheet...')
    with open('GRI_PaSR_index_values.csv', 'w') as myfile:
        wr = csv.writer(myfile, delimiter=',')
        for i in range(len(indexes)):
            if i % 1000 == 0:
                print(i)
            wr.writerow([indexes[i], CSPs[i]])
