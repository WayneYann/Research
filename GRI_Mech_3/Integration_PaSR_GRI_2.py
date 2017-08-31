#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 04 10:44 2017

@author: andrewalferman
"""

import os as os
import numpy as np
import pyjacob as pyjacob
# import scipy as sci
import datetime
import time as timer
import warnings

# from scipy.integrate import odeint
from scipy.integrate import ode

global functioncalls


def firstderiv(time, state, press):
    """Force the integrator to use the right arguments."""
    # Need to make sure that N2 is at the end of the state array
    dy = np.zeros_like(state)
    pyjacob.py_dydt(time, press, state, dy)
    global functioncalls
    functioncalls += 1
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


def jacvdp(x, y, eta):
    """Find the local Jacobian matrix of the Van der Pol equation."""
    return np.array([[0., 1.], [-1. - 2*y[0]*y[1]*eta, eta-eta*y[0]**2]])


def dydx(x, y, eta):
    """Find the local vector of the first derivative of the Van der Pol eqn."""
    # Unpack the y vector
    y1 = y[0]
    y2 = y[1]

    # Create dydx vector (y1', y2')
    f = np.array([y2, eta*y2 - y1 - eta*y2*y1**2.])
    # print(f)
    return f


def d2ydx2(x, y, eta):
    """Find the local vector of the 2nd derivative of the Van der Pol eqn."""
    # Unpack the y vector
    y1 = y[0]
    y2 = y[1]

    # Create vector of the second derivative
    y2prime = eta*y2 - y1 - eta*y2*y1**2.
    f = np.array([y2prime, eta*y2prime - y2 - 2*eta*y1*y2 - eta*y2prime*y1**2])
    return f


def derivcd4(vals, dx):
    """Take the derivative of a series using 4th order central differencing.

    Given a list of values at equally spaced points, returns the first
    derivative using the fourth order central difference formula, or forward/
    backward differencing at the boundaries.
    """
    deriv = []
    for i in range(2):
        deriv.append((-3 * vals[i] + 4 * vals[i + 1] - vals[i + 2]) / (2 * dx))
    for i in range(2, len(vals) - 2):
        deriv.append(((-1 * vals[i + 2]) + (8 * vals[i + 1]) -
                     (8 * vals[i - 1]) + vals[i - 2]) /
                     (12 * dx)
                     )
    for i in range((len(vals) - 2), len(vals)):
        deriv.append((3 * vals[i] - 4 * vals[i - 1] + vals[i - 2]) / 2 * dx)
    return deriv


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
    for i in range(len(solution)):
        dydxlist.append(dfun(xlist[i], solution[i, :], funcparams[0]))
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
    return indexlist  # , dydxlist


def stiffnessindicator(time, solution, jfun, *args):
    """
    Find the local stiffness indicator after calculating local solution.

    Given the value of the solution, find the stiffness indicator as defined by
    Soderlind 2013.
    """
    funcparams = []
    for arg in args:
        funcparams.append(arg)

    try:
        jacobian = jfun(time, solution, funcparams[0])
    except ValueError:
        jacobian = jfun(time[0], solution[0], funcparams[0])
    Hermitian = 0.5 * (jacobian + np.transpose(jacobian))
    eigvals = np.linalg.eigvals(Hermitian)
    return 0.5 * (min(eigvals) + max(eigvals))


def reftimescale(indicatorval, Tlen):
    """
    Find the local reference timescale for the stiffness indicator.

    Given the stiffness indicator values as defined by Soderlind 2013, finds
    the reference time scale.
    """
    if indicatorval >= 0:
        timescale = Tlen
    else:
        timescale = min(Tlen, -1/indicatorval)
    return timescale


def CEMA(xlist, solution, jfun, *args):
    """
    Find values for the chemical explosive mode analysis.

    Same thing as finding the maximum eigenvalue across the solution.
    """
    funcparams = []
    for arg in args:
        funcparams.append(arg)

    values = []
    try:
        for i in range(len(solution)):
            jacobian = jfun(xlist[i], solution[i], funcparams[0])
            values.append(max(np.linalg.eigvals(jacobian)))
    except TypeError:
        jacobian = jfun(xlist, solution, funcparams[0])
        values.append(max(np.linalg.eigvals(jacobian)))
    return values


def stiffnessratio(xlist, solution, jfun, *args):
    """
    Find values of the stiffness ratio.

    Ratio of the eigenvalue with the largest absolute value over the eigenvalue
    with the smallest absolute value. Ignores eigenvalues of zero.
    """
    funcparams = []
    for arg in args:
        funcparams.append(arg)

    values = []
    try:
        for i in range(len(solution)):
            jacobian = jfun(xlist[i], solution[i], funcparams[0])
            eigvals = np.array([abs(j) for j in np.linalg.eigvals(jacobian)
                                if j != 0])
            values.append(max(eigvals)/min(eigvals))
    except TypeError:
        jacobian = jfun(xlist, solution, funcparams[0])
        eigvals = np.array([abs(j) for j in np.linalg.eigvals(jacobian)
                            if j != 0])
        values.append(max(eigvals)/min(eigvals))
    return values


def loadpasrdata():
    """Load the initial conditions from the full PaSR file."""
    print('Loading data...')
    filepath = os.path.join(os.getcwd(), 'ch4_full_pasr_data.npy')
    return np.load(filepath)


def rearrangepasr(Y, N2_pos):
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


# Finding the current time to time how long the simulation takes
starttime = datetime.datetime.now()
print('Start time: {}'.format(starttime))

"""
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
All of the values that need to be adjusted should be in this section.
"""
# Specify if you want to save the data
savedata = True
# Specify if you want all of the stiffness metrics
getmetrics = False
# Possible options will be 'VDP', 'Autoignition', or 'Oregonator'
# Oregonator not yet implemented
equation = 'Autoignition'
# Possible options are 'Stiffness_Index', 'Stiffness_Indicator', 'CEMA',
# 'Stiffness_Ratio'
# method = 'Stiffness_Indicator'
# Options are 'vode' and 'dopri5'
intmode = 'dopri5'
# Make this true if you want to test all of the values across the PaSR.
# Non-PaSR currently not functional
PaSR = True
# Define the range of the computation.
dt = 1.e-8
tstart = 0.
tstop = 0.2
# ODE Solver parameters.
# Tightest tolerances that worked were abserr = 1.0e-17 and relerr = 1.0e-15
abserr = 1.0e-17
relerr = 1.0e-15
# Keep this at false, something isn't working with using the jacobian yet.
usejac = False
# Decide if you want to give pyJac N2 or not.
useN2 = False
# Used if you want to check that the PaSR data is being properly conditioned.
displayconditions = False
# Display the solution shape for plotting/debugging.
displaysolshapes = False
# To be implemented later.
makesecondderivplots = False
"""
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
"""

warnings.filterwarnings("ignore", category=UserWarning)

if equation == 'VDP':
    # Makes no sense to have PaSR for this, so it won't be allowed.
    PaSR = False
    RHSfunction = dydx
    EQjac = jacvdp
    EQ2deriv = d2ydx2
    initcond = [2, 0]
    RHSparam = 1000.
elif equation == 'Autoignition':
    RHSfunction = firstderiv
    EQjac = jacobval
    # Load the initial conditions from the PaSR files
    pasr = loadpasrdata()
    numparticles = len(pasr[:, 0])

# Loop through the PaSR file for initial conditions
if PaSR:
    print('Code progress:')
    particlelist = range(numparticles)
    # We don't want long integrations for every point in the PaSR
    tstart = 0.
    tstop = 4 * dt
else:
    particlelist = [92]
    # Second set of coords was p=679, t=646
    # Third set was p = 877, t = 865

# Create the list of times to compute
tlist = np.arange(tstart, tstop + 0.5 * dt, dt)

functionwork, tstepsneeded, ratiovals, indexvals, indicatorvals, CEMAvals,\
    inttimes = [], [], [], [], [], [], []

for particle in particlelist:
    if PaSR:
        # Provide code progress
        if particle % 1000 == 0:
            print(particle)
    # Set up the initial conditions for autoignition
    if equation == 'Autoignition':
        Y = pasr[particle, :].copy()
        initcond, RHSparam = rearrangepasr(Y, 50)
        # Display the conditions if we're running diagnostics
        if displayconditions:
            print('Initial Condition:')
            for i in Y:
                print(i)
            print('Modified condition:')
            for i in initcond:
                print(i)

    if not PaSR:
        print('Integrating...')
    # Put the initial condition into the solution vector
    solution = [initcond]
    failflag = False

    # Specify the integrator
    if usejac:
        intj = EQjac
    else:
        intj = None

    if intmode == 'vode':
        solver = ode(RHSfunction,
                     jac=intj
                     ).set_integrator(intmode,
                                      method='bdf',
                                      nsteps=1,
                                      # atol=abserr,
                                      # rtol=relerr,
                                      with_jacobian=usejac,
                                      first_step=dt,
                                      # min_step=dt,
                                      max_step=dt
                                      )
    elif intmode == 'dopri5':
        # Assign the explicit solver
        # Note that we can figure out how many intermediate steps for
        # dopri5 by calculating [(functioncalls - 1) / 3]
        solver = ode(RHSfunction,
                     jac=intj
                     ).set_integrator(intmode,
                                      nsteps=1e10,
                                      # atol=abserr,
                                      # rtol=relerr
                                      # min_step=dt,
                                      first_step=dt,
                                      max_step=dt
                                      )
    else:
        raise Exception("Error: Select 'vode' or 'dopri5'.")

    # Set initial conditions
    solver.set_initial_value(initcond, tstart)
    solver.set_f_params(RHSparam)
    solver.set_jac_params(RHSparam)
    solver._integrator.iwork[2] = -1

    # Integrate the ODE across all steps
    tnext = tstart + dt
    halfflag = False
    stepstaken = 0
    timetwo = 0.
    fcallstwo = 0
    while solver.t < tstop:
        # Integrate until hitting the next tstep
        while solver.t < tnext:
            functioncalls = 0
            # Do this to force it to stop at every dt
            # Obtain the previous state values
            prevsol = solver.y
            prevtime = solver.t
            # Reinitialize
            if failflag or intmode == 'vode':
                solver = ode(RHSfunction,
                             jac=intj
                             # Set up for vode
                             ).set_integrator('vode',
                                              method='bdf',
                                              nsteps=1,
                                              # atol=abserr,
                                              # rtol=relerr,
                                              with_jacobian=usejac,
                                              first_step=dt,
                                              # min_step=dt,
                                              max_step=(tnext - solver.t)
                                              )
                # Reset IC's
                solver.set_initial_value(prevsol, prevtime)
                solver.set_f_params(RHSparam)
                solver.set_jac_params(RHSparam)
                solver._integrator.iwork[2] = -1
            else:
                solver = ode(RHSfunction,
                             jac=intj
                             # Set up for dopri5
                             ).set_integrator('dopri5',
                                              # method='bdf',
                                              nsteps=1,
                                              first_step=dt,
                                              max_step=(tnext - solver.t)
                                              )
                # Reset IC's
                solver.set_initial_value(prevsol, prevtime)
                solver.set_f_params(RHSparam)
                solver.set_jac_params(RHSparam)
                solver._integrator.iwork[2] = -1
            if intmode == 'vode':
                time0 = timer.time()
                solver.integrate(tnext, step=True)
                time1 = timer.time()
            else:
                time0 = timer.time()
                solver.integrate(tnext)
                time1 = timer.time()
            if prevtime >= (tstart + dt) and prevtime < (tstart + 2*dt):
                timetwo += time1 - time0
                if intmode == 'vode':
                    stepstaken += 1
                fcallstwo += functioncalls
            # if solver.t <= prevtime:
            #     raise Exception('Error: Simulation not advancing!')
            # Save the solution
            if solver.t >= tnext:
                solution.append(solver.y)
            # Reinitialize if dopri5 fails.
            if intmode == 'dopri5' and not solver.successful():
                # Assuming that the only way this will happen is if dopri5
                # fails.  Switch to vode and do it over again.
                failflag = True
                if not PaSR:
                    raise Exception('dopri5 failed!')
                solution = [initcond]
                # print('dopri5 failed at particle {}, tstep {}!'.format(
                #         particle, tstep))
                solver = ode(RHSfunction,
                             jac=intj
                             ).set_integrator('vode',
                                              method='bdf',
                                              nsteps=1,
                                              # atol=abserr,
                                              # rtol=relerr,
                                              with_jacobian=usejac,
                                              first_step=dt,
                                              # min_step=dt,
                                              max_step=dt
                                              )
                solver.set_initial_value(initcond, tstart)
                solver.set_f_params(RHSparam)
                solver.set_jac_params(RHSparam)
                solver._integrator.iwork[2] = -1
                stepstaken = 0
                halfflag = False
                timetwo = 0.0
                fcallstwo = 0
            else:
                localtemp = solver.y[0]
                if PaSR:
                    # Calculate metrics at the midpoint
                    # and get functioncalls
                    if (solver.t >= (tstart + 2*dt)) and not halfflag:
                        # solutiontimes.append(time1 - time0)
                        # Calculate the indicator, ratio, and CEM
                        if getmetrics:
                            indicator = stiffnessindicator(solver.t,
                                                           solver.y,
                                                           EQjac,
                                                           RHSparam
                                                           )
                            stiffratio = stiffnessratio(solver.t,
                                                        solver.y,
                                                        EQjac,
                                                        RHSparam
                                                        )
                            chemexmode = CEMA(solver.t,
                                              solver.y,
                                              EQjac,
                                              RHSparam
                                              )
                        # print('Halfway value: {}'.format(tstart + 2*dt))
                        # print('Current value: {}'.format(solver.t))
                        halfflag = True
                    # Save the metrics and work when done
                    if solver.t >= tstop:
                        # Print the solution shape
                        # print('Solution shape: {}'.format(
                        #     np.shape(solution)))
                        # Save all the metrics in the lists
                        if getmetrics:
                            solution = np.array(solution)
                            stiffindices = stiffnessindex(tlist,
                                                          solution,
                                                          RHSfunction,
                                                          EQjac,
                                                          RHSparam
                                                          )
                            ratiovals.append(stiffratio)
                            indexvals.append(stiffindices[2])
                            indicatorvals.append(indicator)
                            CEMAvals.append(chemexmode)
                        # Save the functionwork
                        if failflag:
                            functionwork.append(-1)
                            inttimes.append(-1)
                            tstepsneeded.append(-1)
                        else:
                            functionwork.append(fcallstwo)
                            inttimes.append(timetwo)
                            if intmode == 'dopri5':
                                stepstaken = (fcallstwo - 1) / 6
                            tstepsneeded.append(stepstaken)
                # Part of the code for non-PaSR.  Non-functional currently.
                else:
                    solution.append(solver.y)
                    if getmetrics:
                        indicator = stiffnessindicator(solver.t,
                                                       solver.y,
                                                       EQjac,
                                                       RHSparam
                                                       )
                        indicatorvals.append(indicator)
                        stiffratio = stiffnessratio(solver.t,
                                                    solver.y,
                                                    EQjac,
                                                    RHSparam
                                                    )
                        ratiovals.append(stiffratio)
                        chemexmode = CEMA(solver.t,
                                          solver.y,
                                          EQjac,
                                          RHSparam
                                          )
                        CEMAvals.append(chemexmode)
                    # solutiontimes.append(time1 - time0)
                    functionwork.append(functioncalls)
        tnext += dt

    # print('Timesteps needed:')
    # for i in tstepsneeded:
    #     print(i)
    # print(functionwork)
    # for i in inttimes:
    #     print(i)
    # print('Stop solution:')
    # for i in solution:
    #     print(i)
    # print(ratiovals)
    # print(indexvals)
    # print(indicatorvals)
    # print(CEMAvals)
    # print('tstop: {}'.format(tstop))
    # raise Exception('Test run, solver t at {}'.format(solver.t))

    # Display the conditions if we're running diagnostics
    if displayconditions:
        print('Final time:')
        print(solver.t)
        print('Last solution value:')
        for i in solver.y:
            print(i)
        lastjac = jacobval(0.2, solver.y, RHSparam)
        print('Last Jacobian value:')
        for i in lastjac:
            print(i)

# Convert the solution to an array for ease of use.  Maybe just using
# numpy function to begin with would be faster?
solution = np.array(solution)
functionwork = np.array(functionwork)
tstepsneeded = np.array(tstepsneeded)
inttimes = np.array(inttimes)

if getmetrics and not PaSR:
    print('Finding stiffness index...')
    indexvals = stiffnessindex(tlist,
                               solution,
                               RHSfunction,
                               EQjac,
                               RHSparam
                               )

ratiovals = np.array(ratiovals)
indexvals = np.array(indexvals)
indicatorvals = np.array(indicatorvals)
CEMAvals = np.array(CEMAvals)

# Get the current working directory
output_folder = 'Output_Plots/'
data_folder = 'Output_Data/'

if len(solution) != len(tlist):
    print('Solution shape: {}'.format(np.shape(solution)))
    print('tlist shape: {}'.format(np.shape(tlist)))

if savedata:
    solfilename = equation + '_Solution_' + intmode + '_' + str(dt)
    workfilename = equation + '_FunctionWork_' + intmode + '_' + str(dt)
    stepsfilename = equation + '_Timesteps_' + intmode + '_' + str(dt)
    inttimingfilename = equation + '_Int_Times_' + intmode + '_' +\
        str(dt) + '_' + timer.strftime("%m_%d")
    ratiofilename = equation + '_Stiffness_Ratio_' + str(dt)
    indexfilename = equation + '_Stiffness_Index_' + str(dt)
    indicatorfilename = equation + '_Stiffness_Indicator_' + str(dt)
    CEMAfilename = equation + '_CEMA_' + str(dt)
    if PaSR:
        workfilename = 'PaSR_' + workfilename
        stepsfilename = 'PaSR_' + stepsfilename
        inttimingfilename = 'PaSR_' + inttimingfilename
        ratiofilename = 'PaSR_' + ratiofilename
        indexfilename = 'PaSR_' + indexfilename
        indicatorfilename = 'PaSR_' + indicatorfilename
        CEMAfilename = 'PaSR_' + CEMAfilename
    else:
        np.save(data_folder + solfilename, solution)
    np.save(data_folder + workfilename, functionwork)
    np.save(data_folder + stepsfilename, tstepsneeded)
    np.save(data_folder + inttimingfilename, inttimes)
    if getmetrics:
        np.save(data_folder + ratiofilename, ratiovals)
        np.save(data_folder + indexfilename, indexvals)
        np.save(data_folder + indicatorfilename, indicatorvals)
        np.save(data_folder + CEMAfilename, CEMAvals)

finishtime = datetime.datetime.now()
print('Finish time: {}'.format(finishtime))
print('Duration: {}'.format(finishtime - starttime))
