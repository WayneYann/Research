#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 09:42:33 2017

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


def newtonsmethod(derivative, x0, dt, tolerance):
    while True:
        x1 = x0 -

    pass

# Finding the current time to time how long the simulation takes
starttime = datetime.datetime.now()
print('Start time: {}'.format(starttime))

# Define the range of the computation
dt = 1.e-7
tstart = 0
tstop = 0.2
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

particle = 92
tstep = 4

# Get the initial condition.
Y = pasr[tstep, particle, :].copy()

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

# Call the integrator
print('Integrating...')
solution = sci.integrate.odeint(firstderiv, # Call the dydt function
                                # Pass it initial conditions
                                Ys,
                                # Pass it time steps to be evaluated
                                tlist,
                                # Pass any additional information needed
                                args=(Y_press,),
                                # Pass it the Jacobian (not sure if needed)
                                # Dfun=jacobval,
                                # Pass it the absolute and relative tolerances
                                atol=abserr, rtol=relerr,
                                # Print a message stating if it worked or not
                                printmessg=0
                                )

speciesnames = ['H', 'H$_2$', 'O', 'OH', 'H$_2$O', 'O$_2$', 'HO$_2$',
                'H$_2$O$_2$', 'Ar', 'He', 'CO', 'CO$_2$', 'N$_2$']

# Plot the values of the thermochemical composition vector
fs = 12
print('Plotting...')
pyl.figure(0,figsize=(6, 4.5), dpi=400)
pyl.xlabel('Time (s)')
pyl.ylabel('Temperature (K)')
for i in range(1,14):
    pyl.figure(i,figsize=(6, 4.5), dpi=400)
    pyl.ylabel(speciesnames[i-1] + ' Mass Fraction', fontsize=fs)
    pyl.xlabel('Time (s)', fontsize=fs)

for i in range(14):
    pyl.figure(i)
    pyl.xlim(tstart,tstop)
    pyl.plot(tlist, solution[:,i])

pyl.show()