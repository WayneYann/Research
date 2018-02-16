#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import CSPfuncs
import time as time
from scipy.integrate import ode


def intDriver(tim, dt, y_global, mu, setup, CSPtols):
    t0 = tim

    pflag = False

    # Unpack variables
    derivfun, jacfun, mode = setup
    eps_a, eps_r, eps = CSPtols

    y0_local = y_global.copy()

    while tim < t0 + dt:
        M, tau, Qs, Rc = get_slow_projector(tim, y0_local, eps, derivfun,
                                            jacfun, CSPtols)

        # time step size (controlled by fastest slow mode)
        h = mu * tau
        # print(tim, M, Y)

        # Call some integrator
        if mode == 'RK4':
            tim, yn_local = RK4(derivfun, y0_local, tim, dt, Qs, 1)
            #tim, yn_local = RK4(tim, h, y0_local, Qs)
        else:
            solver.set_initial_value(Y, t0)
            solver.set_f_params(Qs, 0)
            # solver.set_jac_params(eps)
            solver.integrate(t0 + dt)
            tim = solver.t
            yn_local = solver.y

        rc_array = radical_correction(tim, yn_local, Rc)

        for i in range(NN):
            y0_local[i] = yn_local[i] - rc_array[i]

            # check if any > 1 or negative
            if y0_local[i] > 1.0 or y0_local[i] < 0.0:
                print("Something was less than 1 or negative.")
                print(tim, M, y0_local)
    return tim, yn_local

# CSP Tolerances
eps_r = 1.0e-3  # Real CSP tolerance
eps_a = 1.0e-3  # Absolute CSP tolerance
eps = 1.0e-2  # Stiffness factor

mu = 0.005  # Timestep factor

NUM = 1  # Number of threads to solve simultaneously (not implemented yet)
NN = 4

# Simulation parameters
t0 = 0.0  # Start time (sec)
tend = 5.0  # End time (sec)
tim = t0  # Current time (sec), initialized at zero
dt = 1.0e-2  # Printing time step

# Options are 'RK4', 'vode'
mode = 'vode'
problem = 'CSPtest'
CSPon = True

# Set initial conditions
Y = []
for i in range(NN):
    Y.append(1.0)

# Initialize the specific problem
if problem == 'CSPtest':
    derivfun = testfunc
    jacfun = testjac
setup = (derivfun, jacfun, mode)
CSPtols = eps_a, eps_r, eps

# Need to also reshape because this was originally written in C
Qs = np.reshape(np.identity(NN), (NN**2,))

if mode != 'RK4':
    solver = ode(derivfun).set_integrator(mode,
                                          #method='bdf',
                                          nsteps=1e6,
                                          # atol=abserr,
                                          # rtol=relerr,
                                          #with_jacobian=False,
                                          #first_step=dt,
                                          # min_step=dt,
                                          max_step=dt
                                          )
    solver.set_initial_value(Y, t0)
    if problem == 'CSPtest':
        solver.set_f_params(Qs, 0)
        # solver.set_jac_params(eps)
    solver._integrator.iwork[2] = -1

# Timer
t_start = time.time()
while tim < tend:
    if dt < 1.0e-6 and tim >= 1.0e-6:
        dt = 1.0e-6
    elif dt < 1.0e-5 and tim >= 1.0e-5:
        dt = 1.0e-5
    elif dt < 1.0e-4 and tim >= 1.0e-4:
        dt = 1.0e-4
    elif dt < 1.0e-3 and tim >= 1.0e-3:
        dt = 1.0e-3
    elif dt < 1.0e-2 and tim >= 1.0e-2:
        dt = 1.0e-2

    if CSPon:
        tim, Y = intDriver(tim, dt, Y, mu, setup, CSPtols)
        print('t = {:.2f}'.format(tim), Y)
    else:
        solver.integrate(t0 + dt)
        print('t = {:.2f}'.format(solver.t), solver.y)
        tim += dt
        t0 = tim

t_end = time.time()

cpu_time = t_end - t_start

print("Time taken: {}".format(cpu_time))
