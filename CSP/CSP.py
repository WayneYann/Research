#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from CSPfuncs import *
from StiffnessFuncs import *
import time as time
from scipy.integrate import ode
import warnings


def CSPwrap(tim, y0_local, RHSparam, derivfun, jacfun, CSPtols):
    """Needs updates before this function will work."""
    raise Exception('CSPwrap function not supported yet.')
    M, tau, Qs, Rc = get_slow_projector(tim, y0_local, RHSparam, derivfun,
                                        jacfun, CSPtols)
    ydot = testfunc(tim, y0_local, Qs, 1)
    return ydot


def intDriver(tim, dt, y_global, mu, setup, CSPtols, *RHSparam):

    if RHSparam:
        RHSparam = RHSparam[0]
    t0 = tim

    pflag = False

    # Unpack variables
    derivfun, jacfun, mode, CSPon, abserr, relerr, dt, constantdt, noRHSparam = setup

    y0_local = y_global[:]

    # Initialize the ode solver
    if mode != 'RK4':
        # This probably isn't working yet
        if CSPon:
            raise Exception('CSP not yet supported.')
            solver = ode(CSPwrap).set_integrator(mode,
                                                 #method='bdf',
                                                 nsteps=1e6,
                                                 atol=abserr,
                                                 rtol=relerr,
                                                 #with_jacobian=False,
                                                 #first_step=dt,
                                                 #min_step=dt - 1e-10,
                                                 max_step=dt
                                                 )
        else:
            solver = ode(derivfun).set_integrator(mode,
                                              #method='bdf',
                                              nsteps=1e6,
                                              atol=abserr,
                                              rtol=relerr,
                                              #with_jacobian=False,
                                              #first_step=dt,
                                              # min_step=dt,
                                              max_step=dt
                                              )
    solver.set_initial_value(Y, t0)
    if noRHSparam:
        pass
    else:
        solver.set_f_params(RHSparam)
        # solver.set_jac_params(eps)

    #solver._integrator.iwork[2] = -1

    while tim < t0 + dt:
        if noRHSparam:
            M, tau, Qs, Rc, stiffness = get_slow_projector(tim, y0_local,
                                                           derivfun, jacfun,
                                                           CSPtols)
        else:
            M, tau, Qs, Rc, stiffness = get_slow_projector(tim, y0_local,
                                                           derivfun, jacfun,
                                                           CSPtols, RHSparam
                                                           )

        # time step size (controlled by fastest slow mode)
        h = mu * tau
        # print(tim, M, Y)

        # Call some integrator - this isn't working yet for CSPon
        if CSPon:
            raise Exception('CSPon not yet supported.')
            if mode == 'RK4':
                tim, yn_local = RK4(derivfun, y0_local, tim, dt, Qs, 1)
                #tim, yn_local = RK4(tim, h, y0_local, Qs)
            else:
                solver.set_initial_value(Y, t0)
                solver.set_f_params(RHSparam, derivfun, jacfun, CSPtols)
                # solver.set_jac_params(eps)
                tstart_step = time.time()
                solver.integrate(t0 + dt)
                comp_time = time.time() - tstart_step
                tim = solver.t
                yn_local = solver.y
            rc_array = radical_correction(tim, yn_local, Rc)
            for i in range(NN):
                y0_local[i] = yn_local[i] - rc_array[i]
        else:
            if mode == 'RK4':
                # Also probably not working yet
                raise Exception('RK4 not yet supported.')
                tim, yn_local = RK4(derivfun, y0_local, tim, dt, Qs, 0)
                #tim, yn_local = RK4(tim, h, y0_local, Qs)
            else:
                tstart_step = time.time()
                solver.integrate(t0 + dt)
                comp_time = time.time() - tstart_step
                tim = solver.t
                y0_local = solver.y

        # for i in range(NN):
        #     # check if any > 1 or negative
        #     if y0_local[i] > 1.0 or y0_local[i] < 0.0:
        #         print("Something was less than 1 or negative.")
        #         print(tim, M, y0_local)
    print('tim: {}'.format(tim))
    print('solver.t: {}'.format(solver.t))
    print('t0: {}'.format(t0))
    print('dt: {}'.format(dt))
    del solver
    return tim, y0_local, comp_time, stiffness, M

# CSP Tolerances
eps_r = 1.0e-3  # Real CSP tolerance
eps_a = 1.0e-3  # Absolute CSP tolerance

# vode tolerances
abserr = 1e-24
relerr = 1e-15

mu = 0.005  # Timestep factor

NUM = 1  # Number of threads to solve simultaneously (not implemented yet)

# Simulation parameters
t0 = 0.0  # Start time (sec)
tim = t0  # Current time (sec), initialized at zero

# Options are 'RK4', 'vode'
mode = 'vode'
# Options are 'CSPtest', 'VDP', 'Oregonator', 'H2', 'GRIMech'
problem = 'CSPtest'
CSPon = False  # Decides if the integration actually will use CSP, not working yet
constantdt = False
# Make this either human readable or better for saving into a table
humanreadable = False

# Filter out the warnings
warnings.filterwarnings('ignore')
# Set initial conditions
noRHSparam = False
if problem == 'CSPtest':
    dt = 1.0e-9  # Integrating time step
    tend = 10.0  # End time (sec)
    NN = 4  # Size of problem
    RHSparam = 1.0e-2  # Stiffness factor
    Y = []
    for i in range(NN):
        Y.append(1.0)
    derivfun = testfunc
    jacfun = testjac
elif problem == 'VDP':
    dt = 1.0e-1  # Integrating time step
    tend = 3000.0  # End time (sec)
    NN = 2  # Size of problem
    Y = [2, 0]
    noRHSparam = True
    derivfun = dydxvdp
    jacfun = jacvdp
elif problem == 'Oregonator':
    dt = 1.0e-1  # Integrating time step
    tend = 320.0  # End time (sec)
    NN = 3  # Size of problem
    Y = [1, 1, 2]
    noRHSparam = True
    derivfun = oregonatordydt
    jacfun = oregonatorjac
elif problem == 'H2':
    dt = 1.0e-3
    tend = 2.0
    particle = 877
    timestep = 865
    pasr = loadpasrdata(problem)
    Y = pasr[timestep, particle, :].copy()
    NN = len(Y)
    Y, RHSparam = rearrangepasr(Y, problem)
    derivfun = firstderiv
    jacfun = jacobval
elif problem == 'GRIMech':
    dt = 1.0e-3
    tend = 2.0
    particle = 92
    Y = pasr[particle, :].copy()
    NN = len(Y)
    Y, RHSparam = rearrangepasr(Y, problem)
    pasr = loadpasrdata(problem)
    derivfun = firstderiv
    jacfun = jacobval

# Initialize the specific problem
setup = (derivfun, jacfun, mode, CSPon, abserr, relerr, dt, constantdt, noRHSparam)
CSPtols = eps_a, eps_r

# Need to also reshape because this was originally written in C
Qs = np.reshape(np.identity(NN), (NN**2,))

# Timer
t_start = time.time()
printstep = 0
if problem == 'CSPtest':
    printevery = 1
# elif problem == 'VDP':
#     printevery = 0
# elif problem == 'Oregonator':
else:
    printevery = 0
while tim < tend:
    # Make it so that there's not millions of data points for CSPtest
    if problem == 'CSPtest':
        printstep += 1
        if constantdt:
            if tim >= 1.0e-8:
                printevery = 10
            if tim >= 1.0e-7:
                printevery *= 10
            if tim >= 1.0e-6:
                printevery *= 10
            if tim >= 1.0e-5:
                printevery *= 10
            if tim >= 1.0e-4:
                printevery *= 10
            if tim >= 1.0e-3:
                printevery *= 10
            if tim >= 1.0e-2:
                printevery *= 10
        else:
            if dt < 1.0e-6 and tim >= 1.0e-6:
                dt = 1.0e-6
            if dt < 1.0e-5 and tim >= 1.0e-5:
                dt = 1.0e-5
            if dt < 1.0e-4 and tim >= 1.0e-4:
                dt = 1.0e-4
            if dt < 1.0e-3 and tim >= 1.0e-3:
                dt = 1.0e-3
            if dt < 1.0e-2 and tim >= 1.0e-2:
                dt = 1.0e-2

    if noRHSparam:
        tim, Y, comp_time, stiffness, M = intDriver(tim, dt, Y, mu, setup,
                                                    CSPtols)
        ratio, indicator, CEM = stiffmetrics(tim, Y, jacfun)
    else:
        tim, Y, comp_time, stiffness, M = intDriver(tim, dt, Y, mu, setup,
                                                    CSPtols, RHSparam)
        ratio, indicator, CEM = stiffmetrics(tim, Y, jacfun, RHSparam)

    comp_speed = dt / comp_time
    if printstep == printevery:
        printstep = 0
        if humanreadable:
            print('t={:<8.2g} M={} speed_comp={:<8.4g}\ts={:<10.8g}\ty:'.format(
                                                                     tim,
                                                                     M,
                                                                     comp_speed,
                                                                     stiffness
                                                                     ),
                  ''.join(('{:<12.8g}'.format(Y[i])
                  for i in range(len(Y)))), ratio, indicator, CEM.real)
        else:
            output = np.array2string(np.hstack((tim, comp_time, M, stiffness,
                                                ratio, indicator, CEM.real,
                                                Y)),
                                     separator=',')
            print(''.join(output.strip('[]').split()))

t_end = time.time()

cpu_time = t_end - t_start

# print("Time taken: {}".format(cpu_time))
