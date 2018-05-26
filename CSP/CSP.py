#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from CSPfuncs import *
from StiffnessFuncs import *
import time as time
from scipy.integrate import ode
import warnings
import os as os
import sys as sys


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
    derivfun, jacfun, mode, CSPon, abserr, relerr, constantdt, noRHSparam = setup

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
                                              nsteps=1e10,
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
    del solver
    return tim, y0_local, comp_time, stiffness, M

# CSP Tolerances
eps_r = 1.0e-3  # Real CSP tolerance
eps_a = 1.0e-3  # Absolute CSP tolerance

# vode tolerances
# abserr = 1e-24
# relerr = 1e-15
abserr = 1e-18
relerr = 1e-14

mu = 0.005  # Timestep factor

NUM = 1  # Number of threads to solve simultaneously (not implemented yet)

# Simulation parameters
t0 = 0.0  # Start time (sec)
tim = t0  # Current time (sec), initialized at zero

# Options are 'RK4', 'vode'
mode = 'vode'
# Options are 'CSPtest', 'VDP', 'Oregonator', 'H2', 'GRIMech'
problem = 'GRIMech'
autoignition = True
CSPon = False  # Decides if the integration actually will use CSP, not working yet
constantdt = False
# Make this either human readable or better for saving into a table
humanreadable = False
printic = False
reducedCSPtol = False

if printic:
    useN2 = True
else:
    useN2 = False

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
    if reducedCSPtol:
        eps_r = 1.0e-6  # Real CSP tolerance
        eps_a = 1.0e-6  # Absolute CSP tolerance
    if autoignition:
        pasr = loadpasrdata(problem)
        dt = 1.0e-4
        tend = 1.0
        particle = 877
        timestep = 865
        Y = pasr[timestep, particle, :].copy()
        Y, RHSparam = rearrangepasr(Y, problem, useN2)
        NN = len(Y)
        if printic:
            if RHSparam > 1000.0:
                RHSparam /= 101325.0
            species = ["H", "H2", "O", "OH", "H2O", "O2", "HO2", "H2O2", "AR",
                       "HE", "CO", "CO2", "N2"]
            print('Temperature (K):')
            print(Y[0])
            print('Pressure (atm):')
            print(RHSparam)
            printstring = ""
            for i in range(len(Y) - 1):
                printstring = printstring + species[i] + '={},'.format(Y[i+1])
            printstring = printstring[:-1]
            print('Species Concentrations:')
            print(printstring)
            print('accelerInt string:')
            printstring = str(Y[0]) + ',' + str(RHSparam) + ',' + printstring
            sys.exit(printstring)
        if RHSparam < 1000.0:
            RHSparam *= 101325.0
    else:
        dt = 1.0e-9
        tend = 5 * dt
        loadfile = '../../accelerInt/initials/H2_CO/ign_data.bin'
        numbin = 16
    derivfun = firstderiv
    jacfun = jacobval
elif problem == 'GRIMech':
    if reducedCSPtol:
        eps_r = 1.0e-6  # Real CSP tolerance
        eps_a = 1.0e-6  # Absolute CSP tolerance
    if autoignition:
        pasr = loadpasrdata(problem)
        dt = 1.0e-4
        tend = 0.4
        particle = 230761
        Y = pasr[particle, :].copy()
        NN = len(Y)
        Y, RHSparam = rearrangepasr(Y, problem, useN2)
        if printic:
            if RHSparam > 1000.0:
                RHSparam /= 101325.0
            species = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "C", "CH",
                       "CH2", "CH2\(S\)", "CH3", "CH4", "CO", "CO2", "HCO", "CH2O",
                       "CH2OH", "CH3O", "CH3OH", "C2H", "C2H2", "C2H3", "C2H4",
                       "C2H5", "C2H6", "HCCO", "CH2CO", "HCCOH", "N", "NH", "NH2",
                       "NH3", "NNH", "NO", "NO2", "N2O", "HNO", "CN", "HCN",
                       "H2CN", "HCNN", "HCNO", "HOCN", "HNCO", "NCO", "C3H7",
                       "C3H8", "CH2CHO", "CH3CHO", "AR", "N2"]
            print('Temperature (K):')
            print(Y[0])
            print('Pressure (atm):')
            print(RHSparam)
            printstring = ""
            for i in range(len(Y) - 1):
                printstring = printstring + species[i] + '={},'.format(Y[i+1])
            printstring = printstring[:-1]
            print('Species Concentrations:')
            print(printstring)
            print('accelerInt string:')
            printstring = str(Y[0]) + ',' + str(RHSparam) + ',' + printstring
            sys.exit(printstring)
        if RHSparam < 1000.0:
            RHSparam *= 101325.0
    else:
        dt = 1.0e-9
        tend = 5 * dt
        loadfile = '../../accelerInt/initials/GRI_Mech_3/ign_data.bin'
        numbin = 56
    derivfun = firstderiv
    jacfun = jacobval

# Initialize the specific problem
setup = (derivfun, jacfun, mode, CSPon, abserr, relerr, constantdt, noRHSparam)
CSPtols = eps_a, eps_r

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
if ((problem != 'H2' and problem != 'GRIMech') or autoignition):
    # Need to also reshape because this was originally written in C
    Qs = np.reshape(np.identity(NN), (NN**2,))
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
                output = np.array2string(np.hstack((tim.real, comp_time.real, M.real, stiffness.real,
                                                    ratio.real, indicator.real, CEM.real,
                                                    Y.real)),
                                         separator=',')
                print(''.join(output.strip('[]').split()))

    t_end = time.time()

    cpu_time = t_end - t_start

    # print("Time taken: {}".format(cpu_time))
else:
    pasr = np.fromfile(loadfile)
    for i in range(int(len(pasr)/numbin)):
        tim = 0.0
        Y = pasr[i*numbin:(i+1)*numbin]
        Y, RHSparam = rearrangepasr(Y, problem, useN2)
        sol, xlist = [], []
        while tim < tend:
            if tim < 0.9 * dt:
                ratio, indicator, CEM = stiffmetrics(tim, Y, jacfun, RHSparam)
            tim, Y, comp_time, stiffness, M = intDriver(tim, dt, Y, mu, setup,
                                                        CSPtols, RHSparam)
            if tim < 1.5 * dt:
                stiffval, Mval = stiffness, M
            xlist.append(tim)
            sol.append(Y)
            comp_speed = dt / comp_time
            #Add a small buffer to ensure it only prints once
            if tim > 4.1 * dt:
                sol = np.array(sol)
                xlist = np.array(xlist)
                index = stiffnessindex(xlist, sol, firstderiv, jacobval, RHSparam)[2]
                if humanreadable:
                    print('p={}, t={:<8.2g} M={} speed_comp={:<8.4g}\ts={:<10.8g}\ty:'.format(
                                                                             i,
                                                                             tim,
                                                                             Mval,
                                                                             comp_speed,
                                                                             stiffval
                                                                             ),
                          ''.join(('{:<12.8g}'.format(Y[j])
                          for j in range(len(Y)))), ratio, indicator, CEM.real)
                else:
                    output = np.array2string(np.hstack((i.real,
                                                        tim.real,
                                                        comp_time.real,
                                                        Mval.real,
                                                        stiffval.real,
                                                        ratio.real,
                                                        indicator.real,
                                                        CEM.real,
                                                        index.real)),
                                             separator=',')
                    print(''.join(output.strip('[]').split()))

    t_end = time.time()

    cpu_time = t_end - t_start

    # print("Time taken: {}".format(cpu_time))
