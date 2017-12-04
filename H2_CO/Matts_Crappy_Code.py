# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:02:15 2016

@author: mattz
@Advanced Combustion Assignment 2
Problem 6

Modified by Andrew Alferman for a MTH 654 term project
12/1/17
"""

import numpy as np
# import matplotlib.pyplot as plt
import cantera as ct
import os as os


def loadpasrdata(num):
    """Load the initial conditions from the PaSR files."""
    pasrarrays = []
    print('Loading data...')
    for i in range(num):
        filepath = os.path.join(os.getcwd(),
                                'pasr_out_h2-co_' +
                                str(i) +
                                '.npy')
        filearray = np.load(filepath)
        pasrarrays.append(filearray)
    return np.concatenate(pasrarrays, 1)


def getH2O2ics(pasr, p, t):
    """Unpack the initial conditions in a legible format."""
    temperature = pasr[t, p, :][1]
    pressure = pasr[t, p, :][2]
    H = pasr[t, p, :][3]
    H2 = pasr[t, p, :][4]
    O = pasr[t, p, :][5]
    OH = pasr[t, p, :][6]
    H2O = pasr[t, p, :][7]
    O2 = pasr[t, p, :][8]
    HO2 = pasr[t, p, :][9]
    H2O2 = pasr[t, p, :][10]
    N2 = pasr[t, p, :][11]
    AR = pasr[t, p, :][12]
    HE = pasr[t, p, :][13]
    CO = pasr[t, p, :][14]
    CO2 = pasr[t, p, :][15]
    return temperature, pressure, H, H2, O, OH, H2O, O2, HO2, H2O2, N2, AR, \
        HE, CO, CO2


def reactionsetup(time, pasr, p, t):
    """Set up the reaction."""
    # reset the ignition time flag and temperature
    T_burn = np.zeros(len(time))
    # utilize a constant pressure reactor from cantera
    # to set equilivalence ratios.
    gas = ct.Solution('../chem.cti')
    # gas.TPX = 1000.0, ct.one_atm, ...
    #   'CH4:{0},O2:2.0,N2:7.52'.format(phi[eq])
    temperature, pressure, H, H2, O, OH, H2O, O2, HO2, H2O2, N2, AR, HE, \
        CO, CO2 = getH2O2ics(pasr, p, t)
    canterastring = ('H:{},H2:{},O:{},OH:{},H2O:{},O2:{},' +
                     'HO2:{},H2O2:{},N2:{},AR:{},HE:{},CO:{},' +
                     'CO2:{}').format(
                     H, H2, O, OH, H2O, O2, HO2, H2O2, N2, AR, HE,
                     CO, CO2)
    gas.TPX = temperature, ct.one_atm, canterastring
    react = ct.IdealGasConstPressureReactor(gas)
    soln = ct.ReactorNet([react])
    return T_burn, react, soln, temperature, pressure, canterastring


# phi = 0.5,1.0,2.0
t = 2.0
dt = 0.001
n = int(t/dt)
# initalize time and a storage for when the temperature spikes
time = np.linspace(0, t, n)
# Trogdor_time = np.zeros(len(phi))
# Burnating_temp=np.zeros(len(phi))
change = 0.0
# plt.figure(num=None, figsize=(12, 8), dpi=900, facecolor='w', edgecolor='k')

pasr = loadpasrdata(1)
numparticles = len(pasr[0, :2, 0])
numtsteps = len(pasr[:, 0, 0])

ignited = False
solutionexists = False
maxburndelta = 0.0

print('Searching through conditions...')
# for eq in range(len(phi)):
for p in range(numparticles):
    print(p)
    for t in range(numtsteps):
        T_burn, react, soln, temperature, pressure, canterastring = \
            reactionsetup(time, pasr, p, t)

        for i in range(len(time)):
            soln.advance(time[i])
            # now for each time step pull store the temperature
            T_burn[i] = react.T
            if i > 0:
                change = abs(T_burn[i] - T_burn[i-1])
            # if change > 5 and temperature < 900.0:
            if change > 5:
                # if not ignited:
                #     igntime = time[i]
                #     igntemp = T_burn[i]
                solutionexists = True
                ignited = True
                # Trogdor_time[eq] = time[i]
                # Burnating_temp[eq] = T_burn[i]
        if ignited:
            ignited = False
            burndelta = react.T - temperature
            if burndelta > maxburndelta:
                maxburndelta = burndelta
                burnp = p
                burnt = t

if not solutionexists:
    raise Exception('Error - Nothing combusted!')

T_burn, react, soln, temperature, pressure, canterastring = \
    reactionsetup(time, pasr, burnp, burnt)
ignited = False
for i in range(len(time)):
    soln.advance(time[i])
    # now for each time step pull store the temperature
    T_burn[i] = react.T
    if i > 0:
        change = abs(T_burn[i] - T_burn[i-1])
    # if change > 5 and temperature < 900.0:
    if change > 5:
        if not ignited:
            igntime = time[i]
            igntemp = T_burn[i]
        ignited = True
        # Trogdor_time[eq] = time[i]
        # Burnating_temp[eq] = T_burn[i]

print('Best ignition occured at p = {}, t = {}'.format(burnp, burnt))
print('Ignition detected after {} s'.format(igntime))
print('Initial temperature was {}'.format(temperature))
print('Ignition temperature is {}'.format(igntemp))
print('Final temperature is {}'.format(react.T))
print('Initial conditions of particle:')
print('{},{},'.format(temperature, pressure) +
      canterastring.replace(':', '='))

# plt.plot(time, T_burn)

# plt.xlabel('Residence Time')
# plt.ylabel('Temperature')
# plt.title('The Trogdor Curve [Burninating CH4 & Air]')
# plt.plot(Trogdor_time,Burnating_temp, "x", markersize=15)
# plt.legend(['phi=0.5','phi=1.0','phi=2.0','Burninating!'],loc=2)

# plt.show()
