#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 09:50 2017

@author: andrewalferman
"""

import os as os
import numpy as np
import pyjacob as pyjacob
import matplotlib.pyplot as plt
# import scipy as sci


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


def rearrangepasr(Y, useN2):
    """Rearrange the PaSR data so it works with pyJac."""
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
    # Y_x = Ys[newarlen - 1]
    for i in range(N2_pos, newarlen - 1):
        Ys[i] = Ys[i + 1]
    Ys[newarlen - 1] = Y_N2
    if useN2:
        initcond = Ys
    else:
        initcond = Ys[:-1]
    return initcond, Y_press


# Load the initial conditions from the PaSR files
pasr = loadpasrdata(1)
numparticles = len(pasr[0, :, 0])
numtsteps = len(pasr[:, 0, 0])

for i in pasr[469, 91, :]:
    print(i)



# # All of the species names, after the data has been rearranged
# speciesnames = ['H', 'H$_2$', 'O', 'OH', 'H$_2$O', 'O$_2$', 'HO$_2$',
#                 'H$_2$O$_2$', 'Ar', 'He', 'CO', 'CO$_2$', 'N$_2$']
# # Keep in mind that the states also have temperature data
#
# states = np.empty((14, numtsteps*numparticles))
#
# # Rearrange all of the particles so that histograms can be made
# count = 0
# for i in range(numparticles):
#     for j in range(numtsteps):
#         particle, press = rearrangepasr(pasr[j, i, :], True)
#         for k in range(len(particle)):
#             states[k, count] = particle[k]
#         count += 1
#
# # Clear all previous figures and close them all
# for i in range(15):
#     plt.figure(i)
#     plt.clf()
# plt.close('all')
#
# # Make the histograms and plots
# print('Plotting...')
# for i in range(7):
#     plt.figure(i)
#     if i == 0:
#         title = 'Temperature'
#     else:
#         title = speciesnames[i-1]
#     print(title)
#     plt.hist(states[i, :100100], bins='auto')
#     plt.title(title)
#
# plt.show()
