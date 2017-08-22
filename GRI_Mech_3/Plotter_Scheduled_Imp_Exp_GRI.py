#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:34 2017

@author: andrewalferman
"""

import numpy as np
import pylab as pyl
import datetime
import os as os
import time as timer
import matplotlib as plt


def loadpasrdata():
    """Load the initial conditions from the full PaSR file."""
    print('Loading data...')
    filepath = os.path.join(os.getcwd(), 'ch4_full_pasr_data.npy')
    return np.load(filepath)


"""
-------------------------------------------------------------------------------
All of the values that need to be adjusted should be in this section.
"""
# Describe the data files to be loaded
# Explicit run date
targetdate = '08_20'
# Implicit run date
targetdate2 = '08_21'
# targetdate = timer.strftime("%m_%d")
# Possible options are 'Stiffness_Index', 'Stiffness_Indicator', 'CEMA',
# 'Stiffness_Ratio'
method = 'Stiffness_Indicator'
# Possible options will be 'VDP', 'Autoignition', or 'Oregonator'
# Oregonator not yet implemented
equation = 'Autoignition'
# Make this true if you want to plot the reference timescale of the stiffness
# indicator.
findtimescale = False
# Make this true if you want normalized timescales
normtime = False
# Make this true if you want to test all of the values across the PaSR.
# Otherwise, this will run a single autoignition at particle 92, timestep 4.
PaSR = False
pasrfilesloaded = 9
diffcolors = False
# Define the range of the computation.
dt = 1.e-6
tstart = 0.
tstop = 0.2
# Make the plot of the stiffness across the entire PaSR data range.
makerainbowplot = False
# Plot how long it takes to compute stiffness metric
showstiffcomptime = False
# To be implemented later.
makesecondderivplots = False

# Configure the output plots
figformat = 'png'
savefigures = 1
"""
-------------------------------------------------------------------------------
"""
starttime = datetime.datetime.now()
print('Start time: {}'.format(starttime))

# Load the data
output_folder = 'Scheduler/Output_Plots/'
data_folder = 'Scheduler/Output_Data/'

if equation == 'Autoignition':
    pasr = loadpasrdata()
    numparticles = len(pasr[:, 0])
# Figure out the filenames
exsolfilename = equation + '_Solution_Explicit_' + str(dt)
exinttimingfilename = equation + '_Int_Times_Explicit_' + str(dt) + '_' +\
    targetdate
exworkfilename = equation + '_FunctionWork_Explicit_' + str(dt)
impsolfilename = equation + '_Solution_Implicit_' + str(dt)
impinttimingfilename = equation + '_Int_Times_Implicit_' + str(dt) + '_' +\
    targetdate2
impworkfilename = equation + '_FunctionWork_Implicit_' + str(dt)
# Append 'PaSR' to the filename if it is used
if PaSR:
    exinttimingfilename = 'PaSR_' + exinttimingfilename
    exworkfilename = 'PaSR_' + exworkfilename
    impinttimingfilename = 'PaSR_' + impinttimingfilename
    impworkfilename = 'PaSR_' + impworkfilename
else:
    exsolution = np.load(os.path.join(os.getcwd(),
                                      data_folder +
                                      exsolfilename +
                                      '.npy'))
    impsolution = np.load(os.path.join(os.getcwd(),
                                       data_folder +
                                       impsolfilename +
                                       '.npy'))
exsolutiontimes = np.load(os.path.join(os.getcwd(),
                                       data_folder +
                                       exinttimingfilename +
                                       '.npy'))
exfunctionwork = np.load(os.path.join(os.getcwd(),
                                      data_folder +
                                      exworkfilename +
                                      '.npy'))
impsolutiontimes = np.load(os.path.join(os.getcwd(),
                                        data_folder +
                                        impinttimingfilename +
                                        '.npy'))
impfunctionwork = np.load(os.path.join(os.getcwd(),
                                       data_folder +
                                       impworkfilename +
                                       '.npy'))

speciesnames = ['H', 'H$_2$', 'O', 'OH', 'H$_2$O', 'O$_2$', 'HO$_2$',
                'H$_2$O$_2$', 'Ar', 'He', 'CO', 'CO$_2$', 'N$_2$']

tlist = np.arange(tstart, tstop + 0.5 * dt, dt)

print('Plotting...')

# Clear all previous figures and close them all
for i in range(15):
    pyl.figure(i)
    pyl.clf()
pyl.close('all')

# Something is causing a bug in the tlist and this is intended to fix it
if not PaSR:
    impprimaryvals = np.array(impsolution[:, 0])
    exprimaryvals = np.array(exsolution[:, 0])
    if np.shape(impsolution) != np.shape(exsolution):
        print(np.shape(impsolution))
        print(np.shape(exsolution))
        raise Exception('Error: Solutions did not come out to the same shape!')
    if (len(tlist) == len(exprimaryvals) + 1 or
            len(tlist) == len(exfunctionwork) + 1):
        tlist = tlist[1:]
    # if (len(tlist) == len(impprimaryvals) + 1 or
    #         len(tlist) == len(impfunctionwork) + 1):
    #     tlist = tlist[1:]

plotnum = 0
if PaSR:
    # Print the average stiffness computation and solution times
    datanum = len(exsolutiontimes)
    exsolavg = 0.0
    impsolavg = 0.0
    for i in range(len(exsolutiontimes)):
        exsolavg += exsolutiontimes[i]
        exsolavg += impsolutiontimes[i]
    exsolavg = 1000. * (exsolavg / datanum)
    impsolavg = 1000. * (impsolavg / datanum)
    print("Average explicit integration time (ms): {:.7f}".format(exsolavg))
    print("Maximum explicit integration time (ms): {:.7f}".format(
        max(exsolutiontimes) * 1000.))
    print("Average implicit integration time (ms): {:.7f}".format(impsolavg))
    print("Maximum implicit integration time (ms): {:.7f}".format(
        max(impsolutiontimes) * 1000.))

    # Plot of integration times vs. computation number
    pyl.figure(0)
    pyl.xlim(0, datanum)
    pyl.ylim(0, max(max(exsolutiontimes, max(impsolutiontimes))))
    pyl.xlabel('Computation Number')
    pyl.ylabel('Integration Time')
    pyl.scatter(range(datanum), exsolutiontimes, 1.0, c='r', label='Explicit',
                lw=0)
    pyl.scatter(range(datanum), impsolutiontimes, 1.0, c='b', label='Implicit',
                lw=0)
    pyl.legend(fontsize='small')
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    'PaSR_Integration_Times_' +
                    str(dt) +
                    '_' + targetdate2 +
                    '.' + figformat)
    plotnum += 1

    # Plot of function calls vs. computation number
    pyl.figure(0)
    pyl.xlim(0, datanum)
    pyl.ylim(0, max(max(impfunctionwork, max(exfunctionwork))))
    pyl.xlabel('Computation Number')
    pyl.ylabel('Function Calls')
    pyl.scatter(range(datanum), exfunctionwork, 1.0, c='r', label='Explicit',
                lw=0)
    pyl.scatter(range(datanum), impfunctionwork, 1.0, c='b', label='Implicit',
                lw=0)
    pyl.legend(fontsize='small')
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    'PaSR_Fn_Work_' +
                    str(dt) +
                    '_' + targetdate2 +
                    '.' + figformat)
    plotnum += 1

else:
    # Deleting a few values that weren't found using CD formula
    if method == 'Stiffness_Index':
        tlist = tlist[:-3]
        exprimaryvals = exprimaryvals[:-3]
        exsolutiontimes = exsolutiontimes[:-3]
        exfunctionwork = exfunctionwork[:-3]
        impprimaryvals = impprimaryvals[:-3]
        impsolutiontimes = impsolutiontimes[:-3]
        impfunctionwork = impfunctionwork[:-3]
    # Create a list of normalized dt values
    normtlist = []
    for i in tlist:
        normtlist.append(i / (tstop - tstart))
    if normtime and method == 'Stiffness_Indicator':
        tlist = normtlist
        plotx = 0, 1
    else:
        plotx = tstart, tstop

    print(np.shape(tlist))
    print(np.shape(exprimaryvals))
    print(np.shape(exsolution))

    # Plot the solution of the temperature or Y value
    pyl.figure(plotnum)
    if equation == 'VDP':
        ylab = 'Y_Value'
        pyl.ylabel(ylab)
    elif equation == 'Autoignition':
        ylab = 'Temperature'
        pyl.ylabel(ylab + ' (K)')
    pyl.xlabel('Time (sec)')
    pyl.xlim(plotx)
    if diffcolors:
        pyl.scatter(tlist, exprimaryvals, c=tlist, cmap='jet', lw=0,
                    label='Explicit')
        pyl.scatter(tlist, impprimaryvals, c=tlist, cmap='jet', lw=0,
                    label='Implicit')
    else:
        pyl.scatter(tlist, exprimaryvals, 1.0, c='r', lw=0, label='Explicit')
        pyl.scatter(tlist, impprimaryvals, 1.0, c='b', lw=0, label='Implicit')
    pyl.grid(b=True, which='both')
    pyl.legend(fontsize='small')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    equation + '_' +
                    ylab + '_' +
                    str(dt) +
                    '_' + targetdate2 +
                    '.' + figformat)
    plotnum += 1

    # Plot the time per integration
    pyl.figure(plotnum)
    pyl.xlabel('Time (sec)')
    pyl.ylabel('Integration time (sec)')
    pyl.ylim(0, max(max(impsolutiontimes), max(exsolutiontimes)))
    pyl.xlim(plotx)
    if diffcolors:
        pyl.scatter(tlist, impsolutiontimes, 1.0, c=tlist, cmap='jet', lw=0,
                    label='Implicit')
        pyl.scatter(tlist, impsolutiontimes, 1.0, c=tlist, cmap='jet', lw=0,
                    label='Explicit')
    else:
        pyl.scatter(tlist, impsolutiontimes, 1.0, c='b', lw=0,
                    label='Implicit')
        pyl.scatter(tlist, exsolutiontimes, 1.0, c='r', lw=0, label='Explicit')
    pyl.grid(b=True, which='both')
    pyl.legend(fontsize='small')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    equation + '_Integration_Times_' +
                    str(dt) +
                    '_' + targetdate2 +
                    '.' + figformat)
    plotnum += 1

    # Plot the function calls per integration
    pyl.figure(plotnum)
    pyl.xlabel('Time (sec)')
    pyl.ylabel('Function Calls')
    pyl.ylim(0, max(max(impfunctionwork), max(exfunctionwork)))
    pyl.xlim(plotx)
    if diffcolors:
        pyl.scatter(tlist, impfunctionwork, 1.0, c=tlist, cmap='jet', lw=0,
                    label='Implicit')
        pyl.scatter(tlist, exfunctionwork, 1.0, c=tlist, cmap='jet', lw=0,
                    label='Explicit')
    else:
        pyl.scatter(tlist, impfunctionwork, 1.0, c='b', lw=0, label='Implicit')
        pyl.scatter(tlist, exfunctionwork, 1.0, c='r', lw=0, label='Explicit')
    pyl.grid(b=True, which='both')
    pyl.legend(fontsize='small')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    equation + '_Function_Work_' +
                    str(dt) +
                    '_' + targetdate2 +
                    '.' + figformat)
    plotnum += 1

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

"""

finishtime = datetime.datetime.now()
print('Finish time: {}'.format(finishtime))

# pyl.close('all')
pyl.show()
