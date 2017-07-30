#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:34:13 2017

@author: andrewalferman
"""

import numpy as np
import pylab as pyl
import datetime
import os as os
import time as timer


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


"""
-------------------------------------------------------------------------------
All of the values that need to be adjusted should be in this section.
"""
# Describe the data files to be loaded
targetdate = '07_28'
# targetdate = timer.strftime("%m_%d")
# Possible options are 'Stiffness_Index', 'Stiffness_Indicator', 'CEMA',
# 'Stiffness_Ratio'
method = 'Stiffness_Index'
# Possible options will be 'VDP', 'Autoignition', or 'Oregonator'
# Oregonator not yet implemented
equation = 'Autoignition'
# Make this true if you want to plot the reference timescale of the stiffness
# indicator.
findtimescale = False
# Make this true if you want to test all of the values across the PaSR.
# Otherwise, this will run a single autoignition at particle 92, timestep 4.
PaSR = True
pasrfilesloaded = 9
# Define the range of the computation.
dt = 1.e-6
tstart = 0.
tstop = 0.2
# Make the plot of the stiffness across the entire PaSR data range.
makerainbowplot = False
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
output_folder = 'Output_Plots/'
data_folder = 'Output_Data/'
if equation == 'Autoignition':
    pasr = loadpasrdata(pasrfilesloaded)
    numparticles = len(pasr[0, :, 0])
    numtsteps = len(pasr[:, 0, 0])
# Figure out the filenames
solfilename = equation + '_Solution_' + str(dt)
metricfilename = equation + '_' + method + '_Vals_' + str(dt)
inttimingfilename = equation + '_Int_Times_' + str(dt) + '_' + targetdate
metrictimingfilename = equation + '_' + method + '_Timing_' + str(dt) +\
    targetdate
timescalefilename = equation + '_Indicator_Timescales_' + str(dt)
# Append 'PaSR' to the filename if it is used
if PaSR:
    metricfilename = 'PaSR_' + metricfilename
    inttimingfilename = 'PaSR_' + inttimingfilename
    metrictimingfilename = 'PaSR_' + metrictimingfilename
    timescalefilename = 'PaSR_' + timescalefilename
    pasrstiffnessfilename = 'PaSR_Stiffnesses_' + method + '_' + str(dt)
# Load everything
if PaSR:
    stiffvals = np.load(os.path.join(os.getcwd(),
                                     data_folder +
                                     metricfilename +
                                     '.npy'))
    if makerainbowplot:
        pasrstiffnesses = np.load(os.path.join(os.getcwd(),
                                               data_folder +
                                               pasrstiffnessfilename +
                                               '.npy'))
else:
    solution = np.load(os.path.join(os.getcwd(),
                                    data_folder +
                                    solfilename +
                                    '.npy'))
    stiffvalues = np.load(os.path.join(os.getcwd(),
                                       data_folder +
                                       metricfilename +
                                       '.npy'))
if findtimescale:
    timescales = np.load(os.path.join(os.getcwd(),
                                      data_folder +
                                      timescalefilename +
                                      '.npy'))
solutiontimes = np.load(os.path.join(os.getcwd(),
                                     data_folder +
                                     inttimingfilename +
                                     '.npy'))
stiffcomptimes = np.load(os.path.join(os.getcwd(),
                                      data_folder +
                                      metrictimingfilename +
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
    primaryvals = np.array(solution[:, 0])
    if len(tlist) == len(primaryvals) + 1:
        tlist = tlist[1:]

plotnum = 0
if PaSR:
    # Ratios of the stiffness computation times to integration times
    ratios = []
    for i in range(len(solutiontimes)):
        ratios.append(stiffcomptimes[i] / solutiontimes[i])

    # Print the average stiffness computation and solution times
    datanum = len(solutiontimes)
    solavg = 0.0
    stiffavg = 0.0
    for i in range(len(solutiontimes)):
        solavg += solutiontimes[i]
        stiffavg += stiffcomptimes[i]
    solavg = 1000. * (solavg / datanum)
    stiffavg = 1000. * (stiffavg / datanum)
    print("Average integration time (ms): {:.7f}".format(solavg))
    print("Average stiffness metric comp time (ms): {:.7f}".format(stiffavg))
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
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    'PaSR_Integration_Times_' +
                    str(dt) +
                    '_' + targetdate +
                    '.' + figformat)
    plotnum += 1

    # Plot of stiffness computation times vs. computation number
    pyl.figure(plotnum)
    pyl.xlim(0, datanum)
    pyl.ylim(0, max(stiffcomptimes))
    pyl.xlabel('Computation Number')
    pyl.ylabel('Stiffness Metric Computation Time')
    pyl.scatter(range(datanum), stiffcomptimes, 0.1)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    'PaSR_' +
                    method +
                    '_Comp_Times_' +
                    str(dt) +
                    '_' + targetdate +
                    '.' + figformat)
    plotnum += 1

    # Plot of ratio of stiffness computation times vs. integration times
    pyl.figure(plotnum)
    pyl.xlim(0, datanum)
    pyl.ylim(0, max(ratios))
    pyl.xlabel('Particle Number')
    pyl.ylabel('Ratio')
    pyl.scatter(range(datanum), ratios, 0.1)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    'PaSR_' +
                    method +
                    '_Comp_Ratios_' +
                    str(dt) +
                    '_' + targetdate +
                    '.' + figformat)
    plotnum += 1

    # Plot of stiffness computation times vs. stiffness metric
    pyl.figure(plotnum)
    pyl.xlabel(method)
    pyl.ylabel(method + ' Computation Time')
    pyl.xlim(min(stiffvals), max(stiffvals))
    pyl.ylim(0., max(stiffcomptimes))
    if method == 'Stiffness_Index':
        pyl.xscale('log')
    pyl.scatter(stiffvals, stiffcomptimes, 0.1)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    'PaSR_' +
                    method + '_' +
                    'Times_Vals_' +
                    str(dt) +
                    '_' + targetdate +
                    '.' + figformat)
    plotnum += 1

    # Plot of integration times vs. stiffness metric
    pyl.figure(plotnum)
    pyl.xlabel(method)
    pyl.ylabel('Integration Time')
    pyl.ylim(0., max(solutiontimes))
    pyl.xlim(min(stiffvals), max(stiffvals))
    if method == 'Stiffness_Index':
        pyl.xscale('log')
    elif method == 'CEMA':
        pyl.ylim(0, 0.001)
        pyl.xlim(0, 1.e6)
    pyl.scatter(stiffvals, solutiontimes, 0.1)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    'PaSR_Int_Times_' +
                    method +
                    '_' + str(dt) +
                    '_' + targetdate +
                    '.' + figformat)
    plotnum += 1

    if makerainbowplot:
        # Plot the stiffness at every point in the PaSR simulation
        # Create a mesh to plot on
        xcoords = np.arange(numparticles)
        ycoords = np.arange(numtsteps)
        xmesh, ymesh = np.meshgrid(xcoords, ycoords)

        pyl.figure(plotnum)
        # pyl.xlim(0,numparticles)
        # pyl.ylim(0,numtsteps)
        pyl.xlabel('Particle Number')
        pyl.ylabel('PaSR Timestep')
        plot = pyl.contourf(xmesh, ymesh, pasrstiffnesses, 50)
        pyl.grid(b=True, which='both')
        cb = pyl.colorbar(plot)
        if method == 'Stiffness_Index':
            label = cb.set_label('log$_{10}$ (' + method + ')')
        else:
            label = cb.set_label(method)
        if savefigures == 1:
            pyl.savefig(output_folder +
                        'PaSR_' +
                        method +
                        '_' + targetdate +
                        '.' + figformat)
        plotnum += 1

else:
    # Deleting a few values that weren't found using CD formula
    if method == 'Stiffness_Index':
        tlist = tlist[:-3]
        primaryvals = primaryvals[:-3]
        solutiontimes = solutiontimes[:-3]
        stiffvalues = stiffvalues[:-3]
    # Create a list of normalized dt values
    normtlist = []
    for i in tlist:
        normtlist.append(i / (tstop - tstart))
    if method == 'Stiffness_Indicator':
        tlist = normtlist
        plotx = 0, 1
    else:
        plotx = tstart, tstop

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
    pyl.plot(tlist, primaryvals)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    equation + '_' +
                    ylab + '_' +
                    str(dt) +
                    '_' + targetdate +
                    '.' + figformat)
    plotnum += 1

    # Plot the time per integration
    pyl.figure(plotnum)
    pyl.xlabel('Time (sec)')
    pyl.ylabel('Integration time (sec)')
    # pyl.ylim(0, 0.005)
    pyl.xlim(plotx)
    pyl.plot(tlist, solutiontimes)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    equation + '_Integration_Times_' +
                    str(dt) +
                    '_' + targetdate +
                    '.' + figformat)
    plotnum += 1

    # Plot the stiffness metric vs. time
    pyl.figure(plotnum)
    pyl.xlabel('Time (sec)')
    pyl.ylabel(method)
    pyl.xlim(plotx)
    if method == 'Stiffness_Ratio':
        pyl.scatter(tlist, stiffvalues, 0.1)
    else:
        pyl.plot(tlist, stiffvalues)
    if method == 'Stiffness_Index' or method == 'Stiffness_Ratio':
        pyl.yscale('log')
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    equation + '_' +
                    method + '_' +
                    str(dt) +
                    '_' + targetdate +
                    '.' + figformat)
    plotnum += 1

    # Plot the stiffness metric vs. integration time
    pyl.figure(plotnum)
    pyl.xlabel(method)
    pyl.ylabel('Integration Time (sec)')
    pyl.xlim(min(stiffvalues), max(stiffvalues))
    if method == 'CEMA':
        pyl.ylim(0, 0.00025)
    else:
        pyl.ylim(0, max(solutiontimes))
    pyl.scatter(stiffvalues, solutiontimes, 0.1)
    if method == 'Stiffness_Index':
        pyl.xscale('log')
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    equation + '_' +
                    'Int_Times_' +
                    method + '_' +
                    str(dt) +
                    '_' + targetdate +
                    '.' + figformat)
    plotnum += 1

    if findtimescale:
        # Plot the reference timescales vs. time
        pyl.figure(plotnum)
        pyl.xlabel('Time (sec)')
        pyl.ylabel('Reference Timescale')
        pyl.xlim(plotx)
        pyl.plot(tlist, timescales)
        pyl.grid(b=True, which='both')
        if savefigures == 1:
            pyl.savefig(output_folder +
                        equation + '_Ref_Timescale_' +
                        str(dt) +
                        '_' + targetdate +
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
