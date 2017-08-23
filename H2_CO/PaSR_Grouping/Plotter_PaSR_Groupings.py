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
# Possible options will be 'VDP', 'Autoignition', or 'Oregonator'
# Oregonator not yet implemented
equation = 'Autoignition'
# Specify if you want all of the stiffness metrics
getmetrics = False
# Make this true if you want to test all of the values across the PaSR.
# Otherwise, this will run a single autoignition.
PaSR = True
pasrfilesloaded = 9
# Figure out a way of doing this later.
# diffcolors = False
# Define the range of the computation.
dt = 1.e-6
tstart = 0.
tstop = 0.2
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
exsolfilename = equation + '_Solution_dopri5_' + str(dt)
exworkfilename = equation + '_FunctionWork_dopri5_' + str(dt)
impsolfilename = equation + '_Solution_vode_' + str(dt)
impworkfilename = equation + '_FunctionWork_vode_' + str(dt)
ratiofilename = equation + '_Stiffness_Ratio_' + str(dt)
indexfilename = equation + '_Stiffness_Index_' + str(dt)
indicatorfilename = equation + '_Stiffness_Indicator_' + str(dt)
CEMAfilename = equation + '_CEMA_' + str(dt)
# Append 'PaSR' to the filename if it is used
if PaSR:
    exworkfilename = 'PaSR_' + exworkfilename
    impworkfilename = 'PaSR_' + impworkfilename
    ratiofilename = 'PaSR_' + ratiofilename
    indexfilename = 'PaSR_' + indexfilename
    indicatorfilename = 'PaSR_' + indicatorfilename
    CEMAfilename = 'PaSR_' + CEMAfilename
else:
    exsolution = np.load(os.path.join(os.getcwd(),
                                      data_folder +
                                      exsolfilename +
                                      '.npy'))
    impsolution = np.load(os.path.join(os.getcwd(),
                                       data_folder +
                                       impsolfilename +
                                       '.npy'))
exfunctionwork = np.load(os.path.join(os.getcwd(),
                                      data_folder +
                                      exworkfilename +
                                      '.npy'))
impfunctionwork = np.load(os.path.join(os.getcwd(),
                                       data_folder +
                                       impworkfilename +
                                       '.npy'))
ratiovals = np.load(os.path.join(os.getcwd(),
                                 data_folder +
                                 ratiofilename +
                                 '.npy'))
indexvals = np.load(os.path.join(os.getcwd(),
                                 data_folder +
                                 indexfilename +
                                 '.npy'))
indicatorvals = np.load(os.path.join(os.getcwd(),
                                     data_folder +
                                     indicatorfilename +
                                     '.npy'))
CEMAvals = np.load(os.path.join(os.getcwd(),
                                data_folder +
                                CEMAfilename +
                                '.npy'))

print(np.shape(impfunctionwork))
print(np.shape(exfunctionwork))
print(np.shape(ratiovals))
print(np.shape(indexvals))
print(np.shape(indicatorvals))
print(np.shape(CEMAvals))

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
    datanum = len(impfunctionwork)
    exworkavg = 0.0
    impworkavg = 0.0
    for i in range(datanum):
        exworkavg += exfunctionwork[i]
        impworkavg += impfunctionwork[i]
    exworkavg = (exworkavg / datanum)
    impworkavg = (impworkavg / datanum)
    print("Average explicit RHS function calls: {:.7f}".format(exworkavg))
    print("Maximum explicit RHS function calls: {:.7f}".format(
        max(exfunctionwork)))
    print("Average implicit RHS function calls: {:.7f}".format(impworkavg))
    print("Maximum implicit RHS function calls: {:.7f}".format(
        max(impfunctionwork)))

    # Plot of function calls vs. computation number
    pyl.figure(0)
    pyl.xlim(0, datanum)
    pyl.ylim(0, max(max(impfunctionwork), max(exfunctionwork)))
    pyl.xlabel('Computation Number')
    pyl.ylabel('Function Calls')
    pyl.scatter(range(datanum), exfunctionwork, 1.0, c='r', label='dopri5',
                lw=0)
    pyl.scatter(range(datanum), impfunctionwork, 1.0, c='b', label='vode',
                lw=0)
    pyl.legend(fontsize='small')
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    'PaSR_Fn_Work_' +
                    str(dt) +
                    '.' + figformat)
    plotnum += 1

    # Plot of function calls vs. stiffness ratio
    fig = pyl.figure(plotnum)
    pyl.xlabel('Stiffness Ratio')
    pyl.ylabel('Function Calls')
    pyl.ylim(min(min(impfunctionwork), min(exfunctionwork)),
             max(max(impfunctionwork), max(exfunctionwork)))
    # pyl.xlim(min(ratiovals), max(ratiovals))
    pyl.xscale('log')
    colors = plt.cm.spectral(np.linspace(0, 1, pasrfilesloaded))
    ax = fig.add_subplot(111)
    # if diffcolors:
    #     for i in range(pasrfilesloaded):
    #         ax2.scatter(ratiovals[i*100100:(i+1)*100100],
    #                     impfunctionwork[i*100100:(i+1)*100100],
    #                     color=colors[i],
    #                     label='PaSR ' + str(i) + ' Data'
    #                     )
    #     ax2.legend(fontsize='small')
    # else:
    ax.scatter(ratiovals[:, 0], impfunctionwork, 1.0, c='b', label='vode',
               lw=0)
    ax.scatter(ratiovals[:, 0], exfunctionwork, 1.0, c='r', label='dopri5',
               lw=0)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR_Fn_Work_Stiffness_Ratio_' + str(dt)
        # if diffcolors:
        #     name += '_color'
        pyl.savefig(name + '.' + figformat)
    plotnum += 1

    # Plot of function calls vs. stiffness index
    fig1 = pyl.figure(plotnum)
    pyl.xlabel('Stiffness Index')
    pyl.ylabel('Function Calls')
    pyl.ylim(min(min(impfunctionwork), min(exfunctionwork)),
             max(max(impfunctionwork), max(exfunctionwork)))
    pyl.xlim(min(indexvals), max(indexvals))
    pyl.xscale('log')
    # colors = plt.cm.spectral(np.linspace(0, 1, pasrfilesloaded))
    ax1 = fig1.add_subplot(111)
    # if diffcolors:
    #     for i in range(pasrfilesloaded):
    #         ax2.scatter(ratiovals[i*100100:(i+1)*100100],
    #                     impfunctionwork[i*100100:(i+1)*100100],
    #                     color=colors[i],
    #                     label='PaSR ' + str(i) + ' Data'
    #                     )
    #     ax2.legend(fontsize='small')
    # else:
    ax1.scatter(indexvals, impfunctionwork, 1.0, c='b', label='vode', lw=0)
    ax1.scatter(indexvals, exfunctionwork, 1.0, c='r', label='dopri5', lw=0)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR_Fn_Work_Stiffness_Index_' + str(dt)
        # if diffcolors:
        #     name += '_color'
        pyl.savefig(name + '.' + figformat)
    plotnum += 1

    # Plot of function calls vs. stiffness indicator
    fig2 = pyl.figure(plotnum)
    pyl.xlabel('Stiffness Indicator')
    pyl.ylabel('Function Calls')
    pyl.ylim(min(min(impfunctionwork), min(exfunctionwork)),
             max(max(impfunctionwork), max(exfunctionwork)))
    pyl.xlim(min(indicatorvals), max(indicatorvals))
    # colors = plt.cm.spectral(np.linspace(0, 1, pasrfilesloaded))
    ax2 = fig2.add_subplot(111)
    # if diffcolors:
    #     for i in range(pasrfilesloaded):
    #         ax2.scatter(ratiovals[i*100100:(i+1)*100100],
    #                     impfunctionwork[i*100100:(i+1)*100100],
    #                     color=colors[i],
    #                     label='PaSR ' + str(i) + ' Data'
    #                     )
    #     ax2.legend(fontsize='small')
    # else:
    ax2.scatter(indicatorvals, impfunctionwork, 1.0, c='b', label='vode', lw=0)
    ax2.scatter(indicatorvals, exfunctionwork, 1.0, c='r', label='dopri5',
                lw=0)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR_Fn_Work_Stiffness_Indicator_' + str(dt)
        # if diffcolors:
        #     name += '_color'
        pyl.savefig(name + '.' + figformat)
    plotnum += 1

    # Plot of function calls vs. chemical explosive mode
    fig3 = pyl.figure(plotnum)
    pyl.xlabel('Chemical Explosive Mode')
    pyl.ylabel('Function Calls')
    pyl.ylim(min(min(impfunctionwork), min(exfunctionwork)),
             max(max(impfunctionwork), max(exfunctionwork)))
    pyl.xlim(min(CEMAvals[:, 0]), max(CEMAvals[:, 0]))
    pyl.xscale('log')
    # colors = plt.cm.spectral(np.linspace(0, 1, pasrfilesloaded))
    ax3 = fig3.add_subplot(111)
    # if diffcolors:
    #     for i in range(pasrfilesloaded):
    #         ax2.scatter(ratiovals[i*100100:(i+1)*100100],
    #                     impfunctionwork[i*100100:(i+1)*100100],
    #                     color=colors[i],
    #                     label='PaSR ' + str(i) + ' Data'
    #                     )
    #     ax2.legend(fontsize='small')
    # else:
    ax3.scatter(CEMAvals[:, 0], impfunctionwork, 1.0, c='b', label='vode',
                lw=0)
    ax3.scatter(CEMAvals[:, 0], exfunctionwork, 1.0, c='r', label='dopri5',
                lw=0)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR_Fn_Work_CEMA_' + str(dt)
        # if diffcolors:
        #     name += '_color'
        pyl.savefig(name + '.' + figformat)
    plotnum += 1

else:
    # Deleting a few values that weren't found using CD formula for the index
    tlist = tlist[:-3]
    exprimaryvals = exprimaryvals[:-3]
    exfunctionwork = exfunctionwork[:-3]
    impprimaryvals = impprimaryvals[:-3]
    impfunctionwork = impfunctionwork[:-3]

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
    # if diffcolors:
    #     pyl.scatter(tlist, exprimaryvals, c=tlist, cmap='jet', lw=0,
    #                 label='Explicit')
    #     pyl.scatter(tlist, impprimaryvals, c=tlist, cmap='jet', lw=0,
    #                 label='Implicit')
    # else:
    pyl.scatter(tlist, exprimaryvals, 1.0, c='r', lw=0, label='Explicit')
    pyl.scatter(tlist, impprimaryvals, 1.0, c='b', lw=0, label='Implicit')
    pyl.grid(b=True, which='both')
    pyl.legend(fontsize='small')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    equation + '_' +
                    ylab + '_' +
                    str(dt) +
                    '.' + figformat)
    plotnum += 1

    # Plot the function calls per integration
    pyl.figure(plotnum)
    pyl.xlabel('Time (sec)')
    pyl.ylabel('Function Calls')
    pyl.ylim(0, max(max(impfunctionwork), max(exfunctionwork)))
    pyl.xlim(plotx)
    # if diffcolors:
    #     pyl.scatter(tlist, impfunctionwork, 1.0, c=tlist, cmap='jet', lw=0,
    #                 label='Implicit')
    #     pyl.scatter(tlist, exfunctionwork, 1.0, c=tlist, cmap='jet', lw=0,
    #                 label='Explicit')
    # else:
    pyl.scatter(tlist, impfunctionwork, 1.0, c='b', lw=0, label='Implicit')
    pyl.scatter(tlist, exfunctionwork, 1.0, c='r', lw=0, label='Explicit')
    pyl.grid(b=True, which='both')
    pyl.legend(fontsize='small')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    equation + '_Function_Work_' +
                    str(dt) +
                    '.' + figformat)
    plotnum += 1

finishtime = datetime.datetime.now()
print('Finish time: {}'.format(finishtime))

# pyl.close('all')
pyl.show()
