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
# Select the method to determine what was fastest
# Can be either 'clock', 'RHS', or 'tsteps'
fastermethod = 'clock'
# Explicit and implicit target dates
impdate = '08_30'
exdate = '08_30'
# Make this true if you want to test all of the values across the PaSR.
# Otherwise, this will run a single autoignition.
PaSR = True
pasrfilesloaded = 9
# Figure out a way of doing this later.
# diffcolors = False
# Define the range of the computation.
dt = 1.e-8
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
    pasr = loadpasrdata(9)
    numparticles = len(pasr[0, :, 0])
    numtsteps = len(pasr[:, 0, 0])

# Figure out the filenames
exsolfilename = equation + '_Solution_dopri5_' + str(dt)
exworkfilename = equation + '_FunctionWork_dopri5_' + str(dt)
exstepsfilename = equation + '_Timesteps_dopri5_' + str(dt)
exinttimingfilename = equation + '_Int_Times_dopri5_' +\
    str(dt) + '_' + exdate
impsolfilename = equation + '_Solution_vode_' + str(dt)
impworkfilename = equation + '_FunctionWork_vode_' + str(dt)
impstepsfilename = equation + '_Timesteps_vode_' + str(dt)
impinttimingfilename = equation + '_Int_Times_vode_' +\
    str(dt) + '_' + impdate
ratiofilename = equation + '_Stiffness_Ratio_' + str(dt)
indexfilename = equation + '_Stiffness_Index_' + str(dt)
indicatorfilename = equation + '_Stiffness_Indicator_' + str(dt)
CEMAfilename = equation + '_CEMA_' + str(dt)
# Append 'PaSR' to the filename if it is used
if PaSR:
    exworkfilename = 'PaSR_' + exworkfilename
    impworkfilename = 'PaSR_' + impworkfilename
    exstepsfilename = 'PaSR_' + exstepsfilename
    impstepsfilename = 'PaSR_' + impstepsfilename
    exinttimingfilename = 'PaSR_' + exinttimingfilename
    impinttimingfilename = 'PaSR_' + impinttimingfilename
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
extstepsneeded = np.load(os.path.join(os.getcwd(),
                                      data_folder +
                                      exstepsfilename +
                                      '.npy'))
imptstepsneeded = np.load(os.path.join(os.getcwd(),
                                       data_folder +
                                       impstepsfilename +
                                       '.npy'))
exinttimes = np.load(os.path.join(os.getcwd(),
                                  data_folder +
                                  exinttimingfilename +
                                  '.npy'))
impinttimes = np.load(os.path.join(os.getcwd(),
                                   data_folder +
                                   impinttimingfilename +
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

print('impfunctionwork: {}'.format(np.shape(impfunctionwork)))
print('exfunctionwork: {}'.format(np.shape(exfunctionwork)))
print('extstepsneeded: {}'.format(np.shape(extstepsneeded)))
print('imptstepsneeded: {}'.format(np.shape(imptstepsneeded)))
print('exinttimes: {}'.format(np.shape(exinttimes)))
print('impinttimes: {}'.format(np.shape(impinttimes)))
print('ratiovals: {}'.format(np.shape(ratiovals)))
print('indexvals: {}'.format(np.shape(indexvals)))
print('indicatorvals: {}'.format(np.shape(indicatorvals)))
print('CEMAvals: {}'.format(np.shape(CEMAvals)))

speciesnames = ['H', 'H$_2$', 'O', 'OH', 'H$_2$O', 'O$_2$', 'HO$_2$',
                'H$_2$O$_2$', 'Ar', 'He', 'CO', 'CO$_2$', 'N$_2$']

tlist = np.arange(tstart, tstop + 0.5 * dt, dt)

for i in range(10):
    print(imptstepsneeded[i])
    print(extstepsneeded[i])
    print(impinttimes[i])
    print(impinttimes[i])
    print(impfunctionwork[i])
    print(exfunctionwork[i])
    print('----')

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

# Group the data by which method was fastest
# Create lists of data to group the information
impfmeasure, impfratio, impfindex, impfindicator, impfCEM =\
    [], [], [], [], []
exfmeasure, exfratio, exfindex, exfindicator, exfCEM =\
    [], [], [], [], []
eqmeasure, eqratio, eqindex, eqindicator, eqCEM =\
    [], [], [], [], []
failimpmeasure, failratio, failindex, failindicator, failCEM =\
    [], [], [], [], []

if fastermethod == 'RHS':
    ylabel = 'Function Calls'
    impmeasure = impfunctionwork.copy()
    exmeasure = exfunctionwork.copy()
    savestring = '_Fn_Work_'
    for i in range(len(impfunctionwork)):
        if exfunctionwork[i] == -1:
            failimpmeasure.append(impfunctionwork[i])
            failratio.append(ratiovals[i])
            failindex.append(indexvals[i])
            failindicator.append(indicatorvals[i])
            failCEM.append(CEMAvals[i])
        elif impfunctionwork[i] < exfunctionwork[i]:
            impfmeasure.append(impfunctionwork[i])
            impfratio.append(ratiovals[i])
            impfindex.append(indexvals[i])
            impfindicator.append(indicatorvals[i])
            impfCEM.append(CEMAvals[i])
        elif impfunctionwork[i] > exfunctionwork[i]:
            exfmeasure.append(exfunctionwork[i])
            exfratio.append(ratiovals[i])
            exfindex.append(indexvals[i])
            exfindicator.append(indicatorvals[i])
            exfCEM.append(CEMAvals[i])
        else:
            eqmeasure.append(impfunctionwork[i])
            eqratio.append(ratiovals[i])
            eqindex.append(indexvals[i])
            eqindicator.append(indicatorvals[i])
            eqCEM.append(CEMAvals[i])
elif fastermethod == 'clock':
    ylabel = 'Clock Time (s)'
    impmeasure = impinttimes.copy()
    exmeasure = exinttimes.copy()
    savestring = '_Int_Times_'
    for i in range(len(impfunctionwork)):
        if exfunctionwork[i] == -1:
            failimpmeasure.append(impinttimes[i])
            failratio.append(ratiovals[i])
            failindex.append(indexvals[i])
            failindicator.append(indicatorvals[i])
            failCEM.append(CEMAvals[i])
        elif impinttimes[i] < exinttimes[i]:
            impfmeasure.append(impinttimes[i])
            impfratio.append(ratiovals[i])
            impfindex.append(indexvals[i])
            impfindicator.append(indicatorvals[i])
            impfCEM.append(CEMAvals[i])
        elif impinttimes[i] > exinttimes[i]:
            exfmeasure.append(exinttimes[i])
            exfratio.append(ratiovals[i])
            exfindex.append(indexvals[i])
            exfindicator.append(indicatorvals[i])
            exfCEM.append(CEMAvals[i])
        else:
            eqmeasure.append(impinttimes[i])
            eqratio.append(ratiovals[i])
            eqindex.append(indexvals[i])
            eqindicator.append(indicatorvals[i])
            eqCEM.append(CEMAvals[i])
elif fastermethod == 'tsteps':
    ylabel = 'Time Steps Taken'
    impmeasure = imptstepsneeded.copy()
    exmeasure = extstepsneeded.copy()
    savestring = '_Timesteps_'
    for i in range(len(impfunctionwork)):
        if exfunctionwork[i] == -1:
            failimpmeasure.append(imptstepsneeded[i])
            failratio.append(ratiovals[i])
            failindex.append(indexvals[i])
            failindicator.append(indicatorvals[i])
            failCEM.append(CEMAvals[i])
        elif imptstepsneeded[i] < extstepsneeded[i]:
            impfmeasure.append(imptstepsneeded[i])
            impfratio.append(ratiovals[i])
            impfindex.append(indexvals[i])
            impfindicator.append(indicatorvals[i])
            impfCEM.append(CEMAvals[i])
        elif imptstepsneeded[i] > extstepsneeded[i]:
            exfmeasure.append(extstepsneeded[i])
            exfratio.append(ratiovals[i])
            exfindex.append(indexvals[i])
            exfindicator.append(indicatorvals[i])
            exfCEM.append(CEMAvals[i])
        else:
            eqmeasure.append(imptstepsneeded[i])
            eqratio.append(ratiovals[i])
            eqindex.append(indexvals[i])
            eqindicator.append(indicatorvals[i])
            eqCEM.append(CEMAvals[i])

# Print out some statistics on how whatever ran faster
print('Implicit faster:')
print(np.shape(impfmeasure))
print('Explicit faster:')
print(np.shape(exfmeasure))
print('Equal speed:')
print(np.shape(eqmeasure))
print('Explicit fails:')
print(np.shape(failimpmeasure))

plotnum = 0
if PaSR:
    # Print the average stiffness computation and solution times
    # RHS function calls
    datanum = len(impfunctionwork)
    exworkavg = 0.0
    impworkavg = 0.0
    for i in range(datanum):
        if exfunctionwork[i] > 0:
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
    # Wall clock time
    exclockavg = 0.0
    impclockavg = 0.0
    for i in range(datanum):
        if exinttimes[i] > 0:
            exclockavg += exinttimes[i]
        impclockavg += impinttimes[i]
    exclockavg = (exclockavg / datanum)
    impclockavg = (impclockavg / datanum)
    print("Average explicit clock time: {:.7f}".format(exclockavg))
    print("Maximum explicit clock time: {:.7f}".format(
        max(exinttimes)))
    print("Average implicit clock time: {:.7f}".format(impclockavg))
    print("Maximum implicit clock time: {:.7f}".format(
        max(impinttimes)))
    # Timesteps taken
    exstepsavg = 0.0
    impstepsavg = 0.0
    for i in range(datanum):
        if extstepsneeded[i] > 0:
            exstepsavg += extstepsneeded[i]
        impstepsavg += imptstepsneeded[i]
    exstepsavg = (exstepsavg / datanum)
    impstepsavg = (impstepsavg / datanum)
    print("Average explicit timesteps taken: {:.7f}".format(exstepsavg))
    print("Maximum explicit timesteps taken: {:.7f}".format(
        max(extstepsneeded)))
    print("Average implicit timesteps taken: {:.7f}".format(impstepsavg))
    print("Maximum implicit timesteps taken: {:.7f}".format(
        max(imptstepsneeded)))

    raise Exception

    # Plot of function calls vs. computation number
    pyl.figure(0)
    pyl.xlim(0, datanum)
    pyl.ylim(0, max(max(impfunctionwork), max(exfunctionwork)))
    pyl.xlabel('Computation Number')
    pyl.ylabel('Function Calls')
    pyl.scatter(range(datanum), impfunctionwork, 1.0, c='b', label='vode',
                lw=0)
    pyl.scatter(range(datanum), exfunctionwork, 1.0, c='r', label='dopri5',
                lw=0)
    pyl.legend(fontsize='small', markerscale=5)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    'PaSR_Fn_Work_' +
                    str(dt) +
                    '.' + figformat)
    plotnum += 1

    # Plot of clock time vs. computation number
    pyl.figure(0)
    pyl.xlim(0, datanum)
    pyl.ylim(0, max(max(impinttimes), max(exinttimes)))
    pyl.xlabel('Computation Number')
    pyl.ylabel('Clock Time (s)')
    pyl.scatter(range(datanum), impinttimes, 1.0, c='b', label='vode',
                lw=0)
    pyl.scatter(range(datanum), exinttimes, 1.0, c='r', label='dopri5',
                lw=0)
    pyl.legend(fontsize='small', markerscale=5)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    'PaSR_Int_Times_' +
                    str(dt) +
                    '.' + figformat)
    plotnum += 1

    # Plot of time steps needed vs. computation number
    pyl.figure(0)
    pyl.xlim(0, datanum)
    pyl.ylim(0, max(max(imptstepsneeded), max(extstepsneeded)))
    pyl.xlabel('Computation Number')
    pyl.ylabel(ylabel)
    pyl.scatter(range(datanum), imptstepsneeded, 1.0, c='b', label='vode',
                lw=0)
    pyl.scatter(range(datanum), extstepsneeded, 1.0, c='r', label='dopri5',
                lw=0)
    pyl.legend(fontsize='small', markerscale=5)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        pyl.savefig(output_folder +
                    'PaSR_Timesteps_' +
                    str(dt) +
                    '.' + figformat)
    plotnum += 1

    # Plot of timesteps neeeded vs. stiffness ratio
    fig = pyl.figure(plotnum)
    pyl.xlabel('Stiffness Ratio')
    pyl.ylabel(ylabel)
    pyl.ylim(min(min(impmeasure), min(exmeasure)),
             max(max(impmeasure), max(exmeasure)))
    # pyl.xlim(min(ratiovals), max(ratiovals))
    pyl.xscale('log')
    # colors = plt.cm.spectral(np.linspace(0, 1, pasrfilesloaded))
    ax = fig.add_subplot(111)
    ax.scatter(ratiovals, impmeasure, 1.0, c='b', label='vode',
               lw=0)
    ax.scatter(ratiovals, exmeasure, 1.0, c='r', label='dopri5',
               lw=0)
    ax.legend(fontsize='small', markerscale=5)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR' + savestring + 'Stiffness_Ratio_' +\
            str(dt)
        pyl.savefig(name + '.' + figformat)
    plotnum += 1

    # Plot of timing measure vs. stiffness index
    fig1 = pyl.figure(plotnum)
    pyl.xlabel('Stiffness Index')
    pyl.ylabel(ylabel)
    pyl.ylim(min(min(impmeasure), min(exmeasure)),
             max(max(impmeasure), max(exmeasure)))
    pyl.xlim(min(indexvals), max(indexvals))
    pyl.xscale('log')
    # colors = plt.cm.spectral(np.linspace(0, 1, pasrfilesloaded))
    ax1 = fig1.add_subplot(111)
    ax1.scatter(indexvals, impmeasure, 1.0, c='b', label='vode',
                lw=0)
    ax1.scatter(indexvals, exmeasure, 1.0, c='r', label='dopri5',
                lw=0)
    ax1.legend(fontsize='small', markerscale=5)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR' + savestring + 'Stiffness_Index_' +\
            str(dt)
        pyl.savefig(name + '.' + figformat)
    plotnum += 1

    # Plot of timing measure vs. stiffness indicator
    fig2 = pyl.figure(plotnum)
    pyl.xlabel('Stiffness Indicator')
    pyl.ylabel(ylabel)
    pyl.ylim(min(min(impmeasure), min(exmeasure)),
             max(max(impmeasure), max(exmeasure)))
    pyl.xlim(min(indicatorvals), max(indicatorvals))
    # colors = plt.cm.spectral(np.linspace(0, 1, pasrfilesloaded))
    ax2 = fig2.add_subplot(111)
    ax2.scatter(indicatorvals, impmeasure, 1.0, c='b',
                label='vode', lw=0)
    ax2.scatter(indicatorvals, exmeasure, 1.0, c='r',
                label='dopri5',
                lw=0)
    ax2.legend(fontsize='small', markerscale=5)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR' + savestring + 'Stiffness_Indicator_' +\
            str(dt)
        pyl.savefig(name + '.' + figformat)
    plotnum += 1

    # Plot of timing measure vs. chemical explosive mode
    fig3 = pyl.figure(plotnum)
    pyl.xlabel('Chemical Explosive Mode')
    pyl.ylabel(ylabel)
    pyl.ylim(min(min(impmeasure), min(exmeasure)),
             max(max(impmeasure), max(exmeasure)))
    pyl.xlim(1e-16, max(CEMAvals))
    pyl.xscale('log')
    # colors = plt.cm.spectral(np.linspace(0, 1, pasrfilesloaded))
    ax3 = fig3.add_subplot(111)
    ax3.scatter(CEMAvals, impmeasure, 1.0, c='b', label='vode',
                lw=0)
    ax3.scatter(CEMAvals, exmeasure, 1.0, c='r', label='dopri5',
                lw=0)
    ax3.legend(fontsize='small', markerscale=5)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR' + savestring + 'CEMA_' + str(dt)
        pyl.savefig(name + '.' + figformat)
    plotnum += 1

    # Plot the stiffness ratio of the groupings
    fig4 = pyl.figure(plotnum)
    pyl.xlabel('Stiffness Ratio')
    pyl.ylabel(ylabel)
    pyl.ylim(min(min(impmeasure), min(exmeasure)),
             max(max(impfmeasure),
                 max(exfmeasure),
                 max(eqmeasure)  # ,
                 # max(failimpwork)
                 ))
    pyl.xscale('log')
    ax4 = fig4.add_subplot(111)
    ax4.scatter(impfratio, impfmeasure, 1.0, c='b', label='vode Faster',
                lw=0)
    ax4.scatter(exfratio, exfmeasure, 1.0, c='r', label='dopri5 Faster',
                lw=0)
    ax4.scatter(eqratio, eqmeasure, 1.0, c='g', label='Equal Speeds',
                lw=0)
    ax4.scatter(failratio, failimpmeasure, 1.0, c='y',
                label='dopri5 Failed',
                lw=0)
    ax4.legend(fontsize='small', markerscale=5)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR' + savestring + 'Ratio_Groupings_' +\
            str(dt)
        pyl.savefig(name + '.' + figformat)
    plotnum += 1

    # Plot the stiffness index of the groupings
    fig5 = pyl.figure(plotnum)
    pyl.xlabel('Stiffness Index')
    pyl.ylabel(ylabel)
    pyl.ylim(min(min(impmeasure), min(exmeasure)),
             max(max(impfmeasure),
                 max(exfmeasure),
                 max(eqmeasure)  # ,
                 # max(failimpwork)
                 ))
    # pyl.xlim(min(ratiovals), max(ratiovals))
    pyl.xscale('log')
    ax5 = fig5.add_subplot(111)
    ax5.scatter(impfindex, impfmeasure, 1.0, c='b', label='vode Faster',
                lw=0)
    ax5.scatter(exfindex, exfmeasure, 1.0, c='r', label='dopri5 Faster',
                lw=0)
    ax5.scatter(eqindex, eqmeasure, 1.0, c='g', label='Equal Speeds',
                lw=0)
    ax5.scatter(failindex, failimpmeasure, 1.0, c='y',
                label='dopri5 Failed',
                lw=0)
    ax5.legend(fontsize='small', markerscale=5)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR' + savestring + 'Index_Groupings_' +\
            str(dt)
        # if diffcolors:
        #     name += '_color'
        pyl.savefig(name + '.' + figformat)
    plotnum += 1

    # Plot the stiffness indicator of the groupings
    fig6 = pyl.figure(plotnum)
    pyl.xlabel('Stiffness Indicator')
    pyl.ylabel(ylabel)
    pyl.ylim(min(min(impmeasure), min(exmeasure)),
             max(max(impfmeasure),
                 max(exfmeasure),
                 max(eqmeasure)  # ,
                 # max(failimpwork)
                 ))
    ax6 = fig6.add_subplot(111)
    ax6.scatter(impfindicator, impfmeasure, 1.0, c='b',
                label='vode Faster',
                lw=0)
    ax6.scatter(exfindicator, exfmeasure, 1.0, c='r',
                label='dopri5 Faster',
                lw=0)
    ax6.scatter(eqindicator, eqmeasure, 1.0, c='g', label='Equal Speeds',
                lw=0)
    ax6.scatter(failindicator, failimpmeasure, 1.0, c='y',
                label='dopri5 Failed',
                lw=0)
    ax6.legend(fontsize='small', markerscale=5)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR' + savestring + 'Indicator_Groupings_' +\
            str(dt)
        # if diffcolors:
        #     name += '_color'
        pyl.savefig(name + '.' + figformat)
    plotnum += 1

    # Plot the chemical explosive mode of the groupings
    fig7 = pyl.figure(plotnum)
    pyl.xlabel('Chemical Explosive Mode')
    pyl.ylabel(ylabel)
    pyl.ylim(min(min(impmeasure), min(exmeasure)),
             max(max(impfmeasure),
                 max(exfmeasure),
                 max(eqmeasure)  # ,
                 # max(failimpwork)
                 ))
    pyl.xscale('log')
    pyl.xlim(1e-16, max(CEMAvals))
    # colors = plt.cm.spectral(np.linspace(0, 1, pasrfilesloaded))
    ax7 = fig7.add_subplot(111)
    ax7.scatter(impfCEM, impfmeasure, 1.0, c='b', label='vode Faster',
                lw=0)
    ax7.scatter(exfCEM, exfmeasure, 1.0, c='r', label='dopri5 Faster',
                lw=0)
    ax7.scatter(eqCEM, eqmeasure, 1.0, c='g', label='Equal Speeds',
                lw=0)
    ax7.scatter(failCEM, failimpmeasure, 1.0, c='y',
                label='dopri5 Failed',
                lw=0)
    ax7.legend(fontsize='small', markerscale=5)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR' + savestring + 'CEMA_Groupings_' +\
            str(dt)
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
    pyl.scatter(tlist, exprimaryvals, 1.0, c='r', lw=0, label='dopri5')
    pyl.scatter(tlist, impprimaryvals, 1.0, c='b', lw=0, label='vode')
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
    pyl.scatter(tlist, impfunctionwork, 1.0, c='b', lw=0, label='vode')
    pyl.scatter(tlist, exfunctionwork, 1.0, c='r', lw=0, label='dopri5')
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
