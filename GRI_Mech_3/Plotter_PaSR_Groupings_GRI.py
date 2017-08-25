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
    pasr = loadpasrdata()
    numparticles = len(pasr[:, 0])

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

# Group the data by which method was fastest
# Create lists of data to group the information
impfwork, impfratio, impfindex, impfindicator, impfCEM = [], [], [], [], []
exfwork, exfratio, exfindex, exfindicator, exfCEM = [], [], [], [], []
eqwork, eqratio, eqindex, eqindicator, eqCEM = [], [], [], [], []
failimpwork, failratio, failindex, failindicator, failCEM = [], [], [], [], []

for i in range(len(impfunctionwork)):
    if exfunctionwork[i] == -1:
        failimpwork.append(impfunctionwork[i])
        failratio.append(ratiovals[i])
        failindex.append(indexvals[i])
        failindicator.append(indicatorvals[i])
        failCEM.append(CEMAvals[i])
    elif impfunctionwork[i] < exfunctionwork[i]:
        impfwork.append(impfunctionwork[i])
        impfratio.append(ratiovals[i])
        impfindex.append(indexvals[i])
        impfindicator.append(indicatorvals[i])
        impfCEM.append(CEMAvals[i])
    elif impfunctionwork[i] > exfunctionwork[i]:
        exfwork.append(exfunctionwork[i])
        exfratio.append(ratiovals[i])
        exfindex.append(indexvals[i])
        exfindicator.append(indicatorvals[i])
        exfCEM.append(CEMAvals[i])
    else:
        eqwork.append(impfunctionwork[i])
        eqratio.append(ratiovals[i])
        eqindex.append(indexvals[i])
        eqindicator.append(indicatorvals[i])
        eqCEM.append(CEMAvals[i])

print('Implicit faster:')
print(np.shape(impfwork))
print('Explicit faster:')
print(np.shape(exfwork))
print('Equal speed:')
print(np.shape(eqwork))
print('Explicit fails:')
print(np.shape(failimpwork))

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
    ax.scatter(ratiovals, impfunctionwork, 1.0, c='b', label='vode',
               lw=0)
    ax.scatter(ratiovals, exfunctionwork, 1.0, c='r', label='dopri5',
               lw=0)
    ax.legend(fontsize='small', markerscale=5)
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
    ax1.scatter(indexvals, impfunctionwork, 1.0, c='b', label='vode',
                lw=0)
    ax1.scatter(indexvals, exfunctionwork, 1.0, c='r', label='dopri5',
                lw=0)
    ax1.legend(fontsize='small', markerscale=5)
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
    ax2.scatter(indicatorvals, impfunctionwork, 1.0, c='b',
                label='vode', lw=0)
    ax2.scatter(indicatorvals, exfunctionwork, 1.0, c='r',
                label='dopri5',
                lw=0)
    ax2.legend(fontsize='small', markerscale=5)
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
    pyl.xlim(1e-16, max(CEMAvals))
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
    ax3.scatter(CEMAvals, impfunctionwork, 1.0, c='b', label='vode',
                lw=0)
    ax3.scatter(CEMAvals, exfunctionwork, 1.0, c='r', label='dopri5',
                lw=0)
    ax3.legend(fontsize='small', markerscale=5)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR_Fn_Work_CEMA_' + str(dt)
        # if diffcolors:
        #     name += '_color'
        pyl.savefig(name + '.' + figformat)
    plotnum += 1

    # Plot the stiffness ratio of the groupings
    fig4 = pyl.figure(plotnum)
    pyl.xlabel('Stiffness Ratio')
    pyl.ylabel('Function Calls')
    pyl.ylim(min(min(impfunctionwork), min(exfunctionwork)),
             max(max(impfwork),
                 max(exfwork),
                 max(eqwork)  # ,
                 # max(failimpwork)
                 ))
    # pyl.xlim(min(ratiovals), max(ratiovals))
    pyl.xscale('log')
    # colors = plt.cm.spectral(np.linspace(0, 1, pasrfilesloaded))
    ax4 = fig4.add_subplot(111)
    # if diffcolors:
    #     for i in range(pasrfilesloaded):
    #         ax2.scatter(ratiovals[i*100100:(i+1)*100100],
    #                     impfunctionwork[i*100100:(i+1)*100100],
    #                     color=colors[i],
    #                     label='PaSR ' + str(i) + ' Data'
    #                     )
    #     ax2.legend(fontsize='small')
    # else:
    ax4.scatter(impfratio, impfwork, 1.0, c='b', label='vode Faster',
                lw=0)
    ax4.scatter(exfratio, exfwork, 1.0, c='r', label='dopri5 Faster',
                lw=0)
    ax4.scatter(eqratio, eqwork, 1.0, c='g', label='Equal Speeds',
                lw=0)
    ax4.scatter(failratio, failimpwork, 1.0, c='y',
                label='dopri5 Failed',
                lw=0)
    ax4.legend(fontsize='small', markerscale=5)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR_Fn_Work_Ratio_Groupings_' + str(dt)
        # if diffcolors:
        #     name += '_color'
        pyl.savefig(name + '.' + figformat)
    plotnum += 1

    # Plot the stiffness index of the groupings
    fig5 = pyl.figure(plotnum)
    pyl.xlabel('Stiffness Index')
    pyl.ylabel('Function Calls')
    pyl.ylim(min(min(impfunctionwork), min(exfunctionwork)),
             max(max(impfwork),
                 max(exfwork),
                 max(eqwork)  # ,
                 # max(failimpwork)
                 ))
    # pyl.xlim(min(ratiovals), max(ratiovals))
    pyl.xscale('log')
    # colors = plt.cm.spectral(np.linspace(0, 1, pasrfilesloaded))
    ax5 = fig5.add_subplot(111)
    # if diffcolors:
    #     for i in range(pasrfilesloaded):
    #         ax2.scatter(ratiovals[i*100100:(i+1)*100100],
    #                     impfunctionwork[i*100100:(i+1)*100100],
    #                     color=colors[i],
    #                     label='PaSR ' + str(i) + ' Data'
    #                     )
    #     ax2.legend(fontsize='small')
    # else:
    ax5.scatter(impfindex, impfwork, 1.0, c='b', label='vode Faster',
                lw=0)
    ax5.scatter(exfindex, exfwork, 1.0, c='r', label='dopri5 Faster',
                lw=0)
    ax5.scatter(eqindex, eqwork, 1.0, c='g', label='Equal Speeds',
                lw=0)
    ax5.scatter(failindex, failimpwork, 1.0, c='y',
                label='dopri5 Failed',
                lw=0)
    ax5.legend(fontsize='small', markerscale=5)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR_Fn_Work_Index_Groupings_' + str(dt)
        # if diffcolors:
        #     name += '_color'
        pyl.savefig(name + '.' + figformat)
    plotnum += 1

    # Plot the stiffness indicator of the groupings
    fig6 = pyl.figure(plotnum)
    pyl.xlabel('Stiffness Indicator')
    pyl.ylabel('Function Calls')
    pyl.ylim(min(min(impfunctionwork), min(exfunctionwork)),
             max(max(impfwork),
                 max(exfwork),
                 max(eqwork)  # ,
                 # max(failimpwork)
                 ))
    # pyl.xlim(min(ratiovals), max(ratiovals))
    # pyl.xscale('log')
    # colors = plt.cm.spectral(np.linspace(0, 1, pasrfilesloaded))
    ax6 = fig6.add_subplot(111)
    # if diffcolors:
    #     for i in range(pasrfilesloaded):
    #         ax2.scatter(ratiovals[i*100100:(i+1)*100100],
    #                     impfunctionwork[i*100100:(i+1)*100100],
    #                     color=colors[i],
    #                     label='PaSR ' + str(i) + ' Data'
    #                     )
    #     ax2.legend(fontsize='small')
    # else:
    ax6.scatter(impfindicator, impfwork, 1.0, c='b',
                label='vode Faster',
                lw=0)
    ax6.scatter(exfindicator, exfwork, 1.0, c='r',
                label='dopri5 Faster',
                lw=0)
    ax6.scatter(eqindicator, eqwork, 1.0, c='g', label='Equal Speeds',
                lw=0)
    ax6.scatter(failindicator, failimpwork, 1.0, c='y',
                label='dopri5 Failed',
                lw=0)
    ax6.legend(fontsize='small', markerscale=5)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR_Fn_Work_Indicator_Groupings_' + str(dt)
        # if diffcolors:
        #     name += '_color'
        pyl.savefig(name + '.' + figformat)
    plotnum += 1

    # Plot the stiffness index of the groupings
    fig7 = pyl.figure(plotnum)
    pyl.xlabel('Chemical Explosive Mode')
    pyl.ylabel('Function Calls')
    pyl.ylim(min(min(impfunctionwork), min(exfunctionwork)),
             max(max(impfwork),
                 max(exfwork),
                 max(eqwork)  # ,
                 # max(failimpwork)
                 ))
    pyl.xscale('log')
    pyl.xlim(1e-16, max(CEMAvals))
    # colors = plt.cm.spectral(np.linspace(0, 1, pasrfilesloaded))
    ax7 = fig7.add_subplot(111)
    # if diffcolors:
    #     for i in range(pasrfilesloaded):
    #         ax2.scatter(ratiovals[i*100100:(i+1)*100100],
    #                     impfunctionwork[i*100100:(i+1)*100100],
    #                     color=colors[i],
    #                     label='PaSR ' + str(i) + ' Data'
    #                     )
    #     ax2.legend(fontsize='small')
    # else:
    ax7.scatter(impfCEM, impfwork, 1.0, c='b', label='vode Faster',
                lw=0)
    ax7.scatter(exfCEM, exfwork, 1.0, c='r', label='dopri5 Faster',
                lw=0)
    ax7.scatter(eqCEM, eqwork, 1.0, c='g', label='Equal Speeds',
                lw=0)
    ax7.scatter(failCEM, failimpwork, 1.0, c='y',
                label='dopri5 Failed',
                lw=0)
    ax7.legend(fontsize='small', markerscale=5)
    pyl.grid(b=True, which='both')
    if savefigures == 1:
        name = output_folder + 'PaSR_Fn_Work_CEMA_Groupings_' + str(dt)
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
