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
# Select the method to determine what was fastest
# Can be either 'clock', 'RHS', or 'tsteps'
fastermethod = 'clock'
# Explicit and implicit target dates
impdate = '09_11'
exdate = '09_12'
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

# print('impfunctionwork: {}'.format(np.shape(impfunctionwork)))
# print('exfunctionwork: {}'.format(np.shape(exfunctionwork)))
# print('extstepsneeded: {}'.format(np.shape(extstepsneeded)))
# print('imptstepsneeded: {}'.format(np.shape(imptstepsneeded)))
# print('exinttimes: {}'.format(np.shape(exinttimes)))
# print('impinttimes: {}'.format(np.shape(impinttimes)))
# print('ratiovals: {}'.format(np.shape(ratiovals)))
# print('indexvals: {}'.format(np.shape(indexvals)))
# print('indicatorvals: {}'.format(np.shape(indicatorvals)))
# print('CEMAvals: {}'.format(np.shape(CEMAvals)))

speciesnames = ['H', 'H$_2$', 'O', 'OH', 'H$_2$O', 'O$_2$', 'HO$_2$',
                'H$_2$O$_2$', 'Ar', 'He', 'CO', 'CO$_2$', 'N$_2$']

tlist = np.arange(tstart, tstop + 0.5 * dt, dt)

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
print('Statistics for ' + fastermethod + ':')
print('Implicit faster:')
print(np.shape(impfmeasure))
print('Explicit faster:')
print(np.shape(exfmeasure))
print('Equal speed:')
print(np.shape(eqmeasure))
print('Explicit fails:')
print(np.shape(failimpmeasure))
print('-----------')

# Print out statistics on possible speedups
# Start with the total integration speeds of explicit/implicit
imptotalsteps = sum(imptstepsneeded)
imptotaltime = sum(impinttimes)
imptotalRHS = sum(impfunctionwork)
extotalsteps = sum(extstepsneeded)
extotaltime = sum(exinttimes)
extotalRHS = sum(exfunctionwork)
print('Total implicit integration time: {}'.format(imptotaltime))
print('Total implicit time steps: {}'.format(imptotalsteps))
print('Total implicit RHS function calls: {}'.format(imptotalRHS))
print('Total explicit integration time: {}'.format(extotaltime))
print('Total explicit time steps: {}'.format(extotalsteps))
print('Total explicit RHS function calls: {}'.format(extotalRHS))
print('-----------')
# Figure out what a perfect switching mechanism would look like
fastesttime = 0.0
leaststeps = 0
leastcalls = 0
points = 0
for i in range(len(impfunctionwork)):
    if impinttimes[i] <= exinttimes[i]:
        fastesttime += impinttimes[i]
        points += 1
    else:
        fastesttime += exinttimes[i]
        points += 1
    if imptstepsneeded[i] <= extstepsneeded[i]:
        leaststeps += imptstepsneeded[i]
    else:
        leaststeps += extstepsneeded[i]
    if impfunctionwork[i] <= exfunctionwork[i]:
        leastcalls += impfunctionwork[i]
    else:
        leastcalls += exfunctionwork[i]
print('Fastest integration time w/ perfect scheduler: {}'.format(fastesttime))
print('Fewest time steps w/ perfect scheduler: {}'.format(leaststeps))
print('Fewest RHS function calls w/ perfect scheduler: {}'.format(leastcalls))
print('Perfect scheduler speedups:')
print('Implicit time speedup: {}'.format(1 / (fastesttime / imptotaltime)))
print('Explicit time speedup: {}'.format(1 / (fastesttime / extotaltime)))
print('Implicit steps speedup: {}'.format(1 / (leaststeps / imptotalsteps)))
print('Explicit steps speedup: {}'.format(1 / (leaststeps / extotalsteps)))
print('Implicit calls speedup: {}'.format(1 / (leastcalls / imptotalRHS)))
print('Explicit calls speedup: {}'.format(1 / (leastcalls / extotalRHS)))
print('-----------')

# Use the stiffness metrics to determine scheduling
ratioschedtime = 0.0
ratioschedsteps = 0
ratioschedcalls = 0
CEMschedtime = 0.0
CEMschedsteps = 0
CEMschedcalls = 0
indexschedtime = 0.0
indexschedsteps = 0
indexschedcalls = 0
for i in range(len(impfunctionwork)):
    if indicatorvals[i] <= -1e9:
        CEMschedtime += impinttimes[i]
        CEMschedsteps += imptstepsneeded[i]
        CEMschedcalls += impfunctionwork[i]
    else:
        if exinttimes[i] > 0:
            CEMschedtime += exinttimes[i]
            CEMschedsteps += extstepsneeded[i]
            CEMschedcalls += exfunctionwork[i]
        else:
            CEMschedtime += impinttimes[i]
            CEMschedsteps += imptstepsneeded[i]
            CEMschedcalls += impfunctionwork[i]
    # if indexvals[i] >= 5e10:
    #     indexschedtime += impinttimes[i]
    #     indexschedsteps += imptstepsneeded[i]
    #     indexschedcalls += impfunctionwork[i]
    # else:
    #     indexschedtime += exinttimes[i]
    #     indexschedsteps += extstepsneeded[i]
    #     indexschedcalls += exfunctionwork[i]
    # if ratiovals[i] >= 1e22:
    #     ratioschedtime += impinttimes[i]
    #     ratioschedsteps += imptstepsneeded[i]
    #     ratioschedcalls += impfunctionwork[i]
    # else:
    #     ratioschedtime += exinttimes[i]
    #     ratioschedsteps += extstepsneeded[i]
    #     ratioschedcalls += exfunctionwork[i]
# print('Ratio Scheduler (default vode):')
# print('Integration time w/ scheduler: {}'.format(ratioschedtime))
# print('Time steps w/ scheduler: {}'.format(ratioschedsteps))
# print('RHS function calls w/ scheduler: {}'.format(ratioschedcalls))
# print('Implicit time speedup: {}'.format(1 / (ratioschedtime /
#                                               imptotaltime)))
# print('Implicit steps speedup: {}'.format(1 / (ratioschedsteps /
#                                                imptotalsteps)))
# print('Implicit calls speedup: {}'.format(1 / (ratioschedcalls /
#                                                imptotalRHS)))
print('-----------')
print('Indicator Scheduler (default vode):')
print('Integration time w/ scheduler: {}'.format(CEMschedtime))
print('Time steps w/ scheduler: {}'.format(CEMschedsteps))
print('RHS function calls w/ scheduler: {}'.format(CEMschedcalls))
print('Implicit time speedup: {}'.format(1 / (CEMschedtime / imptotaltime)))
print('Implicit steps speedup: {}'.format(1 / (CEMschedsteps / imptotalsteps)))
print('Implicit calls speedup: {}'.format(1 / (CEMschedcalls / imptotalRHS)))
print('-----------')


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

finishtime = datetime.datetime.now()
print('Finish time: {}'.format(finishtime))

# pyl.close('all')
pyl.show()
