#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 2017

@author: andrewalferman
"""

import numpy as np
import csv as csv
import matplotlib.pyplot as plt
import sys as sys


def readsolver(solver, problem, csptol, dt):
    """Take the input file and return the QoI."""
    particles, inttimes = [], []
    timingname = 'speciesdata-' + problem + '-' + solver + '-' + dt + '.csv'
    with open(timingname, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        fail = False
        for row in reader:
            if not fail:
                try:
                    particles.append(int(row[0]))
                    inttimes.append(float(row[2]))
                except ValueError:
                    fail = True
            else:
                fail = False
    orderedtimes = np.empty_like(inttimes)
    for i, particle in enumerate(particles):
        orderedtimes[particle] = inttimes[i]
    # These lines only necessary because I messed up the file naming convention
    if problem == 'grimech':
        fname = 'GRIMech-'
    elif problem == 'h2':
        fname = 'H2-'
    CSPstiff, ratio, indicator, CEM, index = [[] for i in range(5)]
    metricname = 'PaSRStiffvals-' + fname + csptol + '.csv'
    with open(metricname, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            #Mval.append(int(float(row[3])))
            CSPstiff.append(float(row[4]))
            ratio.append(float(row[5]))
            indicator.append(float(row[6]))
            CEM.append(float(row[7]))
            index.append(float(row[8]))
    #Mval = np.array(Mval)
    CSPstiff = np.array(CSPstiff)
    ratio = np.array(ratio)
    indicator = np.array(indicator)
    CEM = np.array(CEM)
    index = np.array(index)
    return np.array([ratio, indicator, CEM, index, CSPstiff, orderedtimes])

dts = ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4']
figformat = 'png'
output_folder = './'

# Implemented solvers are 'cvodes', 'exp4', 'exprb43', 'radau2a', 'rkc'
solvers = ['cvodes', 'exp4', 'exprb43', 'radau2a', 'rkc']
# Implemented metrics are 'Ratios', 'Indicators', 'CEM', 'Indexes'
xlabels = ['Ratios', 'Indicators', 'CEM', 'Indexes', 'CSP Stiffness']
problem = 'h2'
csptol = '1e-3'

for t in range(len(dts)):
    # Clear all previous figures and close them all
    # for i in range(len(xlabels)):
    #     plt.figure(i)
    #     plt.clf()
    plt.close('all')

    print('Loading ' + dts[t] + '...')
    data = {}
    for key in solvers:
        data[key] = readsolver(key, problem, csptol, dts[t])

    print('Plotting ' + dts[t] + '...')

    # Loop to plot all of the scatter points and set the xmin/xmax
    ymax = 0
    ymin = 9999
    xmax = np.zeros(len(xlabels))
    xmin = np.zeros(len(xlabels))
    # First loop through the solvers to get min/max for each metric
    for key in solvers:
        ymax = max(ymax, max(data[key][-1]))
        ymin = min(ymin, min(data[key][-1]))
        for i in range(len(xlabels)):
            if key == solvers[0]:
                xmin[i] = min(data[key][i])
                xmax[i] = max(data[key][i])
            else:
                xmin[i] = min(xmin[i], min(data[key][i]))
                xmax[i] = max(xmax[i], max(data[key][i]))

    # Now loop through to modify failed vals to 95% ymax and plot results
    # Cheat a little and set ymax manually to make better plots, as needed
    # if dts[t] == '1e-8':
    #     ymax = 0.000003
    # elif dts[t] == '1e-7':
    #     ymax = 0.000010
    # elif dts[t] == '1e-6':
    #     ymax = 0.00004
    # elif dts[t] == '1e-5':
    #     ymax = 0.00015
    for key in solvers:
        for i in range(len(data[key][-1])):
            if data[key][-1][i] < 0:
                data[key][-1][i] = 0.95 * ymax

    # Set up the labels and options, then plot
    for i in range(len(xlabels)):
        fig, ax = plt.subplots()
        for key in solvers:
            ax.scatter(data[key][i], data[key][-1], 1.0, lw=0, label=key)
        legend = ax.legend(solvers, loc='upper right', fontsize='small',
                           markerscale=5)
        plt.ylim(ymin, ymax)
        # plt.yscale('log')
        # Set the limits and unique values for each plot
        if xlabels[i] == 'Ratios':
            # Parameters for ratio plot
            plt.xscale('log')
            plt.xlim(xmin[0], xmax[0])
        elif xlabels[i] == 'Indicators':
            # Parameters for indicator plot
            plt.xlim(xmin[1], xmax[1])
        elif xlabels[i] == 'CEM':
            # Parameters for CEM plot
            plt.xlim(max(1e-9, xmin[2]), xmax[2])
        elif xlabels[i] == 'Indexes':
            # Parameters for index plot
            plt.xscale('log')
            plt.xlim(max(1e-9, xmin[3]), xmax[3])
        elif xlabels[i] == 'CSP Stiffness':
            # Parameters for CSP plot
            plt.xscale('log')
            plt.xlim(xmin[4], xmax[4])
        #elif xlabels[i] == 'Fast Modes':
            # Parameters for fast modes plot
            #plt.xlim(max(0, xmin[5]), xmax[5])
        plt.title(xlabels[i] + ' vs. Int Times, dt={}'.format(dts[t]))
        plotxlabel = xlabels[i]
        # if plotxlabel.endswith('s'):
        #     plotxlabel = plotxlabel[:-1]
        # if plotxlabel.endswith('e'):
        #     plotxlabel = plotxlabel[:-1]
        plt.xlabel(plotxlabel + ' Values')
        plt.ylabel('Integration Times')
        ax.grid(b=True, which='both')
        plt.tight_layout()
        figname = output_folder + xlabels[i] + '_' + problem + '_Int_Times_' + dts[t] + '.' + figformat
        figname.replace(" ", "_")
        plt.savefig(figname, dpi=600)

plt.show()
