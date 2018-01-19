#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 2017

@author: andrewalferman
"""

import numpy as np
import csv as csv
import matplotlib.pyplot as plt


def readsolver(solver, dt):
    """Take the input file and return the QoI."""
    ratios, indicators, CEMAvals, inttimes, indexes, CSPs = \
        [], [], [], [], [], []
    with open('speciesdata-' + solver + '-' + dt + '.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            ratios.append(float(row[1]))
            indicators.append(float(row[2]))
            CEMAvals.append(float(row[3]))
            CSPs.append(float(row[4]))
            inttimes.append(float(row[-1]))
    ratios = np.array(ratios)
    indicators = np.array(indicators)
    CEMAvals = np.array(CEMAvals)
    inttimes = np.array(inttimes)
    with open('GRI_PaSR_index_values.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            indexes.append(float(row[0]))
    indexes = np.array(indexes)
    # MAKE SURE TO RETURN inttimes LAST!!!
    return np.array([ratios, indicators, CEMAvals, indexes, CSPs, inttimes])

dts = ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4']
figformat = 'png'
output_folder = 'Output_Plots/'

# Implemented solvers are 'cvodes', 'exp4', 'exprb43', 'radau2a', 'rkc'
solvers = ['cvodes', 'exp4', 'exprb43', 'radau2a', 'rkc']
# Implemented metrics are 'Ratios', 'Indicators', 'CEM', 'Indexes'
xlabels = ['Ratios', 'Indicators', 'CEM', 'Indexes', 'CSPs']

for t in range(len(dts)):
    # Clear all previous figures and close them all
    # for i in range(len(xlabels)):
    #     plt.figure(i)
    #     plt.clf()
    plt.close('all')

    print('Loading ' + dts[t] + '...')
    data = {}
    for key in solvers:
        data[key] = readsolver(key, dts[t])

    print('Plotting ' + dts[t] + '...')

    # Loop to plot all of the scatter points and set the xmin/xmax
    ymax = 0
    xmax = np.zeros(len(xlabels))
    xmin = np.zeros(len(xlabels))
    # First loop through the solvers to get min/max for each metric
    for key in solvers:
        ymax = max(ymax, max(data[key][-1]))
        for i in range(len(xlabels)):
            if key == solvers[0]:
                xmin[i] = min(data[key][i])
                xmax[i] = max(data[key][i])
            else:
                xmin[i] = min(xmin[i], min(data[key][i]))
                xmax[i] = max(xmax[i], max(data[key][i]))

    # Now loop through to modify failed vals to 95% ymax and plot results
    # Cheat a little and set ymax manually to make better plots, as needed
    if dts[t] == '1e-8':
        ymax = 0.000003
    elif dts[t] == '1e-7':
        ymax = 0.000010
    elif dts[t] == '1e-6':
        ymax = 0.00004
    elif dts[t] == '1e-5':
        ymax = 0.00015
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
        plt.ylim(0, ymax)
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
        elif xlabels[i] == 'CSPs':
            # Parameters for CSP plot
            plt.xscale('log')
            plt.xlim(xmin[4], xmax[4])
        plt.title(xlabels[i] + ' vs. Int Times, dt={}'.format(dts[t]))
        plotxlabel = xlabels[i]
        if plotxlabel.endswith('s'):
            plotxlabel = plotxlabel[:-1]
        if plotxlabel.endswith('e'):
            plotxlabel = plotxlabel[:-1]
        plt.xlabel(plotxlabel + ' Values')
        plt.ylabel('Integration Times')
        ax.grid(b=True, which='both')
        plt.tight_layout()
        plt.savefig(output_folder + xlabels[i] + '_' + 'Int_Times_' + dts[t] +
                    '.' + figformat, dpi=600)

plt.show()
