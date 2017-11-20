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
    ratios, indicators, CEMAvals, inttimes = [], [], [], []
    with open('speciesdata-' + solver + '-' + dt + '.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            ratios.append(float(row[1]))
            indicators.append(float(row[2]))
            CEMAvals.append(float(row[3]))
            inttimes.append(float(row[4]))
    ratios = np.array(ratios)
    indicators = np.array(indicators)
    CEMAvals = np.array(CEMAvals)
    inttimes = np.array(inttimes)
    return np.array([ratios, indicators, CEMAvals, inttimes])


# dts = ['1e-6', '1e-5', '1e-4']
dts = ['1e-8', '1e-7']
figformat = 'png'
output_folder = 'Output_Plots/'

solvers = ['cvodes', 'radau2a', 'exprb43', 'radau2a', 'rkc']
xlabels = ['Ratios', 'Indicators', 'CEM']

# Clear all previous figures and close them all
for i in range(15):
    plt.figure(i)
    plt.clf()
plt.close('all')

for t in range(len(dts)):
    data = {}
    for key in solvers:
        data[key] = readsolver(key, dts[t])

    print('Plotting ' + dts[t] + '...')

    # Loop to plot all of the scatter points and set the xmin/xmax
    ymax = 0
    xmax = np.zeros(3)
    xmin = np.zeros(3)
    # First loop through to get min/max
    for key in solvers:
        ymax = max(ymax, max(data[key][3]))
        for i in range(3):
            if key == solvers[0]:
                xmin[i] = min(data[key][i])
                xmax[i] = max(data[key][i])
            else:
                xmin[i] = min(xmin[i], min(data[key][i]))
                xmax[i] = max(xmax[i], max(data[key][i]))
    # Now loop through to modify failed vals to 95% ymax and plot results
    for key in solvers:
        for i in range(len(data[key][3])):
            if data[key][3][i] < 0:
                data[key][3][i] = 0.95 * ymax
        for i in range(3):
            plt.figure(t * len(dts) + i)
            plt.scatter(data[key][i], data[key][3], 1.0, lw=0, label=key)

    # Set the limits and unique values for each plot
    plt.figure(t * len(dts) + 0)
    plt.xscale('log')
    plt.xlim(xmin[0], xmax[0])
    plt.ylim(0, ymax)

    plt.figure(t * len(dts) + 1)
    plt.xlim(xmin[1], xmax[1])
    plt.ylim(0, ymax)

    plt.figure(t * len(dts) + 2)
    plt.xlim(max(1e-9, xmin[2]), xmax[2])
    plt.ylim(0, ymax)

    # Set up the labels and options, then plot
    for i in range(len(xlabels)):
        plt.figure(i)
        plt.title(xlabels[i] + ' vs. Int Times, dt={}'.format(dts[t]))
        plt.xlabel(xlabels[i] + ' Values')
        plt.ylabel('Integration Times')
        plt.grid(b=True, which='both')
        plt.legend(fontsize='small', markerscale=5)
        plt.tight_layout()
        plt.savefig(output_folder + xlabels[i] + '_' + 'Int_Times_' + dts[t] +
                    '.' + figformat)

plt.show()
