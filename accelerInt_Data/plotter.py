#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:34 2017

@author: andrewalferman
"""

import numpy as np
import csv as csv
import matplotlib.pyplot as plt


def readsolver(solver, dt):
    """Take the input file and return the QoI."""
    ratios, indicators, CEMAvals, inttimes = [], [], [], []
    with open('speciesdata-' + solver + dt + '.csv', newline='') as csvfile:
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
    return [ratios, indicators, CEMAvals, inttimes]


dt = '-1e-5'

solvers = ['cvodes', 'radau2a', 'exprb43', 'radau2a', 'rkc']
data = {}
for key in solvers:
    data[key] = readsolver(key, dt)

# Clear all previous figures and close them all
for i in range(15):
    plt.figure(i)
    plt.clf()
plt.close('all')

print('Plotting...')



# Loop to plot all of the scatter points
ymax = 0
xmax = np.zeros(3)
xmin = np.zeros(3)
for key in solvers:
    ymax = max(ymax, max(data[key][3]))
    for i in range(3):
        plt.figure(i)
        plt.scatter(data[key][i], data[key][3], 1.0, lw=0, label=key)
        xmax[i] = max(xmax[i], max([i for i in data[key][i] if i != -1.0]))
        xmin[i] = min(xmin[i], min([i for i in data[key][i] if i != -1.0]))

ymax = 0.00005

plt.figure(0)
plt.title('Ratios vs. Int Times')
plt.xlabel('Ratio Values')
plt.xscale('log')
plt.xlim(1e-9, xmax[0])
plt.ylim(0, ymax)

plt.figure(1)
plt.title('Indicators vs. Int Times')
plt.xlabel('Indicator Values')
plt.xlim(xmin[1], xmax[1])
plt.ylim(0, ymax)

plt.figure(2)
plt.title('CEM vs. Int Times')
plt.xlabel('CEM Values')
plt.xlim(max(1e-9, xmin[2]), xmax[2])
plt.ylim(0, ymax)

for i in range(3):
    plt.figure(i)
    plt.ylabel('Integration Times')
    plt.grid(b=True, which='both')
    plt.legend(fontsize='small', markerscale=5)
    plt.tight_layout()

plt.show()
