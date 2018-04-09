#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 2017

@author: andrewalferman
"""

import numpy as np
import csv as csv
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
from StiffnessFuncs import *
from CSPfuncs1 import *
import sys as sys


problem = 'VDP'

[ts, ts_timing, Ms, comptimes, CSPstiffness, Y1s, Y2s, Y3s, Y4s, sol, ratios,
    indicators, CEMs] = [[] for i in range(13)]
if problem == 'CSPtest':
    with open('csptoyproblem2.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            ts.append(float(row[0]))
            Ms.append(float(row[1]))
            CSPstiffness.append(float(row[3]))
            Y1s.append(float(row[4]))
            Y2s.append(float(row[5]))
            Y3s.append(float(row[6]))
            Y4s.append(float(row[7]))
            sol.append([float(row[i]) for i in range(4,8)])
            ratios.append(float(row[8]))
            indicators.append(float(row[9]))
            CEMs.append(float(row[10]))
    with open('csptoyproblem.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            ts_timing.append(float(row[0]))
            comptimes.append(float(row[2]))
elif problem == 'VDP':
    with open('vdp.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            ts.append(float(row[0]))
            Ms.append(float(row[1]))
            CSPstiffness.append(float(row[3]))
            Y1s.append(float(row[4]))
            Y2s.append(float(row[5]))
            sol.append([float(row[i]) for i in range(4,6)])
            ratios.append(float(row[6]))
            indicators.append(float(row[7]))
            CEMs.append(float(row[8]))
elif problem == 'Oregonator':
    with open('oregonatorcsp.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            ts.append(float(row[0]))
            Ms.append(float(row[1]))
            CSPstiffness.append(float(row[3]))
            Y1s.append(float(row[4]))
            Y2s.append(float(row[5]))
            Y3s.append(float(row[6]))
            sol.append([float(row[i]) for i in range(4,7)])
            ratios.append(float(row[7]))
            indicators.append(float(row[8]))
            CEMs.append(float(row[9]))
ts = np.array(ts)
ts_timing = np.array(ts_timing)
Ms = np.array(Ms)
comptimes = np.array(comptimes)
CSPstiffness = np.array(CSPstiffness)
Y1s = np.array(Y1s)
Y2s = np.array(Y2s)
Y3s = np.array(Y3s)
Y4s = np.array(Y4s)
sol = np.array(sol)
ratios = np.array(ratios)
indicators = np.array(indicators)
CEMs.append(CEMs)

# Calculating values of the stiffness index here for convenince
if problem == 'CSPtest':
    derivfun = testfunc
    jacfun = testjac
elif problem == 'VDP':
    derivfun = dydxvdp
    jacfun = jacvdp
elif problem == 'Oregonator':
    derivfun = oregonatordydt
    jacfun = oregonatorjac

# Needed to bring over the epsilon from the driving code
indexes = stiffnessindex(ts, sol, derivfun, jacfun, 1.0e-2)

plt.close('all')

figformat = 'png'
output_folder = 'Output_Plots/'

plt.figure(num=None, figsize=(8, 4.5))
host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.70)

par1 = host.twinx()
par2 = host.twinx()
par3 = host.twinx()

offset = 60
new_fixed_axis = par1.get_grid_helper().new_fixed_axis
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
new_fixed_axis = par3.get_grid_helper().new_fixed_axis
par1.axis["right"] = new_fixed_axis(loc="right",
                                    axes=par1,
                                    offset=(0, 0))

par1.axis["right"].toggle(all=True)
par2.axis["right"] = new_fixed_axis(loc="right",
                                    axes=par2,
                                    offset=(offset+10, 0))

par2.axis["right"].toggle(all=True)

par3.axis["right"] = new_fixed_axis(loc="right",
                                    axes=par3,
                                    offset=(2*offset, 0))

par3.axis["right"].toggle(all=True)

host.set_xlabel('Time [-]')
host.set_ylabel('Y Value')
par1.set_ylabel('Computation Speed [-]')
par2.set_ylabel('Number Slow Modes')
par3.set_ylabel('CSP Stiffness')

par2.set_yticks([0, 1, 2, 3, 4])

if problem == 'Oregonator':
    host.set_yscale('log')

# host.title("CSP Toy Problem")
if problem == 'CSPtest':
    host.set_xscale('log')
par3.set_yscale('log')
#ymin = min(CSPstiffness)
#par3.set_ylim(ymin,2)
if problem == 'Oregonator':
    p1, = host.plot(ts, Y2s, label='Y1')
else:
    p1, = host.plot(ts, Y1s, label='Y1')
# p2, = host.plot(ts, Y2s, label='Y2')
if problem == 'CSPtest':
    p3, = host.plot(ts, Y3s, label='Y3')
    p4, = host.plot(ts, Y4s, label='Y4')
# p5, = par1.plot(ts_timing, comptimes, label='Comp Speed, dt=1e-4')
p6, = par2.plot(ts, Ms, label='M', linestyle='--')
p7, = par3.plot(ts, CSPstiffness, label='CSP Stiffness', linestyle=':')

host.legend(bbox_to_anchor=(0.42, 0.4), fontsize='x-small', markerscale=3)

#         plt.xlabel(plotxlabel + ' Values')
#         plt.ylabel('Integration Times')
#         ax.grid(b=True, which='both')
#         plt.tight_layout()
#         plt.savefig(output_folder + xlabels[i] + '_' + 'Int_Times_' + dts[t] +
#                     '.' + figformat, dpi=600)

plt.show()
