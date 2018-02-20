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


ts, ts_timing, Ms, comptimes, CSPstiffness, Y1s, Y2s, Y3s, Y4s = \
    [], [], [], [], [], [], [], [], []
with open('csptoyproblem-vals.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        ts.append(float(row[0]))
        Ms.append(float(row[1]))
        CSPstiffness.append(float(row[3]))
        Y1s.append(float(row[4]))
        Y2s.append(float(row[5]))
        Y3s.append(float(row[6]))
        Y4s.append(float(row[7]))
with open('csptoyproblem.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        ts_timing.append(float(row[0]))
        comptimes.append(float(row[2]))
ts = np.array(ts)
ts_timing = np.array(ts_timing)
Ms = np.array(Ms)
comptimes = np.array(comptimes)
CSPstiffness = np.array(CSPstiffness)
Y1s = np.array(Y1s)
Y2s = np.array(Y2s)
Y3s = np.array(Y3s)
Y4s = np.array(Y4s)

figformat = 'png'
output_folder = 'Output_Plots/'

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.70)

par1 = host.twinx()
par2 = host.twinx()
par3 = host.twinx()

offset = 60
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
new_fixed_axis2 = par3.get_grid_helper().new_fixed_axis
par2.axis["right"] = new_fixed_axis(loc="right",
                                    axes=par2,
                                    offset=(30, 0))

par2.axis["right"].toggle(all=True)

par3.axis["right"] = new_fixed_axis(loc="right",
                                    axes=par3,
                                    offset=(90, 0))

par3.axis["right"].toggle(all=True)

host.set_xlabel('Time [-]')
host.set_ylabel('Y Value')
par1.set_ylabel('Computation Speed [-]')
par2.set_ylabel('Number Slow Modes')
par3.set_ylabel('CSP Stiffness')
par2.set_yticks([0, 1, 2, 3, 4])

# host.title("CSP Toy Problem")
host.set_xscale('log')
par3.set_yscale('log')
p1, = host.plot(ts, Y1s, label='Y1')
p2, = host.plot(ts, Y2s, label='Y2')
p3, = host.plot(ts, Y3s, label='Y3')
p4, = host.plot(ts, Y4s, label='Y4')
p5, = par2.plot(ts, Ms, label='M', linestyle='--')
p6, = par3.plot(ts, CSPstiffness, label='CSP Stiffness', linestyle=':')
p7, = par1.plot(ts_timing, comptimes, label='Comp Speed, dt=1e-4')

host.legend(bbox_to_anchor=(0.3, 0.4), fontsize='x-small', markerscale=3)

#         plt.xlabel(plotxlabel + ' Values')
#         plt.ylabel('Integration Times')
#         ax.grid(b=True, which='both')
#         plt.tight_layout()
#         plt.savefig(output_folder + xlabels[i] + '_' + 'Int_Times_' + dts[t] +
#                     '.' + figformat, dpi=600)

plt.show()
