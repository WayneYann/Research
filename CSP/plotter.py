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
#import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
from StiffnessFuncs import *
from CSPfuncs import *
import sys as sys
from matplotlib.ticker import NullFormatter

#plt.ioff()

problem = 'VDP'

[ts, ts_timing, Ms, comptimes, CSPstiffness, Y1s, Y2s, Y3s, Y4s, sol, ratios,
    indicators, CEMs] = [[] for i in range(13)]

with open(problem + '.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        try:
            ts.append(float(row[0]))
            comptimes.append(float(row[1]))
            Ms.append(float(row[2]))
            CSPstiffness.append(float(row[3]))
            ratios.append(float(row[4]))
            indicators.append(float(row[5]))
            CEMs.append(float(row[6]))
            sol.append([float(row[i]) for i in range(7,len(row))])
        except ValueError:
            ts.append(complex(row[0]).real)
            comptimes.append(complex(row[1]).real)
            Ms.append(complex(row[2]).real)
            CSPstiffness.append(complex(row[3]).real)
            ratios.append(complex(row[4]).real)
            indicators.append(complex(row[5]).real)
            CEMs.append(complex(row[6]).real)
            sol.append([complex(row[i]).real for i in range(7,len(row))])

ts = np.array(ts)
ts_timing = np.array(ts_timing)
Ms = np.array(Ms)
comptimes = np.array(comptimes)
CSPstiffness = np.array(CSPstiffness)
sol = np.array(sol)
ratios = np.array(ratios)
indicators = np.array(indicators)
CEMs = np.array(CEMs)

# Calculating values of the stiffness index here for convenince
noRHSparam = False
if problem == 'CSPtest':
    derivfun = testfunc
    jacfun = testjac
    RHSparam = 1.0e-2
elif problem == 'VDP':
    derivfun = dydxvdp
    jacfun = jacvdp
    noRHSparam = True
elif problem == 'Oregonator':
    derivfun = oregonatordydt
    jacfun = oregonatorjac
    noRHSparam = True
elif problem == 'H2':
    derivfun = firstderiv
    jacfun = jacobval
    particle = 877
    timestep = 865
    pasr = loadpasrdata(problem)
    Y = pasr[timestep, particle, :].copy()
    Y, RHSparam = rearrangepasr(Y, problem, False)
elif problem == 'GRIMech':
    derivfun = firstderiv
    jacfun = jacobval
    particle = 230761
    pasr = loadpasrdata(problem)
    Y = pasr[particle, :].copy()
    Y, RHSparam = rearrangepasr(Y, problem, False)

if noRHSparam:
    indexes = stiffnessindex(ts, sol, derivfun, jacfun)
else:
    indexes = stiffnessindex(ts, sol, derivfun, jacfun, RHSparam)

for i in range(10):
    plt.figure(i)
    plt.clf()
plt.close('all')

figformat = 'png'
output_folder = 'Output_Plots/'

# plt.figure(num=None, figsize=(8, 4.5))
# host = host_subplot(111, axes_class=AA.Axes)
# plt.subplots_adjust(right=0.70)
#
# par1 = host.twinx()
# par2 = host.twinx()
# par3 = host.twinx()
#
# offset = 60
# new_fixed_axis = par1.get_grid_helper().new_fixed_axis
# new_fixed_axis = par2.get_grid_helper().new_fixed_axis
# new_fixed_axis = par3.get_grid_helper().new_fixed_axis
# par1.axis["right"] = new_fixed_axis(loc="right",
#                                     axes=par1,
#                                     offset=(0, 0))
#
# par1.axis["right"].toggle(all=True)
# par2.axis["right"] = new_fixed_axis(loc="right",
#                                     axes=par2,
#                                     offset=(offset+10, 0))
#
# par2.axis["right"].toggle(all=True)
#
# par3.axis["right"] = new_fixed_axis(loc="right",
#                                     axes=par3,
#                                     offset=(2*offset, 0))
#
# par3.axis["right"].toggle(all=True)
#
# host.set_xlabel('Time [-]')
# host.set_ylabel('Y Value')
# par1.set_ylabel('Computation Speed [-]')
# par2.set_ylabel('Number Slow Modes')
# par3.set_ylabel('CSP Stiffness')
#
# par2.set_yticks([0, 1, 2, 3, 4])
#
# if problem == 'Oregonator':
#     host.set_yscale('log')
#
# # host.title("CSP Toy Problem")
# if problem == 'CSPtest':
#     host.set_xscale('log')
# par3.set_yscale('log')
# #ymin = min(CSPstiffness)
# #par3.set_ylim(ymin,2)
# if problem == 'Oregonator':
#     p1, = host.plot(ts, Y2s, label='Y1')
# else:
#     p1, = host.plot(ts, Y1s, label='Y1')
# # p2, = host.plot(ts, Y2s, label='Y2')
# if problem == 'CSPtest':
#     p3, = host.plot(ts, Y3s, label='Y3')
#     p4, = host.plot(ts, Y4s, label='Y4')
# # p5, = par1.plot(ts_timing, comptimes, label='Comp Speed, dt=1e-4')
# p6, = par2.plot(ts, Ms, label='M', linestyle='--')
# p7, = par3.plot(ts, indicators, label='CSP Stiffness', linestyle=':')
#
# host.legend(bbox_to_anchor=(0.42, 0.4), fontsize='x-small', markerscale=3)
#
# #         plt.xlabel(plotxlabel + ' Values')
# #         plt.ylabel('Integration Times')
# #         ax.grid(b=True, which='both')
# #         plt.tight_layout()
# #         plt.savefig(output_folder + xlabels[i] + '_' + 'Int_Times_' + dts[t] +
# #                     '.' + figformat, dpi=600)

posCEM = [i for i in CEMs if i > 0]
if posCEM:
    CEMmin = min(posCEM)
    CEMmax = max(posCEM)

f, axarr = plt.subplots(4, 2, sharex='col', figsize=(9.0, 6.0))
#plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.suptitle('Plots for ' + problem)
f.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off',
                right='off')
if problem == 'CSPtest':
    showsols = 4
    axarr[0,0].set_xscale('log')
    axarr[0,1].set_xscale('log')
    plt.xlabel('x Value')
elif problem == 'Oregonator':
    showsols = 3
    axarr[0,0].set_yscale('log')
    plt.xlabel('Time')
else:
    showsols = 1
    plt.xlabel('x Value')
for i in range(showsols):
    lab = 'Y' + str(i+1)
    axarr[0,0].plot(ts, sol[:, i], label=lab)
if problem == 'CSPtest' or problem == 'Oregonator':
    axarr[0,0].legend(loc='upper left', bbox_to_anchor=(1,1),
                      fontsize='x-small')
axarr[0,0].set_title('Solution')
axarr[0,1].plot(ts, Ms)
axarr[0,1].set_title('CSP Fast Modes')
axarr[1,0].plot(ts, CSPstiffness)
axarr[1,0].set_title('CSP Stiffness')
axarr[1,0].set_yscale('log')
if max(CSPstiffness) > 1.0:
    axarr[1,0].set_ylim(min(CSPstiffness)*0.6,3.0)
axarr[1,1].plot(ts, indicators)
axarr[1,1].set_title('Stiffness Indicator')
axarr[2,0].plot(ts[:-5], indexes[:-5])
axarr[2,0].set_title('Stiffness Index')
axarr[2,0].set_yscale('log')
axarr[2,1].plot(ts, ratios)
axarr[2,1].set_title('Stiffness Ratio')
axarr[2,1].set_yscale('log')
axarr[3,0].plot(ts, CEMs)
axarr[3,0].set_title('Chemical Explosive Mode')
if posCEM:
    axarr[3,0].set_ylim(CEMmin*0.5, CEMmax*10.0)
    axarr[3,0].set_yscale('log')

# Fine-tune figure; make subplots farther from each other.
f.subplots_adjust(hspace=0.3)
f.subplots_adjust(wspace=0.3)

plt.savefig(problem + '.png', dpi=600)

plt.show()
