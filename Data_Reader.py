#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:09:32 2017

@author: alfermaa
"""

import pylab as pyl
import numpy as np

filename = 'IndexVals_1e-07'
[indexvalues, tlist, solution, pasrtimestep, particle, order, dt,
 abserr, relerr]  = np.load(filename+'.npy')

tstart, tstop = tlist[0], tlist[-1]

for p in range(1,3):
    pyl.figure(p, figsize=(6, 4.5), dpi=600)
    pyl.xlabel('Time (s)', fontsize=14)
    pyl.xlim(tstart,tstop)
    pyl.grid(True)
    pyl.hold(True)

#Set the linewidth to make plotting look nicer
lw = 1

# Set all of the parameters that we want to apply to each plot specifically.
pyl.figure(1)
pyl.ylabel('Temperature (K)', fontsize=14)
pyl.plot(tlist, solution[:,0], linewidth=lw)
#for i in range(1,len(solution[0,:-1])):
#    if i != 5:
#        pyl.plot(tlist, solution[:,i], linewidth=lw)
#pyl.title('Temperature Graph, Time={}, Particle={}'.format(
#            pasrtimestep,particle), fontsize=16)
pyl.xlim(0,0.005)
pyl.savefig(filename+'_temp_NT.png')

pyl.figure(2)
pyl.ylabel('Stiffness Index Value', fontsize=14)
pyl.plot(tlist, indexvalues, 'b', linewidth=lw)
#pyl.title('Stiffness Index, Order = {}'.format(order), fontsize=16)
pyl.yscale('log')
pyl.xlim(0, 0.1)
pyl.ylim(1.e5,1.e17)
#pyl.text(0,0.999,'dt = {}, Abs Error = {}, Rel Error = {}'.format(
#            dt,abserr,relerr))
pyl.savefig(filename+'_0_NT.png')

pyl.show()