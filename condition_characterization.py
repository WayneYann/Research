#! /usr/bin/env python

#condition characteriser.py
#reports how close the "average" condition in our databases
#are to adiabatic flame temperature

import numpy as np
import cantera as ct
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import plot_styles as ps
import os
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.rc('text.latex', unicode=True)
plt.rc('text.latex',
    preamble=r'\usepackage{amsmath},\usepackage[version=3]{mhchem}')
plt.rc('font', family='serif')

npart = 100

home = './'
out_dir = './PaSR_Characterization/'
mech_dir = os.path.expanduser('~/mechs')
mechs = {'H2' : {'gas' : 'h2.cti', 'fuel' : 'H2', 'size' : 16},
         'CH4' : {'gas' : 'grimech30.cti', 'fuel' : 'CH4', 'size' : 56}
         }

large_font = 20
#check for valid models
for directory in [d for d in os.listdir(home) if
                os.path.isdir(os.path.join(home, d))
                and any(mech in d for mech in mechs)]:


    mech = mechs[directory]
    gas = ct.Solution(os.path.join(mech_dir, mechs[directory]['gas']))
    #check directories of valid models for yaml files

    counter = 0
    arr = np.fromfile(os.path.join(home, directory, 'data.bin'))

    arr = arr.reshape(-1, npart, mech['size'])

    times = np.unique(arr[:, 0, 0])
    per_run = times.shape[0]

    offset = 3
    fuel_ind = gas.species_index(mech['fuel']) + offset
    o2_ind = gas.species_index('O2') + offset

    for i in range(arr.shape[0] / per_run):
        run_offset = i * per_run
        T_ad = arr[run_offset, 0, 1]
        Y_ad = arr[run_offset, 0, fuel_ind]

        ts = np.unique(arr[run_offset:run_offset + per_run, :, 0])
        ys = arr[run_offset:run_offset + per_run, :, fuel_ind].flatten()
        Ts = arr[run_offset:run_offset + per_run, :, 1].flatten()
        fig, ax = plt.subplots()
        def get_lim(array, log=False):
            minv = np.min(array)
            maxv = np.max(array)
            if log:
                minv = np.floor(np.log10(minv))
                maxv = np.ceil(np.log10(maxv))
            return minv, maxv

        xmin, xmax = get_lim(ys, True)
        ymin, ymax = get_lim(Ts)

        #plt.gca().set_xscale('log')
        hb = plt.hexbin(ys, Ts, xscale='log', bins='log',
                gridsize=75, cmap='Blues', mincnt=1,
                extent=(xmin, xmax, ymin - 50, T_ad + 50))
        cb = fig.colorbar(hb, ax=ax)
        label = cb.set_label('\\math{\\log_{10}\\lvert \\text{count} + 1\\rvert}',
                                fontsize=ps.tick_font_size)
        fuel_str = r'\ce{{{}}} mass fraction'.format(mech['fuel'])
        #plot adiabatic flame temp
        xmin, xmax = plt.xlim()
        xv = np.linspace(xmin, xmax)
        plt.plot(xv, np.ones_like(xv) * T_ad, 'k--')
        factor = 0.3 if directory == 'CH4' else 0.4
        plt.text(np.power(10, factor * (np.log10(xmax) + np.log10(xmin))), T_ad - 150, r'$\text{T}_{\text{ad}}$',
            fontsize=large_font, color='k')
        #fiddle with ticks
        for tl in ax.get_xticklines() + ax.get_yticklines():
            tl.set_color('w')
        #set limits
        plt.xlim([xmin, xmax])
        plt.ylim([None, T_ad + 50])
        plt.xlabel(fuel_str)
        plt.ylabel('Temperature (K)')
        ps.finalize()
        plt.savefig('{}{}dist_{}.pdf'.format(out_dir, directory, str(i)))
        plt.close()

        #now do the average temperature
        plt.plot(ts, np.mean(arr[run_offset:run_offset + per_run, :, 1], axis=1))
        plt.ylabel(r'Average Temperature (K)')
        plt.xlabel('time (s)')
        xmin, xmax = np.min(times), np.max(times)
        xv = np.linspace(xmin, xmax)
        plt.plot(xv, np.ones_like(xv) * T_ad, 'k--')
        ymin, ymax = plt.ylim()
        ydiff = 0.075 * (ymax - ymin)
        plt.text(0.6 * (xmax + xmin), T_ad - ydiff, r'$\text{T}_{\text{ad}}$',
            fontsize=large_font)
        plt.xlim([xmin, xmax])
        plt.ylim([None, ymax + ydiff])
        ps.finalize()
        plt.savefig('{}{}tbar_{}.pdf'.format(out_dir, directory, str(i)))
        plt.close()