#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:54:13 2017

@author: andrewalferman
"""

import pyjac as pyj
import pyjacob as pyjacob
import pylab as pyl
import scipy as sci

"""Module for testing function (accuracy) of pyJac.
"""

# Python 2 compatibility
#from __future__ import division
#from __future__ import print_function

# Standard libraries
import os
import re
import sys
import subprocess
import pickle
from argparse import ArgumentParser
import multiprocessing
import glob

# Related modules
import numpy as np

try:
    import cantera as ct
    from cantera import ck2cti
except ImportError:
    print('Error: Cantera must be installed.')
    raise

# Compiler based on language
cmd_compile = dict(c='gcc',
                   cuda='nvcc',
                   fortran='gfortran'
                   )

# Flags based on language
flags = dict(c=['-std=c99'],
             cuda=['-arch=sm_20',
                   '-I/usr/local/cuda/include/',
                   '-I/usr/local/cuda/samples/common/inc/',
                   '-dc'],
             fortran='')

libs = dict(c=['-lm', '-std=c99'],
            cuda='-arch=sm_20',
            fortran='')


class ReactorConstPres(object):
    """Object for constant pressure ODE system.
    """
    def __init__(self, gas):
        """
        Parameters
        ----------
        gas : `cantera.Solution`
            `cantera.Solution` object with the kinetic system

        Returns
        -------
        None

        """
        self.gas = gas
        self.P = gas.P

    def __call__(self):
        """Return the ODE function, y' = f(t,y)

        Parameters
        ----------
        None

        Returns
        -------
        `numpy.array` with dT/dt + dY/dt

        """
        # State vector is [T, Y_1, Y_2, ... Y_K]
        rho = self.gas.density

        wdot = self.gas.net_production_rates
        dTdt = - (np.dot(self.gas.partial_molar_enthalpies, wdot) /
                  (rho * self.gas.cp)
                  )
        dYdt = wdot * self.gas.molecular_weights / rho

        return np.hstack((dTdt, dYdt))


class AutodiffJacob(object):
    """Class for
    """
    def __init__(self, pressure, fwd_spec_map):
        """Initialize autodifferentation object.

        Parameters
        ----------
        pressure : float
            Pressure in Pascals
        fwd_spec_map : `numpy.array`
            Map of original species indices to new indices

        Returns
        -------
        None

        """
        self.pres = pressure
        self.fwd_spec_map = fwd_spec_map
        import adjacob
        self.jac = adjacob

    def eval_jacobian(self, gas):
        """Evaluate finite difference Jacobian using Adept \
        autodifferentiation package.

        Parameters
        ----------
        gas : `cantera.Solution`
            `cantera.Solution` object with the kinetic system

        Returns
        -------
        jacob : `numpy.array`
            Jacobian matrix evaluated using autodifferentiation

        """

        y = np.hstack((gas.T, gas.Y[self.fwd_spec_map][:-1]))

        jacob = np.zeros((gas.n_species * gas.n_species))

        self.jac.ad_eval_jacobian(0, gas.P, y, jacob)
        return jacob


def run_pasr(pasr_input_file, mech_filename, pasr_output_file=None):
    """Run PaSR simulation to get thermochemical data for testing.

    Parameters
    ----------
    pasr_input_file : str
        Name of PaSR input file in YAML format
    mech_filename : str
        Name of Cantera-format mechanism file
    pasr_output_file : str
        Optional; filename for saving PaSR output data

    Returns
    -------
    state_data : ``numpy.array``
        Array with state data (time, temperature, pressure, mass fractions)

    """
    # Run PaSR to get data
    pasr_input = pasr.parse_input_file(pasr_input_file)
    state_data = pasr.run_simulation(
                    mech_filename,
                    pasr_input['case'],
                    pasr_input['temperature'],
                    pasr_input['pressure'],
                    pasr_input['equivalence ratio'],
                    pasr_input['fuel'],
                    pasr_input['oxidizer'],
                    pasr_input['complete products'],
                    pasr_input['number of particles'],
                    pasr_input['residence time'],
                    pasr_input['mixing time'],
                    pasr_input['pairing time'],
                    pasr_input['number of residence times']
                    )
    if pasr_output_file:
        np.save(pasr_output_file, state_data)

    return state_data



class cpyjac_evaluator(object):
    """Class for pyJac-based Jacobian matrix evaluator
    """
    def __copy(self, B):
        """Copy NumPy array into new array
        """
        A = np.empty_like(B)
        A[:] = B
        return A

    def check_numbers(self, build_dir, gas, filename='mechanism.h'):
        """Ensure numbers of species and forward reaction match.

        Parameters
        ----------
        build_dir : str
            Path of directory for build objects
        gas : `cantera.Solution`
            Object with kinetic system
        filename : str
            Optional; filename of header with mechanism information

        Returns
        -------
        None

        """
        with open(os.path.join(build_dir, filename), 'r') as file:
            n_spec = None
            n_reac = None
            for line in file.readlines():
                if n_spec is None:
                    match = re.search(r'^#define NSP (\d+)$', line)
                    if match:
                        n_spec = int(match.group(1))

                if n_reac is None:
                    match = re.search(r'^#define FWD_RATES (\d+)$', line)
                    if match:
                        n_reac = int(match.group(1))

                if n_spec is not None and n_reac is not None:
                    break

        if n_spec != gas.n_species:
            print('Error: species counts do not match between '
                  'mechanism.h and Cantera.'
                  )
            raise
        if n_reac != gas.n_reactions:
            print('Error: forward reaction counts do not match between '
                  'mechanism.h and Cantera.'
                  )
            raise


    def check_optimized(self, build_dir, gas, filename='mechanism.h'):
        """Check if pyJac files were cache-optimized (and thus rearranged)

        Parameters
        ----------
        build_dir : str
            Path of directory for build objects
        gas : `cantera.Solution`
            Object with kinetic system
        filename : str
            Optional; filename of header with mechanism information

        Returns
        -------
        None

        """
        self.fwd_spec_map = np.array(range(gas.n_species))
        with open(os.path.join(build_dir, filename), 'r') as file:
            opt = False
            last_spec = None
            for line in file.readlines():
                if 'Cache Optimized' in line:
                    opt = True
                match = re.search(r'^//last_spec (\d+)$', line)
                if match:
                    last_spec = int(match.group(1))
                if opt and last_spec is not None:
                    break

        self.last_spec = last_spec
        self.cache_opt = opt
        self.dydt_mask = np.array([0] + [x + 1 for x in range(gas.n_species)
                                  if x != last_spec]
                                  )
        if self.cache_opt:
            with open(os.path.join(build_dir, 'optimized.pickle'), 'rb') as file:
                dummy = pickle.load(file)
                dummy = pickle.load(file)
                self.fwd_spec_map = np.array(pickle.load(file))
                self.fwd_rxn_map = np.array(pickle.load(file))
                self.back_spec_map = np.array(pickle.load(file))
                self.back_rxn_map = np.array(pickle.load(file))
        elif last_spec != gas.n_species - 1:
            #still need to treat it as a cache optimized
            self.cache_opt = True

            (self.fwd_spec_map,
             self.back_spec_map
             ) = utils.get_species_mappings(gas.n_species, last_spec)

            self.fwd_spec_map = np.array(self.fwd_spec_map)
            self.back_spec_map = np.array(self.back_spec_map)

            self.fwd_rxn_map = np.array(range(gas.n_reactions))
            self.back_rxn_map = np.array(range(gas.n_reactions))
        else:
            self.fwd_spec_map = list(range(gas.n_species))
            self.back_spec_map = list(range(gas.n_species))
            self.fwd_rxn_map = np.array(range(gas.n_reactions))
            self.back_rxn_map = np.array(range(gas.n_reactions))

        #assign the rest
        n_spec = gas.n_species
        n_reac = gas.n_reactions

        self.fwd_dydt_map = np.array([0] + [x + 1 for x in self.fwd_spec_map])

        self.fwd_rev_rxn_map = np.array([i for i in self.fwd_rxn_map
                                        if gas.reaction(i).reversible]
                                        )
        rev_reacs = self.fwd_rev_rxn_map.shape[0]
        self.back_rev_rxn_map = np.sort(self.fwd_rev_rxn_map)
        self.back_rev_rxn_map = np.array(
                                    [np.where(self.fwd_rev_rxn_map == x)[0][0]
                                    for x in self.back_rev_rxn_map]
                                    )
        self.fwd_rev_rxn_map = np.array(
                                    [np.where(self.back_rev_rxn_map == x)[0][0]
                                    for x in range(rev_reacs)]
                                    )

        self.fwd_pdep_map = [self.fwd_rxn_map[i] for i in range(n_reac)
                             if is_pdep(gas.reaction(self.fwd_rxn_map[i]))
                             ]
        pdep_reacs = len(self.fwd_pdep_map)
        self.back_pdep_map = sorted(self.fwd_pdep_map)
        self.back_pdep_map = np.array([self.fwd_pdep_map.index(x)
                                      for x in self.back_pdep_map]
                                      )
        self.fwd_pdep_map = np.array([np.where(self.back_pdep_map == x)[0][0]
                                     for x in range(pdep_reacs)]
                                     )

        self.back_dydt_map = np.array([0] +
                                      [x + 1 for x in self.back_spec_map]
                                      )


    def __init__(self, build_dir, gas, module_name='pyjacob',
                 filename='mechanism.h'
                 ):
        self.check_numbers(build_dir, gas, filename)
        self.check_optimized(build_dir, gas, filename)
        self.pyjac = __import__(module_name)

    def eval_conc(self, temp, pres, mass_frac, conc):
        """Evaluate species concentrations at current state.

        Parameters
        ----------
        temp : float
            Temperature, in Kelvin
        pres : float
            Pressure, in Pascals
        mass_frac : ``numpy.array``
            Species mass fractions
        conc : ``numpy.array``
            Species concentrations, in kmol/m^3

        Returns
        -------
        None

        """
        mw_avg = 0
        rho = 0
        if self.cache_opt:
            test_mass_frac = self.__copy(mass_frac[self.fwd_spec_map])
            self.pyjac.py_eval_conc(temp, pres, test_mass_frac,
                                    mw_avg, rho, conc
                                    )
            conc[:] = conc[self.back_spec_map]
        else:
            self.pyjac.py_eval_conc(temp, pres, mass_frac, mw_avg, rho, conc)


    def eval_rxn_rates(self, temp, pres, conc, fwd_rates, rev_rates):
        """Evaluate reaction rates of progress at current state.

        Parameters
        ----------
        temp : float
            Temperature, in Kelvin
        pres : float
            Pressure, in Pascals
        conc : ``numpy.array``
            Species concentrations, in kmol/m^3
        fwd_rates : ``numpy.array``
            Reaction forward rates of progress, in kmol/m^3/s
        rev_rates : ``numpy.array``
            Reaction reverse rates of progress, in kmol/m^3/s

        Returns
        -------
        None

        """
        if self.cache_opt:
            test_conc = self.__copy(conc[self.fwd_spec_map])
            self.pyjac.py_eval_rxn_rates(temp, pres, test_conc,
             fwd_rates, rev_rates)
            fwd_rates[:] = fwd_rates[self.back_rxn_map]
            if self.back_rev_rxn_map.size:
                rev_rates[:] = rev_rates[self.back_rev_rxn_map]
        else:
            self.pyjac.py_eval_rxn_rates(temp, pres, conc,
                                         fwd_rates, rev_rates
                                         )


    def get_rxn_pres_mod(self, temp, pres, conc, pres_mod):
        """Evaluate reaction rate pressure modifications at current state.

        Parameters
        ----------
        temp : float
            Temperature, in Kelvin
        pres : float
            Pressure, in Pascals
        conc : ``numpy.array``
            Species concentrations, in kmol/m^3
        pres_mod : ``numpy.array``
            Reaction rate pressure modification

        Returns
        -------
        None

        """
        if self.cache_opt:
            test_conc = self.__copy(conc[self.fwd_spec_map])
            self.pyjac.py_get_rxn_pres_mod(temp, pres, test_conc, pres_mod)
            pres_mod[:] = pres_mod[self.back_pdep_map]
        else:
            self.pyjac.py_get_rxn_pres_mod(temp, pres, conc, pres_mod)


    def eval_spec_rates(self, fwd_rates, rev_rates, pres_mod, spec_rates):
        """Evaluate species overall production rates at current state.

        Parameters
        ----------
        fwd_rates : ``numpy.array``
            Reaction forward rates of progress, in kmol/m^3/s
        rev_rates : ``numpy.array``
            Reaction reverse rates of progress, in kmol/m^3/s
        pres_mod : ``numpy.array``
            Reaction rate pressure modification
        spec_rates : ``numpy.array``
            Reaction reverse rates of progress, in kmol/m^3/s

        Returns
        -------
        None

        """
        if self.cache_opt:
            test_fwd = self.__copy(fwd_rates[self.fwd_rxn_map])
            if self.fwd_rev_rxn_map.size:
                test_rev = self.__copy(rev_rates[self.fwd_rev_rxn_map])
            else:
                test_rev = self.__copy(rev_rates)
            if self.fwd_pdep_map.size:
                test_pdep = self.__copy(pres_mod[self.fwd_pdep_map])
            else:
                test_pdep = self.__copy(pres_mod)
            self.pyjac.py_eval_spec_rates(test_fwd, test_rev,
                                          test_pdep, spec_rates
                                          )
            spec_rates[:] = spec_rates[self.back_spec_map]
        else:
            self.pyjac.py_eval_spec_rates(fwd_rates, rev_rates,
                                          pres_mod, spec_rates
                                          )


    def dydt(self, t, pres, y, dydt):
        """Evaluate derivative

        Parameters
        ----------
        t : float
            Time, in seconds
        pres : float
            Pressure, in Pascals
        y : ``numpy.array``
            State vector (temperature + species mass fractions)
        dydt : ``numpy.array``
            Derivative of state vector

        Returns
        -------
        None

        """
        if self.cache_opt:
            test_y = self.__copy(y[self.fwd_dydt_map])
            test_dydt = np.zeros_like(test_y)
            self.pyjac.py_dydt(t, pres, test_y, test_dydt)
            dydt[self.dydt_mask] = test_dydt[self.back_dydt_map[self.dydt_mask]]
        else:
            self.pyjac.py_dydt(t, pres, y, dydt)


    def eval_jacobian(self, t, pres, y, jacob):
        """Evaluate the Jacobian matrix

        Parameters
        ----------
        t : float
            Time, in seconds
        pres : float
            Pressure, in Pascals
        y : ``numpy.array``
            State vector (temperature + species mass fractions)
        jacob : ``numpy.array``
            Jacobian matrix

        Returns
        -------
        None

        """
        if self.cache_opt:
            test_y = self.__copy(y[self.fwd_dydt_map][:])
            self.pyjac.py_eval_jacobian(t, pres, test_y, jacob)
        else:
            self.pyjac.py_eval_jacobian(t, pres, y, jacob)


    def update(self, index):
        """Updates evaluator index

        Parameters
        ----------
        index : int
            Index of data for evaluating quantities

        Returns
        -------
        None

        """
        self.index = index


    def clean(self):
        pass


class tchem_evaluator(cpyjac_evaluator):
    """Class for TChem-based Jacobian matrix evaluator
    """
    def __init__(self, build_dir, gas, state_data, mechfile, thermofile,
                 module_name='py_tchem', filename='mechanism.h'
                 ):
        self.tchem = __import__(module_name)

        if thermofile == None:
            thermofile = mechfile

        # TChem needs array of full species mass fractions
        self.y_mask = np.array([0] + [x + 2 for x in range(gas.n_species)])
        def czeros(shape):
            arr = np.zeros(shape)
            return arr.flatten(order='c')
        def reshaper(arr, shape, reorder=None):
            arr = arr.reshape(shape, order='c').astype(np.dtype('d'), order='c')
            if reorder is not None:
                arr = arr[:, reorder]
            return arr

        states = state_data[:, 1:]
        num_cond = states.shape[0]
        #init vectors
        test_conc = czeros((num_cond, gas.n_species))
        test_fwd_rates = czeros((num_cond,gas.n_reactions))
        test_rev_rates = czeros((num_cond,gas.n_reactions))
        test_spec_rates = czeros((num_cond,gas.n_species))
        test_dydt = czeros((num_cond, gas.n_species + 1))
        test_jacob = czeros((num_cond, (gas.n_species) * (gas.n_species)))

        pres = states[:, 1].flatten(order='c')
        y_dummy = states[:, [x for x in self.y_mask]
                         ].flatten(order='c').astype(np.dtype('d'), order='c')

        self.tchem.py_eval_jacobian(mechfile, thermofile, num_cond,
                                    pres, y_dummy, test_conc, test_fwd_rates,
                                    test_rev_rates, test_spec_rates,
                                    test_dydt, test_jacob
                                    )

        #reshape for comparison
        self.test_conc = reshaper(test_conc, (num_cond, gas.n_species))
        self.test_fwd_rates = reshaper(test_fwd_rates,
                                       (num_cond, gas.n_reactions)
                                       )
        self.test_rev_rates = reshaper(test_rev_rates,
                                       (num_cond, gas.n_reactions)
                                       )
        self.test_spec_rates = reshaper(test_spec_rates,
                                        (num_cond, gas.n_species)
                                        )
        self.test_dydt = reshaper(test_dydt, (num_cond, gas.n_species + 1))
        self.test_jacob = reshaper(test_jacob, (num_cond,
                                   (gas.n_species) * (gas.n_species))
                                   )
        self.index = 0

    def get_conc(self, conc):
        """Evaluate species concentrations at current state.

        Parameters
        ----------
        conc : ``numpy.array``
            Species concentrations, in kmol/m^3

        Returns
        -------
        None

        """
        conc[:] = self.test_conc[self.index, :]


    def get_rxn_rates(self, fwd_rates, rev_rates):
        """Evaluate reaction rates of progress at current state.

        Parameters
        ----------
        fwd_rates : ``numpy.array``
            Reaction forward rates of progress, in kmol/m^3/s
        rev_rates : ``numpy.array``
            Reaction reverse rates of progress, in kmol/m^3/s

        Returns
        -------
        None

        """
        fwd_rates[:] = self.test_fwd_rates[self.index, :]
        rev_rates[:] = self.test_rev_rates[self.index, :]


    def get_spec_rates(self, spec_rates):
        """Evaluate species overall production rates at current state.

        Parameters
        ----------
        spec_rates : ``numpy.array``
            Reaction reverse rates of progress, in kmol/m^3/s

        Returns
        -------
        None

        """
        spec_rates[:] = self.test_spec_rates[self.index, :]


    def get_dydt(self, dydt):
        """Evaluate derivative

        Parameters
        ----------
        dydt : ``numpy.array``
            Derivative of state vector

        Returns
        -------
        None

        """
        dydt[:] = self.test_dydt[self.index, :-1]


    def get_jacobian(self, jacob):
        """Evaluate the Jacobian matrix

        Parameters
        ----------
        jacob : ``numpy.array``
            Jacobian matrix

        Returns
        -------
        None

        """
        jacob[:] = self.test_jacob[self.index, :]



def derivcd4(vals, dx):
    """Given a list of values at equally spaced points, returns the first
    derivative using the fourth order central difference formula, or forward/
    backward differencing at the boundaries."""
    deriv = []
    for i in range(len(vals)):
        try:
            deriv.append((-1*vals[i-2] + 8*vals[i-1] + 8*vals[i+1] -\
                          vals[i+2]) / 12*dx)
        except IndexError:
            try:
                deriv.append((-3*vals[i] + 4*vals[i+1] - vals[i+2]) / 2*dx)
            except IndexError:
                deriv.append((3*vals[i] - 4*vals[i-1] + vals[i-2]) / 2*dx)
    return deriv


def weightednorm(matrix, weights):
    """Weighted average norm function as defined in 1985 Shampine.  Takes a
    matrix and 2 weights and returns the maximum value (divided by wi) of the
    sum of each value in each row multiplied by wj."""
    # Unpack the parameters
    wi, wj = weights

    # Initialize a list that will be called later to obtain the maximum value
    ivalues = []

    matrix = np.array(matrix)

    # A few try statements are used to figure out the shape of the matrix
    # Try statements are used because the code would otherwise return an
    # exception if the matrix is one dimensional.  The shape of the matrix is
    # needed to iterate across the rows and columns later.
    try:
        num_rows, num_columns = matrix.shape
        dimensions = 2
    except ValueError:
        dimensions = 1
        try:
            num_rows = 1
            num_columns = matrix.shape[0]
        except IndexError:
            num_rows = matrix.shape[1]
            num_columns = 1
    # Sums up the values across each of the columns and applies the weights,
    # then finds the maximum value (the weighted matrix norm) after the weights
    # have been applied.
    for i in range(num_columns):
        columnsum = 0.
        for j in range(num_rows):
            if dimensions == 2:
                columnsum += np.abs(matrix[j][i]) * wj
            else:
                columnsum += np.abs(matrix[i]) * wj
        ivalues.append(columnsum / wi)
    return np.max(ivalues)


def stiffnessindex(sp, jacobian, derivativevals, normweights):
    '''Function that uses stiffness parameters (sp), the local Jacobian matrix,
    and a vector of the local function values to determine the local stiffness
    index as defined in 1985 Shampine'''
    # Method 1 uses the weighted norm of the Jacobian, Method 2 uses the
    # spectral radius of the Jacobian.
    method = 2

    # Unpack the parameters
    tolerance, order, xi, gamma = sp

    # The second derivative normally will come in one row for this program,
    # however we want to be taking the weighted norm of the second derivative
    # values in one column instead.  The derivative values must then be
    # transposed.  Should try to make this smarter by checking the number of
    # rows/columns before transposing.
    np.asarray(derivativevals).T.tolist()

    if method == 1:
        exponent = 1./(order + 1)
        index = tolerance**exponent *\
            weightednorm(jacobian, normweights) *\
             weightednorm(derivativevals, normweights)**exponent *\
             ((np.abs(xi)**exponent) / np.abs(gamma))
    else:
        exponent = 1./(order + 1)
        index = tolerance**exponent *\
            np.max(np.abs(np.linalg.eigvals(jacobian))) *\
             weightednorm(derivativevals, normweights)**exponent *\
             ((np.abs(xi)**exponent) / np.abs(gamma))
    return index

# Stiffness index parameter values to be sent to the stiffness index function
gamma = 1.
xi = 1.
order = 1
tolerance = 1.
stiffnessparams = tolerance, order, xi, gamma

# Weighted norm parameters to be sent to the weighted norm function
wi = 1.
wj = 1.
normweights = wi, wj

## Define the range of the computation
tstart = 0
tstop = 0.01
dt = 1.e-3
tlist = np.arange(tstart, tstop+dt, dt)

t0 = 0.050

# ODE Solver parameters
abserr = 1.0e-10
relerr = 1.0e-8

# Load the initial conditions from the pasr file
ic = np.load('/Users/andrewalferman/Desktop/Research/pasr_out_h2-co_0.npy')

# Call the integrator
#solution = sci.integrate.ode(pyjacob.py_dydt, pyjacob.py_eval_jacobian).\
#                            set_integrator('zvode', method='bdf')
#solution.set_initial_value(ic[500,0,:], t0)
#t1 = 0.100
#while solution.successful() and solution.t < t1:
#    print(solution.t + dt, solution.integrate(solution.t+dt))
solution = sci.integrate.odeint(pyjacob.py_dydt, # Call the dydt function
                                # Pass it initial conditions
                                ic[500,0,1:],
                                # Pass it time steps to be evaluated
                                tlist,
                                # Pass whatever additional information is needed
                                args=(solution,tchem_evaluator.get_dydt(pyjacob.py_dydt)),
                                # Pass it the Jacobian (not sure if needed)
                                Dfun=pyjacob.py_eval_jacobian,
                                # Pass it the absolute and relative tolerances
                                atol=abserr, rtol=relerr,
                                # Print a message stating if it worked or not
                                printmessg=1
                                )





#print(pyjacob.py_dydt(0.05,2418.195016625938,ic[500,0,:],solution))

## Obtain the derivative values
#for i in solution:
#    firstderiv = pyjacob.py_dydt(initconditions)
#    jacobian = pyjacob.py_eval_jacobian(initconditions)
#
#secondderiv = derivcd4(firstderiv)
#
## Find the stiffness index across the range of the solution by using the above
## functions to get the Jacobian matrix and second derivative
#indexvalues = []
#for i in solution:
#    localstiffness = stiffnessindex(stiffnessparams, jacobian[i],
#                                    secondderiv[i], normweights)
#    indexvalues.append(localstiffness)
#
## Plot the solution.  This loop just sets up some of the parameters that we
## want to modify in all of the plots.
#for p in range(1, 3):
#    pyl.figure(p, figsize=(6, 4.5), dpi=400)
#    pyl.xlabel('x Value')
#    pyl.grid(True)
#    pyl.hold(True)
#
##Set the linewidth to make plotting look nicer
#lw = 1
#
## Set all of the parameters that we want to apply to each plot specifically.
#pyl.figure(1)
#pyl.ylabel('y1 Value')
#pyl.plot(tlist, solution[:,0], 'b', linewidth=lw)
#pyl.title('Temperature Graph')
#
#pyl.figure(2)
#pyl.ylabel('Index Value')
#pyl.plot(tlist, indexvalues, 'b', linewidth=lw)
#pyl.title('IA-Stiffness Index, Order = {}'.format(order))
#pyl.yscale('log')
#
#pyl.show()
