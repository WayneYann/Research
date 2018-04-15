#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 6 12:14 2018

@author: andrewalferman
"""

import os as os
import numpy as np
import pyjacob as pyjacob


def derivcd4(vals, dx):
    """Take the derivative of a series using 4th order central differencing.

    Given a list of values at equally spaced points, returns the first
    derivative using the fourth order central difference formula, or forward/
    backward differencing at the boundaries.
    """
    deriv = []
    for i in range(2):
        deriv.append((-3 * vals[i] + 4 * vals[i + 1] - vals[i + 2]) / (2 * dx))
    for i in range(2, len(vals) - 2):
        deriv.append(((-1 * vals[i + 2]) + (8 * vals[i + 1]) -
                     (8 * vals[i - 1]) + vals[i - 2]) /
                     (12 * dx)
                     )
    for i in range((len(vals) - 2), len(vals)):
        deriv.append((3 * vals[i] - 4 * vals[i - 1] + vals[i - 2]) / 2 * dx)
    return deriv


def weightednorm(matrix, weights):
    """Weighted average norm function as defined in 1985 Shampine.

    Takes a matrix and 2 weights and returns the maximum value (divided by
    wi) of the sum of each value in each row multiplied by wj.  Needs to be
    passed either a matrix of m x n dimensions where m,n > 1, or a column
    vector.
    """
    # Unpack the parameters
    wi, wj = weights

    # Initialize a list that will be called later to obtain the maximum value
    colsums = np.zeros(len(matrix))

    # Try loop used because numpy didn't seem to like 1D arrays for the
    # weighted norm
    try:
        for i in range(len(matrix)):
            colsums += wj * np.abs(matrix[i])
        return np.max(colsums) / wi
    except TypeError:
        matrixcol = wj * np.abs(matrix)
        return np.sum(matrixcol) / wi


def stiffnessindex(xlist, solution, dfun, jfun, *args, **kwargs):
    """Determine the stiffness index across a solution vector.

    Function that uses stiffness parameters, the local Jacobian matrix,
    and a vector of the local function values to determine the local stiffness
    index as defined in 1985 Shampine.
    """
    SIparams = {'method': 2,
                'gamma': 1,
                'xi': 1,
                'order': 1,
                'tolerance': 1,
                'wi': 1,
                'wj': 1
                }

    for key, value in kwargs.items():
        SIparams[key] = value

    funcparams = []
    for arg in args:
        funcparams.append(arg)

    # Method 2 uses the weighted norm of the Jacobian, Method 1 uses the
    # spectral radius of the Jacobian.
    method = SIparams['method']
    # Stiffness index parameter values
    gamma = SIparams['gamma']
    xi = SIparams['xi']
    order = SIparams['order']
    tolerance = SIparams['tolerance']
    # Weighted norm parameters
    wi = SIparams['wi']
    wj = SIparams['wj']

    normweights = wi, wj

    # Obtain the derivative values for the derivative of order p
    dx = xlist[1] - xlist[0]
    dydxlist = []
    for i in range(len(solution)):
        dydxlist.append(dfun(xlist[i], solution[i, :], funcparams[0]))
    # Raise the derivative to the order we need it
    for i in range(order):
        dydxlist = derivcd4(dydxlist, dx)
    dydxlist = np.array(dydxlist)

    # Create a list to return for all the index values in a function
    indexlist = []

    # Figure out some of the values that will be multiplied many times, so that
    # each computation only needs to happen once.
    exponent = 1. / (order + 1)
    xiterm = ((np.abs(xi)**(-1 * exponent)) / np.abs(gamma))
    toleranceterm = tolerance**exponent

    # Actual computation of the stiffness index for the method specified.
    for i in range(len(solution)):
        jacobian = jfun(xlist[i], solution[i, :], funcparams[0])
        if method == 1:
            eigenvalues = np.linalg.eigvals(jacobian)
            index = toleranceterm *\
                np.max(np.abs(eigenvalues)) *\
                weightednorm(dydxlist[i, :], normweights)**(-1 * exponent) *\
                xiterm
        else:
            index = toleranceterm *\
                weightednorm(jacobian, normweights) *\
                weightednorm(dydxlist[i, :], normweights)**(-1 * exponent) *\
                xiterm
        indexlist.append(index)
    indexlist = np.array(indexlist)
    return indexlist  # , dydxlist


def stiffnessindicator(time, solution, jfun, *args):
    """
    Find the local stiffness indicator after calculating local solution.

    Given the value of the solution, find the stiffness indicator as defined by
    Soderlind 2013.
    """
    funcparams = []
    for arg in args:
        funcparams.append(arg)

    try:
        jacobian = jfun(time, solution, funcparams[0])
    except ValueError:
        jacobian = jfun(time[0], solution[0], funcparams[0])
    Hermitian = 0.5 * (jacobian + np.transpose(jacobian))
    eigvals = np.linalg.eigvals(Hermitian)
    return 0.5 * (min(eigvals) + max(eigvals))


def reftimescale(indicatorval, Tlen):
    """
    Find the local reference timescale for the stiffness indicator.

    Given the stiffness indicator values as defined by Soderlind 2013, finds
    the reference time scale.
    """
    if indicatorval >= 0:
        timescale = Tlen
    else:
        timescale = min(Tlen, -1/indicatorval)
    return timescale


def CEMA(xlist, solution, jfun, *args):
    """
    Find values for the chemical explosive mode analysis.

    Same thing as finding the maximum eigenvalue across the solution.
    """
    funcparams = []
    for arg in args:
        funcparams.append(arg)

    values = []
    try:
        for i in range(len(solution)):
            jacobian = jfun(xlist[i], solution[i], funcparams[0])
            values.append(max(np.linalg.eigvals(jacobian)))
    except TypeError:
        jacobian = jfun(xlist, solution, funcparams[0])
        values.append(max(np.linalg.eigvals(jacobian)))
    return values


def stiffnessratio(xlist, solution, jfun, *args):
    """
    Find values of the stiffness ratio.

    Ratio of the eigenvalue with the largest absolute value over the eigenvalue
    with the smallest absolute value. Ignores eigenvalues of zero.
    """
    funcparams = []
    for arg in args:
        funcparams.append(arg)

    values = []
    try:
        for i in range(len(solution)):
            jacobian = jfun(xlist[i], solution[i], funcparams[0])
            eigvals = np.array([abs(j) for j in np.linalg.eigvals(jacobian)
                                if j != 0])
            values.append(max(eigvals)/min(eigvals))
    except TypeError:
        jacobian = jfun(xlist, solution, funcparams[0])
        eigvals = np.array([abs(j) for j in np.linalg.eigvals(jacobian)
                            if j != 0])
        values.append(max(eigvals)/min(eigvals))
    return values


def stiffmetrics(xlist, solution, jfun, *args):
    """
    Find values of the stiffness ratio, stiffness indicator, and CEMA.
    """
    if args:
        funcparams = []
        for arg in args:
            funcparams.append(arg)

    ratios, indicators, CEMs = [], [], []
    if isinstance(xlist, float):
        if args:
            jacobian = jfun(xlist, solution, funcparams[0])
        else:
            jacobian = jfun(xlist, solution)
        eigvals1 = np.array([abs(j) for j in np.linalg.eigvals(jacobian)
                            if j != 0])
        Hermitian = 0.5 * (jacobian + np.transpose(jacobian))
        eigvals2 = np.linalg.eigvals(Hermitian)
        ratios = (max(eigvals1)/min(eigvals1))
        indicators = 0.5 * (min(eigvals2) + max(eigvals2))
        CEMs = max(np.linalg.eigvals(jacobian))
    else:
        for i in range(len(solution)):
            if args:
                jacobian = jfun(xlist[i], solution[i], funcparams[0])
            else:
                jacobian = jfun(xlist[i], solution[i])
            eigvals1 = np.array([abs(j) for j in np.linalg.eigvals(jacobian)
                                if j != 0])
            Hermitian = 0.5 * (jacobian + np.transpose(jacobian))
            eigvals2 = np.linalg.eigvals(Hermitian)
            ratios.append(max(eigvals1)/min(eigvals1))
            indicators.append(0.5 * (min(eigvals2) + max(eigvals2)))
            CEMs.append(max(np.linalg.eigvals(jacobian)))
    return ratios, indicators, CEMs
