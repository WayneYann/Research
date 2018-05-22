#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg
import math as mth
import sys as sys
import os as os
import pyjacob as pyjacob


def RK4(func, y0, t, h, Q, qflag):
    """Perform a single step of RK4.

    Currently just configured for the test problem
    """
    # Should turn Q, qflag, eps into optional arguments to be more general
    Y1 = y0
    Y2 = y0 + 0.5 * h * func(t, Y1, Q, qflag)
    Y3 = y0 + 0.5 * h * func(t + h * 0.5, Y2, Q, qflag)
    Y4 = y0 + h * func(t + h * 0.5, Y3, Q, qflag)
    y = y0 + (h / 6.0) * (func(t, Y1, Q, qflag)
                          + 2 * func(t + 0.5 * h, Y2, Q, qflag)
                          + 2 * func(t + 0.5 * h, Y3, Q, qflag)
                          + func(t + h, Y4, Q, qflag)
                         )
    t += h
    return t, y


def jacvdp(x, y, *eta):
    """Find the local Jacobian matrix of the Van der Pol equation."""
    # Note that the dummy value is just because I originally set this up
    # for the test problem and thought I'd be making a lot of changes to eps

    # Change this parameter to modify the stiffness of the problem.
    if eta:
        eta = eta[0]
    else:
        eta = 1.0E3
    eta = 1.0E3

    return np.array([[0., 1.], [-1. - 2*y[0]*y[1]*eta, eta-eta*y[0]**2]])


def dydxvdp(x, y, *eta):
    """Find the local vector of the first derivative of the Van der Pol eqn."""

    # Change this parameter to modify the stiffness of the problem.
    if eta:
        eta = eta[0]
    else:
        eta = 1.0E3
    eta = 1.0E3

    # Unpack the y vector
    y1 = y[0]
    y2 = y[1]

    # Create dydx vector (y1', y2')
    f = np.array([y2, eta*y2 - y1 - eta*y2*y1**2.])

    return f


def d2ydx2vdp(x, y):
    """Find the local vector of the 2nd derivative of the Van der Pol eqn."""

    # Change this parameter to modify the stiffness of the problem.
    eta = 1

    # Unpack the y vector
    y1 = y[0]
    y2 = y[1]

    # Create vector of the second derivative
    y2prime = eta*y2 - y1 - eta*y2*y1**2.
    f = np.array([y2prime, eta*y2prime - y2 - 2*eta*y1*y2 - eta*y2prime*y1**2])
    return f


def oregonatordydt(tim, y, *eps):
    """Find the local vector of the first derivative of the Oregonator."""

    # Stiffness factor - change this here if you want the problem to change
    if eps:
        s = eps[0]
        q = eps[1]
        w = eps[2]
    else:
        s = 77.27
        q = 8.375E-6
        w = 0.161

    #print(qflag)

    ydot = np.empty(3)

    NN = len(y)

    ydot[0] = s * (y[0] - y[0] * y[1] + y[1] - (q * y[0]**2))
    ydot[1] = (y[2] - y[1] - y[0] * y[1]) / s
    ydot[2] = w * (y[0] - y[2])

    return np.array(ydot)

def oregonatorjac(tim, y, *eps):
    """Find the local vector of the first derivative of the Oregonator."""
    # Note that the dummy value is just because I originally set this up
    # for the test problem and thought I'd be making a lot of changes to eps
    if eps:
        s = eps[0]
        q = eps[1]
        w = eps[2]
    else:
        s = 77.27
        q = 8.375E-6
        w = 0.161

    row1 = [s * (-1 * y[1] + 1 - q * 2 * y[0]), s * (1 - y[0]), 0]
    row2 = [-1 * y[1] / s, (-1 - y[0])/s, 1/s]
    row3 = [w, 0, -1 * w]

    return np.array([row1, row2, row3])


def testfunc(tim, y, eps):
    """Derivative (dydt) source term for model problem.

    param[in]  tim     the time (sec)
    param[in]  y       the data array, size neq
    param[in]  Q       slow-manifold projector matrix, size NN*NN
    param[in]  qflag   projector flag (1 to use)
    return     ydot    derivative array, size neq
    """

    NN = len(y)

    ydot = np.ones(NN)

    for i in range(NN - 1):
        sumterm = 0.0
        for j in range(i, NN - 1):
            sumterm += (y[j+1] / ((1 + y[j+1])**2))
        ydot[i] = ((1 / (eps**(NN - (i+1))))
                   * (-1 * y[i] + (y[i+1] / (1 + y[i + 1])))
                   - sumterm)
    ydot[-1] = -1 * y[-1]

    return ydot


def testjac(tim, y, eps):
    """
    Jacobian for model problem.

    param[in]  tim  	the time (sec)
    param[in]  y	 	the data array, size neq
    param[in]  eps     stiffness factor
    return     dfdy	Jacobian, size (neq,neq)
    """

    dfdy = np.empty(len(y)**2)

    dfdy[0]  = -1.0 / (eps**3)
    dfdy[1]  = 0.0
    dfdy[2]  = 0.0
    dfdy[3]  = 0.0
    dfdy[4]  = ((1.0 / (eps**3 * (1.0 + y[1])**2))
                + ((y[1] - 1.0) / ((y[1] + 1.0)**3)))
    dfdy[5]  = -1.0 / (eps**2)
    dfdy[6]  = 0.0
    dfdy[7]  = 0.0
    dfdy[8]  = (y[2] - 1.0) / ((y[2] + 1.0)**3)
    dfdy[9]  = ((1.0 / (eps**2 * (1.0 + y[2])**2))
                + ((y[2] - 1.0) / ((y[2] + 1.0)**3)))
    dfdy[10] = -1.0 / eps
    dfdy[11] = 0.0
    dfdy[12] = (y[3] - 1.0) / ((y[3] + 1.0)**3)
    dfdy[13] = (y[3] - 1.0) / ((y[3] + 1.0)**3)
    dfdy[14] = ((1.0 / (eps * (1.0 + y[3])**2))
                + ((y[3] - 1.0) / ((y[3] + 1.0)**3)))
    dfdy[15] = -1.0
    return np.reshape(np.array(dfdy),(4,4))


def firstderiv(time, state, press):
    """Force the integrator to use the right arguments."""
    # Need to make sure that N2 is at the end of the state array
    dy = np.zeros_like(state)
    pyjacob.py_dydt(time, press, state, dy)
    return dy


def jacobval(time, state, press):
    """Force the integrator to use the right arguments."""
    # Need to get rid of N2 because PyJac doesn't compute it.
    # new = state[:-1]
    # print('Jacobian function called.')
    a = len(state)
    jacobian = np.zeros(a**2)
    # Obtain the jacobian from pyJac
    pyjacob.py_eval_jacobian(time, press, state, jacobian)
    jacobian = np.reshape(jacobian, (a, a))
    # Re-add the zeros back in
    # jacobian = np.insert(jacobian, a, np.zeros(a), axis=1)
    # jacobian = np.vstack((jacobian, np.zeros(a+1)))
    return jacobian


def loadpasrdata(problem):
    """Load the initial conditions from the full PaSR file."""
    # print('Loading data...')
    if problem == 'GRIMech':
        filepath = os.path.join(os.getcwd(), '../GRI_Mech_3/ch4_full_pasr_data.npy')
        return np.load(filepath)
    elif problem == 'H2':
        pasrarrays = []
        for i in range(9):
            filepath = os.path.join(os.getcwd(),
                                    '../H2_CO/pasr_out_h2-co_' +
                                    str(i) +
                                    '.npy')
            filearray = np.load(filepath)
            pasrarrays.append(filearray)
        return np.concatenate(pasrarrays, 1)
    else:
        raise Exception('No PaSR data found.')


def rearrangepasr(Y, problem, useN2):
    """Rearrange the PaSR data so it works with pyJac."""
    press_pos = 2
    temp_pos = 1
    if problem == 'GRIMech':
        N2_pos = 50
    elif problem == 'H2':
        N2_pos = 11
    arraylen = len(Y)

    Y_press = Y[press_pos]
    Y_temp = Y[temp_pos]
    Y_species = Y[3:arraylen]
    Y_N2 = Y[N2_pos]
    Ys = np.hstack((Y_temp, Y_species))
    Ys = np.delete(Ys, N2_pos - 2, 0)
    Ys = np.hstack((Ys, Y_N2))

    if useN2:
        initcond = Ys
    else:
        initcond = Ys[:-1]
    return initcond, Y_press


def radical_correction(ydot, Q):
    """
    Function that applies radical correction to data array

    Performs matrix-vector multiplication of radical correction tensor
    with data array.

    param[in]  tim  time at new time step (sec)
    param[in]  y    array of data already integrated, size NN
    param[in]  Rc   Radical correction tensor, size NN*NN
    return     g    array holding radical corrections, size NN
    """
    NN = len(Y)
    for i in range(NN):
        sum_row = 0.0
        for j in range (NN):
            sum_row += Q[j][i] * ydot[j]
        ydotn[i] = sum_row
    return ydotn


def get_slow_projector(tim, y, derivfun, jacfun, CSPtols, *RHSparams):
    """
    Function that performs CSP analysis and returns vectors.

    param[in]   tim   current time (s)
    param[in]   y     array of values at current time, size NN
    param[in]   eps   stiffness factor
    param[out]  Qs    slow-manifold projector matrix, size NN*NN
    param[out]  Rc    radical correction tensor, size NN*NN
    return      taum1 time scale of fastest slow mode (sec)
    return      M     number of slow modes
    """
    if RHSparams:
        RHSparam = RHSparams[0]

    NN = len(y)

    # CSP vectors and covectors
    # a_csp = np.empty((NN, NN))  # Array with CSP vectors
    # b_csp = np.empty((NN, NN))  # Array with CSP covectors

    if RHSparams:
        M, tau, a_csp, b_csp = get_fast_modes(tim, y, derivfun, jacfun, CSPtols, RHSparam)
    else:
        M, tau, a_csp, b_csp = get_fast_modes(tim, y, derivfun, jacfun, CSPtols)

    # Rc starts as a matrix of zeros
    Rc = np.zeros((NN, NN))

    # Qs starts as identity matrix
    Qs = np.identity(NN)

    for j in range(NN):
        for i in range(NN):
            # ensure at least 1 exhausted mode
            if M > 0:
                sum_qs = 0.0
                sum_rc = 0.0

                # sum over slow modes
                # print([a_csp[r][i] * b_csp[r][j] for r in range(M)])
                sum_qs = mth.fsum([a_csp[r][i] * b_csp[r][j] for r in range(M)])
                sum_rc = mth.fsum([(a_csp[r][i] * b_csp[r][j] * tau[r]) for r in range(M)])

                # Qs = Id - sum_r^M a_r*b_r
                Qs[j][i] -= sum_qs

                # Rc = sum_r^M a_r*tau_r*b_r
                Rc[j][i] = sum_rc
    taum1 = abs(tau[M])
    try:
        stiffness = float(abs(tau[0])) / float(taum1)
    except ZeroDivisionError:
        stiffness = 1e99
    if stiffness < 1.0e-50:
        print(tau)
    return M, taum1, Qs, Rc, stiffness


def get_csp_vectors(tim, y, jacfun, *RHSparams):
    """
    Function that performs CSP analysis and returns vectors.

    Performs eigendecomposition of Jacobian to get the eigenvalues (inverse of
    mode timescales), CSP vectors (from right eigenvectors) and CSP covectors
    (from left eigenvectors). Uses LAPACK Fortran subroutines, also
    sorts (based on eigenvalue magnitude) and normalizes eigenvectors.

    Explosive modes (positive eigenvalues) will always be retained, since sorted
    in ascending order (and all others are negative). Complex eigenvalues (with
    associated complex eigenvectors) are handled also by taking the sum and
    difference of the real and imaginary parts of the eigenvectors.

    param[in]   tim     current time (s)
    param[in]   y       array of values at current time, size NN
    param[in]   eps     stiffness factor
    param[in]   jacfun  function of the jacobian
    return      tau     array of CSP mode timescales (s), size NN
    return      a_csp   CSP vectors, size NN*NN
    return      b_csp   CSP covectors, size NN*NN
    """
    if RHSparams:
        RHSparam = RHSparams[0]

    # Initializations that could not be done in c
    ERROR = False
    NN = len(y)
    tau = np.empty_like(y)
    a_csp = np.empty((NN, NN))
    b_csp = np.empty((NN, NN))

    # Obtain the jacobian
    if RHSparams:
        jac = jacfun(tim, y, RHSparam)
    else:
        jac = jacfun(tim, y)

    # Obtain the eigenvalues and left and right eigenvectors of the jacobian
    evale, evecl, evecr = linalg.eig(np.reshape(jac, (NN, NN)), left=True)

    # Split the eigenvectors into real and imaginary parts, as was done before
    evalr = [np.real(e) for e in evale]
    evali = [np.imag(e) for e in evale]

    # Check if there are imaginary eigenvalues or eigenvectors
    # complexflag = False
    # for i in evali:
    #     if abs(i) > np.finfo(float).resolution:
    #         print('Complex eigs detected at t={}'.format(tim))
    #         complexflag = True
    # for i in evecl:
    #     for j in i:
    #         if abs(np.imag(j)) > np.finfo(float).resolution:
    #             print('Complex evecl detected at t={}'.format(tim))
    #             complexflag = True
    # for i in evecr:
    #     for j in i:
    #         if abs(np.imag(j)) > np.finfo(float).resolution:
    #             print('Complex evecr detected at t={}'.format(tim))
    #             complexflag = True
    # if complexflag:
    #     raise Exception('Imaginary values detected.')

    # Sort the eigenvalues
    order = insertion_sort([abs(i) for i in evalr])
    # print(evalr)
    # print(order)

    for i in range(NN):
        try:
            tau[i] = abs(1.0 / float(evalr[order[i]]))  # time scales, inverse of eigenvalues
        except ZeroDivisionError:
            tau[i] = 1.0E99
        for j in range(NN):
            # CSP vectors, right eigenvectors
            a_csp[i][j] = evecr[order[i]][j]
            # CSP covectors, left eigenvectors
            b_csp[i][j] = evecl[order[i]][j]

    # print(tau)
    # print(insertion_sort(tau))
    # eliminate complex components of eigenvectors if complex eigenvalues,
    # and normalize dot products (so that bi*aj = delta_ij).
    flag = 1
    for i in range(NN):
        # check if imaginary part of eigenvalue
        # (skip 2nd eigenvalue of complex pair)
        if abs(evali[i]) > np.finfo(float).resolution and flag == 1:
            # complex eigenvalue

            # normalize
            #needs to be able to handle complex eigenvectors
            sum_r = 0.0  # real part
            sum_i = 0.0  # imaginary part

            for j in range(NN):
                # need to treat left eigenvector as complex conjugate
                # (so take negative of imag part)
                sum_r = mth.fsum([sum_r, (np.real(a_csp[i][j])
                                         * np.real(b_csp[i][j])
                                         + np.imag(a_csp[i][j])
                                         * np.imag(b_csp[i][j])
                                        )])
                sum_i = mth.fsum([sum_i, (np.imag(a_csp[i][j])
                                         * np.real(b_csp[i][j])
                                         - np.real(a_csp[i][j])
                                         * np.imag(b_csp[i][j])
                                        )])

            sum2 = sum_r**2 + sum_i**2

            # print(sum2)
            # ensure sum is not zero
            if abs(sum2) > np.finfo(float).resolution:
                for j in range(NN):
                    # normalize a, and set a1=real, a2=imag
                    a_old = a_csp[i][j]
                    a_csp[i][j] = (np.real(a_old) * sum_r
                                   + np.imag(a_csp[i][j]) * sum_i * 1j) / sum2
                    a_csp[i][j] = (np.imag(a_csp[i][j]) * sum_r
                                   - np.real(a_old) * sum_i * 1j) / sum2

                    # set b1=2*real, b2=-2*imag
                    b_csp[i][j] = 2.0 * b_csp[i][j]

            # skip next (conjugate of current)
            flag = 2

        elif flag == 2:
            # do nothing, this is second of complex pair
            flag = 1
        else:
            # real eigenvalue
            flag = 1

            # More accurate summation
            sumj = mth.fsum([a_csp[i][j] * b_csp[i][j] for j in range(NN)])

            # ensure dot product is not zero
            if abs(sumj) > np.finfo(float).resolution:
                for j in range(NN):
                    # just normalize a
                    a_csp[i][j] /= sumj

    if ERROR:
        # ensure a and b are inverses
        for i in range(NN):
            I = np.empty(NN)
            for j in range(NN):
                I[j] = 0
                for k in range(NN):
                    I[j] += a_csp[j][k] * b_csp[i][k]
                if i == j:
                    if (abs(I[j] - 1.0) > 1.0e-14):
                        print("CSP vectors not orthogonal")
                    else:
                        if (abs(I[j] ) > 1.0e-14):
                            print("CSP vectors not orthogonal")
                # printf("%17.10le ", I[j]);
            #printf("\n");

        # check that new CSP vectors and covectors recover
        # standard base vectors e_i
        for ei in range(NN):
            # fill e_i
            e = np.empty(NN)
            for i in range(NN):
                if ei == i:
                    e[i] = 1.0
                else:
                    e[i] = 0.0

            for i in range(NN):
                # ith component of e
                e_comp = 0.0
                for j in range(NN):
                    # (b.e_i)*a

                    # b.e_i
                    e_sum = 0.0
                    for k in range(NN):
                        e_sum += b_csp[j][k] * e[k]

                    e_sum *= a_csp[j][i]
                    e_comp += e_sum

                # if reconstructed basis not very close to original basis
                if (abs(e[i] - e_comp) > 1.0e-10):
                    print("Error recreating standard basis vectors.")
                    print("e_{:d}, component {:d}, e_comp: {:17.10f} \t error: {:17.10f}".format(ei+1, i, e_comp, abs(e[i] - e_comp)))
    return tau, a_csp, b_csp


def get_fast_modes(tim, y, derivfun, jacfun, CSPtols, *RHSparams):
    """
    Function that returns the number of exhausted modes.

    param[in]   tim     current time (s)
    param[in]   y       array of values at current time, size NN
    param[out]  tau     array of CSP mode timescales (s), size NN
    param[out]  a_csp   CSP vectors, size NN*NN
    param[out]  b_csp   CSP covectors, size NN*NN
    return      M       number of exhausted (fast) modes
    """
    if RHSparams:
        RHSparam = RHSparams[0]

    # Initialize things that weren't initialized in C
    NN = len(y)

    eps_a, eps_r = CSPtols

    # perform CSP analysis to get timescales, vectors, covectors
    if RHSparams:
        tau, a_csp, b_csp = get_csp_vectors(tim, y, jacfun, RHSparam)
    else:
        tau, a_csp, b_csp = get_csp_vectors(tim, y, jacfun)

    # now need to find M exhausted modes
    # first calculate f^i
    f_csp = np.empty(NN)  # f^i is b* operated on g

    Q = np.empty((NN, NN)) # unused projector array
    qflag = 0 # flag telling dydt to not use projector

    # call derivative function
    if RHSparams:
        g_csp = derivfun(tim, y, RHSparam) # array of derivatives (g in CSP)
    else:
        g_csp = derivfun(tim, y) # array of derivatives (g in CSP)

    for i in range(NN):
        # operate b_csp on g

        # More accurate summation
        f_csp[i] = mth.fsum([b_csp[i][j] * g_csp[j] for j in range(NN)])

    M = 0  # start with no slow modes
    mflag = 0
    while M < (NN - 1) and mflag == 0:
        # testing for M = M + 1 slow modes
        # check error
        #y_norm = 0.0
        #err_norm = 0.0
        for i in range(NN):
            #sum_m = 0.0

            # More accurate summation
            sum_m = mth.fsum([a_csp[k][i] * f_csp[k]
                              for k in range(M+1)])

            # if error larger than tolerance, flag
            if abs(tau[M] * sum_m) >= (eps_a + (eps_r * y[i])):
                mflag = 1;

            # ensure below error tolerance and not explosive mode (positive eigenvalue)
            # tau[M+1] is time scale of fastest of slow modes (driving)
            # tau[M] is time scale of slowest exhausted mode (current)

        # add current mode to exhausted if under error tolerance and not explosive mode
        if mflag == 0 and tau[M] < 0.0:
            M += 1  # add current mode to exhausted modes
        else:
            mflag = 1  # explosve mode, stop here

    return M, tau, a_csp, b_csp


def insertion_sort(vals):
    """
    Insertion sort function.

    Performs insertion sort and returns indices in ascending order based on
    input values. Best on very small arrays (n < 20). Based on Numerical Recipes
    in C algorithm.

    param[in]   vals  array of values to be sorted
    return      ords  array of sorted indices

    """
    ##### These didn't work...
    # n = len(vals)
    # ords = np.empty_like(vals)
    #
    # for i in range(n):
    #     ords[i] = i
    #
    # for j in range(1, n):
    #     # pick out one element
    #     val = vals[ords[j]]
    #     ival = ords[j]
    #     i = j - 1;
    #     while (i >= 0 and vals[ords[i]] > val ):
    #         # look where to insert
    #         ords[i + 1] = ords[i]
    #         i -= 1
    #
    #     # insert element
    #     ords[i + 1] = ival;
    # return ords
    # ------------------
    # n = len(vals)
    # ords = np.empty_like(vals)
    #
    # for i in range(n):
    #     ords[i] = i
    #
    # for i in range(1, n):
    #     j = i
    #     val = vals[j]
    #     while i >= 0 and vals[j-1] > val:
    #         # look where to insert
    #         ords[i] = ords[i-1]
    #         i -= 1
    #
    #     # insert element
    #     ords[i + 1] = ival;
    # return ords
    #
    # This one worked, pulled from here:
    # https://stackoverflow.com/questions/5185060/insertion-sort-get-indices
    sorted_data = sorted(enumerate(vals), key=lambda key: key[1])
    indices = list(range(len(vals)))
    indices.sort(key=lambda key: sorted_data[key][0])
    # for i in range(len(vals)):
    #     indices[i] = len(vals) - (indices[i] + 1)
    return indices
