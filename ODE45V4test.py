#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 09:47:08 2017

@author: alfermaa
"""

import numpy as np
import pylab as pyl


def RKexplicitstep():
    pass


def ode45v4(function, t0, tfinal, y0, **keyword_parameters):
    """A copy of the ode45v4 program provided in MTH552"""
    # Coefficient table
    alpha = np.array([.25, .375, 12/13, 1., .5])
    beta  = np.array([ [    1.,     0.,     0.,    0.,     0.,   0.],
                       [    3.,     9.,     0.,    0.,     0.,   0.],
                       [ 1932., -7200.,  7296.,    0.,     0.,   0.],
                       [ 8341.,-32832., 29440., -845.,     0.,   0.],
                       [-6080., 41040.,-28352., 9295., -5643.,   0.]])
    gamma = np.array([ [902880., 0., 3953664., 3855735., -1371249., 277020.],
                       [ -2090., 0.,   22528.,   21970.,   -15048., -27360.]])
    beta[0] = beta[0] * .25
    beta[1] = beta[1] * .03125
    beta[2] = beta[2] / 2197.
    beta[3] = beta[3] / 4104.
    beta[4] = beta[4] / 20520.
    gamma[0] = gamma[0] / 7618050.
    gamma[1] = gamma[1] / 752400.
    beta = np.transpose(beta)
    gamma = np.transpose(gamma)

    # Initialization
    if ('tol' in keyword_parameters):
        tol = keyword_parameters['tol']
    else:
        tol = 1.e-6
    t = t0
    hmax = (tfinal - t) * .0625
    h = hmax * .125
    try:
        y = y0[:]
    except TypeError:
        y = np.array([y0])
    f = np.zeros((len(y), 6))
    chunk = 128
    tout = np.zeros((chunk,1))
    yout = np.zeros((chunk,len(y)))
    k = 0
    tout[k] = t
    yout[k,:] = np.transpose(y)
    power = 0.2

    # Main loop
    while ((t < tfinal) and (t + h > t)):
        # Makes the next step smaller if it would overshoot tfinal
        if (t + h) > tfinal:
            h = tfinal - t

        # Compute the slopes
        temp = function(t, y)
        f[:,0] = temp[:]
        for i in range(5):
            temp = function((t + alpha[i]*h), (y+h*np.dot(f,beta[:,i])))
            f[:,i] = temp[:]

        # Estimate the error and the acceptable error
        delta = np.linalg.norm(h*f*gamma[:,1], np.inf)
        tau = tol*max(np.linalg.norm(y, np.inf),1.0)

        # Update the solution only if the error is acceptable
        if delta <= tau:
            t += h
            y += h*np.dot(f,gamma[:,1])
            k += 1
            if k > len(tout):
                tout = np.concatenate([tout, np.zeros((chunk,1))])
                yout = np.concatenate([yout, np.zeros((chunk,len(y)))])
            tout[k-1] = t
            yout[k-1] = np.transpose(y)

        # Update the step size
        if delta != 0.0:
            h = min(hmax, 0.8*h*(tau/delta)**power)

    if t < tfinal:
        print('Singularity likely.')
        print(t)

    return tout[:k], yout[:k,:]


def Oregonator(x,y):
    k = [1.34, 1.6e9, 8.e3, 4.e7, 1.0]
    f = np.zeros(np.shape(y))
    f[0] = -k[0]*y[0]*y[1] - k[2]*y[0]*y[2]
    f[1] = -k[0]*y[0]*y[1] - k[1]*y[1]*y[2] + k[4]*y[4]
    f[2] = k[0]*y[0]*y[1] - k[1]*y[1]*y[2] + k[2]*y[0]*y[2] - 2.*k[3]*y[2]**2.
    f[3] = k[1]*y[1]*y[2] - k[3]*y[2]**2.
    f[4] = k[2]*y[0]*y[2] - k[4]*y[4]
    return f

t0 = 0
tfinal = 1
U0 = [0.06, 3.3e-7, 5.01e-11, 0.03, 2.4e-8]
tolerance = 1.e-12

t, U = ode45v4(Oregonator, t0, tfinal, U0, tol=tolerance)

for i in range(len(U[0,:])):
    pyl.figure(i)
    pyl.plot(t, U[:,i])
    pyl.title('C' + str(i + 1))
    #pyl.yscale('log')

pyl.show()
