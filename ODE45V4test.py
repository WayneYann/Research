#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 09:47:08 2017

@author: alfermaa
"""

import numpy as np
import pylab as pyl
import scipy as sci


def forwardeulercoeff():
    A = np.array([0])
    b = np.array([1])
    c = np.array([0])
    return A, b, c


def RK4coeff():
    A = np.array([[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,1,0]])
    b = np.array([1/6,1/3,1/3,1/6])
    c = np.array([0,0.5,0.5,1])
    return A, b, c


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
        temp = function(y,t)
        f[:,0] = temp[:]
        for i in range(5):
            temp = function((y+h*np.dot(f,beta[:,i])), (t + alpha[i]*h))
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


def RKexplicitstep(function, t, U0, dt, A, b, c):
    """A copy of the general RK explicit step method provided in MTH552"""
    r = len(b)
    m = len(U0)
    K = np.zeros((m,r))
    K[:,0] = function(U0, t)
    for j in range(1,r):
        Y = U0 + np.dot(dt*K[:,0:j],np.transpose(A[j,0:j]))
        K[:,j] = function(Y, t + c[j]*dt)
    return U0 + dt*np.dot(K,b)


def orbitODE(y,t):
    mu = 0.012277471
    muhat = 1. - mu
    d1 = (((y[0] + mu)**2.) + (y[1]**2.))**1.5
    d2 = (((y[0] - muhat)**2.) + (y[1]**2.))**1.5
    yprime = np.zeros_like(y)
    yprime[0] = y[2]
    yprime[1] = y[3]
    yprime[2] = y[0] + 2.*y[3] - muhat*((y[0]+mu)/d1) - mu*((y[0]-muhat)/d2)
    yprime[3] = y[1] - 2.*y[2] - muhat*(y[1]/d1) - mu*(y[1]/d2)
    return yprime


def Oregonator(y,x):
    k = [1.34, 1.6e9, 8.e3, 4.e7, 1.0]
    f = np.zeros(np.shape(y))
    f[0] = -k[0]*y[0]*y[1] - k[2]*y[0]*y[2]
    f[1] = -k[0]*y[0]*y[1] - k[1]*y[1]*y[2] + k[4]*y[4]
    f[2] = k[0]*y[0]*y[1] - k[1]*y[1]*y[2] + k[2]*y[0]*y[2] - 2.*k[3]*y[2]**2
    f[3] = k[1]*y[1]*y[2] - k[3]*y[2]**2
    f[4] = k[2]*y[0]*y[2] - k[4]*y[4]
    return f


def exactsoltest(t):
    """Note that this function only applies when U0 = 5"""
    return 5 * np.exp(-0.5*t)


def testproblem(y,t):
    return -0.5*y

t0 = 0
tfinal = 1
#U0 = [0.06, 3.3e-7, 5.01e-11, 0.03, 2.4e-8]
tolerance = 1.e-4
U0 = np.array([5.])
nstep = int(2e7)
dt = (tfinal - t0) / nstep
t = np.arange(t0, tfinal + .5*dt, dt)




t, U = ode45v4(testproblem, t0, tfinal, U0, tol=tolerance)

"""

A, b, c = RK4coeff()
U0 = np.array([0.994,0.,0.,-2.00158510637908252240537862224])
#U0 = np.array([5])
tol = 3.e-3

t0 = 0
tfinal = 171.
tspan = [t0, tfinal]
dt = (tfinal - t0) / nstep
m = len(U0)
U = np.zeros((m,nstep+1))
U[:,0] = U0
t = np.arange(t0, tfinal + .5*dt, dt)

exact = [exactsoltest(i) for i in t]

for i in range(int(nstep)):
    U[:,i+1] = ode45v4(orbitODE,t0,tfinal,U0,tol=tol)

pyl.plot(U[0,:], U[1,:])
#pyl.plot(t, U[0,:])
#pyl.plot(t, exact)



solution = sci.integrate.odeint(Oregonator, # Call the dydt function
                                # Pass it initial conditions
                                U0,
                                # Pass it time steps to be evaluated
                                t
                                # Pass whatever additional information is needed
                                #args=(Y_press,),
                                # Pass it the Jacobian (not sure if needed)
#                                Dfun=jacobval,
                                # Pass it the absolute and relative tolerances
                                #atol=abserr, rtol=relerr,
                                # Print a message stating if it worked or not
                                #printmessg=0
                                )

#pyl.plot(solution[:,1], solution[:,1])

"""

for i in range(len(U[0,:])):
    pyl.figure(i)
    pyl.plot(t, U[:,i])
    pyl.title('C' + str(i + 1))
    #pyl.yscale('log')

pyl.show()

