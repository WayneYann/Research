# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import numpy as np
#import matplotlib.pyplot as plt
#import scipy as sci
#
#
#def fun(x):
#    return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
#            0.5 * (x[1] - x[0])**3 + x[1]]
#
#
#def jac(x):
#    return np.array([[1 + 1.5 * (x[0] - x[1])**2,
#                       -1.5 * (x[0] - x[1])**2],
#                      [-1.5 * (x[1] - x[0])**2,
#                       1 + 1.5 * (x[1] - x[0])**2]])
#
#sol = sci.optimize.root(fun, [0., 0.], jac=jac, method='hybr')
#
#print(sol.x)

import numpy as np
import scipy as sci
import pylab as pyl
import matplotlib.font_manager as fnt


def vectorfield(w, t, p):
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
        w :  vector of the state variables:
                  w = [x1,y1,x2,y2]
        t :  time
        p :  vector of the parameters:
                  p = [m1,m2,k1,k2,L1,L2,b1,b2]
    """
    x1, y1, x2, y2 = w
    m1, m2, k1, k2, L1, L2, b1, b2 = p

    # Create f = (x1',y1',x2',y2'):
    f = [y1,
         (-b1 * y1 - k1 * (x1 - L1) + k2 * (x2 - x1 - L2)) / m1,
         y2,
         (-b2 * y2 - k2 * (x2 - x1 - L2)) / m2]
    return f


# Parameter values
# Masses:
m1 = 1.0
m2 = 4.0
# Spring constants
k1 = 88.0
k2 = 80.0
# Natural lengths
L1 = 0.5
L2 = 1.0
# Friction coefficients
b1 = 0.8
b2 = 0.5

# Initial conditions
# x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
x1 = 0.1
y1 = 0.0
x2 = 2.25
y2 = 0.0

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 10.0
numpoints = 250

# Create the time samples for the output of the ODE solver.
# I use a large number of points, only because I want to make
# a plot of the solution that looks nice.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

# Pack up the parameters and initial conditions:
p = [m1, m2, k1, k2, L1, L2, b1, b2]
w0 = [x1, y1, x2, y2]

# Call the ODE solver.
wsol = sci.integrate.odeint(vectorfield, w0, t, args=(p,),
              atol=abserr, rtol=relerr)

x1sol = []
x2sol = []
for i in range(len(wsol)):
    x1sol.append(wsol[i][0])
    x2sol.append(wsol[i][2])

# Plot the solution
pyl.xlabel('t')
pyl.grid(True)
pyl.hold(True)
lw = 1

pyl.figure(1, figzize=(6, 4.5))
pyl.xlabel
pyl.plot(t, x1sol, 'b', linewidth=lw)
pyl.plot(t, x2sol, 'g', linewidth=lw)
pyl.title('Mass Displacements')
pyl.show()