#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 7 13:52 2017

@author: andrewalferman
"""

import numpy as np


def backeulerstep(y0, y, x, dx, derivativefunc):
    """Take one step of the Euler method."""
    return y0 + dx * derivativefunc(x, y)


def impeulerstep(RHSfunction, y0, x0, dx, Newtontol):
    """Take advance one timestep using backward Euler and Newton's method."""
    # print('---------------')
    y = y0
    x1 = x0 + dx
    # print('y: {}'.format(y))
    # print('x1: {}'.format(x1))
    y1 = y
    # y1 = Eulerstep(y0, x0, dx, RHSfunction)
    # y = y1 + 2 * Newtontol
    # print('y1: {}'.format(y1))
    # print('---------------')
    for i in range(10000):
        # print('y: {}'.format(y))
        f = backeulerstep(y0, y1, x1, dx, RHSfunction)
        # print('f: {}'.format(f))
        # fprime = (y1 - y) / dx
        # if y1 == y:
        #     fprime = RHSfunction(x1)
        fprime = RHSfunction(x1, y1)
        # print('fprime: {}'.format(fprime))
        y1 = y - (f / fprime)
        # print('y1: {}'.format(y1))
        # print('i: {}'.format(i))
        # print('---------------')
        if abs(y1 - y) < Newtontol:
            print('Within tolerance at y = {}'.format(y1))
            print('Needed {} iterations'.format(i))
            break
        y = y1
    # print('Return: {}'.format(Eulerstep(y0, x1, dx, RHSfunction)))
    # raise Exception('y1: {}'.format(y1))
    return y1


def examplederiv(x, y):
    """Derivative of y = 1/x."""
    return -1/x**2


def examplederiv2(x, y):
    """Derivative of y = (x^2 + 1)^0.5."""
    return x/y


yval = 1.0
t = 0.0
tstop = 3.0
dt = 0.1
tolerance = 1e-12

while t < tstop:
    # print([t, val])
    val = impeulerstep(examplederiv2, yval, t, dt, tolerance)
    t += dt

print('Final value: {}'.format(val))
