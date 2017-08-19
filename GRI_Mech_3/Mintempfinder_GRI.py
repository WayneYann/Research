#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 23:55:53 2017

@author: andrewalferman
"""

import numpy as np
import os as os


def loadpasrdata():
    """Load the initial conditions from the full PaSR file."""
    print('Loading data...')
    filepath = os.path.join(os.getcwd(), 'ch4_full_pasr_data.npy')
    return np.load(filepath)


ic = loadpasrdata()
print(np.shape(ic))

coords = (99999999, 99999999)

dudlist = ()

mintemp = 999999.

for i in range(len(ic[:, 0])):
    if ic[i, 1] < mintemp:
        if len([k for k in dudlist if i == k]) == 0:
            if ic[i, 1] > 875.:
                mintemp = ic[i, 1]
                coords = (i)

print('min temp: {}'.format(mintemp))
print('coords: {}'.format(coords))
