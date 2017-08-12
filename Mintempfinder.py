#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 23:55:53 2017

@author: andrewalferman
"""

import numpy as np
import os as os


def loadpasrdata(num):
    """Load the initial conditions from the PaSR files."""
    pasrarrays = []
    print('Loading data...')
    for i in range(num):
        filepath = os.path.join(os.getcwd(),
                                'pasr_out_h2-co_' +
                                str(i) +
                                '.npy')
        filearray = np.load(filepath)
        pasrarrays.append(filearray)
    return np.concatenate(pasrarrays, 1)


ic = loadpasrdata(9)
print(np.shape(ic))

coords = (99999, 99999)

dudlist = ((162, 91), (264, 91), (378, 91), (255, 91))

mintemp = 999999.

for i in range(len(ic[:, 0, 0])):
    for j in range(len(ic[0, :, 0])):
        if ic[i, j, 1] < mintemp:
            if len([k for k in dudlist if (i, j) == k]) == 0:
                if ic[i, j, 1] > 900.:
                    if j >= 600 and j < 700:
                        mintemp = ic[i, j, 1]
                        coords = (i, j)

print('min temp: {}'.format(mintemp))
print('coords: {}'.format(coords))
