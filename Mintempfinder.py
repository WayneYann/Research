#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 23:55:53 2017

@author: andrewalferman
"""

import numpy as np

ic = np.load('/Users/andrewalferman/Desktop/Research/pasr_out_h2-co_0.npy')

coords = (99999, 99999)

dudlist = ((162,91),(264,91),(378,91),(255,91))

mintemp = 999999.

for i in range(len(ic[:,0,0])):
    for j in range(len(ic[0,:,0])):
        if ic[i,j,1] < mintemp:
            if len([k for k in dudlist if (i,j) == k]) == 0:
                if ic[i,j,1] > 855:
                    mintemp = ic[i,j,1]
                    coords = (i,j)

print('min temp: {}'.format(mintemp))
print('coords: {}'.format(coords))