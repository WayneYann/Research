# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import os as os


def loadpasrdata():
    """Load the initial conditions from the full PaSR file."""
    print('Loading data...')
    filepath = os.path.join(os.getcwd(), 'ch4_full_pasr_data.npy')
    return np.load(filepath)


dt = 1.e-8
output_folder = 'Output_Plots/'
data_folder = 'Output_Data/'
equation = 'PaSR_Autoignition'
ratiofilename = equation + '_Stiffness_Ratio_' + str(dt)
CEMAfilename = equation + '_CEMA_' + str(dt)
ratiovals = np.load(os.path.join(os.getcwd(),
                                 data_folder +
                                 ratiofilename +
                                 '.npy'))
CEMAvals = np.load(os.path.join(os.getcwd(),
                                data_folder +
                                CEMAfilename +
                                '.npy'))

print(np.shape(ratiovals))
print(np.shape(CEMAvals))

data = loadpasrdata()

ratiovals = ratiovals[:, 0]
CEMAvals = CEMAvals.reshape(len(CEMAvals[:,0]))

np.save(data_folder + CEMAfilename, CEMAvals)
np.save(data_folder + ratiofilename, ratiovals)

print(np.shape(ratiovals))
print(np.shape(CEMAvals))
