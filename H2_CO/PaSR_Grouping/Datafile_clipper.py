# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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


dt = 1.e-6
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

data = loadpasrdata(9)

#ratiovals = ratiovals[:, 0]
#CEMAvals = CEMAvals[:, 0]

#np.save(data_folder + CEMAfilename, CEMAvals)
#np.save(data_folder + ratiofilename, ratiovals)