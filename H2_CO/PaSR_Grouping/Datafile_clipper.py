# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import os as os

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

#ratiovals = ratiovals[:, 0]
#CEMAvals = CEMAvals[:, 0]

#np.save(data_folder + CEMAfilename, CEMAvals)
#np.save(data_folder + ratiofilename, ratiovals)