#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:30 2017

A script to check if a file exists within a folder every second, and changes
the name if it does.  Useful if you're accidentally saving two files with the
same name in a long computation.

Make sure to put this script in the same folder that you want to be checking.

@author: andrewalferman
"""

import os as os
import time
from pathlib import Path

filename = 'PaSR_Autoignition_Int_Times_1e-07_08_28.npy'
newfilename = 'PaSR_Autoignition_Int_Times_REVIEW_1e-07_08_28.npy'

currentdir = os.getcwd()
new_file = Path(currentdir + '/' + newfilename)

while time.time() < 1504640790:
    my_file = Path(currentdir + '/' + filename)
    if my_file.exists():
        os.rename(filename, newfilename)
        break
    time.sleep(1)
