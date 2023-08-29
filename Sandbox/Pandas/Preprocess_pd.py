'''
This is a Pandas reimplementation of cBprocess() in bakR.
Includes both the functionality of reliableFeatures() and cBprocess().
'''

import pandas as pd
import numpy as np
import os 

# Load cB
os.chdir('C:\\Users\\isaac\\Documents\\baknpy\\Data')


cB = pd.read_csv("cB.csv.gz")

### Only keep rows with non-ambiguous XF

# Rows of interest
cB_XF = cB[["sample", "XF", "TC", "nT", "n"]]

# No __no_feature or __ambiguous
cB_XF = cB_XF[~cB_XF["XF"].str.contains("__")]

# Add together read count values for identicial rows
cB_XF = cB_XF.groupby(
    ['sample', 'XF', 'TC', 'nT']
    ).agg(
        {'n': 'sum'}
    ).reset_index()
